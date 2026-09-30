#!/usr/bin/env python3
"""
Does negation scope over selectional preferences?
-------------------------------------------------
Scores a typical vs. an atypical filler after an affirmative frame and after a
minimally different negated frame (the frames differ ONLY by the word "not").

  TAXONOMIC  (negation SHOULD flip the preference)
     aff: "A robin is a kind of"       + " bird" / " fish"
     neg: "A robin is not a kind of"   + " bird" / " fish"

  THEMATIC   (negation should NOT flip the preference)
     aff: "The chef was chopping the"      + " onions" / " branches"
     neg: "The chef was not chopping the"  + " onions" / " branches"

For every item and frame we store log P(filler | prefix), summed over the
filler's sub-word tokens. Because the SAME filler strings appear in both frames,
filler frequency and token length cancel out in the negation effect
(delta = pref_neg - pref_aff), which is computed in analyze.py.

Optional --layerwise: applies the "logit lens" to every layer's residual stream
at the last prefix position and records the first-token preference per layer.

Usage
  python run_experiment.py --models gpt2 EleutherAI/pythia-160m --layerwise
"""
import argparse
import os
import re
import time

import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Stimuli
# --------------------------------------------------------------------------- #
def build_trials(stim_dir):
    """Return a list of dicts, one per (item, frame)."""
    trials = []

    tax = pd.read_csv(os.path.join(stim_dir, "taxonomic.csv"))
    for r in tax.itertuples():
        for frame, neg in (("aff", ""), ("neg", " not")):
            trials.append(dict(
                condition="taxonomic", item_id=r.item_id, frame=frame,
                prefix=f"{r.subject_np} is{neg} a kind of",
                typical=r.true_category, atypical=r.false_category))

    them = pd.read_csv(os.path.join(stim_dir, "thematic.csv"))
    for r in them.itertuples():
        for frame, neg in (("aff", ""), ("neg", " not")):
            trials.append(dict(
                condition="thematic", item_id=r.item_id, frame=frame,
                prefix=f"The {r.agent} was{neg} {r.verb_ing} the",
                typical=r.typical, atypical=r.atypical))
    return trials


# --------------------------------------------------------------------------- #
# Tokenisation helpers
# --------------------------------------------------------------------------- #
def encode_prefix(tok, prefix):
    ids = tok(prefix, add_special_tokens=True).input_ids
    # Give the first real token some context: prepend BOS if the tokenizer
    # did not add one (GPT-2 / Pythia). Harmless for scoring the filler.
    if tok.bos_token_id is not None and (len(ids) == 0 or ids[0] != tok.bos_token_id):
        ids = [tok.bos_token_id] + ids
    return ids


def encode_continuation(tok, prefix, cont):
    """Token ids of ' cont' as they appear after `prefix`.

    Tokenise prefix+cont jointly (correct merges at the boundary); fall back to
    separate encoding if the joint encoding does not preserve the prefix.
    """
    p_ids = tok(prefix, add_special_tokens=False).input_ids
    f_ids = tok(prefix + " " + cont, add_special_tokens=False).input_ids
    if f_ids[:len(p_ids)] == p_ids and len(f_ids) > len(p_ids):
        return f_ids[len(p_ids):]
    return tok(" " + cont, add_special_tokens=False).input_ids


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def get_final_norm(model):
    """Final normalisation layer, needed for the logit lens."""
    paths = ["transformer.ln_f",            # GPT-2, GPT-J, BLOOM
             "gpt_neox.final_layer_norm",   # Pythia / GPT-NeoX
             "model.norm",                  # Llama, Mistral, Qwen2, OLMo-2, Gemma
             "model.decoder.final_layer_norm"]  # OPT
    for path in paths:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


@torch.no_grad()
def logprob_of_continuation(model, prefix_ids, cont_ids, device):
    ids = torch.tensor([prefix_ids + cont_ids], device=device)
    logits = model(ids).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for i, t in enumerate(cont_ids):
        pos = len(prefix_ids) + i - 1          # logits at pos predict token pos+1
        total += logprobs[pos, t].item()
    return total


@torch.no_grad()
def layerwise_first_token(model, final_norm, prefix_ids, tok_a, tok_b, device):
    """Logit-lens log-prob difference (tok_a - tok_b) at every layer."""
    ids = torch.tensor([prefix_ids], device=device)
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states                     # (n_layers + 1) x [1, T, d]
    lm_head = model.get_output_embeddings()
    diffs = []
    for layer, h in enumerate(hs):
        if layer == len(hs) - 1:
            logits = out.logits[0, -1].float()     # final layer: exact logits
        else:
            logits = lm_head(final_norm(h[:, -1:, :]))[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        diffs.append((lp[tok_a] - lp[tok_b]).item())
    return diffs


def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(arg, device):
    if arg == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[arg]


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run_model(name, trials, args):
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    print(f"\n=== {name}  (device={device}, dtype={dtype}) ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    major = int(transformers.__version__.split(".")[0])
    dtype_kw = {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(name, **dtype_kw).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    final_norm = get_final_norm(model) if args.layerwise else None
    if args.layerwise and final_norm is None:
        print("  ! architecture not recognised for logit lens; skipping --layerwise")

    rows, lrows = [], []
    for k, tr in enumerate(trials):
        p_ids = encode_prefix(tok, tr["prefix"])
        c_typ = encode_continuation(tok, tr["prefix"], tr["typical"])
        c_aty = encode_continuation(tok, tr["prefix"], tr["atypical"])
        rows.append(dict(
            model=name, n_params=n_params, **tr,
            lp_typical=logprob_of_continuation(model, p_ids, c_typ, device),
            lp_atypical=logprob_of_continuation(model, p_ids, c_aty, device),
            ntok_typical=len(c_typ), ntok_atypical=len(c_aty)))
        if final_norm is not None:
            diffs = layerwise_first_token(model, final_norm, p_ids,
                                          c_typ[0], c_aty[0], device)
            for layer, d in enumerate(diffs):
                lrows.append(dict(model=name, condition=tr["condition"],
                                  item_id=tr["item_id"], frame=tr["frame"],
                                  layer=layer, n_layers=len(diffs) - 1,
                                  pref_first_token=d))
        if (k + 1) % 40 == 0:
            print(f"  {k + 1}/{len(trials)} trials")

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip("./"))
    outdir = os.path.join(args.outdir, safe)
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "item_scores.csv"), index=False)
    if lrows:
        pd.DataFrame(lrows).to_csv(os.path.join(outdir, "layerwise.csv"), index=False)
    print(f"  saved -> {outdir}  ({time.time() - t0:.1f}s, {n_params / 1e6:.1f}M params)")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="HuggingFace model ids or local paths")
    ap.add_argument("--stimuli", default=os.path.join(HERE, "stimuli"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--layerwise", action="store_true",
                    help="also record logit-lens preferences at every layer")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--show_prompts", action="store_true",
                    help="print the first few prompts and exit")
    args = ap.parse_args()

    trials = build_trials(args.stimuli)
    print(f"{len(trials)} trials "
          f"({sum(t['condition'] == 'taxonomic' for t in trials) // 2} taxonomic items, "
          f"{sum(t['condition'] == 'thematic' for t in trials) // 2} thematic items)")
    if args.show_prompts:
        for t in trials[:2] + trials[-2:]:
            print(f"  [{t['condition']}/{t['frame']}] '{t['prefix']}' + "
                  f"' {t['typical']}' | ' {t['atypical']}'")
        return
    for name in args.models:
        run_model(name, trials, args)


if __name__ == "__main__":
    main()
