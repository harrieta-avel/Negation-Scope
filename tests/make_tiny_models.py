#!/usr/bin/env python3
"""
Builds three TINY, RANDOMLY INITIALISED models (GPT-2, GPT-NeoX/Pythia, Llama
architectures) with a small byte-level BPE tokenizer trained on the stimuli.

Purpose: verify the whole pipeline end-to-end without internet access.
Their outputs are MEANINGLESS noise - never report them as results.

    python tests/make_tiny_models.py
    python run_experiment.py --models tests/tiny-gpt2 tests/tiny-neox tests/tiny-llama \
        --layerwise --outdir tests/smoke_results
    python analyze.py --results tests/smoke_results
"""
import os

import pandas as pd
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import (GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM,
                          LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast)

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "..", "stimuli")


def corpus():
    lines = []
    for f in ("taxonomic.csv", "thematic.csv"):
        df = pd.read_csv(os.path.join(STIM, f))
        lines += [" ".join(map(str, r)) for r in df.values]
    lines += ["The man was not doing it. A thing is a kind of thing. is not a kind of"] * 5
    return lines


def make_tokenizer():
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus(), vocab_size=600, min_frequency=1,
                            special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, bos_token="<|endoftext|>",
                                  eos_token="<|endoftext|>", unk_token="<|endoftext|>")
    return tok


def main():
    torch.manual_seed(0)
    tok = make_tokenizer()
    V = len(tok)
    specs = {
        "tiny-gpt2": GPT2LMHeadModel(GPT2Config(vocab_size=V, n_positions=64, n_embd=32,
                                                n_layer=2, n_head=2,
                                                bos_token_id=0, eos_token_id=0)),
        "tiny-neox": GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=V, hidden_size=48,
                                                      num_hidden_layers=3, num_attention_heads=2,
                                                      intermediate_size=96,
                                                      max_position_embeddings=64)),
        "tiny-llama": LlamaForCausalLM(LlamaConfig(vocab_size=V, hidden_size=64,
                                                   num_hidden_layers=4, num_attention_heads=2,
                                                   intermediate_size=128,
                                                   max_position_embeddings=64)),
    }
    for name, model in specs.items():
        path = os.path.join(HERE, name)
        model.save_pretrained(path)
        tok.save_pretrained(path)
        n = sum(p.numel() for p in model.parameters())
        print(f"saved {path}  ({n / 1e3:.0f}k params)")


if __name__ == "__main__":
    main()
