# Does negation scope over selectional preferences?

Code, stimuli and analysis for the seminar paper in `PAPER_DRAFT.md`.

**The question in one line:** do language models know *what* a "not" applies to, or do they only react to seeing the word?

| Frame | Should "not" change the preferred filler? |
|---|---|
| `A robin is (not) a kind of` → *bird* vs *fish* | **Yes** — negation targets category membership |
| `The chef was (not) chopping the` → *onions* vs *branches* | **No** — onions are still what chefs chop |

## Project layout

```
negation_scope/
├── PAPER_DRAFT.md          paper draft; results sections are templates you fill in
├── README.md               this file
├── colab_run.ipynb         easiest way to run: free GPU on Google Colab
├── run_experiment.py       scores fillers under affirmative / negated frames
├── analyze.py              statistics, profile classification, figures, RESULTS.md
├── requirements.txt
├── stimuli/
│   ├── taxonomic.csv       45 items: subject, true category, false category
│   └── thematic.csv        45 items: agent, verb, typical patient, atypical patient
├── figures/
│   └── fig0_predicted_profiles.png   SIMULATED predictions (for the paper's hypotheses section)
└── tests/
    ├── make_tiny_models.py  builds tiny random models for offline testing
    └── test_profiles.py     checks the classifier on simulated data
```

## Option A — Google Colab (recommended, free GPU)

1. Go to <https://colab.research.google.com> → *File → Upload notebook* → choose `colab_run.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*.
3. Run the cells top to bottom. The first cell asks you to upload `negation_scope.zip`.
4. The last cell downloads `results.zip` containing all tables and figures.

## Option B — your own machine

```bash
cd negation_scope
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. sanity check: print the prompts
python run_experiment.py --models x --show_prompts

# 2. small CPU-friendly run (a few minutes on a laptop)
python run_experiment.py --layerwise --models gpt2 EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m

# 3. analyse
python analyze.py
```

Results appear in `results/`. Open `results/RESULTS.md` for paste-ready tables.

## Which models to run

| Goal | Models | Hardware |
|---|---|---|
| Minimum viable paper | `gpt2`, `gpt2-medium`, `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m` | laptop CPU |
| **Scaling analysis (recommended)** | Pythia `70m 160m 410m 1b 1.4b 2.8b` | Colab T4 |
| Second model family | `Qwen/Qwen2.5-0.5B`, `-1.5B`, `-3B`, `-7B` | T4 (7B is tight: add `--dtype float16`) |
| Instruction tuning effect | `Qwen/Qwen2.5-1.5B` vs `Qwen/Qwen2.5-1.5B-Instruct` | T4 |
| Llama family | `meta-llama/Llama-3.2-1B`, `-3B` | gated: accept licence on HuggingFace, then `huggingface-cli login` |

The Pythia suite is the cleanest scaling comparison: every model saw the same data in the same order, so size is the only difference.

## Output files

| File | Content |
|---|---|
| `results/<model>/item_scores.csv` | raw log-probabilities per item and frame |
| `results/<model>/layerwise.csv` | logit-lens preference per layer (with `--layerwise`) |
| `results/item_deltas.csv` | per item: `pref_aff`, `pref_neg`, `delta` |
| `results/summary.csv` | descriptives per model × condition |
| `results/stats.csv` | tests, scope ratio R, profile label per model |
| `results/RESULTS.md` | tables formatted for the paper |
| `fig1_delta_by_model.png` | **main figure**: negation effect per condition and model |
| `fig2_scatter_<model>.png` | every item: affirmative vs negated preference |
| `fig3_scaling.png` | negation effect vs parameter count |
| `fig4_layerwise_<model>.png` | where in the network the effect emerges |

### Reading the numbers

* `pref` = log P(typical) − log P(atypical). Positive means the model prefers the typical filler.
* `delta` = pref_neg − pref_aff. **The negation effect.** Negative means "not" pushed the model away from the typical filler.
* `neg_correct` = share of items handled correctly under negation (taxonomic: flipped below 0; thematic: typical still preferred).
* `scope_ratio_R` = mean Δ_thematic / mean Δ_taxonomic. **≈ 0 → scope-sensitive; ≈ 1 → blunt flipper.** Only computed when the taxonomic effect is reliably negative.
* `profile` is assigned by transparent rules (see `classify()` in `analyze.py`):

| Profile | Rule |
|---|---|
| negation-blind | taxonomic Δ CI includes 0 |
| heuristic flipper | taxonomic Δ < 0, thematic Δ < 0, difference not significant |
| scope-sensitive | taxonomic Δ < 0, significantly more negative than thematic, R < 0.25 |
| partially scope-sensitive | as above but R ≥ 0.25 |

Report the numbers, not just the labels. The labels are a summary device.

## Extending or replacing the stimuli

The stimuli are author-constructed (see Limitations in the paper). To strengthen the study:

* **Thematic:** replace or extend `stimuli/thematic.csv` with items from McRae et al. (1998), Ferretti et al. (2001), or DTFit (Vassallo et al., 2018). Keep the columns `item_id, agent, verb_ing, typical, atypical`.
* **Taxonomic:** add items from NEG-1500-SIMP (Shivagunde et al., 2023) or Negated LAMA (Kassner & Schütze, 2020). Keep `item_id, subject_np, true_category, false_category`.
* Frames are built in `build_trials()` in `run_experiment.py`; change them there if you add new conditions.

Rules for new items: the filler must be a natural continuation right after the frame; affirmative and negated frames must differ **only** by "not"; atypical thematic fillers should be unusual but possible, not absurd.

## Regenerating the paper's tables

The three tables in §5 of `PAPER_DRAFT.md` are generated from `results/summary.csv`, never typed by hand:

```bash
python make_paper_tables.py            # writes paper_tables/tablemain.md, table1.md, table2.md, table3.md
python make_paper_tables.py --inject   # also splices them into PAPER_DRAFT.md
```

`--inject` replaces whatever sits between the `<!-- TABLE1:START -->` / `<!-- TABLE1:END -->` comment markers in the draft. Rerun the experiment, rerun this, and the paper is up to date. The markers are invisible in the rendered PDF or Word file.

`tablemain.md` is the compact table in §5.2 of the body; the other three are the full tables in Appendix B.

Column sources: the full tables come from `results/summary.csv` (`subset` = `all` and `known` respectively); Table 2's `perm_p` and `cohens_d` come from `results/stats.csv`, and show `–` if that file is absent.

## Testing (no internet needed)

```bash
python tests/test_profiles.py        # classifier recognises simulated profiles
python tests/make_tiny_models.py     # builds tiny random GPT-2 / Pythia / Llama models
python run_experiment.py --models tests/tiny-gpt2 tests/tiny-neox tests/tiny-llama \
       --layerwise --outdir tests/smoke_results
python analyze.py --results tests/smoke_results
```

Tiny random models must come out at chance (`aff_acc` ≈ 0.5) and `negation-blind`. **Never report smoke-test numbers as results.**

## Troubleshooting

* **Out of memory:** add `--dtype float16` (GPU) or drop the largest model.
* **Apple Silicon:** `--device mps --dtype float32` (float16 on MPS can produce NaNs).
* **`401` / gated repo:** Llama models need licence acceptance and `huggingface-cli login`.
* **Architecture not recognised for logit lens:** item scores are still computed; only `--layerwise` is skipped. Add the model's final-norm path to `get_final_norm()`.
