#!/usr/bin/env python3
"""
Analysis for "Does negation scope over selectional preferences?"

For every item:
    pref_aff = log P(typical | aff) - log P(atypical | aff)
    pref_neg = log P(typical | neg) - log P(atypical | neg)
    delta    = pref_neg - pref_aff              (the negation effect)

Predictions for a model that knows WHAT negation applies to:
    taxonomic : delta strongly negative (preference flips toward the false category)
    thematic  : delta close to zero      (the typical patient stays typical)

Outputs (in results/):
    summary.csv   per model x condition x subset descriptive statistics
    stats.csv     per model tests + scope ratio R + profile label
    RESULTS.md    paste-ready tables
    fig*.png      figures
Usage:
    python analyze.py                # reads results/*/item_scores.csv
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(0)
COND_COLORS = {"taxonomic": "#c0392b", "thematic": "#2471a3"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(resdir):
    files = glob.glob(os.path.join(resdir, "*", "item_scores.csv"))
    if not files:
        raise SystemExit(f"No item_scores.csv under {resdir}. Run run_experiment.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["pref"] = df["lp_typical"] - df["lp_atypical"]
    wide = df.pivot_table(index=["model", "n_params", "condition", "item_id"],
                          columns="frame", values="pref").reset_index()
    wide = wide.rename(columns={"aff": "pref_aff", "neg": "pref_neg"})
    wide["delta"] = wide["pref_neg"] - wide["pref_aff"]
    wide.columns.name = None
    return wide.sort_values(["n_params", "condition", "item_id"])


def load_layerwise(resdir):
    files = glob.glob(os.path.join(resdir, "*", "layerwise.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def boot_ci(x, n=10000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    means = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def perm_test(a, b, n=10000):
    """Two-sided permutation test on the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n):
        RNG.shuffle(pooled)
        if abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= abs(obs):
            count += 1
    return (count + 1) / (n + 1)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def summarise(wide):
    rows = []
    for subset in ("all", "known"):
        for (m, npar, cond), g in wide.groupby(["model", "n_params", "condition"]):
            if subset == "known":           # model shows the preference when affirmative
                g = g[g["pref_aff"] > 0]
            if len(g) == 0:
                continue
            lo, hi = boot_ci(g["delta"])
            big = g[g["pref_aff"].abs() >= 1.0]
            neg_ok = (g["pref_neg"] < 0) if cond == "taxonomic" else (g["pref_neg"] > 0)
            rows.append(dict(
                model=m, n_params=npar, condition=cond, subset=subset, n=len(g),
                mean_pref_aff=g["pref_aff"].mean(), mean_pref_neg=g["pref_neg"].mean(),
                mean_delta=g["delta"].mean(), delta_ci_lo=lo, delta_ci_hi=hi,
                median_delta=g["delta"].median(),
                median_rel_delta=(big["delta"] / big["pref_aff"].abs()).median()
                if len(big) else np.nan,
                aff_acc=(g["pref_aff"] > 0).mean(),
                neg_correct=neg_ok.mean(),
                wilcoxon_p=stats.wilcoxon(g["delta"]).pvalue if len(g) > 5 else np.nan))
    return pd.DataFrame(rows)


def classify(tax_mean, tax_ci, them_mean, them_ci, p_diff, ratio):
    """Assign a profile. `ratio` is R = mean_them / mean_tax.

    R is compared by MAGNITUDE: the thematic effect can come out with the
    opposite sign to the taxonomic one, and a large negative R still means
    the two conditions shifted by comparable amounts.
    """
    tax_shift = tax_ci[1] < 0                        # CI entirely below zero
    tax_wrong = tax_ci[0] > 0                        # CI entirely ABOVE zero
    them_shift = them_ci[1] < 0 or them_ci[0] > 0    # two-sided
    if tax_wrong:
        return "wrong-direction"
    if not tax_shift:
        return "negation-blind" if not them_shift else "anomalous (thematic-only shift)"
    if p_diff >= 0.05:
        return "heuristic flipper" if them_shift else "inconclusive (difference n.s.)"
    r = abs(ratio) if ratio == ratio else float("inf")
    if r < 0.25:
        return "scope-sensitive"
    if r <= 1.0:
        return "partially scope-sensitive"
    return "both conditions shift"


def model_stats(wide, subset):
    rows = []
    for (m, npar), g in wide.groupby(["model", "n_params"]):
        if subset == "known":
            g = g[g["pref_aff"] > 0]
        tax = g.loc[g.condition == "taxonomic", "delta"].values
        them = g.loc[g.condition == "thematic", "delta"].values
        if len(tax) < 3 or len(them) < 3:
            continue
        p_perm = perm_test(tax, them)
        tax_ci, them_ci = boot_ci(tax), boot_ci(them)
        # R is only interpretable when negation reliably moves the taxonomic items;
        # otherwise we would be dividing by noise around zero.
        ratio = them.mean() / tax.mean() if (tax_ci[1] < 0 or tax_ci[0] > 0) else np.nan
        rows.append(dict(
            model=m, n_params=npar, subset=subset, n_tax=len(tax), n_them=len(them),
            mean_delta_tax=tax.mean(), mean_delta_them=them.mean(),
            scope_ratio_R=ratio,
            welch_t=stats.ttest_ind(tax, them, equal_var=False).statistic,
            welch_p=stats.ttest_ind(tax, them, equal_var=False).pvalue,
            perm_p=p_perm, cohens_d=cohens_d(tax, them),
            profile=classify(tax.mean(), tax_ci, them.mean(), them_ci, p_perm,
                             ratio if not np.isnan(ratio) else 1.0)))
    return pd.DataFrame(rows)


def mixed_model(wide):
    """delta ~ condition * log10(params) with random intercepts per item (>=3 models)."""
    if wide["model"].nunique() < 3:
        return None
    import warnings
    warnings.filterwarnings("ignore")
    import statsmodels.formula.api as smf
    d = wide.copy()
    d["log_params"] = np.log10(d["n_params"])
    d["log_params"] -= d["log_params"].mean()
    d["item_key"] = d["condition"] + "_" + d["item_id"]
    try:
        fit = smf.mixedlm("delta ~ C(condition, Treatment('thematic')) * log_params",
                          d, groups=d["item_key"]).fit(reml=True)
        return fit.summary().as_text()
    except Exception as e:                         # pragma: no cover
        return f"Mixed model failed: {e}"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def short(m):
    return m.split("/")[-1]


def fig_delta(summary, out):
    s = summary[summary.subset == "all"]
    models = list(dict.fromkeys(s.sort_values("n_params")["model"]))
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(models)), 4.2))
    for j, cond in enumerate(("taxonomic", "thematic")):
        c = s[s.condition == cond].set_index("model").reindex(models)
        err = np.vstack([c["mean_delta"] - c["delta_ci_lo"], c["delta_ci_hi"] - c["mean_delta"]])
        ax.bar(x + (j - 0.5) * 0.38, c["mean_delta"], 0.38, yerr=err, capsize=3,
               color=COND_COLORS[cond], label=cond)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [short(m) for m in models], rotation=30, ha="right")
    ax.set_ylabel("negation effect  Δ = pref_neg − pref_aff\n(log-prob units, 95% CI)")
    ax.set_title("Effect of 'not' on the typical-vs-atypical preference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_scatter(wide, model, out):
    g = wide[wide.model == model]
    fig, ax = plt.subplots(figsize=(5, 5))
    for cond, gg in g.groupby("condition"):
        ax.scatter(gg["pref_aff"], gg["pref_neg"], s=22, alpha=0.75,
                   color=COND_COLORS[cond], label=cond)
    lim = np.nanmax(np.abs(g[["pref_aff", "pref_neg"]].values)) * 1.1 + 0.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="no effect of 'not'")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("preference, affirmative frame")
    ax.set_ylabel("preference, negated frame")
    ax.set_title(short(model))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_scaling(summary, out):
    s = summary[summary.subset == "all"]
    if s["model"].nunique() < 2:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, g in s.groupby("condition"):
        g = g.sort_values("n_params")
        ax.errorbar(g["n_params"], g["mean_delta"],
                    yerr=[g["mean_delta"] - g["delta_ci_lo"], g["delta_ci_hi"] - g["mean_delta"]],
                    marker="o", capsize=3, color=COND_COLORS[cond], label=cond)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("parameters")
    ax.set_ylabel("mean Δ (95% CI)")
    ax.set_title("Negation effect vs. model size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def fig_layerwise_delta(lw, model, out):
    """Negation effect (neg - aff) per layer, with 95% CI bands."""
    g = lw[lw.model == model]
    w = g.pivot_table(index=["condition", "item_id", "layer"], columns="frame",
                      values="pref_first_token").reset_index()
    if not {"aff", "neg"} <= set(w.columns):
        return
    w["delta"] = w["neg"] - w["aff"]
    agg = w.groupby(["condition", "layer"]).delta.agg(["mean", "std", "size"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["size"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for cond, gg in agg.groupby("condition"):
        ax.plot(gg.layer, gg["mean"], color=COND_COLORS[cond], lw=2, label=cond)
        ax.fill_between(gg.layer, gg["mean"] - 1.96 * gg.se, gg["mean"] + 1.96 * gg.se,
                        color=COND_COLORS[cond], alpha=.18, lw=0)
    ax.axhline(0, color="k", lw=.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("layer (logit lens; 0 = embeddings)")
    ax.set_ylabel("negation effect Δ at the target position\n(log-prob units, 95% CI)")
    ax.set_title(f"Where the negation effect arises: {short(model)}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_layerwise(lw, model, out):
    g = lw[lw.model == model]
    agg = g.groupby(["condition", "frame", "layer"])["pref_first_token"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (cond, frame), gg in agg.groupby(["condition", "frame"]):
        ax.plot(gg["layer"], gg["pref_first_token"], color=COND_COLORS[cond],
                ls="-" if frame == "aff" else "--", marker=".", label=f"{cond} / {frame}")
    ax.axhline(0, color="k", lw=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("layer (logit lens; 0 = embeddings)")
    ax.set_ylabel("mean first-token preference\n(typical − atypical)")
    ax.set_title(f"Layerwise preference: {short(model)}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def md_table(df, cols, fmt):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.get(c, "{}").format(v) if not (isinstance(v, float) and np.isnan(v)) else "–")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    wide = load(args.results)
    summary = summarise(wide)
    st = pd.concat([model_stats(wide, "all"), model_stats(wide, "known")], ignore_index=True)
    summary.to_csv(os.path.join(args.results, "summary.csv"), index=False)
    st.to_csv(os.path.join(args.results, "stats.csv"), index=False)
    wide.to_csv(os.path.join(args.results, "item_deltas.csv"), index=False)

    figs = []
    fig_delta(summary, os.path.join(args.results, "fig1_delta_by_model.png"))
    figs.append("fig1_delta_by_model.png")
    for m in wide["model"].unique():
        fn = f"fig2_scatter_{short(m)}.png"
        fig_scatter(wide, m, os.path.join(args.results, fn))
        figs.append(fn)
    if fig_scaling(summary, os.path.join(args.results, "fig3_scaling.png")):
        figs.append("fig3_scaling.png")
    lw = load_layerwise(args.results)
    if lw is not None:
        for m in lw["model"].unique():
            fn = f"fig4_layerwise_{short(m)}.png"
            fig_layerwise(lw, m, os.path.join(args.results, fn))
            figs.append(fn)
            fn5 = f"fig5_layerdelta_{short(m)}.png"
            fig_layerwise_delta(lw, m, os.path.join(args.results, fn5))
            figs.append(fn5)

    mm = mixed_model(wide)

    # ---------------- paste-ready report ----------------
    s = summary.copy()
    s["model"] = s["model"].map(short)
    s["delta [95% CI]"] = s.apply(
        lambda r: f"{r.mean_delta:.2f} [{r.delta_ci_lo:.2f}, {r.delta_ci_hi:.2f}]", axis=1)
    t = st.copy()
    t["model"] = t["model"].map(short)
    fmt = {"mean_pref_aff": "{:.2f}", "mean_pref_neg": "{:.2f}", "aff_acc": "{:.2f}",
           "neg_correct": "{:.2f}", "median_rel_delta": "{:.2f}", "wilcoxon_p": "{:.3g}",
           "mean_delta_tax": "{:.2f}", "mean_delta_them": "{:.2f}", "scope_ratio_R": "{:.2f}",
           "welch_p": "{:.3g}", "perm_p": "{:.3g}", "cohens_d": "{:.2f}", "n": "{:d}"}
    with open(os.path.join(args.results, "RESULTS.md"), "w") as f:
        f.write("# Results\n\n## Table 1. Descriptives (all items)\n\n")
        f.write(md_table(s[s.subset == "all"],
                         ["model", "condition", "n", "mean_pref_aff", "mean_pref_neg",
                          "delta [95% CI]", "median_rel_delta", "aff_acc", "neg_correct",
                          "wilcoxon_p"], fmt))
        f.write("\n\n`neg_correct` = share of items where the negated frame behaves correctly "
                "(taxonomic: preference flipped below 0; thematic: typical still preferred).\n")
        f.write("\n## Table 2. Taxonomic vs. thematic negation effect, per model\n\n")
        f.write(md_table(t[t.subset == "all"],
                         ["model", "mean_delta_tax", "mean_delta_them", "scope_ratio_R",
                          "perm_p", "welch_p", "cohens_d", "profile"], fmt))
        f.write("\n\n## Table 3. Same, restricted to items the model 'knows' "
                "(pref_aff > 0)\n\n")
        f.write(md_table(t[t.subset == "known"],
                         ["model", "n_tax", "n_them", "mean_delta_tax", "mean_delta_them",
                          "scope_ratio_R", "perm_p", "profile"],
                         {**fmt, "n_tax": "{:d}", "n_them": "{:d}"}))
        if mm:
            f.write("\n\n## Mixed-effects model (all models pooled)\n\n```\n" + mm + "\n```\n")
        f.write("\n\n## Figures\n\n" + "\n".join(f"![{x}]({x})" for x in figs) + "\n")

    print(open(os.path.join(args.results, "RESULTS.md")).read().split("## Figures")[0])
    print(f"Figures written: {', '.join(figs)}")


if __name__ == "__main__":
    main()
