#!/usr/bin/env python3
"""
Sanity test for the analysis logic: simulate three idealised models and check
that analyze.py assigns the expected profile to each.

    blind  : 'not' changes nothing            -> negation-blind
    flipper: 'not' flips BOTH conditions      -> heuristic flipper
    scope  : 'not' flips taxonomic only       -> scope-sensitive

    python tests/test_profiles.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import analyze  # noqa: E402

rng = np.random.default_rng(1)
N = 45
# effect of 'not' on the preference, per condition
PROFILES = {"sim-blind":   {"taxonomic": 0.0,  "thematic": 0.0},
            "sim-flipper": {"taxonomic": -6.0, "thematic": -6.0},
            "sim-scope":   {"taxonomic": -6.0, "thematic": -0.3}}
EXPECTED = {"sim-blind": "negation-blind", "sim-flipper": "heuristic flipper",
            "sim-scope": "scope-sensitive"}


def simulate(tmp):
    for i, (name, eff) in enumerate(PROFILES.items()):
        rows = []
        for cond in ("taxonomic", "thematic"):
            for k in range(N):
                base = rng.normal(3.0, 1.0)             # model knows the affirmative
                for frame in ("aff", "neg"):
                    pref = base + (eff[cond] + rng.normal(0, 0.6) if frame == "neg" else 0)
                    rows.append(dict(model=name, n_params=10 ** (7 + i), condition=cond,
                                     item_id=f"{cond[:1]}{k}", frame=frame,
                                     lp_typical=-5 + pref / 2, lp_atypical=-5 - pref / 2))
        d = os.path.join(tmp, name)
        os.makedirs(d)
        pd.DataFrame(rows).to_csv(os.path.join(d, "item_scores.csv"), index=False)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        simulate(tmp)
        wide = analyze.load(tmp)
        st = analyze.model_stats(wide, "all").set_index("model")
    ok = True
    for name, exp in EXPECTED.items():
        got = st.loc[name, "profile"]
        R = st.loc[name, "scope_ratio_R"]
        flag = "PASS" if got == exp else "FAIL"
        ok &= got == exp
        print(f"{flag}  {name:12s} expected={exp:18s} got={got:18s} R={R:.2f}"
              if not np.isnan(R) else
              f"{flag}  {name:12s} expected={exp:18s} got={got:18s} R=–")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
