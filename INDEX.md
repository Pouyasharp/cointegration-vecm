# Cointegration and VECM — INDEX

> **Quick navigation** for the Cointegration / VECM deep-dive skill.
> Read `SKILL.md` top-to-bottom for the curriculum; use this INDEX
> to find a specific topic fast.

---

## File layout

```
cointegration-vecm/
├── SKILL.md            # ~500 lines — the curriculum
├── INDEX.md            # this file
├── REFINEMENT.md       # versioned checklist + known gaps
├── LICENSE             # MIT
├── requirements.txt
├── .gitignore
├── README.md           # GitHub front door
├── references/
│   └── statsmodels-vecm-api-quirks.md
├── demos/
│   ├── demo_01_engle_granger.py         # Engle-Granger 2-step
│   ├── demo_02_johansen.py              # Johansen trace + max-eig
│   ├── demo_03_vecm.py                  # VECM fit + interpretation
│   ├── demo_04_granger_in_vecm.py       # Weak + strong exogeneity
│   └── demo_05_structural_breaks.py     # Gregory-Hansen
└── figures/
    ├── demo_01_figure.png
    ├── demo_02_figure.png
    ├── demo_03_figure.png
    ├── demo_04_figure.png
    └── demo_05_figure.png
```

---

## Topic → Section map (SKILL.md)

| Topic | SKILL § | Demo |
|---|---|---|
| Spurious regression | §1 | demo_01 |
| Cointegration definition | §2 | demo_01 |
| Engle-Granger 2-step | §3 | demo_01 |
| Johansen trace + max-eig | §4 | demo_02 |
| VECM representation | §5 | demo_03 |
| Long-run β interpretation | §6 | demo_03 |
| Short-run γ interpretation | §6 | demo_03 |
| Speed of adjustment α | §6 | demo_03 |
| Granger in VECM | §7 | demo_04 |
| Weak exogeneity | §7 | demo_04 |
| Strong exogeneity | §7 | demo_04 |
| Single-equation ECM consistency | §7 | demo_04 |
| Structural breaks | §8 | demo_05 |
| Gregory-Hansen | §8 | demo_05 |
| Self-assessment rubric | §9 | (8 items) |
| Reading list | §10 | (7 papers) |
| Refinement log | §11 | (versioned) |

---

## How to run everything

```bash
cd /path/to/cointegration-vecm
pip install -r requirements.txt

python3 demos/demo_01_engle_granger.py
python3 demos/demo_02_johansen.py
python3 demos/demo_03_vecm.py
python3 demos/demo_04_granger_in_vecm.py
python3 demos/demo_05_structural_breaks.py
```

Each demo runs in ~30-90s, prints a step-by-step narrative to stdout,
and writes a figure to `figures/`.

---

## How to use as a study guide

1. Read `SKILL.md` §0 — get oriented, set up the tool stack.
2. Read §1-§3 (Engle-Granger) and run `demo_01`. Spurious regression
   is a probabilistic phenomenon — the same DGP can give different
   results across runs.
3. Read §4 (Johansen) and run `demo_02`. Trace and max-eigenvalue
   tests can disagree; report both.
4. Read §5-§6 (VECM) and run `demo_03`. The key output is β
   (long-run), α (adjustment), and γ (short-run).
5. Read §7 (Granger in VECM) and run `demo_04`. The concepts of
   weak and strong exogeneity are diagnostic — they tell you
   whether a full multivariate model is necessary.
6. Read §8 (Structural breaks) and run `demo_05`. Always plot
   the residual — a visible shift is a hint.
7. Self-assess with §9 rubric.
8. Read the papers in §10.
9. When you find a gap, add it to `REFINEMENT.md`.

---

*INDEX last updated: 2026-06-15 (v0.2.0)*
