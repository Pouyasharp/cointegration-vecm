# Cointegration and VECM — Scaffold

> **Scaffold / outline version (v0.1.0).** A planned PhD-depth
> treatment of cointegration and Vector Error Correction Models.
> Sections marked `⟳ TBD` are to-be-developed in v0.2.0+.
>
> The one demo that ships with v0.1.0 is a working Engle-Granger
> 2-step cointegration test on synthetic data.
>
> Standalone deep-dive of the cointegration half of Stage 3 of the
> [`econometrics-deep-research`](https://github.com/Pouyasharp/econometrics-deep-research)
> parent curriculum.

---

## Status

This repo is a **scaffold** — it ships one working demo
(`demo_01_engle_granger.py`) plus a detailed outline of the planned
content. The full deep-dive (with 5 demos matching the
`arimax-sarimax-tutorial` pattern) is targeted for v0.2.0+.

**What works now (v0.1.0):**
- ✅ Complete `SKILL.md` outline (sections 0-8 with TBD markers)
- ✅ `INDEX.md` for navigation
- ✅ `REFINEMENT.md` listing what's planned for v0.2.0
- ✅ `references/` directory (placeholder for v0.2.0)
- ✅ One working demo: Engle-Granger 2-step on synthetic data
- ✅ `README.md` (this file)
- ✅ Standard `LICENSE` (MIT), `requirements.txt`, `.gitignore`

**What's planned for v0.2.0:**
- ⟳ §1-§6 fully developed (current outlines are 50-200 words each)
- ⟳ `demo_02_johansen.py` — Johansen trace test on multi-equation system
- ⟳ `demo_03_vecm.py` — VECM estimation and short-run/long-run decomposition
- ⟳ `demo_04_granger_in_vecm.py` — Granger causality within VECM
- ⟳ `demo_05_structural_breaks.py` — Gregory-Hansen, Hatemi-J tests
- ⟳ `references/urca-api-quirks.md` — gotcha bank

---

## Quick start

```bash
git clone https://github.com/Pouyasharp/cointegration-vecm.git
cd cointegration-vecm
pip install -r requirements.txt

python3 demos/demo_01_engle_granger.py
```

The demo prints a step-by-step narrative to stdout and writes a
4-panel figure to `figures/`:

- **A:** Two independent random walks (the spurious regression case)
- **B:** OLS on the two RWs (huge t-stat, garbage slope)
- **C:** Two cointegrated I(1) series (the real long-run relationship)
- **D:** The cointegrating residual (should be stationary)

---

## Repository layout

```
cointegration-vecm/
├── README.md
├── SKILL.md             # full outline (sections 0-8, with TBD markers)
├── INDEX.md             # navigation
├── REFINEMENT.md        # v0.2.0 plan
├── LICENSE              # MIT
├── requirements.txt
├── .gitignore
├── references/          # placeholder for v0.2.0
└── demos/
    └── demo_01_engle_granger.py   # 1 working demo
```

---

## Companion repos

- [`Pouyasharp/arimax-sarimax-tutorial`](https://github.com/Pouyasharp/arimax-sarimax-tutorial) — PhD deep-dive on the ARIMA / SARIMAX half of the time-series stage
- [`Pouyasharp/econometrics-deep-research`](https://github.com/Pouyasharp/econometrics-deep-research) — 5-stage parent curriculum

---

## License

MIT. See `LICENSE`.
