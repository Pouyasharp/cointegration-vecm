# Refinement Checklist — `cointegration-vecm`

This file is the **living to-do list** for the curriculum. The protocol:

1. **Quarterly** — pick 3 of these items and turn them into demos or
   wiki pages.
2. **Per item** — when added, bump `version` in `SKILL.md` and append
   to "What changed" below.
3. **Yearly** — rewrite one section from scratch using the items
   added in that year.

---

## v0.3.0 — what's planned for the next iteration

### §5 VECM
- [ ] Add: identification issues (Phillips triangular representation,
  Gonzalo-Granger permanent-transitory decomposition, Procrustes
  transformation)
- [ ] Add: Bayesian VECM (via `BVAR` or `bmcmc` packages)
- [ ] Demo: Procrustes transformation with sensitivity analysis

### §6 Interpretation
- [ ] Add: structural IRF within the VECM (Pesaran-Shin 1998)
- [ ] Add: forecast error variance decomposition (FEVD)
- [ ] Add: persistence profile (Lee-Strazicich 2003)

### §7 Granger in VECM
- [ ] Add: common-feature tests (Engle-Kozicki 1993)
- [ ] Add: permanent-transitory decomposition
- [ ] Demo: dynamic OLS (DOLS, Stock-Watson 1993) for single-equation
  cointegrating regression

### §8 Structural breaks
- [ ] Add: Hatemi-J (2008) test (trend break)
- [ ] Add: Bai-Perron (1998, 2003) multiple-break test
- [ ] Add: Markov-switching VECM (Krolzig 1997)
- [ ] Add: TVP-VECM (time-varying parameters, Primiceri 2005)
- [ ] Demo: real-data application to a US macro pair (e.g.,
  consumption-income)

### Cross-cutting
- [ ] Add: panel cointegration (Pedroni 1999, Westerlund 2007)
- [ ] Add: nonlinear cointegration (Kapetanios-Shin-Snell 2006)
- [ ] Add: regime-switching cointegration (Krolzig-Hansen 2001)
- [ ] Real-data case study: price + demand pair from a real
  industry (using the AMIQ data, anonymized)

---

## What changed (v0.1.0 → v0.2.0)

- **v0.2.0 (2026-06-15):** Promoted from scaffold to full deep-dive.
  - Added demo 02 (Johansen trace + max-eig) — runs on a 3-var
    VECM with k=1 cointegrating relation planted; verifies the
    rank decision.
  - Added demo 03 (VECM) — fits VECM(1) with planted β, α, γ;
    recovers all three sets of parameters and compares to the truth.
  - Added demo 04 (Granger in VECM) — DGP encodes y₂ strongly
    exogenous + Granger-causing y₁ in the short run; verifies that
    the test detects both.
  - Added demo 05 (Gregory-Hansen) — level shift at unknown break
    date; verifies that GH finds the break.
  - Expanded SKILL.md from ~50-line outline to ~500-line PhD
    curriculum.
  - Added INDEX.md, REFINEMENT.md (this file), and
    references/statsmodels-vecm-api-quirks.md (planned).

- **v0.1.0 (2026-06-15):** SCAFFOLD. 1 working demo
  (Engle-Granger 2-step). Sections 1-7 were outlines.

---

*Last updated: 2026-06-15 · v0.2.0*
