# Dissertation Draft Structure

**Bilingual drafts (2026-07-14):**
- `chapter_1_introduction_bilingual.md`
- `chapter_3_methodology_bilingual.md`
- `chapter_4_results_bilingual.md`
- `chapter_5_discussion_bilingual.md`
- `chapter_6_conclusion_bilingual.md`
- Chapter 2 literature: keep current draft (`00_admin/literature_review.md` / bilingual); do not revise yet.

**Working title:**  
A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

**Positioning:** An empirical integration and comparison study (application + integration + systematic out-of-sample comparison), not a proposal of a new neural architecture.

**Research questions:**
- **RQ1:** Do remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline and the random-walk benchmark?
- **RQ2:** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data?
- **RQ3:** Can modality-level interpretability reveal which signals the model relies on across different market conditions?

### Cross-chapter RQ mapping
| RQ | Methods (Ch3) | Results (Ch4) | Discussion (Ch5) |
|---|---|---|---|
| RQ1 | M0–M4 information sets; nested CW vs M1; DM vs M0 | §4.2–4.3 modality increments | §5.1 |
| RQ2 | Flat vs deep under identical data/protocol | §4.4 **paired** Miˢᵈᵉᵉᵖ vs Mi | §5.2 |
| RQ3 | SHAP / gates / attention plan | §4.6 interpretability | §5.3 |

### Writing caution on architecture claims
- Treat “Deep overall better than Flat” as a **hypothesis to test**, not a settled headline.
- Prefer: *Deep models outperformed their flat counterparts in selected multimodal settings, particularly when shipping information was included.*
- Use the stronger claim (*consistently outperformed*) only if **paired Mi comparisons**, multiple targets/windows, and robustness all support it.

### Indicative word budget (CASA habit)

CASA MSc dissertations in recent Distinction samples typically declare **~10,000–12,000 words** for the main body. Figures, tables, captions, references and appendices are usually **outside** that count (confirm in the current CASA0010 brief).

Taylor (Meeting 03): lit review ≈ **4–5 pages** as a starting guide.

| Chapter | Indicative words | ~Share | Notes |
|---|---:|---:|---|
| **1 Introduction** | **900–1,200** | ~9% | Motivation, gap, RQs, contributions, roadmap |
| **2 Literature Review** | **1,800–2,400** | ~18% | ≈4–6 pages; close on the gap |
| **3 Data & Methods** | **2,500–3,000** | ~24% | Core design in body; variable lists / search grids → Appendix |
| **4 Results** | **2,500–3,200** | ~26% | Organised by experimental logic + paired Flat–Deep |
| **5 Discussion** | **1,600–1,900** | ~15% | RQ-by-RQ explanation (not a results repeat) |
| **6 Conclusion** | **400–700** | ~5% | Tight summary |
| **Main body total** | **≈10,000–12,000** | 100% | Protect Ch5 interpretation space |
| Abstract | ~250–350 | — | |
| References + Appendices | as needed | — | Dictionaries, hyperparameter grids, extra SHAP |

**If over length:** trim Ch2 and move Ch3 detail to appendices **before** cutting Ch5.

---

## Front matter
- Title page, abstract (~250–350 words), acknowledgements, table of contents, list of figures/tables

---

## Chapter 1 — Introduction *(~900–1,200 words)*

### 1.1 Background and motivation
- Weekly Brent is hard to forecast; the no-change / random-walk benchmark is strong
- Alternative data (shipping / AIS, satellite RS) as physical-market proxies
- Most work still flattens heterogeneous sources into one feature table

### 1.2 Research problem
- Incremental value of RS / shipping over a financial baseline remains unclear
- Unclear whether representation-level fusion beats flat concatenation on the same data
- Need leakage-safe evaluation and formal forecast-comparison tests

### 1.3 Research gap
- Multi-source oil studies rarely compare flat vs modality-aware fusion under one protocol
- Few studies jointly report nested increments (vs M1) and absolute skill (vs M0)

### 1.4 Research questions
- State RQ1–RQ3 explicitly

### 1.5 Contributions
- Unified M0–M4 ladder across flat and deep architectures
- Fair rolling-origin backtest with DM / Clark–West
- Paired Flat–Deep comparison by information set
- Interpretability focused on models with predictive value (see §3.10)

### 1.6 Dissertation structure
- Brief roadmap of Chapters 2–6

---

## Chapter 2 — Literature Review *(~1,800–2,400 words; ≈4–6 pages)*

*(Existing draft to be aligned to this TOC after supervisor feedback. Each subsection should end by linking forward to RQ1–3 / the gap.)*

### 2.1 Crude-oil forecasting and benchmark difficulty
- Persistence; Alquist et al. no-change benchmark; why complex models often fail out of sample

### 2.2 Financial and macroeconomic predictors
- Inventories, production/refinery, volatility, rates, FX, GPR, futures/spreads as an economically informed baseline

### 2.3 Shipping activity as an alternative-data source
- AIS / PortWatch proxies; reverse causality; noisy cargo inference; network structure vs flat counts

### 2.4 Remote sensing for energy and economic monitoring
- NTL, NO₂, site imagery / embeddings; indirect mechanisms; cloud and missingness

### 2.5 Multimodal and temporal fusion methods
- Feature-level vs representation-level fusion; gated / cross-attention; EO foundation models; missing-modality training

### 2.6 Forecast evaluation and interpretability
- DM / Clark–West; SHAP and modality-level attribution as complementary to accuracy tests

### 2.7 Research gap and conceptual framework
- Gap table; conceptual link from literature → M0–M4 design and RQ1–3

---

## Chapter 3 — Data and Methods *(~2,500–3,000 words)*

*Keep the chapter readable: full variable dictionaries, hyperparameter search spaces, and long feature tables go to the Appendix.*

### 3.1 Research design
- Empirical comparison: flat feature fusion vs modality-aware representation-level fusion
- Same information sets (M0–M4), same timeline, same metrics

### 3.2 Prediction targets and forecasting timeline
- Target: next-week Brent \(P_{t+1}\); train on \(r_{t+1}=\log(P_{t+1}/P_t)\); evaluate on reconstructed price
- Forecast origin, lookback, sample window

### 3.3 Data sources
- Finance / RS / shipping sources at summary level (details → Appendix A)

### 3.4 Temporal alignment, lagging and missingness
- Release-timestamp alignment (no look-ahead)
- Irregular RS / shipping availability; masks rather than silent fill where relevant

### 3.5 M0–M4 information sets
- M0 random walk; M1 finance; M2 +RS; M3 +shipping; M4 all modalities
- Identical modality definitions for flat and deep arms

### 3.6 Flat models
- Early concatenation; Ridge / XGBoost (and any early-fusion deep baseline if reported)

### 3.7 Deep and representation-level models
- Modality-specific encoders + gated / cross-attention fusion
- Explicit missing-modality / dropout handling

### 3.8 Training and hyperparameter selection
- Summary of selection rule; full grids → Appendix C

### 3.9 Leakage-free validation protocol
- Rolling-origin expanding window; retrain frequency; test period

### 3.10 Evaluation, DM/CW tests and interpretability
- RMSE, MAE, DirAcc, skill vs M0; DM vs M0; Clark–West nested tests vs M1
- **Interpretability rule (refined):**  
  *Interpretability analysis is primarily conducted for the best-performing models that outperform the relevant benchmark. Supplementary SHAP analysis is also reported for models showing statistically significant incremental gains over M1, even where they do not surpass M0.*
- Ethics / reproducibility note (public data; code paths)

---

## Chapter 4 — Results *(~2,500–3,200 words)*

*Organised by experimental logic and research questions — not only “Flat block then Deep block”.*

### 4.1 Descriptive analysis and experimental overview
- Test period, sample size, prediction target, baselines, metrics, significance tests

### 4.2 Flat-model results *(mainly RQ1)*
- M0 vs M1
- M2 / M3 / M4 incremental performance vs M1
- Whether M2 / M3 / M4 beat M0
- Ridge vs XGBoost differences
- CW / DM results; stress that CW significance ≠ beating M0

### 4.3 Deep-model results *(RQ1 within deep arm)*
- Deep M1–M4 vs Deep M0 / no-change
- Effect of adding RS, shipping, and all modalities
- Performance across reported targets / reconstruction metrics

### 4.4 Flat versus deep comparison *(core RQ2)*
- **Paired comparisons by information set**, not only best-vs-best:
  - M1ˢᵈᵉᵉᵖ vs M1ˢᶠˡᵃᵗ
  - M2ˢᵈᵉᵉᵖ vs M2ˢᶠˡᵃᵗ
  - M3ˢᵈᵉᵉᵖ vs M3ˢᶠˡᵃᵗ
  - M4ˢᵈᵉᵉᵖ vs M4ˢᶠˡᵃᵗ
- Separate architecture effect from modality-set effect
- Report where deep wins / fails; avoid over-generalisation

### 4.5 Robustness and sensitivity analysis
- Alternative lookbacks
- Shipping feature arms / subsets
- Random seeds
- Missing modality / modality dropout
- Target definition or evaluation-window variants (as available)
- RS checks (e.g. water-mask, leave-one-AOI-out) where relevant

### 4.6 Interpretability results *(RQ3)*
- Primary: best models with value vs the relevant benchmark (esp. those beating M0)
- Supplementary: short SHAP / gates for models with significant CW gain over M1 (e.g. Flat M3) — framed as *why shipping helps M1*, not as absolute forecast superiority
- Modality / site / chokepoint contributions; association ≠ causation

---

## Chapter 5 — Discussion *(~1,600–1,900 words)*

*Answer RQs; do not re-list tables. Explain mechanisms and meaning.*

### 5.1 Answer to RQ1 — Do alternative data improve forecasting?
- Why RS (M2) gains are limited
- Why shipping (M3) is more stable / informative than RS
- Whether M4 adds complementary information or redundancy
- Why some models beat M1 yet still lose to M0
- Statistical significance vs economic / practical significance

### 5.2 Answer to RQ2 — Is representation-level fusion better than flat concatenation?
- Evidence from **paired** Deep vs Flat comparisons
- Source of any deep advantage: temporal modelling, non-linearity, and/or modality-specific encoding
- Whether gains hold across targets / periods
- Whether added complexity is justified

### 5.3 Answer to RQ3 — Interpretable modality and spatial dependence?
- Gate / SHAP weights across finance, shipping, RS
- Periods when shipping weight rises
- Important ports / chokepoints
- Economic plausibility of attributions
- Explicit limit: model dependence ≠ causal effect

### 5.4 Implications
- Theoretical / empirical: fair multimodal comparison; nested vs absolute skill
- Practical: caution when gains over M0 are small

### 5.5 Limitations
- Weekly horizon; sample length; proxy noise; reverse causality
- Encoder / graph choices; missingness; compute / coverage

### 5.6 Future research
- Longer history for best model; richer AIS graphs; missing-modality stress tests; other horizons / commodities

---

## Chapter 6 — Conclusion *(~400–700 words)*

### 6.1 Summary of findings
- 3–5 sentences answering RQ1–RQ3 without new numbers dumps

### 6.2 Contributions
- Integration + systematic Flat vs Deep comparison under a leakage-safe protocol

### 6.3 Final conclusion
- Strong baselines first; fusion design second; interpretability where predictive value exists

---

## References

## Appendices
- **A.** Full variable dictionaries (M1–M4)
- **B.** Extra result / robustness tables and figures
- **C.** Hyperparameter grids, seeds, software / config notes
- **D.** Supplementary SHAP for incremental-but-not-M0 models (if kept out of main text)
