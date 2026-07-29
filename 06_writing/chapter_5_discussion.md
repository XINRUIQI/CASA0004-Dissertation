# Chapter 5 — Discussion *(~1,600–1,900)*

## 5.1 RQ1 — Do alternative data help?

RQ1 asked whether remote sensing and shipping add out-of-sample value over financial time series and M0. The answer depends on the contrast, and that dependence is itself the finding. No Flat model beats M0. This aligns with the oil-forecasting literature that treats the no-change forecast as a hard short-horizon bar (Alquist, Kilian and Vigfusson, 2013). Within Flat, shipping still helps relative to M1 while remote sensing does not; sparse or cleaned RS variants do not overturn that ranking.

Under Deep, finance plus shipping yields only a small positive skill versus M0, and adding remote sensing often brings no further gain. Shipping is the more informative alternative modality here, but the absolute margin is narrow. That differs from much of the AIS and satellite oil literature in Chapter 2, which more often shows that vessel or Earth-observation proxies carry information about trade, demand or infrastructure without testing one-week-ahead Brent skill against both a financial baseline and M0. The results do not deny physical-market information in those proxies; they show that, at the weekly Brent horizon under leakage-safe evaluation, such information does not automatically become large absolute forecast gains. Nested increments and absolute skill must be reported together.

## 5.2 RQ2 — Does representation-level fusion beat flat fusion?

RQ2 asked whether modality-aware representation-level fusion outperforms flat feature fusion when data and protocol are fixed. Deep outperforms Flat most clearly once shipping enters; Deep on financial time series alone yields limited gains. The architecture gap opens mainly where relational structure can be preserved—here a shipping network.

This sits between two literatures. Flat early fusion remains the convenient default for classical high-dimensional oil-price learners, but does not keep network structure. Gated and modality-aware models (Arevalo et al., 2017; Gohari et al., 2024) show that separate streams can matter, yet those studies are not weekly Brent designs with AIS–PortWatch graphs and a no-change price benchmark. The paired results therefore complement both: preserving shipping structure can help under matched sets, while the same Deep machinery does not rescue remote sensing. Cross-attention can raise a single-seed ceiling but is less stable across seeds than gated fusion, so the architectural claim remains conditional.

## 5.3 RQ3 — What does the model rely on when value exists?

RQ3 asked whether modality-level interpretability reveals which signals the model relies on. Evidence is restricted to specifications with predictive value, mainly Deep M3. Mean gates put substantial weight on financial time series and shipping, but week-level shipping-gate paths are unstable across seeds. Only the Russia–Ukraine announcement window shows a cross-seed co-rising shipping gate; the Red Sea window is not locked. Hormuz is the only chokepoint focus in the top set for all three seeds.

That reading matches a cautious view of attention and gates: such weights describe operations inside a fitted model and need not identify causal features (Jain and Wallace, 2019). It also differs from monitoring narratives that would treat a single disruption window—or one seed’s chokepoint map—as actionable evidence. The useful claim is narrower: when Deep shipping-inclusive forecasts clear M0, Hormuz is the only spatial focus stable enough for the main text, and event-window gate moves must pass a multi-seed filter.

## 5.4 Implications

For evidence design, a shared evaluation protocol matters as much as a new fusion module. Nested M1 contrasts and absolute M0 contrasts jointly show that shipping can help relative to financial time series while absolute gains remain small. Gains over M0 are too small for a forecasting-breakthrough claim. Near-null Flat results are informative: early fusion of alternative data is not automatically useful.

Two agendas already raised in Chapters 1–2 sharpen what the framework can and cannot do. First, oil-price surprises matter for risk management, hedging and budgeting (Chapter 1). The practical implication is not that Deep M3 should replace existing hedges, but that teams who are offered multimodal “signals” can require the same double test used here—nested gain over financial time series *and* skill versus a no-change bar—before treating those signals as decision-relevant at a weekly horizon. Second, the 2019–2025 window includes the 2022 energy crisis and later market adjustment (Chapter 1). In that setting, AIS–PortWatch and satellite products are often promoted as near-real-time monitors of physical disruption. The results caution against converting a single chokepoint map or a single disruption-window gate spike into an operational alert: only multi-seed-stable foci (here Hormuz) and filters that survive across seeds warrant discussion, and even then as model-dependence diagnostics rather than policy instruments.

More broadly, alternative-data and Earth-observation providers can report nested gains and M0 skill together to avoid overselling nested-only improvements. Methods researchers can reuse the matched Flat–Deep design for other commodities or horizons.

## 5.5 Limitations

The study uses a weekly horizon and a modest scored sample after warm-up, so small skill differences should not be over-read. Alternative-data proxies are noisy and may respond to prices as well as lead them. Frozen Earth-observation embeddings, shipping-graph construction and missingness rules affect Deep results; cross-attention is especially seed-sensitive. Matched Flat–Deep comparisons also differ in model class and capacity, so they isolate the overall modelling strategy more cleanly than a single fusion operator.

## 5.6 Future research

Future work can extend the best Deep specifications to longer history and more seeds; enrich the shipping graph and missing-modality stress tests; and apply the same Flat–Deep protocol to other horizons or related energy commodities. That would test whether the shipping-centred, modest-positive-skill pattern survives outside this weekly Brent window.
