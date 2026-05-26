# Reading Note — Paper 05: Multimodal Forecasting with Heterogeneous Data

## Citation

Emami, H., Dang, X.-H., Shah, Y., & Zerfos, P. (2024). Modality-aware Transformer for Financial Time Series Forecasting. In *Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF 2024)*. ACM.

- **arXiv**: [2310.01232](https://arxiv.org/abs/2310.01232) (Preprint: October 2023)
- **Published at**: ICAIF 2024
- **Affiliation**: IBM T.J. Watson Research Center

---

## Core Method

- **Architecture**: Modality-aware Transformer (MAT), a novel multimodal transformer that integrates categorical text and numerical time-series data for target time-series forecasting.
- **Key innovation — Feature-level attention**: Before computing temporal attention, MAT applies learned feature-level attention weights within each modality to focus on the most relevant features, effectively performing automatic feature selection.
- **Three attention mechanisms**:
  1. **Intra-modal MHA**: Captures temporal patterns and feature importance within each individual modality (e.g., within numerical economic indicators, or within text embeddings separately).
  2. **Inter-modal MHA**: Learns cross-modal interactions by attending jointly across modalities, enabling the model to discover how signals in one modality relate to patterns in another.
  3. **Target-modal MHA (Decoder)**: A decoder-side attention mechanism that focuses specifically on extracting information from all modalities relevant to the target time-series prediction.
- **Input modalities**: Financial text (earnings calls, news) encoded via pre-trained language models; numerical time-series (macroeconomic indices, market indicators, company financials).
- **Evaluation**: Tested on financial datasets including macroeconomic forecasting and company-level prediction tasks; compared against unimodal baselines and existing multimodal approaches.

---

## Key Findings

1. **MAT outperforms unimodal models** (both text-only and numerical-only) across all evaluation settings, confirming that multimodal fusion provides genuinely complementary information.
2. **Feature-level attention is critical**: Ablation studies show that removing the feature-level attention layer significantly degrades performance, demonstrating that not all features within a modality are equally useful and dynamic selection is essential.
3. **Cross-modal attention captures non-trivial interactions**: The inter-modal MHA reveals attention patterns where specific textual signals (e.g., management tone in earnings calls) correlate with shifts in numerical economic indicators, providing interpretable cross-modal insights.
4. **Modality-aware design outperforms naive concatenation**: Simply concatenating features from different modalities and feeding them into a standard Transformer performs worse than the structured modality-aware approach, highlighting the importance of respecting modality boundaries in the architecture.
5. **Robust to modality noise**: The feature-level attention mechanism acts as an implicit regulariser, reducing the model's sensitivity to noisy or irrelevant features within any single modality.

---

## Relevance to This Dissertation

| Aspect | Connection |
|--------|------------|
| **Multimodal fusion architecture** | MAT's three-level attention design (intra-modal → inter-modal → target-modal) provides a directly applicable architectural template for fusing the dissertation's four data modalities (market, NLP, remote sensing, shipping). |
| **Feature-level attention** | The learned feature importance within each modality addresses a key challenge in the dissertation: different remote sensing indices, shipping features, and NLP sentiment scores have varying relevance over time. |
| **Text + numerical fusion** | The paper demonstrates how to jointly model text (NLP/sentiment) and numerical time-series (market fundamentals), which directly maps to two of the dissertation's four modalities. |
| **Scalable to more modalities** | The modular intra-/inter-modal design can be extended to accommodate additional modalities (satellite imagery features, AIS-derived shipping indices) beyond the two modalities studied in the paper. |
| **Interpretability via attention** | Cross-modal attention weights provide a built-in interpretability mechanism, allowing the dissertation to analyse which modality drives predictions at different time periods (e.g., does shipping data become more important during supply disruptions?). |

---

## Limitations

1. **Only two modalities tested**: The paper fuses text and numerical data only; it does not include image/spatial modalities (remote sensing, AIS trajectories), so the scalability to 4+ modalities is hypothetical and untested.
2. **Financial sector focus**: Experiments target equity markets and macroeconomic indicators, not commodity/energy markets; transferability to crude oil forecasting needs empirical validation.
3. **No commodity-specific evaluation**: The paper does not test on oil price forecasting, leaving a gap in understanding how well the architecture handles the unique dynamics of energy markets (supply shocks, OPEC decisions, exogenous disruption risk).
4. **Computational cost**: The three-tier attention mechanism introduces significant computational overhead compared to standard Transformers; scalability to large feature sets or long sequences is not analysed.
5. **Static modality structure**: The architecture assumes a fixed set of modalities at training time; it does not handle missing modalities gracefully (e.g., if satellite data is unavailable for certain months due to cloud cover).

---

## Notes for Dissertation Integration

- Adopt the MAT architecture as a starting point or inspiration for the dissertation's multimodal fusion model, extending it from 2 to 4 modalities.
- Implement the feature-level attention mechanism to handle the heterogeneous feature spaces: dense numerical vectors (market data), sentiment scores (NLP), spectral indices (remote sensing), and voyage statistics (AIS).
- Use the intra-modal / inter-modal / target-modal decomposition to analyse which data modality contributes most to Brent price forecasting and how this changes across different market regimes.
- Cite the "naive concatenation vs. modality-aware design" finding to justify a structured fusion approach rather than simply stacking all features into a single model.
- Address the missing-modality limitation by designing a modality-dropout or masking strategy during training.
