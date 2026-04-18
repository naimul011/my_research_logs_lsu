# Experiment 26 Report — Real-to-Real FSGeomNet (CelebA-HQ)

**Date:** 2026-04-16  
**Workspace:** /mnt/data0/naimul  
**Experiment Folder:** /mnt/data0/naimul/ExperimentRoom/Experiment26

---

## Abstract

Experiment 26 evaluates FSGeomNet on real-to-real face swapping using CelebA-HQ crops. The model is fine-tuned with W-space swap supervision to improve geometric consistency and reduce the visible seam artifacts seen in prior experiments. Results show strong source identity retention with low target leakage across 30 real-to-real pairs.

---

## Setup

- **Model:** FSGeomNet initialized from Exp25 (checkpoint epoch_059)
- **Fine-tuning signal:** W-swap supervision with $k=9$ (source layers 0–8, target layers 9–17)
- **Source set:** 000951–000980 (30 images)
- **Target set:** 001001–001030 (30 images)
- **Alignment:** InsightFace detection + 256x256 ArcFace landmark alignment
- **Output:** raw 256x256 crops (no paste-back)

---

## Metrics

Let $\phi(\cdot)$ be the ArcFace embedding (L2-normalized). For a swap output $x_{swap}$, source $x_s$, and target $x_t$:

$$
\text{ID-Ret} = \cos\left(\phi(x_{swap}), \phi(x_s)\right)
$$

$$
\text{ID-Leak} = \cos\left(\phi(x_{swap}), \phi(x_t)\right)
$$

$$
\text{SepGap} = \text{ID-Ret} - \text{ID-Leak}
$$

Higher ID-Ret and SepGap are better. Lower ID-Leak is better.

---

## Results Summary (n = 30)

| Metric | Mean | Std | Min/Max |
|---|---:|---:|---:|
| ID-Ret | 0.9635 | 0.0222 | min 0.8684 / max 0.9886 |
| ID-Leak | 0.0572 | 0.0501 | max 0.1814 |
| SepGap | 0.9063 | 0.0539 | — |

**Change vs. Exp25 (StyleGAN source):**
- $\Delta$ID-Ret = +0.0225
- $\Delta$ID-Leak = -0.0287
- $\Delta$SepGap = +0.0512

---

## Figures

**A. KDE of identity metrics (Exp26 vs Exp25):**

![KDE: source fidelity vs target leakage](../ExperimentRoom/Experiment26/analysis_real/A_kde.png)

**B. Fidelity vs leakage scatter (Exp26 vs Exp25):**

![Scatter: fidelity vs leakage](../ExperimentRoom/Experiment26/analysis_real/D_scatter_comparison.png)

**C. Cross-experiment comparison:**

![Cross-experiment bars](../ExperimentRoom/Experiment26/analysis_real/E_cross_experiment.png)

**D. Pair-wise leakage and fidelity heatmaps:**

![Heatmaps](../ExperimentRoom/Experiment26/analysis_real/F_heatmaps.png)

---

## Qualitative Examples

**Example pair 00 (src | tgt | swap):**

![Pair 00 comparison](../ExperimentRoom/Experiment26/results_real/comparisons/pair00_comparison.png)

**Example pair 09 (src | tgt | swap):**

![Pair 09 comparison](../ExperimentRoom/Experiment26/results_real/comparisons/pair09_comparison.png)

---

## Notes and Interpretation

- The distribution of ID-Ret is tightly concentrated near 0.96+, indicating strong source identity transfer.
- ID-Leak remains low (mean 0.057), with a small tail up to 0.18, consistent with occasional target bleed-through.
- The positive SepGap and improvement over Exp25 suggest that W-swap supervision improves geometric fidelity without increasing leakage.

---

## Reproducibility

Key scripts:
- Inference: /mnt/data0/naimul/ExperimentRoom/Experiment26/inference_real.py
- Analysis: /mnt/data0/naimul/ExperimentRoom/Experiment26/analysis_real.py
- Fine-tuning: /mnt/data0/naimul/ExperimentRoom/Experiment26/finetune.py
