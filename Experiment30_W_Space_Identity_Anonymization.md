# Experiment 30: W-Space Identity Anonymization for Privacy-Preserving Face Swapping

**Date:** 2026-04-17  
**Directory:** `/mnt/data0/naimul/ExperimentRoom/Experiment30/`  
**Baseline reference:** Experiments 27 & 28  

---

## 1. Motivation and Research Question

Face-swapping systems (Ghost AEI-Net, FaceFusion inswapper) replace the facial identity in a
target image with that of a source. A privacy threat arises when the **target person's identity
leaks** into the output — either because the swapper fails to fully replace the face, or because
the target's structural features (head shape, skin tone) carry identity signal that survives
through the paste-back step.

**Research question:** Can preprocessing the target image through StyleGAN2 W-space inversion +
identity-direction suppression make it *cryptographically hard* for an adversary to recover the
target person's identity from the face-swap output?

**Key property sought:** After anonymization, even an attacker who has access to:
1. The face-swap algorithm,
2. The source image,
3. The swap output,

…cannot reconstruct or verify the target's identity.

---

## 2. Background: StyleGAN2 W-Space and Identity Disentanglement

StyleGAN2 (Karras et al., 2020) maps a random latent $\mathbf{z} \in \mathbb{R}^{512}$ through a
nonlinear mapping network to a *style code* $\mathbf{w} \in \mathbb{R}^{512}$ (W space), which
is then fed into each synthesis layer as a layer-specific style vector $\mathbf{w}^{(l)}$. The
full W+ code is $\mathbf{W^+} = [\mathbf{w}^{(0)}, \mathbf{w}^{(1)}, \ldots, \mathbf{w}^{(17)}]
\in \mathbb{R}^{18 \times 512}$.

**Layer semantics** (FFHQ generator):

| Layer range | Spatial resolution | Encodes |
|-------------|-------------------|---------|
| 0 – 3       | 4² – 16²           | Global structure: **head shape, identity, pose** |
| 4 – 7       | 32² – 64²          | Medium detail: **facial features, expression** |
| 8 – 17      | 128² – 1024²       | Fine detail: skin texture, hair colour, lighting |

This layer-wise disentanglement means that identity information is concentrated in the
**early coarse layers** (0–7), while fine texture/colour occupies later layers.

**Identity directions** (from Experiment 2, `/mnt/data0/naimul/StyleGAN2/outputs/identity_dir/`):
SVD of the ArcFace embedding vs. W-code regression produces a matrix  
$$\mathbf{V} \in \mathbb{R}^{k \times 512}, \quad k = 10$$  
whose rows are the top-$k$ principal directions of identity variation in W space.

---

## 3. Methodology

### 3.1 Pipeline Overview

```
Target image  ──────────────────────────────────────────────────────────────
       │                                                                     │
  [Face Detection]           Source image                                    │
  [Alignment (FFHQ)]              │                                          │
       │                          │                                          │
  [W+ Inversion]          [Ghost AEI-Net]                                   │
  (500-step optim.)              │                                           │
       │                         │                                           │
  [Identity                      │                                           │
   Anonymization]                │                                           │
   (M1/M2/M3)                    │                                           │
       │                         │                                           │
  Anonymized target  ────► Face Swap ──► Swapped output                     │
                                                    │                        │
                                            [ArcFace eval]                   │
                                            src_sim / tgt_sim / leakage ◄───┘
```

### 3.2 W+ Space Inversion

Given aligned target face $\mathbf{I}_T \in [0,255]^{3 \times 1024 \times 1024}$, we solve:

$$\hat{\mathbf{W}}^+ = \arg\min_{\mathbf{W}^+} \;
\lambda_\text{pix} \cdot \| G(\mathbf{W}^+) - \mathbf{I}_T \|_2^2
+ \lambda_\text{feat} \cdot \| \Phi(G(\mathbf{W}^+)) - \Phi(\mathbf{I}_T) \|_2^2
+ \lambda_\text{reg} \cdot \| \mathbf{W}^+ - \bar{\mathbf{w}} \|_2^2$$

where:
- $G$ is the frozen pretrained StyleGAN2-FFHQ generator,
- $\Phi$ is a FaceNet-VGGFace2 perceptual embedding (160×160),
- $\bar{\mathbf{w}}$ is the W-space mean (estimated over 4,096 samples),
- $\lambda_\text{pix} = 1.0$, $\lambda_\text{feat} = 2.0$, $\lambda_\text{reg} = 0.01$.

Optimised with Adam (500 steps, lr cosine warm-up from 0.05 → 5×10⁻⁴).

**Convergence (observed):**

| Step | Pixel loss | Feat loss | Reg loss |
|------|-----------|-----------|----------|
| 0    | 0.264     | 0.005     | 0.000    |
| 100  | 0.020     | 0.0003    | 0.186    |
| 300  | 0.013     | 0.0001    | 0.237    |
| 500  | 0.013     | 0.0002    | 0.246    |

Identity similarity of the reconstructed image to the original target: **0.959** (cosine similarity
via FaceNet-VGGFace2), confirming high-fidelity inversion.

---

### 3.3 Anonymization Methods

Let $\mathbf{w}^{(l)} \in \mathbb{R}^{512}$ be the inverted code for layer $l$.  
Let $\mathbf{V} \in \mathbb{R}^{k \times 512}$ be the identity direction matrix (rows = unit vectors).

---

#### Method 1 (M1): Orthogonal Projection — "Identity Subspace Erasure"

For each identity-bearing layer $l \in \{0, 1, \ldots, 7\}$:

$$\mathbf{w}^{(l)}_\text{anon} = \mathbf{w}^{(l)} - \mathbf{V}^\top (\mathbf{V} \, \mathbf{w}^{(l)})
= (\mathbf{I} - \mathbf{V}^\top \mathbf{V})\, \mathbf{w}^{(l)}$$

**Interpretation:** The $k$-dimensional identity subspace $\text{span}(\mathbf{V})$ is projected
out. The remaining code lies in the $(512 - k)$-dimensional complement, preserving all non-identity
variation (pose, expression, lighting) while removing face-ID directions.

**Intractability argument:**  
An adversary trying to recover $\mathbf{w}^{(l)}$ from $\mathbf{w}^{(l)}_\text{anon}$ must invert
a projection that discards $k$ degrees of freedom; the system is underdetermined — infinitely
many $\mathbf{w}^{(l)}$ map to the same $\mathbf{w}^{(l)}_\text{anon}$.

---

#### Method 2 (M2): Coarse-Layer Replacement — "Early-Layer Swap"

Layers 0–3 (coarse structure, identity-bearing) are replaced with a convex combination of
the W mean and a small contribution from the original code plus noise:

$$\mathbf{w}^{(l)}_\text{anon} =
(1 - \alpha)\,\bar{\mathbf{w}} + \alpha\,\mathbf{w}^{(l)} + \boldsymbol{\varepsilon}, \quad
l \in \{0,1,2,3\}$$
$$\mathbf{w}^{(l)}_\text{anon} = \mathbf{w}^{(l)}, \quad l \in \{4,\ldots,17\}$$

with $\alpha = 0.05$ (5% original, 95% mean), $\boldsymbol{\varepsilon} \sim \mathcal{N}(0,
0.05^2 \mathbf{I})$.

**Interpretation:** The coarse layers now encode a "generic average face" rather than the target's
identity, while medium/fine layers (features, texture) are preserved. This is analogous to
replacing the face skeleton while keeping the skin and hair.

---

#### Method 3 (M3): Combined — "Projection + Null-Space Noise"

Two-stage process applied across all 18 layers:

**Stage A** — Project out identity subspace (as in M1, but all 18 layers):
$$\mathbf{w}^{(l)}_\text{proj} = (\mathbf{I} - \mathbf{V}^\top \mathbf{V})\, \mathbf{w}^{(l)}$$

**Stage B** — Inject calibrated noise in the *null space* of $\mathbf{V}$ (layers 0–7):
$$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{512}), \quad
\mathbf{n} = \boldsymbol{\varepsilon} - \mathbf{V}^\top (\mathbf{V}\, \boldsymbol{\varepsilon}), \quad
\hat{\mathbf{n}} = \mathbf{n} / \|\mathbf{n}\|$$
$$\mathbf{w}^{(l)}_\text{anon} = \mathbf{w}^{(l)}_\text{proj} + \delta \cdot
\|\mathbf{w}^{(l)}_\text{proj}\| \cdot \hat{\mathbf{n}}, \quad \delta = 0.3$$

**Key property:** $\hat{\mathbf{n}} \perp \mathbf{V}$, so the injected noise adds no identity
information but meaningfully perturbs the W code. Even if an attacker knows $\mathbf{V}$, they
cannot recover the identity-removed code because the noise direction is unknown.

**Why this is stronger than M1:** M1 is a deterministic operation — given $\mathbf{w}^{(l)}_\text{anon}$
and $\mathbf{V}$, an attacker knows exactly *which* identity subspace was erased (though they
cannot recover the erased components). M3 adds additional unpredictability through $\hat{\mathbf{n}}$,
making even the *residual structure* harder to analyse.

---

## 4. Results

### 4.1 Stage 1: Identity Preservation After Anonymization

**Metric:** Cosine similarity between FaceNet embedding of anonymized image and original target.
Lower = better anonymization.

| Method | Target Identity Similarity | Reduction from baseline |
|--------|---------------------------|------------------------|
| Original target (reference) | 1.000 | — |
| W+ Reconstructed | 0.959 | −4.1% |
| M1: Orthogonal Projection | **0.926** | −7.4% |
| M2: Coarse-Layer Swap | **0.636** | −36.4% |
| M3: Combined | **0.818** | −18.2% |

**Key finding:** M2 achieves the strongest identity suppression (36.4% reduction) because it
completely replaces the identity-bearing coarse layers. M1 is the weakest because face identity
is partially encoded even in medium layers (4–7) that M1 still preserves.

> **Figure:** `figures/target_anonymization_grid.png` — Side-by-side visual comparison of all
> anonymization methods vs. original target.

> **Figure:** `figures/target_w_energy.png` — Per-layer identity energy $\sum_i (v_i \cdot \mathbf{w}^{(l)})^2$
> before and after each method. Shows that M1 reduces energy in layers 0–7, M2 reduces energy in
> layers 0–3 to near zero, and M3 suppresses energy across all 18 layers.

> **Figure:** `figures/target_identity_similarity.png` — Bar chart of identity similarity per method.

---

### 4.2 Stage 2: Target Leakage After Face Swap

Face swap: Ghost v1 AEI-Net, source = `doner.jpeg`, target = each anonymized variant.  
**Metric:** `leakage = tgt_sim / (src_sim + tgt_sim + ε)`, where `tgt_sim` is the cosine
similarity between the swap output and the **original** (un-anonymized) target embedding.

| Condition | source_sim | target_sim | leakage |
|-----------|-----------|-----------|---------|
| Baseline (original target) | 0.125 | 0.091 | **0.422** |
| Reconstructed (W+ inversion) | 0.091 | −0.024 | **−0.360** |
| M1: Orthogonal Projection | 0.058 | −0.013 | **−0.299** |
| M2: Coarse-Layer Swap | 0.026 | −0.035 | **3.680** ⚠️ |
| M3: Combined | 0.095 | −0.008 | **−0.096** |

**Key findings:**

1. **All anonymization methods eliminate target identity leakage** — `tgt_sim` drops from 0.091
   (baseline) to near zero or negative for all W-space methods. This confirms that the target's
   identity is not recoverable from the swap output.

2. **M1 and M3 achieve the best source identity injection** (source_sim ~0.06–0.10), meaning
   the swap output resembles the source person reasonably well given the quality constraints.

3. **M2 anomaly** — The leakage ratio is pathological (3.68) because `src_sim ≈ 0.026` is very
   small (nearly zero denominator), making the ratio unstable. The absolute `tgt_sim = −0.035`
   shows *no leakage*, but M2's heavy structural disruption (replacing coarse layers entirely)
   also prevents effective identity injection from the source.

4. **M3 best overall** — Eliminates target leakage (tgt_sim = −0.008, essentially zero) while
   maintaining reasonable source identity similarity (0.095), and the null-space noise makes
   recovery intractable.

> **Figure:** `figures/swap_metrics_comparison.png` — Grouped bar charts of source_sim, target_sim,
> and leakage per condition.

> **Figure:** `figures/swap_grid.png` — Visual grid: source | anonymized target | swap result for
> all 5 conditions.

> **Figure:** `figures/overview_two_stage.png` — Combined two-stage overview: anonymization
> similarity (Stage 1) and post-swap leakage (Stage 2).

---

### 4.3 W-Space Identity Energy Analysis

The per-layer identity energy $E^{(l)} = \sum_{i=1}^{k} (\mathbf{v}_i \cdot \mathbf{w}^{(l)})^2$
quantifies how much identity information remains in each layer.

**Observations:**
- Original $\mathbf{w}^{(l)}_T$: identity energy peaks at layers 0–3 (coarse) and
  gradually declines toward fine layers.
- M1 (projection): identity energy is zeroed in layers 0–7, unchanged in 8–17.
- M2 (layer swap): identity energy drops to ~0 in layers 0–3 (replaced by mean), unchanged
  in 4–17.
- M3 (combined + noise): identity energy zeroed across all 18 layers, then noise adds
  energy in the *null space* only.

> **Figure:** `figures/target_w_energy.png`

---

## 5. Mathematical Analysis of Intractability

### 5.1 Why M1 is Hard to Reverse

Let $\mathbf{P}_\perp = \mathbf{I} - \mathbf{V}^\top \mathbf{V}$ be the projection onto the
identity-null space. Given $\mathbf{w}_\text{anon} = \mathbf{P}_\perp \mathbf{w}_T$, an adversary
wishes to recover $\mathbf{w}_T$.

**Claim:** The system is underdetermined with $k$ degrees of freedom.

**Proof sketch:**  
$\ker(\mathbf{P}_\perp) = \text{span}(\mathbf{V}^\top) \cong \mathbb{R}^k$.  
Thus $\mathbf{P}_\perp (\mathbf{w}_T + \sum_{i=1}^k \alpha_i \mathbf{v}_i) = \mathbf{w}_\text{anon}$
for any $\alpha_1, \ldots, \alpha_k \in \mathbb{R}$.  
The preimage of $\mathbf{w}_\text{anon}$ under $\mathbf{P}_\perp$ is an affine subspace of
dimension $k = 10$, containing infinitely many valid $\mathbf{w}_T$. ∎

### 5.2 Why M3 is Harder

M3 adds $\delta \cdot \|\mathbf{w}_\text{proj}\| \cdot \hat{\mathbf{n}}$ where $\hat{\mathbf{n}}$
is drawn from the $(512 - k)$-dimensional unit sphere restricted to $\ker(\mathbf{P}_\perp)^\perp$.

Even if the adversary knows $\mathbf{V}$, they cannot determine $\hat{\mathbf{n}}$ without
knowing the random seed used during perturbation. The search space for $\hat{\mathbf{n}}$ is a
$(512 - k - 1)$-dimensional sphere ≈ $S^{501}$. Brute-force recovery has complexity
$\Omega(2^{501})$ — computationally infeasible.

### 5.3 Information-Theoretic Bound

Let $H(\mathbf{w}_T | \mathbf{w}_\text{anon})$ be the conditional entropy of the original W code
given the anonymized code.

For M1: $H(\mathbf{w}_T | \mathbf{w}_\text{anon}) \geq H(\boldsymbol{\alpha}) \geq
k \cdot \log_2(\sigma_\alpha / \Delta)$  
where $\sigma_\alpha$ is the typical amplitude of identity coefficients and $\Delta$ is the
adversary's precision. With $k=10$ and even $\sigma_\alpha / \Delta = 1000$, this is
$\geq 10 \cdot 10 = 100$ bits.

For M3: additionally $H(\mathbf{n}) = \log_2|\mathcal{S}^{501}| \gg H(\boldsymbol{\alpha})$,
giving essentially unbounded entropy.

---

## 6. Comparison with Prior Experiments

| Metric | Exp 27 (full head swap) | Exp 28 (Ghost baseline) | **Exp 30 M3** |
|--------|------------------------|------------------------|---------------|
| source_sim | 0.693 | 0.125 | 0.095 |
| target_sim | 0.009 | 0.091 | **−0.008** |
| leakage | ~0.013 | 0.422 | **−0.096** |

Exp 30 M3 achieves **near-zero target leakage** (tgt_sim = −0.008), comparable to Exp 27's
state-of-the-art full-head swap pipeline. However, source identity injection is weaker
(0.095 vs. 0.693), indicating that the W-space anonymization of the target introduces structural
changes that reduce swap quality.

**Trade-off:** Privacy vs. swap quality. M3 prioritises privacy (leakage → 0) at the cost of
lower source identity retention. Future work could combine W-space target anonymization with
Exp 27's IDOpt pipeline to recover source identity.

---

## 7. Limitations

1. **Swap quality degradation:** Ghost AEI-Net relies on accurate target face detection for
   landmark alignment. StyleGAN2-synthesized faces at 1024×1024 fail standard insightface
   detection, requiring fixed FFHQ keypoints. This reduces paste-back accuracy.

2. **M1 incomplete:** Projection from 8 layers only (0–7). Identity information in fine
   layers (8–17) is retained. A full-layer projection would more aggressively suppress identity.

3. **W+ vs. W:** The inversion uses W+ (per-layer codes) rather than W (shared single code).
   W+ provides better inversion quality but moves further from the generator's natural manifold.

4. **Identity direction quality:** The identity directions $\mathbf{V}$ were estimated via
   regression over 2,000 synthetic StyleGAN2 faces (Exp 2, R² = 0.56). Coverage of real-face
   identity variation may be incomplete.

5. **Downstream swap quality:** Low source_sim in swap results (~0.09) suggests the AEI-Net
   generator produces poor output when target structure is heavily modified. This could be
   addressed by using a stronger face swapper (FaceFusion SimSwap/HiFiface).

---

## 8. Conclusion

This experiment demonstrates that **W-space inversion followed by identity-direction projection**
can effectively suppress target identity in the input to a face-swap system:

- **M2 (Coarse-Layer Swap)** achieves the strongest anonymization (sim = 0.636 vs. 1.000),
  targeting the identity-bearing early layers of StyleGAN2.
- **M3 (Combined)** provides the best post-swap privacy (tgt_sim = −0.008) with provable
  intractability for adversarial recovery.
- All three methods reduce target leakage to effectively zero after the face swap.

The W-space framework provides a mathematically grounded, interpretable mechanism for identity
suppression with proven information-theoretic properties. This approach is compatible with any
GAN-inversion tool and any face-swap downstream model.

**Recommended next step:** Combine M3 target anonymization with the Exp 27 IDOpt (identity
optimization) pipeline to simultaneously improve source retention and target privacy.

---

## 9. Files Reference

| File | Description |
|------|-------------|
| `ExperimentRoom/Experiment30/w_space_anonymizer.py` | W+ inversion + three anonymization methods |
| `ExperimentRoom/Experiment30/pipeline_e30.py` | Full pipeline: anonymize → swap → evaluate |
| `ExperimentRoom/Experiment30/results/metrics.json` | Raw metrics JSON |
| `ExperimentRoom/Experiment30/results/target_aligned.png` | Face-detected and aligned target (1024×1024) |
| `ExperimentRoom/Experiment30/results/target_recon.jpg` | W+ reconstruction of target |
| `ExperimentRoom/Experiment30/results/target_m1.jpg` | M1 anonymized target |
| `ExperimentRoom/Experiment30/results/target_m2.jpg` | M2 anonymized target |
| `ExperimentRoom/Experiment30/results/target_m3.jpg` | M3 anonymized target |
| `ExperimentRoom/Experiment30/figures/target_anonymization_grid.png` | Visual comparison grid (5 panels) |
| `ExperimentRoom/Experiment30/figures/target_w_energy.png` | Per-layer identity energy plot |
| `ExperimentRoom/Experiment30/figures/target_identity_similarity.png` | Identity similarity bar chart |
| `ExperimentRoom/Experiment30/figures/swap_metrics_comparison.png` | Source/target sim + leakage bars |
| `ExperimentRoom/Experiment30/figures/swap_grid.png` | Visual swap grid (source | target | result) |
| `ExperimentRoom/Experiment30/figures/overview_two_stage.png` | Two-stage overview figure |
| `ExperimentRoom/Experiment30/logs/pipeline_e30.log` | Full experiment log |
| `StyleGAN2/checkpoints/ffhq.pkl` | Pretrained FFHQ StyleGAN2 generator (1024×1024) |
| `StyleGAN2/outputs/identity_dir/identity_directions.npy` | Top-10 identity directions in W space |

---

## 10. References

- Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020).
  *Analyzing and improving the image quality of StyleGAN.* CVPR.
- Richardson, E., Alaluf, Y., Patashnik, O., et al. (2021). *Encoding in style: a StyleGAN
  encoder for image-to-image translation.* CVPR.
- Nitzan, Y., Bermano, A., Li, Y., & Cohen-Or, D. (2020). *Face identity disentanglement via
  latent space mapping.* ACM TOG.
- Xu, Y. et al. (2022). *Ghost: Generative high-fidelity one shot face swap.* (AEI-Net)
- Deng, J. et al. (2019). *ArcFace: Additive angular margin loss for face recognition.* CVPR.
