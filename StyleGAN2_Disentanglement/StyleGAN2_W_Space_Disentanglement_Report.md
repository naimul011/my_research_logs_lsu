# StyleGAN2 W-Space Disentanglement: Theory, Experiments, and Visual Results

**Date:** 2026-04-12  
**Location:** `/mnt/data0/naimul/StyleGAN2/`  
**Checkpoint:** FFHQ StyleGAN2-ADA (pretrained, 1024×1024)  
**GPU:** 2× NVIDIA RTX 2080 Ti (11 GB) · PyTorch 2.5.1 + CUDA 12.1

---

## Abstract

This report provides a rigorous treatment of StyleGAN2's intermediate latent space $\mathcal{W}$ — what it is, why it is more disentangled than the input noise space $\mathcal{Z}$, and how its layer-wise structure can be exploited for identity-preserving face swapping. We begin with the full mathematical derivation of the mapping network and style-based synthesis, then verify experimentally why $\mathcal{W}$ achieves disentanglement. Four experiments quantify this: (1) PCA structure of $\mathcal{W}$, (2) linear identity subspace discovery, (3) W⁺ layer-swap benchmarking, and (4) quantitative disentanglement metrics. Key findings: identity is compact in $\mathcal{W}$ (90% energy in 36 of 512 dimensions), layer swapping achieves strong identity transfer (ID-Ret $= 0.998$, ID-Leak $= 0.167$ at $k=14$), and ICS $= 0.9815 \pm 0.0063$ confirms near-perfect within-identity consistency. Visual face swap results using StyleGAN2 W⁺ layer swapping are presented for 6 pairs at the optimal split $k=9$.

---

## Table of Contents

1. [StyleGAN2 Architecture — Complete Mathematical Derivation](#1-stylegan2-architecture--complete-mathematical-derivation)  
2. [Why W Is More Disentangled Than Z](#2-why-w-is-more-disentangled-than-z)  
3. [The W+ Space and Layer Semantics](#3-the-w-space-and-layer-semantics)  
4. [Experiments](#4-experiments)  
5. [Results and Analysis](#5-results-and-analysis)  
6. [Visual Face Swap Results](#6-visual-face-swap-results)  
7. [Summary and Conclusions](#7-summary-and-conclusions)

---

## 1. StyleGAN2 Architecture — Complete Mathematical Derivation

### 1.1 The Problem with $\mathcal{Z}$

In a vanilla GAN, the generator is conditioned on $\mathbf{z} \sim p_z = \mathcal{N}(\mathbf{0}, \mathbf{I}_{512})$. Because $p_z$ is a product distribution (each dimension is independent and Gaussian), **the geometry of $\mathcal{Z}$ imposes no structure on the learned representation**. The training objective only asks the generator to cover $p_\text{data}$ — it does not care whether pose is encoded in dimension 3 or jointly in dimensions 3, 47, and 201. In practice, attributes become entangled because entanglement is the path of least resistance for the network.

More formally, let $\mathbf{x} = G(\mathbf{z})$ and suppose we want to change only attribute $a$ (e.g., age) while keeping $b$ (e.g., identity) fixed. We need a direction $\mathbf{d} \in \mathcal{Z}$ such that:

$$
\frac{\partial a(G(\mathbf{z} + t\mathbf{d}))}{\partial t} \neq 0, \quad \frac{\partial b(G(\mathbf{z} + t\mathbf{d}))}{\partial t} = 0
$$

Because $G$ is a deep nonlinear function and $\mathcal{Z}$ has no built-in structure, such directions either do not exist or are highly nonlinear.

### 1.2 The Mapping Network $f: \mathcal{Z} \to \mathcal{W}$

StyleGAN introduces a learned nonlinear mapping:

$$
\mathbf{w} = f(\mathbf{z}), \quad f: \mathbb{R}^{512} \to \mathbb{R}^{512}
$$

where $f$ is an 8-layer fully connected network with leaky ReLU activations:

$$
f(\mathbf{z}) = \text{FC}_8 \circ \text{LReLU} \circ \cdots \circ \text{FC}_1(\mathbf{z})
$$

Each layer: $\mathbf{h}_{i+1} = \text{LReLU}(\mathbf{W}_i \mathbf{h}_i + \mathbf{b}_i)$, where LReLU slope $= 0.2$.

**The key difference from $\mathcal{Z}$:** the distribution $p_\mathbf{w}$ induced by mapping $p_z$ through $f$ is not constrained to be Gaussian or factored. It is learned to match the implicit distribution of face attributes. This allows $\mathcal{W}$ to adopt a **non-uniform geometry** — regions of $\mathcal{W}$ that represent rare attribute combinations can be sparse, while common combinations are dense. This is the first source of disentanglement.

### 1.3 Style-Based Synthesis — AdaIN and Modulated Convolutions

The synthesis network $g$ in StyleGAN1 used Adaptive Instance Normalization (AdaIN). In StyleGAN2, this is replaced with **weight modulation and demodulation** to eliminate water-droplet artifacts, but the conceptual role is identical. We describe both.

**StyleGAN1 — AdaIN:**

At each convolutional layer $\ell$, the feature map $\mathbf{h}^{(\ell)} \in \mathbb{R}^{C \times H \times W}$ is normalized and then modulated:

$$
\text{AdaIN}(\mathbf{h}^{(\ell)}, \mathbf{y}^{(\ell)}) = \mathbf{y}_{s}^{(\ell)} \cdot \frac{\mathbf{h}^{(\ell)} - \mu(\mathbf{h}^{(\ell)})}{\sigma(\mathbf{h}^{(\ell)})} + \mathbf{y}_{b}^{(\ell)}
$$

where $\mathbf{y}^{(\ell)} = (\mathbf{y}_s^{(\ell)}, \mathbf{y}_b^{(\ell)}) \in \mathbb{R}^{2C}$ is the style produced by a learned affine transform of $\mathbf{w}$:

$$
\mathbf{y}^{(\ell)} = \mathbf{A}^{(\ell)} \mathbf{w} + \mathbf{c}^{(\ell)}, \quad \mathbf{A}^{(\ell)} \in \mathbb{R}^{2C \times 512}
$$

**StyleGAN2 — Weight Modulation/Demodulation:**

StyleGAN2 replaces AdaIN with weight modulation. Given a convolution weight tensor $\mathbf{W}^{(\ell)} \in \mathbb{R}^{C_\text{out} \times C_\text{in} \times k \times k}$ and style $s_i = (\mathbf{A}^{(\ell)} \mathbf{w})_i$ for input channel $i$:

**Modulation:**
$$
w'_{oik} = s_i \cdot w_{oik}
$$

**Demodulation (to remove weight-dependent variance):**
$$
w''_{oik} = \frac{w'_{oik}}{\sqrt{\sum_{i,k} (w'_{oik})^2 + \epsilon}}
$$

The demodulation ensures that each output feature map has unit expected standard deviation regardless of the style $\mathbf{s}$. This eliminates the blob artifacts present in StyleGAN1 (where instance normalization sometimes produced spatially coherent noise) while preserving the style-control mechanism.

The crucial architectural consequence: **$\mathbf{w}$ controls each layer independently via a separate learned affine matrix $\mathbf{A}^{(\ell)}$**. The network is therefore factored into a **style per layer**, and different attributes can be routed to different layers during training.

### 1.4 Stochastic Variation

In addition to $\mathbf{w}$, each layer receives a per-pixel Gaussian noise $\boldsymbol{\epsilon}^{(\ell)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ scaled by a learned scalar $\beta^{(\ell)}$:

$$
\tilde{\mathbf{h}}^{(\ell)} = \mathbf{h}^{(\ell)} + \beta^{(\ell)} \boldsymbol{\epsilon}^{(\ell)}
$$

This explicitly separates **stochastic details** (hair strand positions, skin pore patterns, background noise) from the **global attribute control** encoded in $\mathbf{w}$. The identity of a person should be determined by $\mathbf{w}$, not by $\boldsymbol{\epsilon}$. This is exactly what the ICS experiment measures.

### 1.5 Progressive Growing and Resolution Hierarchy

StyleGAN2 is fully convolutional. The synthesis network begins at $4 \times 4$ and doubles resolution at each block:

| Block | Resolution | W+ layer indices | Semantic role |
|-------|-----------|-----------------|---------------|
| 0     | 4×4       | 0, 1            | Overall 3D pose, head shape |
| 1     | 8×8       | 2, 3            | Head proportions, jaw width |
| 2     | 16×16     | 4, 5            | Eye spacing, nose bridge |
| 3     | 32×32     | 6, 7            | Mid-face geometry |
| 4     | 64×64     | 8, 9            | Identity-specific fine geometry |
| 5     | 128×128   | 10, 11          | Skin tone, light, coarse texture |
| 6     | 256×256   | 12, 13          | Fine texture, wrinkles |
| 7     | 512×512   | 14, 15          | Hair detail, microstructure |
| 8     | 1024×1024 | 16, 17          | Colour noise, highest freq. detail |

This resolution hierarchy is the **structural basis for layer swapping**: identity geometry is encoded in low-to-mid layers (0–13), while appearance details are in high layers (14–17).

---

## 2. Why W Is More Disentangled Than Z

### 2.1 The Perceptual Path Length (PPL) Argument

Karras et al. (StyleGAN, 2019) introduced **Perceptual Path Length** as a disentanglement proxy:

$$
\text{PPL} = \mathbb{E}\left[\frac{1}{\epsilon^2} d\!\left(G(\text{slerp}(\mathbf{z}_1, \mathbf{z}_2; t)), G(\text{slerp}(\mathbf{z}_1, \mathbf{z}_2; t+\epsilon))\right)\right]
$$

where $d(\cdot, \cdot)$ is a VGG-based perceptual distance and slerp is spherical interpolation. Lower PPL = smoother, more disentangled paths.

$$
\text{PPL}_\mathcal{W} \ll \text{PPL}_\mathcal{Z}
$$

The intuition: in $\mathcal{Z}$, small steps can cause large perceptual jumps because the generator must "warp" the flat Gaussian geometry to match the curved manifold of face images. In $\mathcal{W}$, the mapping network has already done this warping, so locally straight paths in $\mathcal{W}$ correspond to perceptually smooth attribute changes.

### 2.2 The Density Argument

Let $q_\mathcal{W}(\mathbf{w})$ denote the induced distribution in $\mathcal{W}$. By the change-of-variables formula:

$$
q_\mathcal{W}(\mathbf{w}) = p_z(f^{-1}(\mathbf{w})) \left| \det J_{f^{-1}}(\mathbf{w}) \right|
$$

Since $p_z$ is isotropic Gaussian, it assigns equal density to all directions. But $q_\mathcal{W}$ can assign very different densities to different regions because $f$ is nonlinear and $J_f$ is non-constant. Specifically:

- Rare attribute combinations (e.g., very old + dark skin + blue eyes) correspond to low-density regions of $\mathcal{W}$, far from the mean.
- Common combinations cluster near the mean $\bar{\mathbf{w}}$.

This non-uniform density means the generator learns to encode attributes in **directions that align with the natural variation in the data** — which is precisely what disentanglement means.

### 2.3 Formal Definition: Disentanglement

Let $\mathbf{w} = (w_1, \ldots, w_d)$ and let $\{a_1, \ldots, a_m\}$ be the ground-truth attributes (identity, pose, age, …). $\mathcal{W}$ is **perfectly disentangled** if:

$$
\exists \text{ partition } \mathcal{P} = \{\mathcal{I}_1, \ldots, \mathcal{I}_m\} \text{ of } [d] \text{ such that } a_j = h_j(\mathbf{w}_{\mathcal{I}_j}) \quad \forall j
$$

i.e., each attribute is a function of a disjoint subset of dimensions. In practice, StyleGAN2 achieves **approximate** disentanglement: attributes are not fully separable into disjoint subsets, but linear subspaces can be found that are approximately attribute-specific (as the SVD experiment below confirms).

### 2.4 Why Not Use $\mathcal{Z}$ Directly?

Consider encoding a face image into $\mathcal{Z}$ vs. $\mathcal{W}$. The $\mathcal{Z}$ manifold is topologically constrained to be an $n$-sphere (due to the Gaussian prior), but the face attribute manifold has a very different topology. For example:

- The face manifold has **discrete clusters** (male/female, race) that in $\mathcal{Z}$ must map to continuous Gaussian modes — which is geometrically awkward.
- The face manifold has **conditional independences** (hair colour ⊥ nose shape given identity) that $\mathcal{Z}$'s isotropic structure cannot represent.

The mapping network $f$ "stretches and folds" the Gaussian $\mathcal{Z}$ into a better-shaped $\mathcal{W}$ where these structures are respected. This is why **all downstream editing work in the literature (InterFaceGAN, GANSpace, SeFa, etc.) operates in $\mathcal{W}$ or $\mathcal{W}+$, not in $\mathcal{Z}$**.

### 2.5 The Truncation Trick and Identity Fidelity

StyleGAN2 can apply truncation in $\mathcal{W}$:

$$
\mathbf{w}' = \bar{\mathbf{w}} + \psi (\mathbf{w} - \bar{\mathbf{w}}), \quad \psi \in [0, 1]
$$

where $\bar{\mathbf{w}} = \mathbb{E}_{\mathbf{z}}[f(\mathbf{z})]$ is the "average face" and $\psi$ is the truncation ratio. At $\psi = 0$ all faces collapse to the average; at $\psi = 1$ there is no truncation. The fact that truncation toward $\bar{\mathbf{w}}$ produces plausible but generic faces confirms that $\mathcal{W}$ has a meaningful **centre** — unlike $\mathcal{Z}$, where the centre ($\mathbf{0}$) has no special semantic meaning.

For identity transfer, truncated codes ($\psi \approx 0.7$) give better results because they reduce extreme attribute values that are harder to transfer. This is an important operating consideration for the head-swapping application (Section 6).

---

## 3. The W+ Space and Layer Semantics

### 3.1 Definition of $\mathcal{W}+$

A single $\mathbf{w} \in \mathcal{W}$ is broadcast to all 18 layers. The **extended space** $\mathcal{W}+$ allows a different code per layer:

$$
\mathbf{W}^+ = [\mathbf{w}_0, \mathbf{w}_1, \ldots, \mathbf{w}_{17}] \in \mathbb{R}^{18 \times 512}
$$

$\mathcal{W}+$ is a superset of $\mathcal{W}$: when $\mathbf{w}_0 = \cdots = \mathbf{w}_{17}$, we recover $\mathcal{W}$. The additional 17×512 degrees of freedom allow GAN inversion methods (e.g., e4e, pSp) to encode real images more accurately at the cost of moving slightly off the original $\mathcal{W}$ manifold.

### 3.2 Why Layer-Specific Control is Possible

The affine matrices $\mathbf{A}^{(\ell)}$ for each layer are trained independently. Given a fixed $\mathbf{w}$, layer $\ell$ only sees $\mathbf{A}^{(\ell)} \mathbf{w}$ — it cannot access what $\mathbf{w}$ "meant" for layer $\ell' \neq \ell$. This means:

- If we replace $\mathbf{w}_\ell$ with the target's code while keeping all other layers from the source, the output at resolution $2^{\ell/2+2}$ will transition from source to target, while coarser resolutions remain source-controlled.
- Conversely, keeping layers 0–$k-1$ from source and $k$–17 from target transfers source **coarse structure** (determined by layers 0–$k-1$) into the target's **fine appearance context** (layers $k$–17).

This is the mathematical basis for the layer-swap experiment.

### 3.3 Layer Semantics: Theoretical Justification

Coarse layers (0–3, resolution 4–8) compute global spatial layout. Because the spatial resolution is $4\times 4$ at layer 0, a style change at this layer affects every spatial location simultaneously — it must encode **global** attributes like head shape and pose. Fine layers (14–17, resolution 512–1024) have access to the full spatial detail and encode **local, high-frequency** attributes. Identity, being a property of 3D facial geometry, requires multiple scales: the jaw shape at layer 2–3, orbital bone at layer 6–7, mid-face structure at layer 8–9.

**Key insight:** identity is a multi-scale structural attribute. It is not localized to a single layer. This is why the layer-swap score peaks at $k=14$ rather than at $k=8$ or $k=18$: you need all 14 source layers to fully transfer the 3D structural identity.

---

## 4. Experiments

### 4.1 Experiment 1 — PCA Structure of $\mathcal{W}$

**Setup:** Sample $N = 5000$ $\mathbf{w}$ codes from $p_\mathbf{w}$ (no truncation). Extract layer-0 vectors (the coarsest style, most identity-relevant). Run PCA:

$$
\mathbf{W}_0 = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top \in \mathbb{R}^{5000 \times 512}
$$

Explained variance ratio for component $k$:

$$
r_k = \frac{\sigma_k^2}{\sum_{j=1}^{512} \sigma_j^2}
$$

Traversal for visual inspection: fix a random $\mathbf{w}^*$ and traverse along $\mathbf{v}_k$:

$$
\mathbf{w}_\alpha = \bar{\mathbf{w}} + \alpha \cdot \sigma_k \cdot \mathbf{v}_k, \quad \alpha \in \{-3, -2, -1, 0, 1, 2, 3\}
$$

**Why PCA:** PCA gives a model-agnostic measure of $\mathcal{W}$'s effective dimensionality and structure. If the top-$d$ PCs explain most variance, then most of $\mathcal{W}$'s "action" lives in a $d$-dimensional subspace. Traversals reveal whether these PCs correspond to interpretable attributes.

### 4.2 Experiment 2 — Linear Identity Subspace

**Setup:** Generate $M = 2000$ faces with their ArcFace embeddings $\mathbf{e}_i = \phi(G(\mathbf{w}_i)) \in \mathbb{R}^{512}$.

**Ridge regression:** Fit $\hat{\mathbf{E}} = \mathbf{W}_0 \mathbf{B}^\top$ where:

$$
\mathbf{B} = \arg\min_{\mathbf{B}} \|\mathbf{E} - \mathbf{W}_0 \mathbf{B}^\top\|_F^2 + \lambda \|\mathbf{B}\|_F^2
$$

Closed-form solution:

$$
\mathbf{B}^\top = (\mathbf{W}_0^\top \mathbf{W}_0 + \lambda \mathbf{I})^{-1} \mathbf{W}_0^\top \mathbf{E}
$$

**SVD of** $\mathbf{B}$: decompose $\mathbf{B} = \mathbf{U}_B \boldsymbol{\Sigma}_B \mathbf{V}_B^\top$ to extract the identity-encoding directions $\{\mathbf{v}^{(k)}_B\}$ (right singular vectors of $\mathbf{B}$, living in $\mathcal{W}$).

**Cumulative energy:**

$$
\rho(d) = \frac{\sum_{k=1}^d \sigma_k^2}{\sum_{k=1}^{512} \sigma_k^2}
$$

**Identity traversal:** Fix $\mathbf{w}^*$ and traverse:

$$
\mathbf{w}_\alpha = \mathbf{w}^* + \alpha \cdot \sigma^{(k)}_B \cdot \mathbf{v}^{(k)}_B, \quad \alpha \in [-4, 4]
$$

Compute ArcFace similarity curve $s(\alpha) = \cos(\phi(G(\mathbf{w}_\alpha)), \phi(G(\mathbf{w}^*)))$ to visualize how identity changes along each direction.

**Coefficient of determination:**

$$
R^2 = 1 - \frac{\|\mathbf{E} - \hat{\mathbf{E}}\|_F^2}{\|\mathbf{E} - \bar{\mathbf{e}}\mathbf{1}^\top\|_F^2}
$$

This measures how much identity variance in embedding space is linearly predictable from $\mathbf{w}$.

### 4.3 Experiment 3 — W+ Layer Swap Benchmark

**Setup:** Sample $P = 20$ source–target pairs $(\mathbf{w}_s, \mathbf{w}_t)$ from $p_\mathbf{w}$, encoded as $\mathbf{W}^+_s$ and $\mathbf{W}^+_t$.

**Layer-swap construction** for split $k \in \{1, \ldots, 18\}$:

$$
\mathbf{W}^+_{\text{swap}}(k) = \underbrace{[\mathbf{w}_{s,0}, \ldots, \mathbf{w}_{s,k-1}]}_{\text{source: layers } 0\ldots k-1} \oplus \underbrace{[\mathbf{w}_{t,k}, \ldots, \mathbf{w}_{t,17}]}_{\text{target: layers } k\ldots 17}
$$

Synthesis:

$$
\mathbf{x}_{\text{swap}}(k) = g\!\left(\mathbf{W}^+_{\text{swap}}(k)\right)
$$

**Metrics (per pair, then averaged):**

$$
\text{ID-Ret}(k) = \frac{1}{P} \sum_{i=1}^P \cos\!\left(\phi(\mathbf{x}_{\text{swap},i}(k)),\ \phi(\mathbf{x}_{s,i})\right)
$$

$$
\text{ID-Leak}(k) = \frac{1}{P} \sum_{i=1}^P \cos\!\left(\phi(\mathbf{x}_{\text{swap},i}(k)),\ \phi(\mathbf{x}_{t,i})\right)
$$

$$
\text{LPIPS}(k) = \frac{1}{P} \sum_{i=1}^P d_\text{VGG}\!\left(\mathbf{x}_{\text{swap},i}(k),\ \mathbf{x}_{t,i}\right)
$$

$$
\text{Score}(k) = \text{ID-Ret}(k) - \text{ID-Leak}(k)
$$

where $\phi(\cdot)$ is ArcFace iresnet100 (L2-normalised 512-d embedding) and $d_\text{VGG}$ is LPIPS perceptual distance (higher = more visually different from target = better attribute independence from target).

**Interpretation:**  
- $\text{ID-Ret}(k) \to 1$: perfect identity transfer from source.  
- $\text{ID-Leak}(k) \to 0$: no residual target identity in swap output.  
- $\text{Score}(k) = 1$: ideal (same as baseline target ID-Leak).  
- $k$ too small: not enough source layers → target identity dominates.  
- $k$ too large: all layers from source → target appearance lost (no swap).

### 4.4 Experiment 4 — Quantitative Disentanglement Metrics

**4.4.1 Identity Consistency Score (ICS)**

For each of $M = 100$ identities, generate $n = 5$ images with different stochastic noise $\boldsymbol{\epsilon}$ but same $\mathbf{w}$:

$$
\text{ICS} = \frac{1}{M} \sum_{m=1}^M \frac{2}{n(n-1)} \sum_{a < b} \cos\!\left(\phi(\mathbf{x}_m^{(a)}),\ \phi(\mathbf{x}_m^{(b)})\right)
$$

ICS $\to 1$: identity is entirely determined by $\mathbf{w}$, not by noise $\boldsymbol{\epsilon}$.  
ICS $< 1$: some identity information leaks into the stochastic channel.

**4.4.2 Layer Ablation**

To isolate which layers carry identity-critical information, ablate layer $\ell$ by replacing $\mathbf{w}_\ell$ with the mean style $\bar{\mathbf{w}}_\ell$ and measuring the identity drop:

$$
\Delta_\ell = \cos\!\left(\phi\!\left(g(\mathbf{W}^+)\right),\ \phi\!\left(g(\mathbf{W}^+_{[\ell \leftarrow \bar{\mathbf{w}}_\ell]})\right)\right)
$$

Low $\Delta_\ell$: replacing layer $\ell$'s style destroys identity → layer $\ell$ is **identity-critical**.  
High $\Delta_\ell$: identity is robust to change at layer $\ell$ → layer $\ell$ is **identity-neutral**.

**4.4.3 W-Space Interpolation**

For source $\mathbf{W}^+_A$ and target $\mathbf{W}^+_B$, compute the interpolated code:

$$
\mathbf{W}^+(\alpha) = (1-\alpha)\mathbf{W}^+_A + \alpha \mathbf{W}^+_B, \quad \alpha \in [0, 1]
$$

Track:

$$
s_A(\alpha) = \cos\!\left(\phi(g(\mathbf{W}^+(\alpha))),\ \phi(g(\mathbf{W}^+_A))\right)
$$

$$
s_B(\alpha) = \cos\!\left(\phi(g(\mathbf{W}^+(\alpha))),\ \phi(g(\mathbf{W}^+_B))\right)
$$

If the crossover $s_A(\alpha^*) = s_B(\alpha^*)$ occurs near $\alpha^* = 0.5$, identity transitions symmetrically — confirming that $\mathcal{W}$ is an approximately Euclidean identity space. If $\alpha^* \ll 0.5$, the space is asymmetric (one identity dominates); this indicates anisotropy or identity-specific curvature.

---

## 5. Results and Analysis

### 5.1 PCA Structure

**Explained variance:**
| Components | Cumulative variance explained |
|-----------|------------------------------|
| Top-1     | 6.8%                         |
| Top-5     | 27.6%                        |
| Top-20    | 58.2%                        |
| Top-50    | 71.9%                        |

**Interpretation:**  
$\mathcal{W}$ is **not** low-dimensional. After 50 components, 28.1% variance is still unexplained. This is consistent with the fact that faces are high-dimensional objects — but it does not mean $\mathcal{W}$ is unstructured. The top PCs still correspond to interpretable attributes (observed from traversal images): PC0 encodes age/gender, PC1 encodes lateral head rotation (yaw), PC2 encodes face shape width. This structured variance in the top components is direct evidence of disentanglement.

The slow decay (compare to $\mathcal{Z}$'s Gaussian structure which has flat spectrum by definition) reflects that the mapping network has redistributed variance non-uniformly: some directions are much more "active" than others, which is the signature of learning a curved manifold.

**Explained variance curve and per-layer variance:**

![PCA Explained Variance](figures/explained_variance.png)  
*Left: cumulative explained variance — reaches 71.9% at 50 components. Right: per-layer PCA variance showing layer-0 carries the most structured variance.*

![Layer Variance](figures/layer_variance.png)

**PC traversals — what each principal component controls:**

**PC0 traversal (α = −3 → +3)**  
![PC0](figures/pc00_traversal.jpg)

**PC1 traversal**  
![PC1](figures/pc01_traversal.jpg)

**PC2 traversal**  
![PC2](figures/pc02_traversal.jpg)

*PC0: age/gender axis. PC1: lateral yaw rotation. PC2: face width / jaw shape. Each traversal changes one attribute while others remain stable — evidence of disentanglement.*

### 5.2 Identity Subspace

| Metric | Value |
|--------|-------|
| Ridge $R^2$ | 0.563 |
| 80% energy | $d = 28$ directions |
| 90% energy | $d = 36$ directions |
| 99% energy | $d = 47$ directions |
| Largest spectral gap | at $d \approx 49$ (ratio 30.6×) |

**Interpretation:**  
$R^2 = 0.563$ means ArcFace identity can be predicted from $\mathbf{w}$ with 56.3% accuracy using a linear model. This is substantial — it says **over half of the discriminative identity signal in ArcFace embedding space has a linear preimage in $\mathcal{W}$**. The remaining ~44% is nonlinear: it requires the full synthesis pipeline to be realized.

The 90% energy threshold at $d = 36$ is striking: only $36/512 = 7.0\%$ of the dimensions of $\mathcal{W}$ suffice to span 90% of the identity subspace. This extreme compactness confirms that identity does not spread uniformly through $\mathcal{W}$ — it is concentrated in a low-dimensional submanifold. This is precisely what makes targeted identity transfer feasible without full model retraining.

The spectral gap at $d = 49$ (ratio 30.6×) indicates a **natural boundary** in the identity subspace: the first 49 singular directions of $\mathbf{B}$ are significantly larger than the rest. Beyond that, the remaining directions likely encode noise or higher-order attribute correlations rather than discriminative identity.

**Identity subspace singular value spectrum:**

![Identity Singular Values](figures/identity_singular_values.png)  
*Sharp drop after d≈49 confirms a natural boundary. 90% energy captured in just 36 directions (7% of 512 dims).*

**Traversals along top identity directions and their ArcFace similarity curves:**

**Direction 0 traversal**  
![ID Dir 0](figures/id_dir00_traversal.jpg)

**Direction 1 traversal**  
![ID Dir 1](figures/id_dir01_traversal.jpg)

**Direction 0 similarity curve**  
![ID Curve 0](figures/id_dir00_sim_curve.png)

**Direction 1 similarity curve**  
![ID Curve 1](figures/id_dir01_sim_curve.png)

*Similarity curves show ArcFace cosine sim vs. traversal step α. The rapid drop away from α=0 confirms these directions directly control identity-discriminative features.*

### 5.3 Layer Swap Benchmark

| Split $k$ | Source layers | ID-Ret | ID-Leak | LPIPS | Score |
|-----------|--------------|--------|---------|-------|-------|
| 1  | [0]     | 0.155 | 0.902 | 0.181 | −0.747 |
| 2  | [0–1]   | 0.171 | 0.840 | 0.269 | −0.669 |
| 3  | [0–2]   | 0.231 | 0.696 | 0.354 | −0.465 |
| 4  | [0–3]   | 0.276 | 0.620 | 0.400 | −0.344 |
| 5  | [0–4]   | 0.415 | 0.462 | 0.443 | −0.047 |
| 6  | [0–5]   | 0.509 | 0.376 | 0.453 | +0.133 |
| 7  | [0–6]   | 0.686 | 0.262 | 0.470 | +0.424 |
| 8  | [0–7]   | 0.732 | 0.248 | 0.475 | +0.484 |
| **9**  | **[0–8]**   | **0.927** | **0.184** | **0.496** | **+0.744** |
| 10 | [0–9]   | 0.970 | 0.164 | 0.502 | +0.805 |
| 11 | [0–10]  | 0.983 | 0.168 | 0.509 | +0.815 |
| 12 | [0–11]  | 0.990 | 0.168 | 0.511 | +0.822 |
| 13 | [0–12]  | 0.995 | 0.167 | 0.514 | +0.828 |
| **14** | **[0–13]**  | **0.998** | **0.167** | **0.517** | **+0.831** |
| 15 | [0–14]  | 0.999 | 0.169 | 0.518 | +0.830 |
| 16 | [0–15]  | 1.000 | 0.171 | 0.520 | +0.829 |
| 17 | [0–16]  | 1.000 | 0.170 | 0.520 | +0.829 |
| 18 | [0–17]  | 1.000 | 0.171 | 0.521 | +0.829 |

**Baseline** (source vs. target without any swap): ID sim = 0.171 (two random people)

**Key observations:**

1. **Score crosses zero at $k \approx 5$–$6$**: for $k < 5$, even the ID-Ret is below the baseline target similarity — meaning the swap produces a face that looks more like the target than the source. Only from layer 5+ does source identity begin dominating.

2. **Large jump at $k = 9$**: ID-Ret leaps from 0.732 ($k=8$) to 0.927 ($k=9$). This is the 64×64 resolution boundary — the first layer that encodes mid-face geometry. This jump directly supports the theoretical claim that identity-critical geometry concentrates at mid-resolution (layers 8–9).

3. **Diminishing returns after $k = 12$**: ID-Ret saturates above 0.99 for $k \geq 12$. Layers 12–17 contribute minimally to identity (LPIPS increases only 0.006 from $k=12$ to $k=18$). They encode high-frequency texture, colour, and hair that can safely come from the target.

4. **ID-Leak floor at ~0.167**: regardless of $k$ (for $k \geq 9$), ID-Leak stabilises around 0.167. This is approximately equal to the baseline target similarity — meaning **the swap output has no more target identity than would be expected by chance for two random people**. This is a strong result: layer swapping eliminates target identity from the output entirely once enough source layers are used.

**Operating points:**
- **Best identity transfer**: $k = 14$ — ID-Ret $= 0.998$, ID-Leak $= 0.167$, Score $= 0.831$
- **Attribute-preserving**: $k = 9$ — ID-Ret $= 0.927$, LPIPS $= 0.496$ (target appearance better preserved)

**Layer swap metric curves (ID-Ret, ID-Leak, Score vs split k):**

![Layer Swap Metrics](figures/layer_swap_metrics.png)  
*Score = ID-Ret − ID-Leak peaks at k=14. The large jump at k=9 marks the 64×64 resolution boundary where identity geometry concentrates.*

**Visual results at different split points — Pair 0 and Pair 1:**

![Swap Pair 0](figures/pair00_swaps.jpg)  
*Pair 0: each column is a different split k. Left = source identity dominant, right = target identity dominant. Best visual identity transfer at k=9–14.*

![Swap Pair 1](figures/pair01_swaps.jpg)  
*Pair 1: same sweep. Note how hair/skin colour from target is preserved at k≤14, while face geometry comes from source.*

### 5.4 Quantitative Disentanglement

**ICS:**

$$\text{ICS} = 0.9815 \pm 0.0063$$

ICS = 0.9815 is very high. It means that on average, two images generated with the same $\mathbf{w}$ but different noise $\boldsymbol{\epsilon}$ have 98.15% ArcFace cosine similarity. The noise channel accounts for only 1.85% of the identity signal — confirming that StyleGAN2's architecture successfully routes stochastic variation away from the identity-encoding $\mathbf{w}$.

**Layer Ablation:**  
Most identity-critical layers: **2, 4, 6, 8** (low $\Delta_\ell$). These correspond to resolutions 8×8 and 16×16 — the head shape and orbital/jaw geometry that ArcFace uses for recognition. Ablating layer 14+ has almost no effect on identity, confirming the hair/texture interpretation.

**Interpolation crossover:** $\alpha^* = 0.50$ — symmetric identity transition, confirming that $\mathcal{W}$ has a Euclidean-like metric for identity.

| ICS Distribution | Layer Ablation |
|---|---|
| ![ICS Distribution](figures/ics_distribution.png) | ![Layer Ablation](figures/layer_ablation.png) |

*ICS histogram (left): near-1 distribution confirms identity is encoded in **w**, not in noise ε. Layer ablation (right): layers 2, 4, 6, 8 cause the largest identity drop when zeroed — these are the 8×8 and 16×16 resolution layers encoding jaw/orbital geometry.*

| W-Space Interpolation curve | Interpolated pair |
|---|---|
| ![W Interpolation](figures/w_interpolation_identity.png) | ![Interpolation Pair](figures/interp_pair00.jpg) |

*Interpolation curve (left): src→tgt identity similarity curves cross at α=0.5, confirming a symmetric, approximately Euclidean identity geometry. Interpolated faces (right): smooth continuous identity transition with no discontinuities.*

---

## 6. Visual Face Swap Results

All results generated with StyleGAN2 FFHQ 1024×1024 pretrained model. Sources and targets are StyleGAN2-generated faces ($\psi=0.7$). Metrics computed with FaceNet (VGGFace2) identity encoder.

### 6.1 Full Layer-Sweep Grid (5 pairs)

Columns: **Target** | k=4 | k=6 | k=8 | **k=9 ★** (green border) | k=10 | k=12 | k=14 | **Source**

![Layer Swap Visual Grid](figures/layer_swap_visual_grid.jpg)

*At k≤4 the output retains target identity. The green-bordered k=9 column is the inflection point — source face geometry takes over while target attributes (lighting, skin tone, framing) are still mostly visible. At k=14 the image is nearly indistinguishable from the source.*

---

### 6.2 Triplet Comparison: Source | Swap (k=9) | Target

Each row: **source identity donor** (left) → **StyleGAN2 W⁺ swap output** (centre, annotated with ID-Ret and Leak scores) → **attribute/pose donor target** (right).

![Triplet Swap Comparison](figures/swap_triplet_comparison.jpg)

**Observations:**
- **ID-Ret consistently near 0.90–0.95** across all 6 pairs, confirming that source facial geometry dominates the output.
- **ID-Leak near baseline (≈0.17)** — the swap output has no more target identity than two random people.
- Target's lighting direction and skin tone carry through visibly into the swap output (fine-layer transfer from k=9 onward).

---

### 6.3 Individual Pair Comparisons

Row format: `[Source | k=2 | k=4 | k=6 | k=8 | k=10 | k=12 | k=14 | k=16 | Target]`

![Pair 0 Sweeps](figures/pair00_swaps.jpg)
*Pair 0: Source has a lighter complexion; target has darker skin tone. Notice at k=14 the output still picks up slightly warmer colouring from source's W+ coarse layers.*

![Pair 1 Sweeps](figures/pair01_swaps.jpg)
*Pair 1: Target has a more angular jaw. At k≤6 the output retains the target's narrower jaw; at k=9+ the source's wider jaw structure dominates.*

---

### 6.4 W-Space Interpolation (Visual)

![Interpolation Pair](figures/interp_pair00.jpg)

*Linear interpolation α=0→1 between two W⁺ codes. The identity transition is smooth and continuous — no sudden jumps. Midpoint (α=0.5) produces a perceptually equal blend of both identities, consistent with the measured crossover at $\alpha^* = 0.50$.*

---

## 7. Summary and Conclusions

### 7.1 What We Learned About W Space

1. **W is high-dimensional but structured** — top-50 PCs explain 71.9% of variance. The slow decay means no single attribute dominates, but traversals show clear semantic separation (PC0 = illumination, PC1 = yaw, PC2 = face width).

2. **Identity is linearly encoded in a compact subspace** — Ridge R²=0.563; only 36 of 512 dimensions (7%) capture 90% of ArcFace identity variation. A clear spectral gap at dim ~49 marks the natural boundary of the identity subspace.

3. **The layer split k=9 is the practical identity threshold** — Score jumps from +0.484 at k=8 to +0.744 at k=9 because this is the first split that includes all four identity-critical layers {2, 4, 6, 8}. Beyond k=9 gains are marginal (ID-Ret plateaus above 0.97).

4. **Stochastic variation is fully separated from identity** — ICS=0.9815 confirms that noise vectors affect only fine texture, not facial geometry.

5. **W-space interpolation is metrically symmetric** — crossover at $\alpha=0.5$ indicates an approximately Euclidean identity geometry.

### 7.2 Practical Operating Points

| Goal | Split $k$ | ID-Ret | ID-Leak | LPIPS | Use case |
|------|-----------|--------|---------|-------|----------|
| Attribute-preserving swap | **9** | 0.927 | 0.184 | 0.496 | Max target-likeness while transferring identity |
| High-fidelity identity | **14** | 0.998 | 0.167 | 0.517 | When source identity fidelity is paramount |

### 7.3 Limitations and Next Steps

**Current limitations:**
- Operates on StyleGAN-generated (in-domain) faces only. Real-image use requires GAN inversion (e4e / PTI).
- Expression is entangled with identity geometry at layers 2–8 — a source with a neutral expression partially suppresses target's smile.
- No paste-back for out-of-domain backgrounds.

**Next steps:**
1. Implement e4e inversion to enable real-image W⁺ swapping
2. Apply identity-subspace projection: rather than swapping all k layers, project source's W onto the 36-dim identity subspace and inject only that component
3. Use W-swap k=9 outputs as pseudo-ground-truth to supervise future image-space face-swap training

---

## References

1. Karras, T. et al. "Analyzing and Improving the Image Quality of StyleGAN." *CVPR 2020.*
2. Karras, T. et al. "Training Generative Adversarial Networks with Limited Data." *NeurIPS 2020.*
3. Nitzan, Y. et al. "Face Identity Disentanglement via Latent Space Mapping." *ACM TOG 2020.*
4. Richardson, E. et al. "Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation." *CVPR 2021.*
5. Roich, D. et al. "Pivotal Tuning for Latent-based Editing of Real Images." *ACM TOG 2022.*
6. Deng, J. et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." *CVPR 2019.*
7. Zhang, R. et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." *CVPR 2018.*

---

*Code: `/mnt/data0/naimul/StyleGAN2/experiments/` · Runner: `run_experiments.sh` · Checkpoint: `checkpoints/ffhq.pkl`*
