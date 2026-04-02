# Problem Statement and Mathematical Formulation of MICA

Learning to reconstruct **3D content from 2D imagery** is an **ill-posed inverse problem**. In particular, monocular RGB-based reconstruction suffers from **depth-scale ambiguity**, where the true physical scale of a face cannot be determined from a single image without additional constraints.

---

## 1. The Mathematical Problem: Depth-Scale Ambiguity

A 3D point  
\[
x \in \mathbb{R}^3
\]  
on a face is projected to a 2D point  
\[
p \in \mathbb{R}^2
\]  
on the image plane using a projection function and a rigid transformation:

\[
p = \pi(R \cdot x + t)
\]

where:

- \( R \in \mathbb{R}^{3 \times 3} \): rotation matrix  
- \( t \in \mathbb{R}^3 \): translation vector  
- \( \pi(\cdot) \): perspective projection  

---

### Scale Ambiguity

Perspective projection is **invariant to scale**. For any scaling factor  
\[
s \in \mathbb{R}
\]

we have:

\[
p = \pi(s \cdot (R \cdot x + t)) = \pi(R \cdot (s \cdot x) + (s \cdot t))
\]

This means:

- Multiple 3D configurations produce the **same 2D projection**
- The reconstruction problem has **infinitely many solutions**

➡️ As a result, self-supervised methods may achieve strong **2D alignment** but fail to recover **true metric geometry**.

---

## 2. The MICA Framework

**MICA (MetrIC fAce)** addresses this issue using **supervised learning** to predict a metrically accurate neutral face shape from a single image.

---

### 2.1 Identity Encoding

The input image \( I \) is processed through a pretrained ArcFace network followed by a mapping function:

\[
z = M(\text{ArcFace}(I)), \quad z \in \mathbb{R}^{300}
\]

where:

- \( z \): identity latent code  
- \( M \): mapping network  

---

### 2.2 Geometry Decoding

#### (a) Model-Based Decoder (3DMM)

MICA uses the FLAME model to reconstruct geometry:

\[
G_{3DMM}(z) = B \cdot z + A
\]

where:

- \( A \): mean face  
- \( B \): principal components (basis)  

---

#### (b) Model-Free Decoder (SIREN)

A neural implicit representation can also be used:

\[
G_{\text{SIREN}}(z) = S(A \mid M'(z))
\]

where:

- \( S \): SIREN network  
- \( M' \): learned conditioning function  

---

## 3. Training and Tracking Objectives

---

### 3.1 Supervised Reconstruction Loss

The network minimizes the **weighted \( L_1 \) distance** between predicted and ground-truth meshes:

\[
L = \sum_{(I,G) \in D} \left| \kappa_{mask} \left( G_{3DMM}(M(\text{ArcFace}(I))) - G \right) \right|
\]

where:

- \( D \): training dataset  
- \( G \): ground-truth mesh  
- \( \kappa_{mask} \): region-specific weighting (e.g., higher weight on facial features)  

---

### 3.2 Photometric Reproduction Error (Tracking)

For face tracking, parameters \( \phi \) are optimized using dense photometric consistency:

\[
E_{dense}(\phi) = \sum_{i \in V} \left| I(\pi(R \cdot p_i(\phi) + t)) - c_i(\phi) \right|
\]

where:

- \( V \): set of visible vertices  
- \( p_i(\phi) \): 3D vertex position  
- \( c_i(\phi) \): rendered color  
- \( I(\cdot) \): input image  

---

## Summary

- 3D face reconstruction from a single image is **ill-posed** due to **scale ambiguity**
- MICA resolves this using:
  - **Identity-driven latent representation**
  - **Supervised metric learning**
  - **Model-based (FLAME) or model-free (SIREN) decoding**
- Training combines:
  - **Geometry supervision**
  - **Photometric consistency for tracking**
