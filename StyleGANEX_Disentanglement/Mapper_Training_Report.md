# StyleGANEX Identity-Disentanglement Mapper — Training Report

**Date:** 2026-04-12  
**Based on:** Nitzan et al. (2020) "Face Identity Disentanglement via Latent Space Mapping"  
**Repository:** `/mnt/data0/naimul/StyleGANEX/`  
**Checkpoint dir:** `training_mapper/checkpoints/`  
**Training log:** `training_mapper/train.log`

---

## 1. Motivation

Standard face-swapping replaces the full W+ code, collapsing expression, pose, lighting and
background from the target. The goal here is to learn a mapping network M that takes:

- **Source image** → FaceNet VGGFace2 identity embedding (512-dim)
- **Target image** → InceptionV3 attribute embedding (2048-dim)

and outputs a W+ code [18×512] that, when decoded by the frozen StyleGANEX generator,
produces a face with the **source identity** but **target expression, pose, lighting, and
background**.

---

## 2. Architecture (StyleGANEX-Aware Improvements over Nitzan et al.)

### 2.1 Mapper (StyleGANEXMapper)

Nitzan et al. mapped to a single W code and broadcast it to all layers. Our mapper outputs
true W+ [18×512] with **layer-guided branches** derived from Exp 4 disentanglement findings:

| Layer Set | Assignment | Source |
|-----------|-----------|--------|
| {6, 8} | ID_ONLY — identity branch | Exp 4: high ID, low expr sensitivity |
| {0,1,3,5,7,9–17} | ATTR_ONLY — attribute branch | Exp 4: high expr/background control |
| {2, 4} | ENTANGLED — learnable sigmoid blend | Exp 4: mixed signal |

**Architecture:**
```
Trunk:  cat(id[512], attr[2048]) → Linear(2560→2048) → LReLU → Linear(2048→1024) → LReLU
ID branch:   cat(trunk[1024], id[512])   → Linear(1536→512) → LReLU
Attr branch: cat(trunk[1024], attr[2048])→ Linear(3072→512) → LReLU
Per-layer heads: Linear(512→512) for each of 18 layers
Entangled layers: w[l] = sigmoid(α_l) · id_head + (1−sigmoid(α_l)) · attr_head
```

Total trainable parameters: **~15M** (mapper + discriminator).

### 2.2 Key improvements over Nitzan et al.

| Feature | Nitzan et al. | This work |
|---------|--------------|-----------|
| W+ output | Broadcast single W | True per-layer [18×512] |
| Layer routing | Uniform | Layer-guided (ID/Attr/Entangled) |
| Perceptual loss | None | LPIPS AlexNet (λ=0.1) |
| Synthesis | Full 1024×1024 | Early-stop at 256×256 (VRAM) |
| ID encoder grad | Standard | Gradient checkpointing |
| Landmark dims | 52 pts / 104-dim | 51 pts / 102-dim (lmk 17–67) |

### 2.3 Loss Function

```
L_total = λ_id · L_id + λ_lnd · L_lnd + λ_rec · L_rec + λ_lpips · L_lpips + L_adv_G
L_D     = L_adv_D + (γ/2) · ||∇D(w_real)||²   (R1 gradient penalty)
```

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| L_id | L1(E_id(src), E_id(out)) | λ=1.0 | Identity preservation |
| L_lnd | MSE(lmk_attr, lmk_out) | λ=1.0 | Pose/expression preservation |
| L_rec | α·(1−SSIM) + (1−α)·L1, α=0.84 | λ=0.001 | Self-consistency (rec mode) |
| L_lpips | LPIPS AlexNet(attr, out) | λ=0.1 | Background/texture preservation |
| L_adv_G | softplus(−D(w_fake)) | — | W distribution matching |
| L_adv_D | non-saturating GAN + R1 | γ=10 | Discriminator |

### 2.4 Training Protocol (Nitzan)

- **Every 3rd step:** Reconstruction mode — I_id = I_attr (self-consistency)
- **Other steps:** Disentanglement mode — I_id ≠ I_attr (different random images)
- **Separate optimizers:** opt_nonadv (lr=5e-5) for non-adversarial losses; opt_adv_G (lr=5e-6) for adversarial; opt_D (lr=2e-5)

---

## 3. Dataset

Generated via `training_mapper/data/generate_dataset.py`:

- **70,000 faces** synthesized by StyleGANEX mapping network (truncation ψ=0.7)
- Saved as: `images/{i:06d}.jpg` (256×256 JPEG) + `wplus/{i:06d}.npy` (float16 [18,512])
- Generation time: ~2.5 hours on RTX 2080 Ti (batch_size=4 to fit VRAM)
- Training used 56,092 samples (loaded at init; gen was still in progress at training start)

---

## 4. Training Run

**Command:**
```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u training_mapper/train.py \
    --data_dir training_mapper/data/dataset \
    --out_dir  training_mapper/checkpoints \
    --steps    50000 --batch_size 6 \
    --lambda_id 1.0 --lambda_lnd 1.0 --lambda_rec 0.001 --lambda_lpips 0.1 \
    --log_every 100 --save_every 1000 --vis_every 500
```

**Hardware:** NVIDIA RTX 2080 Ti (11GB), GPU 1 of 2  
**VRAM peak:** ~10.4 GB  
**Step time:** ~0.6 s/step (GPU face_alignment efficient)  
**Estimated total:** ~8.5 hours for 50K steps

---

## 5. Loss Progression

### 5.1 Key milestones

| Step | L_id | L_lnd* | L_lpips | L_adv_G | L_D | Notes |
|------|------|--------|---------|---------|-----|-------|
| 0 | 0.051 | 0.269 | 0.088 | 0.662 | 8.3 | Initialization |
| 100 | 0.046 | 0.270 | 0.057 | 0.002 | 12.0 | adv_G converged fast |
| 500 | 0.044 | 0.221 | 0.052 | 0.003 | 17.4 | L_D climbing |
| 1000 | 0.045 | — | 0.059 | 0.011 | 23.8 | L_D near peak |
| 2000 | 0.049 | 0.308 | 0.090 | 1.290 | 8.6 | GAN transition |
| 5000 | 0.045 | 0.310 | 0.071 | 0.439 | 4.3 | Stabilizing |
| 10000 | 0.045 | 0.309 | 0.058 | 0.673 | 2.0 | L_D collapsing |
| 15000 | 0.037 | 0.002 | 0.041 | 0.687 | 1.4 | Equilibrium reached |
| 20000 | 0.030 | 0.311 | 0.042 | 0.680 | 1.4 | L_id strong decline |
| 25000 | 0.024 | 0.058 | 0.046 | 0.686 | 1.4 | Continued improvement |
| 28000 | 0.021 | 0.107 | 0.044 | 0.702 | 1.4 | Current (in progress) |

*L_lnd varies heavily by batch: near 0 when face detection fails, ~0.2–0.3 when detected.

### 5.2 Overall improvement (step 0 → step 28K)

| Metric | Step 0 | Step 28K | Relative improvement |
|--------|--------|----------|----------------------|
| L_id   | 0.051  | 0.021    | **−59%** |
| L_lpips | 0.088 | 0.044   | **−50%** |
| L_D    | 8.3    | 1.4      | **−83%** |
| L_adv_G | 0.66  | 0.70    | Stable equilibrium |

### 5.3 Training phases observed

**Phase 1 (steps 0–1000): Adversarial collapse**
- L_adv_G dropped to near 0 immediately (mapper quickly fools discriminator)
- L_D climbed from 8 → 24 (discriminator strengthening, R1 penalty inflating)
- L_lpips: 0.088 → 0.059 (steady improvement)

**Phase 2 (steps 1000–5000): GAN rebalancing**
- L_D collapsed from 24 → 4 (discriminator-generator reached new equilibrium)
- L_adv_G rose to 0.4–1.3 (discriminator got smarter; harder to fool)
- Temporary L_lpips spike to 0.09 (expected during GAN transition)

**Phase 3 (steps 5000–15000): Convergence**
- L_D settled at ~1.4–2.0 (stable minimax equilibrium)
- L_adv_G stabilized at ~0.65–0.70
- L_id began strong decline: 0.045 → 0.037

**Phase 4 (steps 15000–28000): Refinement**
- L_id continuing to improve: 0.037 → 0.019 (best seen)
- L_lpips stable at 0.041–0.046
- Blend weights slowly drifting: layers 2,4 both converging to ~0.514 (slight ID preference)

---

## 6. Learnable Blend Weights

The entangled layers (2 and 4) have learnable sigmoid blend weights α:
- `w[l] = sigmoid(α_l) · id_branch_out + (1 − sigmoid(α_l)) · attr_branch_out`

| Step | Layer 2 (sigmoid) | Layer 4 (sigmoid) |
|------|-------------------|-------------------|
| 0    | 0.5000 | 0.5000 |
| 5000 | 0.5015 | 0.4994 |
| 10000 | 0.5020 | 0.4992 |
| 15000 | 0.5023 | 0.4992 |
| 20000 | 0.5037 | 0.4999 |
| 25000 | 0.5095 | 0.5090 |
| 28000 | 0.5137 | 0.5143 |

Both layers converging to ~0.514 — a slight but growing preference for identity signal in the
entangled layers. Layer 4 was briefly below 0.5 (preferring attr branch) then corrected
after step 20K. This asymmetric evolution suggests the mapper is discovering layer-specific
optimal blends rather than treating entangled layers uniformly.

---

## 7. Implementation Notes & Bug Fixes

### 7.1 IdentityEncoder gradient fix
**Bug:** `@torch.no_grad()` on `IdentityEncoder.forward` blocked the identity loss gradient
from reaching the mapper. L_id was computed but had no gradient path through E_id(imgs_out).

**Fix:** Removed decorator; added conditional gradient checkpointing:
```python
if torch.is_grad_enabled():
    emb = checkpoint(self.model, x, use_reentrant=False)
else:
    emb = self.model(x)
```
This recomputes FaceNet activations during backward instead of storing them, trading compute
for VRAM (avoids storing full InceptionResnetV1 intermediate activations).

### 7.2 LandmarkEncoder size mismatch
**Bug:** `lmk_68[17:]` → 51 landmarks × 2 = 102 coords, but fallback zeros used 104.
`torch.stack` failed with mixed [102] and [104] tensors.

**Fix:** Standardized to 102-dim throughout; updated `N_LANDMARKS=51`, `DIM=102`.

### 7.3 Synthesize 256 with grad
The StyleGANEX 1024×1024 generator was OOMing during training with encoders loaded.
**Fix:** Stop synthesis at loop iteration j=5 (256×256), using only W+[0:14] codes.
This reduces synthesis VRAM from ~7GB to ~3GB, enabling batch_size=6 on 11GB.

### 7.4 L_lnd noisy due to face detection failures
Face_alignment (face detection) fails on ~30–50% of StyleGANEX-generated images,
returning zero landmarks. This makes L_lnd unreliable as a per-step signal but it still
provides meaningful gradient signal for successful detections.

**Mitigation considered:** Pre-compute and cache landmarks for all 70K images. Not
implemented yet — would eliminate per-step detection overhead and make L_lnd stable.

---

## 8. VRAM Budget (batch_size=6)

| Component | VRAM |
|-----------|------|
| StyleGANEX decoder (frozen) | ~1.5 GB |
| FaceNet VGGFace2 (frozen) | ~0.1 GB |
| InceptionV3 (frozen) | ~0.2 GB |
| face_alignment (frozen) | ~0.3 GB |
| LPIPS AlexNet (frozen) | ~0.1 GB |
| Mapper + Discriminator | ~0.2 GB |
| Activations + gradients | ~8.0 GB |
| **Total** | **~10.4 GB** |

---

## 9. Next Steps

1. **Wait for training completion** (~22K steps remaining, ~3.6 hours from step 28K)
2. **Run inference** on held-out source/target pairs:
   ```bash
   ./run_mapper.sh swap --source src.jpg --target tgt.jpg --output out.jpg --show_grid
   ```
3. **Quantitative evaluation:** Measure on test pairs:
   - ID-Ret: cosine similarity between E_id(src) and E_id(out) (↑ better)
   - Expr-KL: KL divergence on DeepFace emotion distributions between tgt and out (↓ better)
   - LPIPS(tgt, out): perceptual similarity to target (↓ better)
4. **Pre-compute landmarks** for next training run to stabilize L_lnd
5. **Blend weight analysis** at final checkpoint: visualize what layer 2 and 4 have learned

---

## 10. Checkpoints

Saved every 1000 steps to `training_mapper/checkpoints/`:
- `step_001000.pt` through `step_028000.pt` (28 checkpoints as of report time)
- `latest.pt` — always the most recent
- Visualizations every 500 steps in `checkpoints/visualizations/step_NNNNNN.jpg`

Resume training: `./run_mapper.sh resume`
