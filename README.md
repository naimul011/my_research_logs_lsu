# Research Logs — LSU Biomedical AI Lab

Research documentation and experiment reports for video understanding, generative models, and face analysis projects.

## Projects

### StyleGAN2 W-Space Disentanglement
**Status:** ✓ Complete
- Investigation of PCA Structure of W
- Linear Identity Subspace Analysis
- Layer-swapping properties

### NICU Distress Detection with VideoMAE
**Status:** ✓ Complete
End-to-end temporal video understanding pipeline for neonatal distress detection using VideoMAE-Base pretrained on Kinetics-400.

- **Model:** VideoMAE-Base (ViT-B backbone, 2-class binary classifier)
- **Dataset:** Cope_Clips_II (650 videos, 42 subjects, subject-aware split)
- **Performance:** Val AUC = 0.9960 (epoch 5), subject-aware generalization
- **Hardware:** 2× NVIDIA RTX 2080 Ti (11GB VRAM each)
- **Framework:** PyTorch 2.5, HuggingFace Transformers, OpenCV

**Key Features:**
- Subject-aware train/val/test split (prevents data leakage)
- Backbone freeze schedule (3 epochs) + differential LR fine-tuning
- Markov smoothing for temporal coherence (4-state trajectory)
- Gradient-based intensity visualization with alert detection
- Real-time detection overlay video generation

---

## Repository Structure

```
ResearchLogs/
├─ README.md                                  (this file)
├─ StyleGAN2_Disentanglement/
│  ├─ StyleGAN2_W_Space_Disentanglement_Report.md
│  └─ W_Space_Analysis.md
└─ VideoMAE_DistressMonitor_Experiment_Report.md
<<<<<<< HEAD
=======
   ├─ 1. Executive Summary
   ├─ 2. Background & Motivation
   ├─ 3. System Architecture
   ├─ 4. Implementation Details
   ├─ 5. Experimental Results
   ├─ 6. Model Performance
   ├─ 7. File Structure
   ├─ 8. Usage Instructions
   ├─ 9. Technical Design Decisions
   ├─ 10. Limitations & Future Work
   ├─ 11. Reproducibility
   ├─ 12. Key Takeaways
   └─ 13. Configuration Reference
```

---

### StyleGAN2 W-Space Disentanglement

**Status:** ✓ Complete

Study of StyleGAN2's W-latent space disentanglement as a basis for training-free face swapping. Four experiments: PCA structure, identity subspace regression, W⁺ layer-swap benchmarking, and quantitative ICS/ablation/interpolation metrics.

- **Model:** FFHQ 1024×1024 StyleGAN2-ADA (NVlabs pretrained)
- **Identity metric:** FaceNet (VGGFace2), LPIPS AlexNet
- **Key result:** ID-Ret=0.927 at layer split k=9 (training-free); identity subspace is ~36-dim of 512-dim W
- **Hardware:** 2× NVIDIA RTX 2080 Ti · PyTorch 2.5.1 + CUDA 12.1

**Key Files:**
- `StyleGAN2_Disentanglement/StyleGAN2_W_Space_Disentanglement_Report.md` — Full report with maths and visual results
  - 1. StyleGAN2 architecture (full derivation: mapping network, AdaIN, weight modulation)
  - 2. Why W is more disentangled than Z (PPL, density, formal definition)
  - 3. W+ space and layer semantics
  - 4. Experiment methodology
  - 5. Results and analysis (PCA, identity subspace, layer swap table, ICS)
  - 6. Visual face swap results (layer-sweep grid, triplet comparison, interpolation)
  - 7. Summary and practical operating points
- `StyleGAN2_Disentanglement/figures/` — All plots and swap images

**Code Location:** `/mnt/data0/naimul/StyleGAN2/experiments/`

---

## How to Use These Reports

1. **Start here:** `VideoMAE_DistressMonitor_Experiment_Report.md` for complete overview
2. **For implementation:** See Sections 4-8 for code details and usage
3. **For troubleshooting:** Section 9 covers technical decisions and FAQs
4. **For deployment:** Section 11 has production checklist

---

## Citation

If using these methods or reports in your research, please cite:

```
VideoMAE Distress Detection Pipeline (2026)
LSU Biomedical AI Lab
Experiment: NICU Temporal Video Understanding
Model: VideoMAE-Base fine-tuned on Cope_Clips_II
>>>>>>> dba140e (StyleGAN2 W-space report: remove Exp25/26 refs, add visual swap results)
```

---

## Contact

For questions or discussions about these reports:
- Check the documentation in the report for FAQs
- Review the production checklist for deployment guidance

---

**Last Updated:** 2026-04-14
