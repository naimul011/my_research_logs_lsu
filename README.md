# Research Logs — LSU Biomedical AI Lab

Research documentation and experiment reports for video understanding and medical imaging projects.

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
```

---

## Contact

For questions or discussions about these reports:
- Check the documentation in the report for FAQs
- Review the production checklist for deployment guidance

---

**Last Updated:** 2026-04-14
