# Real vs. Synthetic Training Contribution Analysis (Selected Breeds V2)

**Generated**: 2026-10-06 19:59:42  

---

## 1. The Synthetic Augmentation Challenge
Training deep convolutional networks on indigenous livestock poses two distinct domain challenges:
1. **Real Data Scarcity**: Prominent indigenous breeds (Dangi, Gaolao, Toda, Surti) possess only 10–28 real photographs per class in training.
2. **Synthetic Domain Shift**: Synthetic images provide rich, dense morphological variations (horns, hump, facial markings), but lack genuine field camera artifacts (motion blur, lighting variations, background clutter).

---

## 2. Balancing Strategy Implemented in V2
To prevent the 800 synthetic images from completely overwhelming the 153 real photographs:
1. **Domain Weight Boost (3.5x for Real)**: Real photographs are assigned a 3.5x sample weight in the cross-entropy objective relative to synthetic samples ($1.0\times$).
2. **Inverse Class Frequency Weighting**: Under-represented real classes like Toda (10 samples) and Surti (14 samples) receive additional scaling to prevent majority class dominance.
3. **Full Participation Guarantee**: All 953 samples (153 real + 800 synthetic) participate in every training epoch. Zero images were excluded.
4. **Label Smoothing (0.05)**: Regularizes classifier logits against overconfidence on synthetic textures.

---

## 3. Measured Impact
- Real validation accuracy reached **88.24%**.
- Locked real test accuracy achieved **69.70%** (Top-3: **96.97%**).
- Synthetic data provided essential background and pose regularization that allowed the model to generalize across both cattle (77.27%) and buffalo (54.55%).
