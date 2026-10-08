# Multi-Model Benchmark Comparison (Selected Breeds)

**Generated**: 2026-10-08 00:16:05  
**Evaluation Scope**: Immutable Frozen Real Test Set (33 Real Photographs, zero synthetic contamination)  

---

## Head-to-Head Comparison Table

| Architecture | Model Version | Params | Locked Test Top-1 | Locked Test Top-3 | Test Macro F1 | Test Cattle Acc | Test Buffalo Acc | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | `selected_breeds_v2` | 5.3M | **69.70% (23/33)** | **96.97% (32/33)** | 0.6694 | 81.82% | 45.45% | ~120 ms |
| **ConvNeXt-Tiny** | `convnext_v2` | 28.6M | **72.73% (24/33)** | 93.94% (31/33) | 0.6657 | 81.82% | 54.55% | ~185 ms |
| **Swin-Transformer (Swin-T)** | `swin_v2` | 28.3M | **78.79% (26/33)** | 90.91% (30/33) | 0.7812 | 86.36% | 63.64% | ~210 ms |
| **EfficientNet-B4** | `selected_breeds_v4` | 19.3M | **51.52% (17/33)** | **75.76%** | 0.5329 | 68.18% | 18.18% | 261.90 ms |
