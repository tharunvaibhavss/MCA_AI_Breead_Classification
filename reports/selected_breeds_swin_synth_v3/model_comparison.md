# Model Generation Comparison

**Generated**: 2026-10-09 00:03:44  

## Comprehensive Multi-Model Benchmark on the Frozen 33-Image Locked Real Test Set

| Model Version | Architecture | Training Pool | Locked Test Top-1 | Locked Test Top-3 | Macro F1 | Correct / 33 | Parameters | Default Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **selected_breeds_swin_v2** | Swin-T | 153 Real + 800 Synth | **78.79%** | **90.91%** | **0.7812** | **26 / 33** | 28.3M | **Current Best / Retained Default** |
| **selected_breeds_swin_synth_v3** | Swin-T | 153 Real + 4,160 Synth | **75.76%** | **100.00%** | **0.7326** | **25 / 33** | 28.3M | Candidate (Evaluated) |
| **convnext_v2** | ConvNeXt-Tiny | 153 Real + 800 Synth | **72.73%** | 93.94% | 0.6657 | 24 / 33 | 28.6M | Retained |
| **selected_breeds_v2** | EfficientNet-B0 | 153 Real + 800 Synth | **69.70%** | 96.97% | 0.6694 | 23 / 33 | 5.3M | Retained |
| **selected_breeds_v4** | EfficientNet-B4 | 153 Real + 4,160 Synth | **51.52%** | 75.76% | 0.5329 | 17 / 33 | 19.3M | Retained |
