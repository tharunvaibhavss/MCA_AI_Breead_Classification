# Application Integration Report: Swin-T Synthetic V3

**Generated**: 2026-10-09 00:03:44  

## 1. Backend FastAPI Registration

- **New Model Key**: `selected_breeds_swin_synth_v3` (also aliases `swin_synth_v3`, `swin_v3`)
- **Default Model Policy**: Per Section 27, `selected_breeds_swin_synth_v3` is registered as a fully selectable model in FastAPI and frontend.
- **Production Default Status**: As instructed, if `selected_breeds_swin_synth_v3` achieves higher than 78.79% on locked test, it becomes the production default; otherwise `selected_breeds_swin_v2` is maintained as the production default.

## 2. Frontend Next.js Integration

- Added `selected_breeds_swin_synth_v3` to the model selector options with full Grad-CAM visualization, Top-3 display, and latency tracking.
