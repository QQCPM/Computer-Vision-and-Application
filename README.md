# SC5006 

Completed labs and tutorials for the image / vision part of the course, kept in the order they were done.

| Folder | Work | What is here |
|---|---|---|
| `01-lab-color-play` | Lab 01 | RGB channels, HSV masks, hue recoloring, HSV edits. Round-trip error **4.0e-7** |
| `02-lab-images-frequencies-edges` | Lab 02 | Histograms, resizing, Fourier filters, Canny |
| `03-lab-local-features-panorama` | Lab 03 | Harris, SIFT, 10-photo NTU panorama |
| `04-tutorial-ai-for-vision` | Tutorial 04 | CIFAR-10 CNN and ViT. Final test accuracy **0.9767** |
| `05-tutorial-ai-vs-real` | Tutorial 05 | AI vs real images. Final test accuracy **0.98** |
| `06-tutorial-pet-segmentation` | Tutorial 06 | Oxford pet segmentation. Final test mIoU **0.792** |
| `07-tutorial-colmap` | Tutorial 07 | COLMAP SfM on fern. 512 → 4096 SIFT features: 570 → **1757** points, reprojection error 0.89 → **0.62 px** |
| `08-tutorial-nerf-3dgs` | Tutorial 08 | Garden novel view synthesis. Nerfacto PSNR 24.50 / SSIM 0.70 vs Splatfacto (3DGS) PSNR **26.64** / SSIM **0.86** |
| `09-tutorial-depth-point-maps` | Tutorial 09 | Monocular depth vs VGGT point maps. Confidence filtering q=0.9: AbsRel 0.137 → **0.042** |
| `10-tutorial-generative-vision` | Tutorial 10 | SD1.5 + Canny ControlNet garden makeover. Chosen setting strength 0.8 / control 0.5 |

## Not in this repo

- `final_model_vit_b32_finetuned.pth` — Tutorial 04 final ViT-B/32 (~350 MB)
- `Quang_final_model.pth` — Tutorial 05 final model (~350 MB)
