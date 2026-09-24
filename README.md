# SC5006 image AI work

Completed labs and tutorials for the image / vision part of the course, kept in the order they were done.

| Folder | Work | What is here |
|---|---|---|
| `02-lab-images-frequencies-edges` | Lab 02 | Histograms, resizing, Fourier filters, Canny |
| `03-lab-local-features-panorama` | Lab 03 | Harris, SIFT, 10-photo NTU panorama |
| `04-tutorial-ai-for-vision` | Tutorial 04 | CIFAR-10 CNN and ViT. Final test accuracy **0.9767** |
| `05-tutorial-ai-vs-real` | Tutorial 05 | AI vs real images. Final test accuracy **0.98** |
| `06-tutorial-pet-segmentation` | Tutorial 06 | Oxford pet segmentation. Final test mIoU **0.792** |

## Not in this repo

GitHub rejects single files over 100 MB. These stay in `~/Downloads`:

- `final_model_vit_b32_finetuned.pth` — Tutorial 04 final ViT-B/32 (~350 MB)
- `Nguyen_Nhat_Quang_final_model.pth` — Tutorial 05 final model (~350 MB)

Tutorial 05 also expects the course dataset `AIvsReal_train_test/` next to the notebook (`~/Downloads/AIvsReal/`). CIFAR-10 is downloaded by Tutorial 04 on first run.

Smaller checkpoints are in `04-tutorial-ai-for-vision/checkpoints/`. Tutorial 06 includes `best_pet_segmentation.pth` and `predictions.zip`.
