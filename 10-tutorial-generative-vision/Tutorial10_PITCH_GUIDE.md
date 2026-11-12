# Tutorial 10: Garden Makeover (study guide + pitch script)

> This is for you to read; don't submit it. The submission is in `Tutorial10_submission/`.

## 0. The big picture in one breath
We take **one real garden photo** and ask Stable Diffusion to **redesign it as a Mediterranean courtyard**,
while **ControlNet** uses the photo's **edge map** to keep the main structure (table, vase, paving) in place.
We try **2 strengths × 3 control scales = 6 images** with the same seed, then measure how much structure
survived (**edge F1**) and how much the image changed (**RGB change**).

---

## 1. Key concepts (read these first)

| Term | Plain-English meaning |
|---|---|
| **Diffusion model** | An AI that learns to remove noise from images step by step. To generate, it starts from noise and "denoises" its way to a picture that matches a text prompt. |
| **Stable Diffusion 1.5 (SD1.5)** | The pretrained text-to-image diffusion model we use. We only run it (*inference*); we never train it. |
| **Image-to-image (img2img)** | Instead of starting from pure noise, start from *your photo with noise added*, then denoise toward the prompt. |
| **Strength** (0–1) | How much noise is added to your photo. 0.5 = half re-imagined, so much of the original survives. 0.8 = mostly re-imagined. Higher strength means a bigger makeover and more risk of losing structure. It also runs more denoising steps (25 × strength), so the two rows don't use equal compute. |
| **ControlNet** | An add-on network that feeds an extra "guide image" (here, edges) into SD so the output follows that layout. |
| **ControlNet conditioning scale** (0, 0.5, 1.0) | How strongly ControlNet's guide is enforced. 0 = off. 1.0 = full force. |
| **Canny edge detector** | A classic algorithm that finds outlines: grayscale → blur (to kill noise) → keep pixels with strong brightness change. Thresholds 100/200 decide what counts as a "strong" edge. |
| **Gaussian blur 5×5, σ=1** | Smooths the image a little so tiny texture noise doesn't become fake edges. |
| **Prompt / negative prompt** | Text describing what we want / what we *don't* want (cartoon, blurry, duplicate table…). |
| **Guidance scale (7.5)** | How hard the model tries to obey the prompt (classifier-free guidance). |
| **Seed / torch.Generator** | The random number starting point. Same seed = same noise = reproducible image. You need a **new** generator per image, because one shared generator would move on to different random numbers after each call. |
| **Scheduler (UniPC), steps (25)** | The step-by-step denoising recipe and how many steps it runs. |
| **Controlled experiment** | Change only one thing at a time. That's why Δ (delta) is measured against the no-control image **at the same strength**. |
| **Edge precision** | Of the edges in the *output*, what fraction sit near a real input edge? Low means **made-up** structure. |
| **Edge recall** | Of the edges in the *input*, what fraction still appear in the output? Low means **lost** structure. |
| **Edge F1** | Harmonic mean of precision and recall, 2PR/(P+R). One number for structural agreement. |
| **Tolerance radius (r = 1, 2, 3 px)** | How far apart two edges can be and still count as matching (done with *dilation*, which thickens edges by r pixels). |
| **Mean absolute RGB change** | Average pixel-colour difference from the input on a 0–1 scale. A measure of how much changed, not of how good it is. |
| **Local evaluation (crop)** | Score only the table region (`TABLE_BOX`), because that's what the designer must keep. |
| **Depth ControlNet / inpainting** | Alternatives: guide with a 3-D depth map, or redesign only a masked region. |

---

## 2. Results (real numbers, T4 GPU, seed 42)

| strength / control | Edge F1 | Δ vs no-control | RGB change | Table-crop F1 |
|---|---|---|---|---|
| 0.5 / 0.0 | 0.530 | 0 | 0.126 | 0.585 |
| 0.5 / 0.5 | 0.714 | +0.184 | 0.110 | 0.746 |
| 0.5 / 1.0 | **0.841** | +0.312 | 0.098 | 0.837 |
| 0.8 / 0.0 | 0.428 | 0 | **0.231** | 0.385 |
| 0.8 / 0.5 | 0.594 | +0.165 | 0.158 | 0.677 |
| 0.8 / 1.0 | 0.834 | **+0.405** | 0.129 | **0.862** |

What you see in the images:
- **0.8/0.0**: a totally new scene (terracotta wall, pots, chairs, white table). Follows the prompt best, but the garden is lost.
- **0.8/1.0**: almost the original layout, but weakly Mediterranean, with grass smears and painterly paving.
- **0.8/0.5**: the balance. Table, base and vase are kept, with warm light and lush planting. **This is our pick.**

---

## 3. Each task: what it asks, what I did, what to say

### Q1.1 Edge map (7 pts)
**Task:** rebuild the Canny condition.
**Code:** `cvtColor(RGB→GRAY)` → `GaussianBlur((5,5), 1.0)` → `Canny(100, 200)`.
**Say:** "This is the structural blueprint ControlNet follows. My version matches the baseline pixel for pixel."

### Q1.2 Experiment grid (9 pts)
**Task:** list all 6 (strength, scale) pairs, with strength as the outer loop.
**Code:** `[(s, c) for s in strengths for c in scales]`.
**Say:** "It's a Cartesian product. Strength on the outside means each figure row has a fixed strength, so I compare across a row."

### Q1.3 Reproducible generator (8 pts)
**Code:** `torch.Generator(device='cpu').manual_seed(seed)`.
**Say:** "A fresh generator per image means every condition starts from the *same* noise, so differences come only from strength and control. Reusing one generator would give each image different noise."

### Q1.4 Precision / recall / F1 (15 pts)
**Code:** `P = (pred & near_ref).sum()/pred.sum()`, `R = (ref & near_pred).sum()/ref.sum()`, `F1 = 2PR/(P+R)`.
**Say:** "Copying the input gets F1 = 1 and zero change. A perfect score with no makeover shows the metric can be gamed. An empty edge map is 0/0, so it's NaN, not perfect. Low precision means invented structure; low recall means lost structure."

### Q1.5 Shared crop (3 pts)
**Code:** `image.crop(box)`. Use the same box for every image so the comparison is fair.

### Q2.1 Controlled comparison (20 pts)
**Say:** "Within each row, more control gives higher F1 and less RGB change. Control matters more at high strength (+0.405 vs +0.312), because at 0.8 the photo is mostly noise and the edge map is what keeps the layout."

### Q2.2 Tolerance robustness (6 pts)
**Say:** "Ranking is identical at r = 1, 2, 3. Scores rise with r only because the rule is more lenient; the images didn't change. The top two are within 0.01, so they're effectively tied."

### Q2.3 Why high F1 can mislead (4 pts)
**Say:** "Our highest-F1 image is the least Mediterranean. F1 rewards *not changing*. I'd pair it with prompt compliance, for example a visual check or a CLIP score."

### Q3.1 Failure cases (6 pts)
**Say:** "0.8/0.0 loses the structure: new table, new chairs, because nothing anchors it at high noise. 0.8/1.0 under-edits and smears textures, because full control forces every edge, even clutter, and fights the prompt."

### Q3.2 Local evaluation (8 pts)
**What I added:** F1 on the table crop only, plus rank comparison, saved to `local_vs_global.csv` and a bar chart.
**Say:** "The top two swap, and 0.8/1.0 becomes #1 on the table. Whole-image F1 is dominated by foliage edges; a designer cares whether the *furniture* survived."

### Q3.3 Responsible use (4 pts)
**Say:** "These are AI concept images, not measured designs. There are no real dimensions, plant suitability or cost. They must be labelled as AI-generated."

### Q4.1 Design choice (4 pts)
**Say:** "I'd choose 0.8/0.5: the table and layout are kept, it's a real makeover, and artefacts are minor."

### Q4.2 Depth vs inpainting (4 pts)
**Say:** "Depth keeps the 3-D layout but is coarse on thin legs. Inpainting keeps the table pixel-exact but needs a mask and can leave seams."

### Q4.3 Next experiment (2 pts)
**Say:** "Repeat the 0.8 row with 4 seeds and report mean ± std. Everything rests on seed 42, and the top two are within 0.01."

---

## 4. 60-second pitch script (say this to the professor)

> "This tutorial is about **controllable image generation**: redesigning a real garden photo without losing its structure.
> I used pretrained **Stable Diffusion 1.5 with a Canny ControlNet**. The **strength** decides how much of the photo is re-imagined, and the **ControlNet scale** decides how strictly the edge map is enforced. I ran a controlled 2-by-3 grid with the same prompt, the same seed and a fresh generator per image, so the only things changing are those two knobs.
>
> To measure structure I implemented **edge precision, recall and F1** with a 2-pixel tolerance. The result is clear: **within each strength, more control means higher F1 and less change**, and control matters most at high strength. F1 jumps by 0.405 at strength 0.8.
>
> But F1 alone is misleading. The top-scoring image barely looks Mediterranean, because F1 rewards *not changing*. So I added a **local table-crop evaluation**, which swaps the top two settings, and I inspected the failures. With no control the table is lost; with full control the image is under-edited and smeared.
>
> My pick for a designer is **strength 0.8, control 0.5**: it keeps the table and layout and still gives a real makeover. The next step would be a **multi-seed repeat**, since everything currently rests on one seed."

### Likely professor questions
- *Why does strength change compute?* img2img skips the first (1 − strength) of the schedule: 25 × 0.5 ≈ 12 steps vs 25 × 0.8 = 20 steps.
- *Why float16 / CPU offload?* To fit in the T4's 16 GB. Peak memory was about 3.4 GB.
- *Why dilate?* Dilation is a cheap way to say "within r pixels". It thickens edges so near-misses count as matches.
- *Is F1 = image quality?* No. It only measures structural agreement; quality and prompt-following must be judged separately.
