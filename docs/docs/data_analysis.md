# LSM Dataset Analysis

**Date:** 2026-05-12
**Total Images:** 1,848
**Format:** JPG
**Classes:** 29 folders (A-Z + LL, Ñ, RR)
**Naming Convention:** `LETTER_N.jpg` (e.g., `A_1.jpg`, `A_2.jpg`, ...)

---

## Images Per Letter

| Letter | Count | Bar | Notes |
|--------|------:|-----|-------|
| A | 153 | ████████████████ | |
| B | 115 | ████████████ | |
| C | 81 | ████████ | |
| D | 64 | ██████ | |
| E | 30 | ███ | low |
| F | 81 | ████████ | |
| G | 29 | ███ | low |
| H | 31 | ███ | low |
| I | 28 | ███ | low |
| J | 28 | ███ | movement sign |
| K | 109 | ███████████ | |
| L | 101 | ██████████ | |
| LL | 6 | █ | critically low |
| M | 30 | ███ | low |
| N | 130 | █████████████ | |
| Ñ | 0 | | empty — no images |
| O | 33 | ███ | |
| P | 100 | ██████████ | |
| Q | 81 | ████████ | |
| R | 32 | ███ | |
| RR | 15 | ██ | very low, movement variant |
| S | 233 | ████████████████████████ | most images |
| T | 137 | ██████████████ | |
| U | 31 | ███ | |
| V | 32 | ███ | |
| W | 31 | ███ | |
| X | 23 | ██ | low |
| Y | 54 | █████ | |
| Z | 30 | ███ | movement sign |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total images | 1,848 |
| Total classes | 29 |
| Mean per class | 63.7 |
| Median per class | 31 |
| Max | 233 (S) |
| Min (non-zero) | 6 (LL) |
| Empty classes | 1 (Ñ) |
| Classes with < 30 images | 7 (G, I, J, X, LL, RR, Ñ) |
| Classes with >= 100 images | 7 (A, B, K, L, N, P, S, T) |

---

## Movement Signs (skip for static classifier)

These letters involve hand movement and are not suitable for single-frame image classification:

| Letter | Images | Notes |
|--------|-------:|-------|
| J | 28 | requires movement |
| Z | 30 | requires movement |
| RR | 15 | movement in some variants |

---

## Static Signs — Training Readiness

Excluding J, Z, and RR, the static dataset has **26 classes** and **1,775 images**.

### Tier Breakdown

**Ready (>= 80 images) — 11 classes:**
A (153), B (115), C (81), F (81), K (109), L (101), N (130), P (100), Q (81), S (233), T (137)

**Usable but would benefit from augmentation (30–79 images) — 12 classes:**
D (64), E (30), H (31), M (30), O (33), R (32), U (31), V (32), W (31), Y (54), I (28), Z (30)

**Critical — need more data (< 30 images) — 3 classes:**
X (23), LL (6), Ñ (0)

---

## Class Imbalance

The dataset is highly imbalanced. The largest class (S = 233) has **39x** more images than the smallest non-empty class (LL = 6). The top 5 classes (S, A, T, N, B) hold **768 images — 42% of the total dataset**.

### Recommendations

1. **Ñ needs images immediately** — 0 samples makes it untrainable
2. **LL needs significant expansion** — 6 images is far below minimum viability
3. **Consider augmentation** (rotation, flipping, brightness, cropping) for classes under 80 images
4. **Consider downsampling or weighted loss** to handle the S/A/T overrepresentation
5. **Target ~100 images per class** as a minimum for reasonable model performance
