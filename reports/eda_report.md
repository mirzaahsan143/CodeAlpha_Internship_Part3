# Exploratory Data Analysis Report

## Leaf Disease Detection - PlantVillage Dataset

**Date:** June 2026

---

## 1. Class Distribution Analysis

### Key Findings:
- **Total Classes:** 15 (3 healthy, 12 diseased)
- **Class Imbalance:** The dataset exhibits moderate imbalance
  - Largest: Tomato Yellow Leaf Curl Virus (3,209 images, 15.5%)
  - Smallest: Potato Healthy (152 images, 0.7%)
  - Ratio: 21:1

### Implications:
- Class imbalance may bias models towards majority classes
- Stratified splitting is essential to maintain class proportions
- Weighted loss functions may benefit rare classes
- Data augmentation should be applied more aggressively to minority classes

## 2. Image Dimension Analysis

### Key Findings:
- **Resolution:** All images are 256×256 pixels
- **Format:** JPEG, RGB (3 channels)
- **Aspect Ratio:** Uniformly 1:1 (square)
- **Consistency:** Perfect dimension uniformity across all 20,639 images

### Implications:
- No resizing discrepancies
- No aspect ratio distortion needed
- Directly compatible with models expecting square inputs
- Uniform dimensions eliminate a common source of preprocessing errors

## 3. Data Quality Assessment

### Key Findings:
- **Corrupted Images:** None detected
- **Duplicate Images:** None detected
- **Invalid Files:** None detected
- **Color Mode Issues:** None detected

### Implications:
- Dataset is clean and ready for modeling
- No data cleaning required beyond standard preprocessing
- High data quality contributes to model performance

## 4. Healthy vs Diseased Balance

| Category | Count | Percentage |
|----------|-------|------------|
| Healthy | 3,221 | 15.6% |
| Diseased | 17,418 | 84.4% |

### Implications:
- Diseased images dominate (84.4%)
- The system is better suited for disease detection
- Healthy class detection may require additional attention
- Real-world application: screening healthy plants is equally important

## 5. Plant Species Distribution

| Plant | Classes | Total Images | Percentage |
|-------|---------|--------------|------------|
| Tomato | 10 | 16,012 | 77.6% |
| Pepper | 2 | 2,475 | 12.0% |
| Potato | 3 | 2,152 | 10.4% |

### Implications:
- Tomato dominates the dataset (77.6%)
- Tomato diseases have more training data
- Pepper and Potato may benefit from additional augmentation
- Cross-plant generalization should be validated
