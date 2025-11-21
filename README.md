# collaborative_cnn_team04

Fork-based collaborative CNN project for cross-dataset Cats vs Dogs classification.

## User 1 (Model v1)

- **Task:** Cat vs Dog classification
- **Dataset (User 1 → Subhadeep):** Cat & Dog Dataset (Tong Python)  
  https://www.kaggle.com/datasets/tongpython/cat-and-dog

### Implementation

- Model code: `models/model_v1.py`
- Training notebook: `notebooks/train_v1.ipynb`
- Trained weights: `models/model_v1.pth`
- Training metrics: `results/metrics_v1.json`
- Grad-CAM samples: `results/gradcam/`

### How to run (User 1 → Subhadeep)

1. Create a virtual environment and install dependencies:

```
pip install -r requirements.txt
```

---

# Instructions for User 2 (as per assignment)

User 2 should:

1. Fork this repository
2. Clone their fork locally
3. Test `models/model_v1.pth` on User 2 dataset (Dogs vs Cats Redux)
4. Save test results as:  
   `results/test_v1_user2.json`
5. Open a GitHub Issue in this repository titled:  
   **Model v1 results on User 2 dataset**  
   and include:
   - test accuracy (if labels exist)
   - or prediction distribution
   - observations / failure cases

---

# Cross-Dataset Analysis

## 1. Domain shift and dataset bias

Although both datasets contain cats and dogs, they differ significantly in:

- image resolution
- background complexity
- lighting conditions
- pose variation
- camera viewpoint
- dataset cleanliness & noise

These changes affect data distribution and cause **domain shift**.  
A model trained on Dataset A may not perform well on Dataset B.

---

## 2. Cross-Dataset Generalization Results

### Model V1 (trained by User 1)

From `metrics_v1.json`:

- **User1 → User1 accuracy:** 63.15%

From `test_v1_user2.json` (12,500 unlabeled images):

- Dogs predicted (class 1): 6854 (54.83%)
- Cats predicted (class 0): 5646 (45.17%)

Since User2 test dataset is **unlabeled**, an accuracy value cannot be computed.  
However, prediction distribution gives insight into bias & generalization.

---

### Model V2 (trained by User 2)

From `metrics_v2.json`:

- **User2 → User2 accuracy:** 81.14%

From evaluation on User1 dataset:

- **User2 → User1 accuracy:** 84.23%

**Key takeaway:**  
Model V2 generalizes far better than Model V1.  
User2’s dataset is more diverse, making V2 more robust.

---

## 3. Cross-Model Comparison Table

| Model  | Train Dataset | Test Dataset | Result               | Notes                               |
| ------ | ------------- | ------------ | -------------------- | ----------------------------------- |
| **V1** | User1         | User1        | 63.15%               | Baseline                            |
| **V1** | User1         | User2        | 6854 dogs, 5646 cats | Cannot compute accuracy (unlabeled) |
| **V2** | User2         | User2        | 81.14%               | Strong performance                  |
| **V2** | User2         | User1        | 84.23%               | Best generalization                 |

---

## 4. Failure Cases & Grad-CAM (Model V1)

Grad-CAMs in `results/gradcam/` show:

- focus on background textures
- confusion between small dogs & cats
- difficulty with dark or low-contrast images
- attention drift in cluttered scenes

This indicates overfitting and weaker domain robustness.

---

## 5. Future Work

- Transfer learning (ResNet18, MobileNetV2)
- Stronger augmentation strategies
- Joint training on both datasets
- Domain adaptation techniques
- Hard example mining

These steps improve cross-dataset performance significantly.

---

## 6. Model V2 (User2 Model) Performance

Validation accuracy progression:

| Epoch | Accuracy   |
| ----- | ---------- |
| 1     | 65.32%     |
| 2     | 74.46%     |
| 3     | 74.96%     |
| 4     | 80.38%     |
| 5     | **81.14%** |

Observations:

- Model V2 learns faster
- More robust to varied images
- Outperforms Model V1 significantly

---
