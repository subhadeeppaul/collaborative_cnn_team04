# collaborative_cnn_team04

Fork-based collaborative CNN project for cross-dataset Cats vs Dogs classification.

## User 1 (Model v1)

- **Task:** Cat vs Dog classification
- **Dataset (User 1 -> Subhadeep):** Cat & Dog Dataset (Tong Python)  
  https://www.kaggle.com/datasets/tongpython/cat-and-dog

### Implementation

- Model code: `models/model_v1.py`
- Training notebook: `notebooks/train_v1.ipynb`
- Trained weights: `models/model_v1.pth`
- Training metrics (loss, accuracy per epoch): `results/metrics_v1.json`

### How to run (User 1 -> Subhadeep)

1. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Next Steps for User 2 (as described in the assignment)

- Fork this repository.
- Clone the fork locally.
- Test `models/model_v1.pth` on User 2's own dataset (Dogs vs Cats Redux).
- Save test results as `results/test_v1_user2.json` in the fork.
- Open a GitHub Issue in this base repository titled:

  `Model v1 results on User 2 dataset`

  and summarize:

  - test accuracy on User 2 dataset
  - any failure cases or observations

## Part C – Cross-Dataset Analysis and Research Discussion

### 1. Domain shift and dataset bias

Model v1 was trained on the TongPython Cats vs Dogs dataset (User 1 dataset).  
User 2 used a different dataset (e.g. Dogs vs Cats Redux) to test the same model.

Even though both datasets contain cats and dogs, the data distribution is different:

- **Image quality & resolution** – User 2 images are generally higher resolution and less compressed.
- **Backgrounds** – User 1 images often have simpler backgrounds, while User 2 images include furniture, floors, outdoor scenes, etc.
- **Pose & viewpoint** – User 2 dataset has many side-views, partial occlusions, and non-centered animals.
- **Lighting** – More variation in brightness, shadows, and color temperature in User 2.

These differences create **domain shift**, which explains why a model that works reasonably on User 1’s dataset does not transfer perfectly to User 2’s dataset.

---

### 2. Cross-dataset generalization

From `results/metrics_v1.json`, the final training accuracy of **Model v1** on the User 1 dataset is approximately:

- **User 1 → User 1:** ~63% training accuracy

From `results/test_v1_user2.json`, we evaluated Model v1 on 12,500 User 2 images and observed:

- **User 1 → User 2:** predictions are roughly
  - Dogs (class 1): 6854 (54.83%)
  - Cats (class 0): 5646 (45.17%)

This suggests that Model v1 is not strongly biased toward a single class, but still its predictions are not calibrated for the new domain. A proper accuracy value cannot be computed here because the User 2 test folder is unlabeled (flat structure), but we can still see how the model behaves.

---

### 3. Cross-Model Comparison Table

| Model  | Train Dataset | Test Dataset | Accuracy / Metric                         | Notes                               |
| ------ | ------------- | ------------ | ----------------------------------------- | ----------------------------------- |
| **V1** | User1         | User1        | **63.15%**                                | Baseline (Subhadeep)                |
| **V1** | User1         | User2        | Dogs: 6854, Cats: 5646 (54.83% vs 45.17%) | Domain shift observed               |
| **V2** | User2         | User2        | **81.14%**                                | Strong performance on User2 dataset |
| **V2** | User2         | User1        | _To be evaluated_                         | Will measure reverse generalization |

---

### 4. Failure cases and Grad-CAM

Grad-CAM visualizations in `results/gradcam/` show where Model v1 focuses when making decisions. Typical observations:

- Sometimes the model focuses on **background textures** (carpets, floors) instead of the animal.
- In some misclassified images, the model attends to **ears or silhouette** and confuses cats and small dogs.
- Dark or low-contrast images are harder for the model: Grad-CAM highlights noisy regions instead of clear facial features.

These patterns are consistent with overfitting to the User 1 training distribution and limited robustness to new domains.

---

### 5. Future work

Several directions can improve cross-dataset performance:

- **Transfer learning:** use pretrained networks (e.g., ResNet18, MobileNetV2) and fine-tune on combined User 1 + User 2 data.
- **Stronger data augmentation:** random crops, flips, color jitter, and slight rotations to simulate variety in pose and lighting.
- **Joint training on both datasets:** mixing images from User 1 and User 2 during training to reduce domain gap.
- **Domain adaptation:** methods that explicitly minimize the difference between feature distributions from the two datasets.
- **Hard-example mining:** manually inspect and include typical failure cases (e.g. dark images, cluttered scenes) into the training set.

These steps would make the model more robust and better suited for real-world deployment across different sources of cat/dog images.

### 6. Model V2 (User2 Model) Performance

User2 trained Model V2 for 5 epochs on the Dogs vs Cats Redux dataset.

**Validation Accuracy Progression:**

| Epoch | Val Accuracy |
| ----- | ------------ |
| 1     | 65.32%       |
| 2     | 74.46%       |
| 3     | 74.96%       |
| 4     | 80.38%       |
| 5     | **81.14%**   |

**Observations:**

- Model V2 achieves significantly higher accuracy than Model V1.
- This confirms that the User2 dataset is inherently easier for V2 since it matches the test domain.
- The strong jump from 65% → 81% shows the model quickly adapted to User2’s distribution.

**Next Step:**  
We will evaluate Model V2 on User1 dataset to study reverse-domain generalization.
