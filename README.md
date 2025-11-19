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
