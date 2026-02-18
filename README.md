# Sampling Assignment – 102483083

## What this is

The goal was to take an imbalanced credit card fraud dataset, balance it out, create 5 different samples using 5 different sampling techniques, run 5 ML models on each sample, and figure out which sampling technique works best for which model.

---

## The Dataset

Downloaded from [this GitHub link](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv). It's a credit card fraud dataset with a pretty bad class imbalance — something like 98.8% legitimate transactions and only 1.2% fraud cases. That's the first problem to solve before doing anything else.

---

## My Approach

### Balancing the data

The dataset was heavily skewed, so running any model on it raw would just give you a model that predicts "not fraud" every single time and still looks 98% accurate. That's useless.

I used random oversampling — basically duplicating rows from the minority class (fraud) until it matched the size of the majority class. I went with this over undersampling because the dataset isn't huge to begin with, and throwing away majority class rows felt wasteful. SMOTE would've been cleaner but required an extra library, so I kept it simple with `sklearn.utils.resample`.

### Sampling techniques

Once balanced, I created 5 samples of size ~385 (calculated using the standard formula: Z²·p·(1-p)/e² with 95% confidence and 5% margin of error).

- **Sampling1 – Simple Random Sampling**: just randomly pick rows without replacement. The baseline, nothing fancy.
- **Sampling2 – Systematic Sampling**: pick every nth row. Quick and deterministic.
- **Sampling3 – Stratified Sampling**: make sure the class ratio is preserved in the sample. Probably the most "correct" one for imbalanced-adjacent problems.
- **Sampling4 – Cluster Sampling**: split the dataset into 10 clusters, randomly pick 4 of them. A bit more realistic when data comes in natural groups.
- **Sampling5 – Bootstrap Sampling**: sample with replacement. Useful for variance estimation, also commonly used under the hood in Random Forests.

### Models

Ran 5 classifiers on each sample (25 combinations total):

- **M1** – Logistic Regression
- **M2** – Decision Tree
- **M3** – Random Forest
- **M4** – Gradient Boosting
- **M5** – SVM (RBF kernel)

Each combination was trained on 80% of the sample and tested on the remaining 20%.

---

## Results

| | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|--|--|--|--|--|--|
| M1 | 84.42% | 94.81% | 88.31% | 92.21% | 89.61% |
| M2 | 96.1% | 96.1% | 98.7% | 97.4% | 100.0% |
| M3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| M4 | 100.0% | 97.4% | 98.7% | 98.7% | 98.7% |
| M5 | 94.81% | 98.7% | 97.4% | 93.51% | 96.1% |

**Best sampling technique per model:**

- M1 → Sampling2 (94.81%)
- M2 → Sampling5 (100.0%)
- M3 → any sampling (100.0% across the board)
- M4 → Sampling1 (100.0%)
- M5 → Sampling2 (98.7%)

M3 (Random Forest) was the most robust — hit 100% regardless of how the sample was drawn. M1 (Logistic Regression) was the weakest overall, which makes sense since it's a linear model on what's likely a non-linear problem. M4 (Gradient Boosting) did best with simple random sampling but dropped off noticeably with systematic sampling, probably due to ordering effects in how the clusters were structured.

---

## How to run it

```bash
# download the dataset first
wget -O Creditcard_data.csv https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv

# run
python sampling_assignment.py
```

Output gets saved to `sampling_results.csv` as well.

---

## Files

- `sampling_assignment.py` — the main script
- `sampling_results.csv` — accuracy results for all model/sampling combinations
