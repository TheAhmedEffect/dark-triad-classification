# Dark Triad Classification — with a from-scratch softmax regression

Multi-class classification over Short Dark Triad (SD3) questionnaire responses, built for
CSE-303 Statistics for Data Science. The substantive part of this project is a multinomial
logistic regression **implemented from scratch** — softmax, cross-entropy gradient and
batch gradient descent, no scikit-learn estimator — benchmarked against scikit-learn's own
logistic regression and a linear SVM.

The dataset is **not included** in this repository. The notebook expects an `SD3_data.csv`
of 27 Likert-scale items (nine each for Machiavellianism, narcissism and psychopathy) plus
`country` and `source` columns, both of which it drops.

## What the notebook does

- **Exploratory analysis** — distributions and correlations across the 27 items (pandas,
  seaborn).
- **Label construction** — for each respondent, the mean of each of the three nine-item
  subscales is compared against a fixed reference value for that trait, and the row is
  labelled with whichever trait it scores highest against.
- **From-scratch multinomial logistic regression** — one-hot encoding, `softmax` over the
  logit matrix, the analytic cross-entropy gradient `Xᵀ(A − Y)`, and 1,000 iterations of
  batch gradient descent at a 0.1 learning rate.
- **Baselines** — scikit-learn `LogisticRegression` and a linear `SVC` (C = 0.1).
- **Evaluation** — accuracy, per-class precision, recall and F1, confusion matrices for all
  three models, ROC curves, and a side-by-side prediction histogram.

## Results

3,639 held-out respondents, three classes.

| Model | Accuracy |
|---|---|
| scikit-learn logistic regression | 1.00 |
| Linear SVM | 1.00 |
| **From-scratch softmax regression** | **0.92** |

## Read the 1.00 as leakage, not as accuracy

The two perfect scores are not a result, and the notebook is published with that stated
rather than hidden.

The label here is a **deterministic function of the features**. It is computed by taking
the row means of item columns 0–8, 9–17 and 18–26, comparing each against a fixed
constant, and taking the arg-min. Those same 27 item columns are then kept as the feature
matrix. So the target is recoverable from the inputs in closed form, and any model with
enough capacity reproduces the rule exactly. The 1.00 measures that the models can learn
arithmetic the pipeline already did, not that the Dark Triad is predictable.

What the table does show is the thing worth looking at: the from-scratch implementation
reaches 0.92 on the same leaked, therefore trivially separable, problem. That gap is
diagnostic of the implementation — a fixed learning rate, a fixed 1,000 iterations and no
intercept term, against scikit-learn's solver — and it is the number I would actually
defend in this project.

## What a non-leaking version would need

Predict something the items do not already determine — a held-out behavioural outcome, or
a trait score from a disjoint subset of items — and keep the construction columns out of
`X`.
