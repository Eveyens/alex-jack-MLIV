# Session 5 — Advanced Anomaly Detection

**Course:** Machine Learning IV (Unsupervised Learning) — Albert School  
**Group:** Alex & Jack  
**Dataset:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) (UCI)

---

## Business Context

A manufacturing firm operates heavy machinery that costs **€100,000/hour** when it fails. The goal is to build an unsupervised anomaly detection pipeline that flags potential failures *before* they occur, while minimising alert fatigue for the maintenance crew.

**Cost matrix:**
- False Positive (unnecessary alert): **€500**
- False Negative (missed failure): **€15,000**

---

## Notebook Structure

### Part 1 — EDA & Cleaning
- Dataset overview, missing values, duplicates
- Class imbalance analysis → true contamination rate ≈ 3.4%
- Feature distributions, boxplots, correlation matrix
- Categorical encoding (`Type`: L/M/H → 0/1/2)
- StandardScaler preprocessing — `y_true` saved for Part 4

### Part 2 — Modeling & Hyperparameter Tuning
Four unsupervised anomaly detection models trained with `contamination = 0.034`:

| Model | Key parameter | Logic |
|-------|--------------|-------|
| `IsolationForest` | `contamination=0.034`, `n_estimators=200` | Random partitioning — short isolation path = anomaly |
| `OneClassSVM` | `nu=0.034`, `kernel='rbf'` | Hypersphere boundary in kernel space |
| `LocalOutlierFactor` | `n_neighbors=20`, `contamination=0.034` | Local density ratio vs neighbours |
| `EllipticEnvelope` | `contamination=0.034` | Mahalanobis distance from robust Gaussian fit (MCD) |

Each model includes a sensitivity analysis on its trade-off parameter.

### Part 3 — Technical Comparison & Visualisations
- PCA 2D & 3D projections with anomaly overlays per model
- t-SNE 2D (subsample of 3,000 points)
- Deep dive: LOF-only vs IsolationForest-only disagreements
- Mathematical explanation of why the two algorithms disagree
- Model consensus heatmap (0–4 models in agreement)

### Part 4 — Managerial Conclusion & Business Strategy
- Confusion matrices for all 4 models (ground truth revealed)
- Business cost per model: FP cost + FN cost
- Baselines: "never alert" and "always alert"
- Precision/Recall scatter with iso-F1 curves
- Actionable recommendations: triage system, deployment strategy, retraining cadence

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Launch the notebook
jupyter notebook Session5_Anomaly_Detection_Notebook.ipynb
```

The dataset is downloaded automatically from UCI at runtime — no manual download needed.

---

## Key Results

The model with the **lowest total business cost** (FP + FN) is recommended for deployment. Given the 30:1 cost asymmetry, **Recall is the dominant metric** — catching more real failures is worth accepting more false alarms.

**Recommended deployment strategy:**
- 4 models agree → Critical alert → immediate intervention
- 2–3 models agree → Planned maintenance → schedule within 48h
- 1 model flags → Watchlist → monitor, no immediate action
