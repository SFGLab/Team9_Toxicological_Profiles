## Data Part (Tox-21 Dataset)
## Baseline Models

Baseline Models (Good Starting Points):

- Logistic Regression (with L1/L2 regularization)
- Support Vector Machines (SVMs with different kernels like RBF)
- **XGBoost** or **Random Forest** using ECFP fingerprints + PubChemFingerprint + a good set of 2D molecular descriptors. This is often hard to beat and provides a solid benchmark.

Neural Networks (for potentially higher performance):

- Multi-Layer Perceptron (MLP): A standard feedforward neural network that works well with vectorized inputs (fingerprints + descriptors).
- Graph Neural Networks (GNNs): State-of-the-art for molecular property prediction. These models operate directly on the graph structure of molecules (derived from SMILES). Examples include:
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)

## Evaluation Metric

- Area Under the Receiver Operating Characteristic Curve (AUC-ROC or ROC AUC): A primary metric, especially for imbalanced datasets. It measures the trade-off between true positive rate and false positive rate.
- Area Under the Precision-Recall Curve (AUC-PRC): Also excellent for imbalanced data, focusing on the performance on the positive (toxic) class.
- F1-Score: Harmonic mean of precision and recall.
Balanced Accuracy: Average of sensitivity (recall) and specificity. Good for imbalance.
- Precision, Recall (Sensitivity), Specificity.
- Confusion Matrix: To understand the types of errors (false positives, false negatives). False negatives (predicting a toxic compound as non-toxic) are often more critical to avoid.

Since those metrics are not suitable for a multi-regression problem, we will evaluate the models using **Mean Absolute Error**.



---

We are devided the paths and work on different aspects, soon we will reach at the fantastic result.

---

## Thanks
