Okay, let's consolidate the entire stepwise process for your toxicity prediction task, from data loading to final model evaluation, incorporating all the insights and decisions we've discussed. This will serve as a comprehensive checklist and narrative of your project.

**Stepwise Process for Compound Toxicity Prediction**

**Phase 1: Data Acquisition and Initial Setup**

1.  **Understand the Goal:**
    *   Predict a continuous "DATA\_Value" (representing toxicity/activity) for a given compound, assay type, and concentration.
    *   Handle multiple assay types and varying numbers of concentration points.
    *   Ensure the model generalizes to new compounds.

2.  **Acquire Data:**
    *   Load pre-split training (`train.csv`), validation (`val.csv`), and test (`test.csv`) datasets. These datasets are initially in a "wide" format where each row is a compound-assay pair, and `DATA0-DATA14` / `CONC0-CONC14` are columns.

3.  **Environment Setup:**
    *   Set up a Python environment (e.g., Conda).
    *   Install necessary libraries: `pandas`, `numpy`, `scikit-learn`, `torch`, `transformers`, `matplotlib`.
    *   Configure GPU environment: Ensure compatible NVIDIA drivers, CUDA Toolkit, and cuDNN are installed and correctly set up for PyTorch to utilize the GPU. Verify with `torch.cuda.is_available()`.

**Phase 2: Data Preprocessing and Feature Engineering**

4.  **Reshape Data (Wide to Long):**
    *   For each dataset (train, val, test):
        *   Transform the data from wide format to long format. Each row in the long format should represent a unique observation: `(PUBCHEM_SID, SMILES, SAMPLE_DATA_TYPE, Concentration, DATA_Value)`.
        *   Handle potential NaNs that arise if some compound-assay pairs have fewer than 15 concentration points by dropping rows where `DATA_Value` or `Concentration` is missing after melting.

5.  **Feature Engineering - Assay Type:**
    *   **One-Hot Encode `SAMPLE_DATA_TYPE`:**
        *   Fit a `OneHotEncoder` from `sklearn.preprocessing` on the `SAMPLE_DATA_TYPE` column of the **training set only**.
        *   Transform this column in the train, validation, and test sets using the fitted encoder. This creates new binary columns (e.g., `SAMPLE_DATA_TYPE_agonist1`, etc.).
        *   Concatenate these new one-hot encoded columns with their respective DataFrames and drop the original `SAMPLE_DATA_TYPE` string column.

6.  **Feature Engineering - Concentration:**
    *   **Log Transform:** Apply a base-10 logarithm to the `Concentration` column to create `Log_Concentration`. Add a small epsilon (`1e-12`) before logging to prevent `log(0)`.
        *   `df['Log_Concentration'] = np.log10(df['Concentration'] + epsilon)`
    *   **Scale `Log_Concentration`:**
        *   Fit a `StandardScaler` from `sklearn.preprocessing` on the `Log_Concentration` column of the **training set only**.
        *   Transform `Log_Concentration` in the train, validation, and test sets using the fitted scaler to create `Log_Concentration_Scaled`.

7.  **Target Variable Preparation - `DATA_Value`:**
    *   **Scale `DATA_Value`:**
        *   Fit a `StandardScaler` on the `DATA_Value` column of the **training set only**.
        *   Transform `DATA_Value` in the train, validation, and test sets using this fitted scaler to create `DATA_Value_Scaled`. Store this scaler as it will be needed to inverse-transform predictions later.

8.  **Feature Engineering - SMILES Embeddings:**
    *   **Choose a Pre-trained Model:** Select `seyonec/ChemBERTa-zinc-base-v1` from the Hugging Face Hub.
    *   **Load Tokenizer and Model:**
        ```python
        from transformers import AutoTokenizer, AutoModel
        smiles_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        smiles_embedding_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        smiles_embedding_model.to(DEVICE) # DEVICE is 'cuda' or 'cpu'
        smiles_embedding_model.eval()
        ```
    *   **Generate Embeddings:**
        *   Collect all unique SMILES strings from the train, validation, and test sets.
        *   Write a function to process a list of SMILES strings in batches:
            *   Tokenize SMILES using `smiles_tokenizer` (with padding, truncation, and special tokens).
            *   Pass tokenized input through `smiles_embedding_model` (in `eval` mode and `with torch.no_grad()`).
            *   Extract the embedding of the first token (e.g., `outputs.last_hidden_state[:, 0, :]`) as the molecule-level embedding (typically 768 dimensions).
        *   Generate embeddings for all unique SMILES.
        *   Create a mapping (dictionary) from each unique SMILES string to its corresponding embedding vector.
        *   Use this map to create the embedding arrays (`X_train_embed`, `X_val_embed`, `X_test_embed`) for each dataset, ensuring the order matches the original DataFrames.

**Phase 3: Model Building and Training (PyTorch)**

9.  **Prepare Final Input Features (X) and Targets (y) for PyTorch:**
    *   **Concatenate Features:** For each set (train, val, test), concatenate:
        *   SMILES embeddings (e.g., `X_train_embed`).
        *   One-hot encoded `SAMPLE_DATA_TYPE` columns.
        *   `Log_Concentration_Scaled` column.
        This forms `X_train_final`, `X_val_final`, `X_test_final`.
    *   **Define Targets:** Use `DATA_Value_Scaled` as the target: `y_train_scaled`, `y_val_scaled`, `y_test_scaled`.
    *   **Convert to PyTorch Tensors:** Convert these NumPy arrays to `torch.Tensor`. Ensure target tensors are reshaped to `[N, 1]`.
    *   **Create `TensorDataset` and `DataLoader`:**
        ```python
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # Repeat for validation and test sets (shuffle=False for val/test)
        ```

10. **Define MLP Model Architecture:**
    *   Create a class that inherits from `torch.nn.Module`.
    *   In `__init__`: Define layers (e.g., `nn.Linear`, `nn.ReLU`, `nn.Dropout`, `nn.BatchNorm1d` (optional)).
    *   In `forward`: Define the computation flow through the layers.
    *   Example (to be tuned):
        ```python
        class ToxicityMLP(nn.Module):
            def __init__(self, input_features):
                super().__init__()
                self.layer_stack = nn.Sequential(
                    nn.Linear(input_features, 256), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
                    nn.Linear(128, 1)
                )
            def forward(self, x): return self.layer_stack(x)
        pytorch_model = ToxicityMLP(input_dim).to(DEVICE)
        ```

11. **Define Loss Function, Optimizer, and Scheduler:**
    *   **Loss:** `criterion = nn.MSELoss()`
    *   **Optimizer:** `optimizer = optim.Adam(pytorch_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)` (tune `LEARNING_RATE` and `WEIGHT_DECAY`).
    *   **Scheduler (Optional but Recommended):** `scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=0.2)`

12. **Implement Training Loop:**
    *   Iterate for a set number of `EPOCHS`.
    *   **Training Phase (per epoch):**
        *   Set model to train mode: `pytorch_model.train()`.
        *   Iterate through `train_loader`.
        *   Move data to `DEVICE`.
        *   Zero gradients: `optimizer.zero_grad()`.
        *   Forward pass: `outputs = pytorch_model(inputs)`.
        *   Calculate loss: `loss = criterion(outputs, targets)`.
        *   Backward pass: `loss.backward()`.
        *   Update weights: `optimizer.step()`.
        *   Accumulate training loss and metrics (e.g., MAE).
    *   **Validation Phase (per epoch):**
        *   Set model to eval mode: `pytorch_model.eval()`.
        *   Use `with torch.no_grad():`.
        *   Iterate through `val_loader`.
        *   Move data to `DEVICE`.
        *   Forward pass.
        *   Calculate validation loss and metrics.
    *   Print epoch statistics (train/val loss, train/val MAE, learning rate).
    *   Update learning rate scheduler: `scheduler.step(epoch_val_loss)`.
    *   **Early Stopping Logic:**
        *   Monitor validation loss. If it doesn't improve for `PATIENCE_EARLY_STOPPING` epochs, break the training loop.
        *   Save the state dictionary of the model with the best validation loss: `torch.save(pytorch_model.state_dict(), 'best_model.pth')`.

**Phase 4: Evaluation and Interpretation**

13. **Load Best Model:**
    *   After training, load the weights from the epoch with the best validation performance: `pytorch_model.load_state_dict(torch.load('best_model.pth'))`.

14. **Evaluate on Test Set (Scaled Values):**
    *   Set model to eval mode: `pytorch_model.eval()`.
    *   Use `with torch.no_grad():`.
    *   Iterate through `test_loader`.
    *   Collect all predictions (`y_pred_scaled_test_pytorch`) and true targets (`y_test_scaled_pytorch`).
    *   Calculate MSE and MAE on these scaled values.

15. **Evaluate on Test Set (Original Scale):**
    *   **Inverse Transform:** Use the `data_value_scaler` (fitted on the training set's `DATA_Value`) to inverse-transform:
        *   `y_pred_original_scale = data_value_scaler.inverse_transform(y_pred_scaled_test_pytorch)`
        *   `y_test_original_scale = data_value_scaler.inverse_transform(y_test_scaled_pytorch)`
    *   **Calculate Final Metrics:** Compute MSE, MAE, and R² score using the original scale values. These are your primary reported metrics.
        ```python
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse_orig = mean_squared_error(y_test_original_scale, y_pred_original_scale)
        # ... etc.
        ```

16. **Analyze Results and Visualize:**
    *   Plot training/validation loss and MAE curves over epochs to diagnose training (e.g., overfitting, underfitting).
    *   Plot a scatter plot of true vs. predicted values (on the original scale) for the test set.
    *   Analyze errors: Are there specific types of compounds, assays, or concentration ranges where the model performs poorly?

**Phase 5: Iteration and Manuscript Preparation**

17. **Iterate and Refine (Hyperparameter Tuning):**
    *   Based on evaluation metrics and learning curves:
        *   Adjust model architecture (layers, neurons).
        *   Tune regularization (dropout rates, L2 `weight_decay`).
        *   Tune optimizer parameters (learning rate, batch size).
        *   Re-train and re-evaluate.
18. **Result:**
19. - R2 score - 0.84
    - MAE = 0.29
