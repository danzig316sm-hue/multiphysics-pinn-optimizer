# Troubleshooting

## Training Issues

### Loss is NaN from the start

**Cause**: exploding gradients or bad input data (zeros/NaNs in CSV).

**Fix**:
1. Check CSV for NaN/Inf values: `pd.read_csv("data.csv").isna().sum()`
2. Lower the learning rate (`learning_rate=1e-4`).
3. Add gradient clipping (not yet built-in — add manually):
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

---

### Physics weight keeps hitting max (5.0)

**Cause**: the model cannot satisfy physics constraints with the current
architecture or data distribution.

**Options**:
1. Increase `physics_weight_max` to allow further boosting.
2. Widen the network (`hidden_dims=[256, 512, 256]`).
3. Lower `residual_tolerance` to prevent premature boosting.
4. Check physics constraint signs and coupling coefficients in
   `models/pinn_model.py` — they may not match your application.

---

### Divergence corrections fire every epoch

**Cause**: learning rate or batch size is too large for the loss landscape.

**Fix**:
1. Reduce initial `learning_rate` (try `5e-4` or `1e-4`).
2. Reduce batch size.
3. Increase `divergence_threshold` to 1.3 to reduce sensitivity.

---

### val_loss is always higher than train_loss by a large margin

**Cause**: overfitting to training data.

**Fix**:
1. Reduce network width/depth.
2. Add dropout (not currently built-in).
3. Increase `test_split` to 0.3 for a larger validation set.
4. Collect more training data.

---

## Data Issues

### `KeyError` or column mismatch in `MultiPhysicsDataset`

**Cause**: CSV columns are not in the expected order (all inputs first,
then thermal, stress, em as the last three).

**Fix**: reorder your CSV columns so the last three are your targets.

---

### Scaler `inverse_transform` gives wrong scale

**Cause**: scalers loaded from a different dataset than the model was
trained on.

**Fix**: always save scalers and the model from the same training run:
```python
dataset.save_scalers("checkpoints/scalers")
torch.save(model.state_dict(), "checkpoints/best_model.pt")
```

---

## Environment Issues

### `ModuleNotFoundError: No module named 'sklearn'`

```bash
pip install scikit-learn
```

### `ModuleNotFoundError: No module named 'gdown'`

```bash
pip install gdown
```

### CUDA out of memory

1. Reduce `batch_size`.
2. Reduce `hidden_dims`.
3. Use CPU: `PINNTrainer(model, device='cpu')`.

---

## Self-Correction Issues

### No corrections fire even when loss is bad

**Cause**: `residual_tolerance` is too high or `divergence_patience` window
is too large.

**Fix**: lower `residual_tolerance` (e.g. `0.01`) and/or lower
`divergence_patience` (e.g. `3`).

---

### Checkpoint restore makes training oscillate

**Cause**: `spike_threshold` is too low, causing frequent restores that
prevent progress.

**Fix**: increase `spike_threshold` to `3.0` or disable with
`restore_on_spike=False`.

---

## Docs Viewer Issues

### `docs_viewer.py: command not found`

Run as a Python script:
```bash
python docs_viewer.py list
```

### Doc search returns no results

Terms are matched case-insensitively against file content. Use shorter
keywords: `search "energy"` instead of `search "energy conservation equation"`.
