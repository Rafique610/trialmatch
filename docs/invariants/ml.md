# ML rules (data, training, evaluation, benchmarks)

## Data

- Only the prescribed dataset is used for training and comparison.
- Use an allow-list of input columns. Columns holding another model's
  predictions or explanations never become inputs or targets.
- Splits are grouped by the entity that must not leak (here: patient). A test
  proves no entity appears in two splits.
- Split manifests (ids + counts) are written to `results/` and committed.

## The test set

- Test data is opened once, for the final evaluation. Prompts, thresholds,
  hyperparameters and checkpoints are chosen on validation only.
- Write the frozen settings down before the test run.

## Reproducibility

- Fixed seeds everywhere (Python, NumPy, torch). Configs live in files, not
  in notebook cells. Every run records model id, adapter, decoding settings,
  date and hardware.
- Notebooks may call scripts; they never hold logic the app or report relies on.

## Evaluation

- Never silently drop or remap unparseable outputs: count them as errors.
- Never treat absent evidence as proof of either outcome.
- State how empty gold / empty predicted sets are scored.

## Benchmarks

- Same model, same inputs across variants. Warm up first. Call
  `torch.cuda.synchronize()` before reading a timer. Repeat (≥5) and report
  mean and std. Reset and read `torch.cuda.max_memory_allocated()` per run.
- Report slowdowns honestly with a measured explanation.
