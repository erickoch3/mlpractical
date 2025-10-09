# Lab 2 Feedback

## mlp/data_providers.py
- `reset()` and `shuffle()` mutate the data arrays in place and immediately reshuffle; because the original ordering is lost, calling `reset()` cannot restore the deterministic presentation that evaluation relies on. The instructors keep a `_current_order` index array, undo that permutation inside `reset()`, and then start the next epoch with `new_epoch()`, so each epoch begins from the unshuffled data before optionally reshuffling. See upstream `reset()`/`new_epoch()` in 0a314c4.
- Guard rails around `batch_size` and `max_num_batches` are missing in our version. The solution adds property setters that validate `batch_size >= 1` and `max_num_batches` not equal to 0, preventing silent failure when misconfigured.
- Our helper `ensure_mlp_data_dir()` replaces the expected `MLP_DATA_DIR` environment variable lookup. The official solution keeps the environment contract explicit, so our notebooks and scripts should rely on that instead of hard-coding project-relative fallbacks.

## mlp/errors.py
- `SumOfSquaredDiffsError.__call__` computes `np.mean((outputs - targets) ** 2)`, which averages over the output dimension. The gradient we return matches the derivative of `0.5 * sum((outputs-targets)^2)` per sample, so cost and gradient are now inconsistent when `output_dim > 1`. The solution keeps the `0.5 * np.mean(np.sum(..., axis=1))` form, aligning with the analytic gradient.
- The Binary and Multi-class cross-entropy error classes were deleted in our copy. Later labs expect these helpers, so removing them forces students to re-implement boilerplate that already exists upstream.

## mlp/layers.py
- `AffineLayer` is missing a `bprop()` implementation. Without propagating gradients back to the inputs, any multilayer model will fail during training. The instructors implement `return grads_wrt_outputs.dot(self.weights)`.
- `LayerWithParameters` in our version lacks a `params` setter, so optimisers cannot write updated parameters generically. The upstream base class defines a setter that derived classes override.
- The sigmoid and softmax layer implementations (with both `fprop` and `bprop`) are absent. Those activations are needed in Lab 2 notebooks; the upstream file provides working versions starting at `SigmoidLayer` and `SoftmaxLayer`.

## notebooks/02_Single_layer_models.ipynb
- In the helper function `error(outputs, targets)` we re-used the incorrect mean-squared implementation noted above, so the training curves under-report the loss by a factor and disagree with the provided gradient check cell.
- One cell permanently appends `/Users/ewkoch/repos/mlp` to `sys.path`. That absolute path will break on any other machine. The official notebook keeps the import hint commented out so students set their environment instead.
- Our call `stats, keys = optimiser.train(...)` assumes only two return values. The solution now returns `(stats, keys, run_history)`; ignoring the history raises a `ValueError` once the API updates, so the instructors capture the third element even if they do not use it immediately.
