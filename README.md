# minrnn

minGRU and minLSTM from "Were RNNs All We Needed?" (Feng et al., 2024), implemented in PyTorch.

Both models reduce to the linear recurrence `h_t = a_t * h_{t-1} + b_t`, which is computed three ways: a sequential loop for inference, a log-space parallel scan for training, and a non-log vanilla scan kept for the numerical-stability comparison.

## Files

- `src/scan.py` - the shared recurrence primitives: the `g` activation, the log-space associative scan, the sequential reference loop, and the vanilla scan.
- `src/min_gru.py` - the minGRU module with sequential, parallel-log, and parallel-vanilla forward paths.
- `src/min_lstm.py` - the minLSTM module with the same three paths and normalized forget/input gates.
- `src/__init__.py` - exports `minGRU`, `minLSTM`, and the scan helpers.
- `smoke.py` - a quick run that checks the three paths agree and gradients flow.
- `tests/test_correctness.py` - the correctness harness: path equivalence, gradient checks, and a tiny overfit test.

## Usage

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
cd minrnn
python smoke.py
python tests/test_correctness.py
```

```python
from src import minGRU

model = minGRU(dim=128)
y = model(x, mode="parallel_log")   # training
y = model(x, mode="sequential")     # inference
```

No GPU is required. The code runs on CPU and uses CUDA or ROCm automatically when available.
