# Tests

`test_amp.py` covers CPU-safe AMP utilities. `test_seam_artifacts.py` checks the committed v2 JSON table. `test_eval_launcher.py` checks that the portable v5 launcher rewrites the frozen path prefix.

Run locally with a Python environment containing NumPy and PyTorch:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Full Genesis simulation/training validation remains hardware- and asset-dependent and is documented separately in `../docs/REPRODUCIBILITY.md`.
