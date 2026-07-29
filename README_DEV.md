Live test runner
----------------

Quick setup for fast edit-and-test feedback while developing locally on Windows.

1) Install dev tools:

```powershell
python -m pip install -r requirements-dev.txt
```

2) Start the watcher (re-runs tests when files change):

```powershell
./tools/watch_tests.ps1
```

Notes
- `ptw` (pytest-watch) watches your project files and re-runs `pytest` whenever a file is saved.
- You can also use the VS Code Testing UI (Python extension) for an integrated experience.
