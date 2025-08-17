import subprocess, sys

def test_predict_help_runs():
    r = subprocess.run([sys.executable, "scripts/predict_existing_model.py", "--help"], capture_output=True)
    assert r.returncode == 0
    assert b"Predict using an existing pickled model" in r.stdout or b"--model-path" in r.stdout

def test_evaluate_help_runs():
    r = subprocess.run([sys.executable, "scripts/evaluate_existing_model.py", "--help"], capture_output=True)
    assert r.returncode == 0
    assert b"Evaluate an existing model on a labeled CSV" in r.stdout or b"--data-csv" in r.stdout
