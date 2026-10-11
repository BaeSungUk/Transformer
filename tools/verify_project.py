"""Validate artifacts and saved-model inference without retraining."""
import ast
import csv
import json
from pathlib import Path
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / ".venv" / "Lib" / "site-packages"))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

notebook = json.loads((root / "bert_irony_project.ipynb").read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and not cell["source"].startswith("%"):
        ast.parse(cell["source"])
        if 'sentence = input(' not in cell["source"]:
            assert cell["execution_count"] is not None, cell["id"]
        assert all(o["output_type"] != "error" for o in cell["outputs"])

out = root / "outputs"
experiment = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
with (out / "test_predictions.csv").open(encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 784
truth = [int(r["true_label"]) for r in rows]
predicted = [int(r["predicted_label"]) for r in rows]
metrics = experiment["test_results"]["BERT fine-tuned"]
assert abs(accuracy_score(truth, predicted) - metrics["accuracy"]) < 1e-10
assert abs(f1_score(truth, predicted, average="macro") - metrics["macro_f1"]) < 1e-10
assert experiment["split_sizes"] == {"train": 2862, "test": 784, "validation": 955}

tokenizer = AutoTokenizer.from_pretrained(out / "best_model", local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(out / "best_model", local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
namespace = {"torch": torch, "np": np, "inference_tokenizer": tokenizer,
             "inference_model": model, "device": device, "MAX_LENGTH": 128,
             "id2label": model.config.id2label}
source = next(c["source"] for c in notebook["cells"] if "def predict_text(" in c["source"])
definition = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
exec(compile(ast.Module(body=[definition], type_ignores=[]), "predict_text", "exec"), namespace)
predict = namespace["predict_text"]
for invalid in ["", "   ", None]:
    try:
        predict(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Empty or invalid input was accepted")
assert predict("hello " * 150)["truncated"]
for row in rows[:5]:
    result = predict(row["text"])
    assert result["prediction"] == model.config.id2label[int(row["predicted_label"])]
    assert abs(result["irony_probability"] - float(row["irony_probability"])) < 1e-5
    assert abs(result["irony_probability"] + result["non_irony_probability"] - 1) < 1e-6

with zipfile.ZipFile(root / "project_results.zip") as archive:
    assert archive.testzip() is None
    assert "outputs/test_metrics.csv" in archive.namelist()
for name in ["learning_curves.png", "confusion_matrix.png", "token_lengths.png"]:
    assert (out / name).stat().st_size > 0
print("PASS: notebook syntax/execution, 784-row metrics, saved-model consistency,")
print("empty inputs, truncation, probabilities, graphs and result ZIP.")
