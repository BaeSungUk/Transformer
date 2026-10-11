"""Execute project cells locally, keeping actual outputs; skip install/input cells."""
import contextlib
import base64
import io
import json
import os
from pathlib import Path
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

root = Path(__file__).resolve().parents[1]
os.chdir(root)
# Reuse the host CUDA PyTorch and scientific packages, supplement missing packages.
sys.path.append(str(root / ".venv" / "Lib" / "site-packages"))
os.environ["HF_HOME"] = str(root / ".cache" / "huggingface")
os.environ["MPLBACKEND"] = "Agg"
notebook_path = root / "bert_irony_project.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
execution_count = 0
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = cell["source"]
    if source.startswith("%") or 'sentence = input(' in source:
        continue
    execution_count += 1
    cell["execution_count"] = execution_count
    output = io.StringIO()
    image_times = {p: p.stat().st_mtime_ns for p in (root / "outputs").glob("*.png")}
    print(f"Executing {cell['id']}", flush=True)
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(source, f"{notebook_path.name}/{cell['id']}", "exec"), namespace)
    except Exception:
        captured = output.getvalue() + traceback.format_exc()
        cell["outputs"] = [{"output_type": "stream", "name": "stderr", "text": captured}]
        notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
        print(captured, flush=True)
        raise
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output.getvalue()}]
    for path in (root / "outputs").glob("*.png"):
        if image_times.get(path) != path.stat().st_mtime_ns:
            cell["outputs"].append({"output_type": "display_data", "metadata": {},
                "data": {"image/png": base64.b64encode(path.read_bytes()).decode("ascii"),
                         "text/plain": path.name}})
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output.getvalue(), flush=True)
print("Completed all noninteractive project cells.", flush=True)
