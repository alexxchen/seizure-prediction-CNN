"""Configure container paths without changing the host settings files."""

import json
from pathlib import Path

for dataset in ("CHBMIT", "FB", "Kaggle2014Pred"):
    filename = Path(f"SETTINGS_{dataset}.json")
    settings = json.loads(filename.read_text())
    settings.update(datadir=f"/data/{dataset}", cachedir=f"/cache/{dataset}",
                    resultdir=f"/output/{dataset}")
    filename.write_text(json.dumps(settings, indent=2) + "\n")
