import subprocess
from pathlib import Path
import argparse

# === Lire les arguments
parser = argparse.ArgumentParser()
parser.add_argument("--yaml", type=str, required=True)
parser.add_argument("-N", type=int, default=1000)
args = parser.parse_args()

# === Construire le chemin absolu vers dfdf_Aime.py
dfdf_path = Path(__file__).resolve().parents[1] / "src" / "darkfield" / "dfdf_Aime.py"

# === Appeler dfdf comme un script, comme sur le cluster
subprocess.run([
    "python",
    str(dfdf_path),
    "-N", str(args.N),
    "--yaml", args.yaml
])
