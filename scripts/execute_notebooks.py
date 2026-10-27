"""Exécuter et enregistrer les sorties des notebooks avec le Python courant.

Les expériences sont lues depuis results/ quand elles existent. Ce contrôle
vérifie le parcours pédagogique complet, sans répéter la recherche coûteuse.
"""

import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/atelier-jupyter-runtime")
os.environ.setdefault("IPYTHONDIR", "/tmp/atelier-ipython")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atelier-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/atelier-cache")

if __name__ == "__main__":
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        # Le kernelspec natif python3 utilise le Python qui lance ce script.
        client = NotebookClient(notebook, timeout=1200, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        cells = sum(cell.cell_type == "code" for cell in notebook.cells)
        print(f"{path.name} : {cells} cellules exécutées sans erreur", flush=True)
