"""Execute the three notebooks in order, keeping outputs in the original files."""

import argparse
import asyncio
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", help="Optional notebook filenames")
    args = parser.parse_args()
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    filenames = args.notebooks or ["01_1nn.ipynb", "02_cnn.ipynb", "03_scattering.ipynb"]
    # Keep Jupyter's runtime files inside this project.
    for name, directory in [("JUPYTER_CONFIG_DIR", "config"), ("JUPYTER_RUNTIME_DIR", "runtime")]:
        os.environ.setdefault(name, str(ROOT / ".jupyter" / directory))
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("IPYTHONDIR", str(ROOT / ".jupyter" / "ipython"))
    Path(os.environ["IPYTHONDIR"]).mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        path = ROOT / filename
        print("Running", path.name, flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=1800, kernel_name="cs5487-digits",
                                resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute()
        finally:
            # Preserve completed cells even if execution is interrupted or fails.
            nbformat.write(notebook, path)
        html, _ = HTMLExporter().from_notebook_node(notebook)
        folder = ROOT / "results" / "notebooks"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (path.stem + ".html")).write_text(html, encoding="utf-8")
        print("Finished", path.name, flush=True)


if __name__ == "__main__":
    main()
