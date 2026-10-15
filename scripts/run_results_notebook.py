"""Execute aggregate evidence review with the invoking Python environment."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def execute_inprocess(notebook, root: Path) -> None:
    """Execute each cell in one fresh IPython process without kernel sockets.

    This mode is for environments where a Jupyter/ZeroMQ kernel cannot bind
    local sockets. It captures native rich displays and records the mode;
    it is not a Jupyter-kernel protocol test.
    """
    import os
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    os.chdir(root)
    shell = InteractiveShell.instance()
    execution_count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        execution_count += 1
        cell.outputs = []
        cell.execution_count = execution_count
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True, silent=False)
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output(
                "display_data", data=output.data, metadata=output.metadata,
            ))
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            raise RuntimeError(f"Notebook cell {index} failed: {type(error).__name__}: {error}") from error
    notebook.metadata["execution_review"] = {
        "mode": "fresh in-process IPython; cells executed top-to-bottom",
        "jupyter_kernel_protocol_verified": False,
        "note": "Use the default kernel engine in environments that permit local Jupyter sockets.",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "notebooks/01_results_review.ipynb")
    parser.add_argument("--engine", choices=("kernel", "inprocess"), default="kernel")
    args = parser.parse_args()
    source = root / "notebooks/01_results_review.ipynb"
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    # A temporary kernel specification prevents a system/global "python3"
    # kernel from silently choosing a different dependency environment.
    if args.engine == "inprocess":
        execute_inprocess(notebook, root)
    else:
        execute_kernel(notebook, root)
        notebook.metadata["execution_review"] = {
            "mode": "Jupyter kernel; cells executed top-to-bottom",
            "jupyter_kernel_protocol_verified": True,
        }
    nbformat.validate(notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"Executed and saved {args.output}")
    print(f"Execution interpreter: {sys.executable}; engine: {args.engine}")
    print("Aggregate evidence review only: no raw dataset/model download or training.")


def execute_kernel(notebook, root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="toxic-comments-review-kernel-") as temporary:
        kernel_name = "toxic-comments-review"
        specification = Path(temporary) / kernel_name
        specification.mkdir()
        (specification / "kernel.json").write_text(
            json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Toxic comments evidence review",
                "language": "python",
            }),
            encoding="utf-8",
        )
        manager = KernelManager(
            kernel_name=kernel_name,
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=120,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(root)}},
        )
        client.execute()


if __name__ == "__main__":
    main()
