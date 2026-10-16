"""Build the retired 03 entry point that directs new Colab runs to notebook 04.

The legacy training implementation remains in Git history. This compatibility
builder keeps old links useful without publishing a second active training path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "03_colab_full_training.ipynb"
LEGACY_REVISION = "46c8f7896456e35ae4fd33d7330a741e7aa401d6"
REPOSITORY_URL = "https://github.com/mobatayneh/toxic-comment-classification"


def build_notebook() -> nbformat.NotebookNode:
    """Return a valid, non-executable migration notice for existing 03 links."""
    sources = (
        """
# Training notebook moved to 04

**Author: Mohammad Albatayneh.** This former training entry point is retired.
For a new Colab run, use the canonical
[04_colab_efficient_training.ipynb](04_colab_efficient_training.ipynb) and the
[efficient training guide](../docs/colab_efficient_training.md).

The efficient notebook is the workflow used for the returned
`colab-efficient-v1` run. It keeps its training code visible and checks the
embedded source bytes. This notice contains no executable training cells.
""",
        """
## Start a new run

1. Open **04_colab_efficient_training.ipynb** in Google Colab.
2. Select a compatible GPU runtime and follow its cells in order.
3. Use its separate run folder and preserve the original CSV bytes.

See [the aggregate results review](01_results_review.ipynb) for the reviewed
validation evidence and its limits. The new validation results are selection
diagnostics; the neural internal test has not been evaluated.
""",
        f"""
## Recover an existing legacy 03 run

An existing `random-roberta-seed42` run requires the **original notebook version
that created it**, together with its saved configuration, exact source bytes,
tokenizer and compatible runtime. Open that version through Git history. The
[published pre-retirement 03 notebook]({REPOSITORY_URL}/blob/{LEGACY_REVISION}/notebooks/03_colab_full_training.ipynb)
is preserved as a reference; use it only if it matches the saved run identity.

The current notebook 04 creates a different experiment and cannot resume a
legacy 03 checkpoint. Keep the legacy run folder intact. Never change saved
identity or checkpoint files to force a match. The
[legacy guide](../docs/colab_training.md) explains this boundary.
""",
    )
    cells = []
    for index, source in enumerate(sources):
        cell = nbformat.v4.new_markdown_cell(source.strip())
        cell.id = f"retired-{index:02d}"
        cells.append(cell)
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"name": OUTPUT.name, "provenance": []},
            "training_workflow": {
                "status": "retired",
                "replacement": "04_colab_efficient_training.ipynb",
                "published_pre_retirement_revision": LEGACY_REVISION,
            },
        },
    )
    nbformat.validate(notebook)
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the saved retirement notice matches its builder")
    args = parser.parse_args()
    rendered = nbformat.writes(build_notebook()) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            parser.exit(1, "Retired Colab entry is stale; regenerate with scripts/build_colab_notebook.py.\n")
        print("Retired Colab entry matches its builder; use notebook 04 for new training.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
