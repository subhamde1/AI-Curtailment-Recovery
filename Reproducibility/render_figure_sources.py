"""Rebuild the vector drawings from the saved study inputs using Matplotlib."""
import argparse
import ast
import json
from pathlib import Path
import runpy
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator

ROOT = Path(__file__).resolve().parent
SOURCE_MANIFEST = ROOT / "figure_sources" / "manifest.json"
INDEX = {row["source"]: (row["number"], row["file"])
         for row in json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))}
METADATA = {"CreationDate": None, "ModDate": None}
SAVEFIG = Figure.savefig


def render(fig, output, destination, rendered):
    key = Path(output).stem
    if key not in INDEX or key in rendered:
        return
    number, name = INDEX[key]
    if key == "final_manager_sequence":
        fig.set_size_inches(8.6, 8.1)
        for artist in fig.axes[0].texts:
            label = artist.get_text()
            if len(label) > 1 and label[0] in "12345" and label[1] == " ":
                x, y = artist.get_position()
                artist.set_position((x, y + .07))
    if key == "revision_reserve_phase":
        fig.set_size_inches(8.1, 3.3)
        label = fig.axes[1].yaxis.label
        label.set_text(label.get_text().replace(" among ", " among\n", 1))
    if key == "final_joint_cost":
        for ax in [axis for axis in fig.axes if axis.images]:
            # Separate adjacent cells, including the uncolored ineligible ones.
            for column in range(5):
                ax.axvline(column + .5, color="#afbac4", lw=.35, zorder=2)
            for row in range(3):
                ax.axhline(row + .5, color="#afbac4", lw=.35, zorder=2)
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            # Keep the original tick locations even if axes are repositioned.
            axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
            axis.set_minor_locator(FixedLocator(axis.get_minorticklocs()))
    fig.canvas.draw()
    text_before = [(artist, artist.get_text(), artist.get_fontsize())
                   for artist in fig.findobj(Text)]
    for artist, _, size in text_before:
        artist.set_fontsize(size + 1)
    if key != "final_manager_sequence" and fig.get_layout_engine() is None:
        # Give axes room for the larger text. The flowchart is drawn in axes
        # coordinates and uses a separate, explicitly designed arrangement.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This figure includes Axes")
            fig.tight_layout(pad=1.4)
    if key == "revision_reserve_phase":
        fig.subplots_adjust(wspace=.72)
    fig.canvas.draw()
    for artist, label, size in text_before:
        assert artist.get_fontsize() == size + 1
        assert artist.get_text() == label, (key, label, artist.get_text())
    fig.set_layout_engine("none")
    box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(.12)
    SAVEFIG(fig, destination / f"{name}.pdf", dpi=1200, bbox_inches=box,
            facecolor="white", metadata=METADATA)
    rendered.add(key)
    print(f"Figure {number}: {name}.pdf", flush=True)


def run_source(relative, **overrides):
    path = ROOT / relative
    with plt.rc_context({"pdf.fonttype": 42, "savefig.facecolor": "white"}):
        sys.path.insert(0, str(path.parent))
        try:
            runpy.run_path(str(path), run_name="__main__", **overrides)
        finally:
            sys.path.pop(0)


def extract_plots(relative, ranges, stop=None, **values):
    """Execute the archived plotting statements with published inputs.

    These two older scripts intermingle computation and rendering inside
    main(). Restrict execution to the plot statements and supply the recorded
    results in place of the costly solver calls. Line ranges refer to the
    immutable versions of the archived scripts distributed in this package.
    """
    path = ROOT / relative
    source = ast.parse(path.read_text(), filename=str(path))
    if stop:
        statements = [node for node in source.body if node.end_lineno <= stop]
    else:
        main = next(node for node in source.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        statements = [node for node in source.body if node.end_lineno < main.lineno]
        statements += [node for node in main.body
                       if any(a <= node.lineno <= b for a, b in ranges)]
    with plt.rc_context({"pdf.fonttype": 42, "savefig.facecolor": "white"}):
        namespace = {"__file__": str(path), "__name__": "__plot_replay__", **values}
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"),
             namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "redrawn_vectors")
    destination = parser.parse_args().output_dir.resolve()
    if destination == SOURCE_MANIFEST.parent:
        parser.error("Use a separate directory for the redrawn vector sources.")
    destination.mkdir(parents=True, exist_ok=True)
    rendered = set()

    def capture(fig, output, *args, **kwargs):
        render(fig, output, destination, rendered)

    Figure.savefig = capture
    try:
        original = ROOT / "earlier_studies"
        results = json.loads((original / "results.json").read_text())
        extensions = json.loads((original / "extension_results.json").read_text())
        extract_plots("earlier_studies/compute_results.py",
                      [(298, 318), (323, 362), (374, 378)], out={},
                      table=results["frontiers"], costs=results["quadratic_costs"])
        extract_plots("earlier_studies/compute_extensions.py", [(375, 389)],
                      out=extensions, reserve=extensions["reserve_curve"])
        for path in ("earlier_studies/plot_revision.py", "earlier_studies/plot_final_revision.py",
                     "figures_hourly.py", "multistage_studies/make_revision_figures.py",
                     "witness_depth2/make_witness_figure.py", "extended_studies/plot_holdout.py"):
            run_source(path)
        extract_plots("structural_studies/write_results.py", [], stop=31)
        arguments = sys.argv[:]
        try:
            sys.argv = [str(ROOT / "render_publication_scaling.py"), "--output", str(destination / "frozen_scaling.png")]
            run_source("render_publication_scaling.py")
        finally:
            sys.argv = arguments
        assert rendered == set(INDEX), sorted(set(INDEX) - rendered)
    finally:
        Figure.savefig = SAVEFIG


if __name__ == "__main__":
    main()
