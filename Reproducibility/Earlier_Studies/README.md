# Numerical studies in Sections 6–12

The input catalogs, solver programs and full-precision results in this
directory support the Paper 3 master manuscript. The experiment specifications
use the original study identifiers. See the [repository guide](../../README.md)
for the current numbering.

Install the Python dependencies from the root requirements.txt. From this
directory, run:

    python3 run_all.py

This replays the numerical studies, draws their plots, and checks the saved
numerical records. The --quick option omits the larger-catalog, scaling, and
matched-depth runs. Use a working copy because a replay writes outputs and
plots.

The repository guide covers the additional studies and figure workflow.
The record in ../publication_snapshot/ identifies the saved
scaling times used in Table 23 and Figure 17.
