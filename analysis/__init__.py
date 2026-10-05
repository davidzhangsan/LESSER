"""Analysis code for Section 5 and Appendices E-F of the LESSER paper.

Each module owns one group of results and states them in its docstring; ``analysis/README.md`` maps
every reported number and figure to the command that regenerates it. CPU steps read the small inputs
in ``data/analysis``; GPU steps (gradient audits, set-gradient sketches, training runs) produce the
large raw outputs from which those inputs are derived.
"""
