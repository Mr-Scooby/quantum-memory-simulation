#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

Generates automatic tex file for a simple line plots. reads .dat file, extracts its columns and genearetes 
the tex code for simple line plot. 
Leaves axis, titles, legend title as placeholders in capitals letters. 


Asssumes first columns as "time_us"
"""


from pathlib import Path
import logging
log = logging.getLogger(__name__)


x_column = "time_us"
# ---------------------------


def read_column_names(path: Path) -> list[str]:
    """Read column names from the first non-empty line."""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                return line.lstrip("#").strip().split()

    raise ValueError(f"No header found in {path}")


def generate_texFile(data_file):
    """ Generates tex file for a plot.
    input .dat file 
    """
    columns = read_column_names(data_file)

    if x_column not in columns:
        raise ValueError(
            f"Column {x_column!r} not found. "
            f"Available columns: {columns}"
        )

    y_columns = [column for column in columns if column != x_column]

    # Defult tikz preamble
    lines = [
        r"\begin{tikzpicture}",
        r"    \begin{axis}[",
        r"        thesisplot,",
        r"        xlabel={XAXIS},",
        r"        ylabel={YAXIS},",
        r"    ]",
        "",
        r"        \addlegendimage{empty legend}",
        r"        \addlegendentry{\textbf{LEGEND TITLE}}",
        "",
    ]

    # generating plot lines
    for index, column in enumerate(y_columns, start=1):
        lines.extend([
            r"        \addplot table[",
            rf"            x={x_column},",
            rf"            y={column}",
            rf"        ]{{{data_file.as_posix()}}};",
            rf"        \addlegendentry{{LABEL {index}}}",
            "",
        ])

    # Close tikz enviroment
    lines.extend([
        r"    \end{axis}",
        r"\end{tikzpicture}",
        "",
    ])

    # Generates file. 
    tex_file = data_file.with_suffix(".tex")
    log.info("generating %s", tex_file) 
    tex_file.parent.mkdir(parents=True, exist_ok=True)
    tex_file.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": 

    # input data
    data_file = Path(input("Data file path: "))
    generate_texFile(data_file)


