# Final Report LaTeX Project

This directory contains the LaTeX source files for the final report.

## Compilation

To compile the document, use:

```bash
pdflatex main.tex
```

For documents with bibliography, run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use `latexmk` for automatic compilation:

```bash
latexmk -pdf main.tex
```

## Structure

- `main.tex` - Main LaTeX document
- `README.md` - This file

## Adding Figures

Place figure files (PNG, PDF, etc.) in a `figures/` subdirectory and reference them using:

```latex
\includegraphics[width=0.8\textwidth]{figures/figure_name}
```

## Adding Bibliography

Create a `references.bib` file and uncomment the bibliography commands in `main.tex`.

