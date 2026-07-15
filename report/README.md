# Master Thesis Preparation Project Report

This directory contains the written report from `/groups/icecube/holgerkc/final`.

- `main.pdf` is the compiled report.
- `main.tex` is the LaTeX entry point.
- `chapters/` contains the report text.
- `figures/` contains all figures referenced by the compiled report.
- `references.bib` contains the bibliography metadata.

Build locally with:

```bash
latexmk -pdf main.tex
```

The external source PDFs that were kept in the local `kilder/` folder are not
tracked in this repository.
