# CASA MSc thesis (migrated draft)

English-only Bookdown build of the CASA0004 dissertation.

## Preview the HTML book

Already built at `docs/index.html`. Open it with either:

```bash
open "docs/index.html"
```

or in Finder: go to `06_writing/CASA-MSc-thesis-main/docs/` and double-click `index.html`.

In Cursor/VS Code you can also right-click `docs/index.html` → **Reveal in Finder**, then open in a browser. A Live Preview / Simple Browser tab also works if you have that extension.

## Build

1. Open `CASA_thesis_template.Rproj` in RStudio (project `renv` will activate).
2. If packages are missing: `renv::restore()`.
3. Build Book → GitBook / PDF, or in the Console:

```r
bookdown::render_book("index.Rmd", output_format = "bookdown::gitbook")
```

Figures are symlinked from `../../05_outputs/figures/` into `figures/`.

Author metadata in `index.Rmd` still uses placeholders (`[Student Name]`, `[Supervisor]`).
