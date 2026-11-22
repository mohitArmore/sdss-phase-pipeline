# Templates

`paper_meta_template.yaml` — copy to `data/raw/papers/<paper_id>/meta.yaml` and fill in from the paper.
Every field is explained inline; the full workflow is in `docs/PAPER_DATASET_GUIDE.md`.
Unknown values stay `null` (the pipeline warns and flags; it never invents them).

Sample IDs are derived as `{paper_id}_f{figure}{panel}_{route}_{condition}_{image_type}`,
with route codes: publisher_figure=pub, pdf_embedded=pdfemb, pdf_render=pdfrend,
supplementary=supp, screenshot=scr, author_supplied=auth.
