# Building the paper-figure dataset — step by step

Goal: for each published figure panel (EBSD phase map, etched optical micrograph,
SEM micrograph) of 2507 SDSS, extract the ferrite/austenite fraction by image
analysis and check it against the value the paper reports.

**Rules**
- Every paper fact (DOI, citation, licence, condition, reported fractions, etchant,
  which phase is dark, palette colours, scale bar) is typed by a human into
  `meta.yaml` from the paper. Unknown → `null`. The code never fills these in.
- Original files are never modified. JPEGs are never re-encoded; all derived
  images and masks are PNG.
- Figures/PDFs are never committed (copyright). Only `meta.yaml`, the manifest,
  CSVs and the markdown report are committed.

Commands below assume the repo root and the default config `configs/data/papers.yaml`.

---

## 1. Download

1. Pick a paper; choose `paper_id` = `{firstauthor}{year}{journalabbrev}`, e.g. `gargalis2023met`.
2. `mkdir -p data/raw/papers/<paper_id>/figures`
3. Get each figure at the best available quality, in this order of preference
   (record which in `acquisition_route`):
   - `publisher_figure` — "Download high-res image" on the article page (often JPEG; keep as is).
   - `supplementary` — supplementary files.
   - `pdf_embedded` — images embedded in the PDF, extracted losslessly:
     `python scripts/ingest_paper.py --paper-dir data/raw/papers/<paper_id> --extract-pdf paper.pdf`
     → `figures/extracted/` + `listing.csv` (page, xref, size, format, sha256).
   - `pdf_render` / `screenshot` — last resort (lowest quality; will be visible in QC).
   - `author_supplied` — files emailed by the authors.
4. Save files with their original bytes (no "Save as…" from an image viewer, no
   format conversion, no resizing). Note the URL and date.
5. Add the paper to `data/LICENSES.md` with its licence copied verbatim.

## 2. Write meta.yaml

`cp templates/paper_meta_template.yaml data/raw/papers/<paper_id>/meta.yaml` and
fill it in. Each field is explained inline. For each panel you need, at minimum:

| field | where it comes from |
|---|---|
| `image_type` | look at the panel: `ebsd_phase` (flat phase colours), `ebsd_phase_iq` (phase colours shaded by image quality), `om`, `sem` |
| `condition`, `build_plane` | figure caption / methods |
| `ebsd_palette` | the figure legend (EBSD); sample hex colours from the image if the legend has no values |
| `etchant`, `dark_phase` | methods + caption (micrographs). For Beraha/KOH etch ferrite is usually dark, but **check the paper** |
| `reported.*` | the number the paper gives for **this** sample; `scope: this_panel` only if measured on this very image; `location_in_paper` e.g. "Table 3" |

`reported.unit`: write `percent` or `fraction` explicitly.

## 3. Preview → crop boxes

```bash
python scripts/ingest_paper.py --paper-dir data/raw/papers/<paper_id> --preview
```

This validates `meta.yaml` (errors + warnings for nulls), hashes every figure and
writes `data/interim/previews/<paper_id>/fig<id>_grid.png`: the original figure
with a pixel grid every 100 px, labelled in **original pixel coordinates**.
Boxes already in `meta.yaml` are drawn: crop = green, ignore = red, scale-bar bbox = magenta.

From the preview, fill in (all `[x0, y0, x1, y1]`, x1/y1 exclusive):
- `crop_box` — the image area of the panel only (exclude panel letter outside, axes, captions).
  A 4 px inner margin is removed automatically (`crop.margin_px`).
- `ignore_boxes` — everything inside the crop that is not microstructure: scale bar,
  its label, panel letters, legends, arrows, insets.
- `scale_bar` — `{um: <length>, bbox: [...]}` with a tight bbox around the bar
  (text inside is OK), or `{um: <length>, px: <measured length>}` if you measured it by hand.

Re-run `--preview` until the boxes sit right.

## 4. Run the pipeline

```bash
python scripts/run_paper_pipeline.py --paper <paper_id>
# or everything:
python scripts/run_paper_pipeline.py --all
```

Stages (each also runs alone: `crop_panels.py`, `calibrate_scale.py`,
`make_masks.py --sample <id>`, `extract_features.py`, `validate.py`):
ingest → crop + ignore map → scale calibration → class mask + overlay →
validation + sensitivity → features.csv → report.

## 5. Review overlays

Open `data/interim/overlays/<sample_id>.png` (blue = austenite, orange = ferrite,
green = sigma, grey = ignored). Check:
- Phases are on the right side (if inverted: `dark_phase` is wrong).
- Legend/scale-bar/labels are grey (ignored); if not, add `ignore_boxes`.
- EBSD: too much grey → palette colours off (sample them again) or raise
  `ebsd.max_delta_e`; boundaries grey is expected with `boundary_policy: ignore`.
- Micrographs: uneven background, scratches, etch artefacts.

## 6. Fix

- Data errors (wrong colour, box, dark phase) → fix `meta.yaml`.
- Method parameters → change `configs/data/papers.yaml` (never the code). The
  config hash is logged and stored in each record, so results stay traceable.
- Re-run step 4 (idempotent: records are upserted by `sample_id`).

## 7. Validate

`reports/validation/summary.md` (MAE, bias, max error; split by image type,
JPEG vs PNG, and paper), `summary.csv` (one row per sample) and
`bland_altman.png`. Tolerances (`validation.tol_pp`): EBSD on the same map 5 pp,
image analysis 8 pp, XRD 10 pp, as-built conditions 2 pp absolute.

QC flags in the manifest/report:

| flag | meaning / action |
|---|---|
| `out_of_tolerance` | error above the applicable tolerance — inspect overlay, then meta.yaml |
| `boundary_sensitive` | ferrite moves > 2 pp when boundary pixels are reassigned/excluded — thin phases or low resolution; report with care |
| `threshold_sensitive` | ferrite moves > 2 pp with ±10 % threshold — weak contrast |
| `jpeg_source` | source was JPEG (expected; compare the JPEG/PNG split in the report) |
| `missing_scale` | no usable scale bar — fractions fine, grain sizes in px only |
| `missing_reported` | no reported value — cannot validate |
| `reported_sum_mismatch` / `reported_unit_ambiguous` | check the reported numbers / set `unit` |
| `small_crop` | crop < `crop.min_side_px` — low statistics |
| `not_segmented` / `missing_extraction` | e.g. `dark_phase` missing — see log |

Logs: `reports/logs/<script>_<confighash>.log`.

## Try it without real papers

```bash
python scripts/run_paper_pipeline.py --all --make-fixture
```

renders the synthetic fixture paper (`tests/fixtures/synthetic_paper/meta.yaml`)
into `data/raw/papers/synthetic_fixture/` and runs everything on it. To keep it
out of the real manifest, use a scratch data root instead:

```bash
python scripts/run_paper_pipeline.py --all --make-fixture --data-root /tmp/sdss_demo --reports-dir /tmp/sdss_demo/reports
```
