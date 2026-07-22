# Datasets

AgriDoctor's local CNN is trained on public datasets pulled from the Hugging Face
Hub. The **images themselves are not committed** — they are large and freely
re-downloadable — so this file plus [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py)
are what make the training corpus fully reproducible. Running the prepare script
downloads every source below and caches it to `data/dataset_v2/` (git-ignored).

```bash
python scripts/prepare_dataset.py --out data/dataset_v2 --oos-per-class 6000
```

## Sources

| Purpose | Hugging Face dataset | License | Notes |
| ------- | -------------------- | ------- | ----- |
| Tomato, potato, pepper, maize diseases **and** the `OOS_OTHER_PLANT` reject class | [`minhhungg/plant-disease-dataset`](https://huggingface.co/datasets/minhhungg/plant-disease-dataset) | Public (PlantVillage-derived) | 70,295 images, 38 classes. 19 map to AgriDoctor crops; the other 19 species (apple, grape, orange, soybean, …) become the "other plant" reject class. Plain studio backgrounds. |
| Rice diseases | [`Project-AgML/rice_leaf_disease_classification`](https://huggingface.co/datasets/Project-AgML/rice_leaf_disease_classification) | Public research | Blast, brown spot, bacterial leaf blight, sheath blight, healthy. |
| Chili diseases | [`Project-AgML/COLD_chili_leaf_disease_classification`](https://huggingface.co/datasets/Project-AgML/COLD_chili_leaf_disease_classification) | Public research | Cercospora, powdery mildew, mites/thrips, healthy. Small (~430 images) — the weakest-supported crop. |
| Cucumber diseases | [`Project-AgML/cucumber_disease_classification`](https://huggingface.co/datasets/Project-AgML/cucumber_disease_classification) | Public research | Anthracnose, downy mildew, bacterial wilt, gummy stem blight, healthy (leaf classes only; fruit classes dropped). |
| `OOS_NOT_PLANT` reject class | [`zh-plus/tiny-imagenet`](https://huggingface.co/datasets/zh-plus/tiny-imagenet) | Public (ImageNet-derived) | General object photos, used so the model can decline non-vegetation images locally. |

> An earlier eggplant source (`Project-AgML/eggplant_leaf_disease_classification`)
> and the pepper classes are supported by the builder but are pruned from the
> deployed 6-crop model (tomato, potato, rice, maize, chili, cucumber).

## Why two out-of-scope classes

A plain N-way classifier must assign one of its known labels to every input,
including a photo of a car. AgriDoctor instead trains two explicit reject
classes so declining is a learned outcome:

- **`OOS_OTHER_PLANT`** — a real leaf, but not a supported crop → escalate to the
  hosted vision model.
- **`OOS_NOT_PLANT`** — not vegetation at all → reject locally, no API call.

See [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py) (`AGML_SOURCES` and
`MAP_RULES`) for the exact class-to-taxonomy mapping, and
[`data/taxonomy.json`](data/taxonomy.json) for the label definitions.

## Provenance and honesty notes

- **Mixed sources.** The four original crops (PlantVillage) have plain studio
  backgrounds; the rice/chili/cucumber sources (Project-AgML) have varied field
  backgrounds. This is disclosed because a model can exploit background
  differences as a shortcut; `scripts/evaluate_cnn.py` reports the out-of-scope
  false-acceptance rate and a corruption-robustness suite precisely to surface
  that risk.
- **Clean-data accuracy overstates field performance.** All sources are curated
  imagery; held-out accuracy is in-distribution and higher than real-world use.
  The evaluation's corruption suite is the honest proxy for field conditions.
- **No private or scraped data** is used; every source is a public Hugging Face
  dataset, downloaded at build time and never redistributed in this repository.
