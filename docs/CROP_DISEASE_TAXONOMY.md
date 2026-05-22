# AgriDoctor AI - Crop Disease Taxonomy

## Overview

This taxonomy covers **6 crops** relevant to South Asia and UK hobby farming, with **38 disease/condition labels** organized into broad categories.

---

## Label Categories

| Category  | Code | Description                                |
| --------- | ---- | ------------------------------------------ |
| Fungal    | `F`  | Diseases caused by fungal pathogens        |
| Bacterial | `B`  | Diseases caused by bacterial infections    |
| Viral     | `V`  | Diseases caused by viral infections        |
| Pest      | `P`  | Damage from insects, mites, or other pests |
| Nutrient  | `N`  | Deficiencies or toxicities of nutrients    |
| Healthy   | `H`  | No disease detected                        |
| Unknown   | `U`  | Cannot classify / needs expert review      |

---

## Complete Taxonomy by Crop

### 1. Tomato (10 labels)

| Label ID           | Disease Name       | Category | Description                                          |
| ------------------ | ------------------ | -------- | ---------------------------------------------------- |
| `TOM_EARLY_BLIGHT` | Early Blight       | F        | Brown spots with concentric rings on lower leaves    |
| `TOM_LATE_BLIGHT`  | Late Blight        | F        | Water-soaked lesions, white mold on underside        |
| `TOM_LEAF_MOLD`    | Leaf Mold          | F        | Yellow spots on upper leaves, olive-green mold below |
| `TOM_SEPTORIA`     | Septoria Leaf Spot | F        | Small dark spots with gray centers                   |
| `TOM_SPIDER_MITES` | Spider Mites       | P        | Fine webbing, yellow stippling on leaves             |
| `TOM_MOSAIC`       | Mosaic Virus       | V        | Mottled yellow-green pattern, distorted leaves       |
| `TOM_BACL_SPOT`    | Bacterial Spot     | B        | Small, dark, water-soaked spots on leaves/fruit      |
| `TOM_BLOSSOM_ROT`  | Blossom End Rot    | N        | Calcium deficiency, dark sunken fruit bottom         |
| `TOM_HEALTHY`      | Healthy            | H        | No visible disease symptoms                          |
| `TOM_UNKNOWN`      | Unknown/Other      | U        | Unclassifiable condition                             |

### 2. Potato (8 labels)

| Label ID           | Disease Name  | Category | Description                             |
| ------------------ | ------------- | -------- | --------------------------------------- |
| `POT_EARLY_BLIGHT` | Early Blight  | F        | Target-shaped brown lesions on leaves   |
| `POT_LATE_BLIGHT`  | Late Blight   | F        | Dark water-soaked lesions, rapid spread |
| `POT_BLACKLEG`     | Blackleg      | B        | Black slimy stem base, wilting          |
| `POT_SCAB`         | Common Scab   | B        | Rough, corky patches on tubers          |
| `POT_VIRAL`        | Viral Mosaic  | V        | Mottled leaves, stunted growth          |
| `POT_APHIDS`       | Aphid Damage  | P        | Curled leaves, honeydew presence        |
| `POT_HEALTHY`      | Healthy       | H        | No visible disease symptoms             |
| `POT_UNKNOWN`      | Unknown/Other | U        | Unclassifiable condition                |

### 3. Rice (8 labels)

| Label ID             | Disease Name          | Category | Description                                    |
| -------------------- | --------------------- | -------- | ---------------------------------------------- |
| `RICE_BLAST`         | Rice Blast            | F        | Diamond-shaped lesions with gray centers       |
| `RICE_BROWN_SPOT`    | Brown Spot            | F        | Oval brown spots with gray center              |
| `RICE_BACT_BLIGHT`   | Bacterial Leaf Blight | B        | Water-soaked streaks turning yellow then white |
| `RICE_TUNGRO`        | Tungro Virus          | V        | Yellow-orange discoloration, stunting          |
| `RICE_SHEATH_BLIGHT` | Sheath Blight         | F        | Irregular gray-green water-soaked lesions      |
| `RICE_STEMB`         | Stem Borer            | P        | Dead hearts, white heads                       |
| `RICE_HEALTHY`       | Healthy               | H        | No visible disease symptoms                    |
| `RICE_UNKNOWN`       | Unknown/Other         | U        | Unclassifiable condition                       |

### 4. Maize/Corn (7 labels)

| Label ID        | Disease Name         | Category | Description                            |
| --------------- | -------------------- | -------- | -------------------------------------- |
| `MAIZE_NLB`     | Northern Leaf Blight | F        | Long elliptical gray-green lesions     |
| `MAIZE_RUST`    | Common Rust          | F        | Reddish-brown pustules on leaves       |
| `MAIZE_GLS`     | Gray Leaf Spot       | F        | Rectangular gray lesions between veins |
| `MAIZE_SMUT`    | Common Smut          | F        | Large gray galls on ears/stalks        |
| `MAIZE_BORER`   | Stem Borer           | P        | Holes in stalks, frass, lodging        |
| `MAIZE_HEALTHY` | Healthy              | H        | No visible disease symptoms            |
| `MAIZE_UNKNOWN` | Unknown/Other        | U        | Unclassifiable condition               |

### 5. Chili/Pepper (7 labels)

| Label ID          | Disease Name    | Category | Description                                  |
| ----------------- | --------------- | -------- | -------------------------------------------- |
| `CHILI_ANTHRAC`   | Anthracnose     | F        | Dark sunken spots on fruits with pink spores |
| `CHILI_BACT_WILT` | Bacterial Wilt  | B        | Sudden wilting, vascular browning            |
| `CHILI_LEAF_CURL` | Leaf Curl Virus | V        | Leaf curling, puckering, stunted growth      |
| `CHILI_POWDERY`   | Powdery Mildew  | F        | White powdery coating on leaves              |
| `CHILI_THRIPS`    | Thrips Damage   | P        | Silvery streaks, distorted leaves            |
| `CHILI_HEALTHY`   | Healthy         | H        | No visible disease symptoms                  |
| `CHILI_UNKNOWN`   | Unknown/Other   | U        | Unclassifiable condition                     |

### 6. Cucumber (8 labels)

| Label ID      | Disease Name          | Category | Description                                  |
| ------------- | --------------------- | -------- | -------------------------------------------- |
| `CUC_POWDERY` | Powdery Mildew        | F        | White powdery patches on leaves              |
| `CUC_DOWNY`   | Downy Mildew          | F        | Angular yellow spots, purple mold underneath |
| `CUC_ANGULAR` | Angular Leaf Spot     | B        | Angular water-soaked spots, turn tan         |
| `CUC_MOSAIC`  | Cucumber Mosaic Virus | V        | Mottled leaves, distorted fruit              |
| `CUC_ANTHRAC` | Anthracnose           | F        | Circular tan spots with dark margins         |
| `CUC_APHIDS`  | Aphid Damage          | P        | Curled leaves, sooty mold                    |
| `CUC_HEALTHY` | Healthy               | H        | No visible disease symptoms                  |
| `CUC_UNKNOWN` | Unknown/Other         | U        | Unclassifiable condition                     |

---

## Summary Statistics

| Metric             | Count |
| ------------------ | ----- |
| Total Crops        | 6     |
| Total Labels       | 38    |
| Fungal Diseases    | 18    |
| Bacterial Diseases | 6     |
| Viral Diseases     | 5     |
| Pest Damage        | 6     |
| Nutrient Issues    | 1     |
| Healthy Labels     | 6     |
| Unknown Labels     | 6     |

---

## Label Distribution Guidance

For balanced training:

- Ensure each crop has ≥ 100 samples per disease class
- Healthy class should be ~15-20% of total
- Unknown class used for edge cases, < 5% of total
- Rare diseases can use GAN/Diffusion augmentation

---

## JSON Schema Reference

```json
{
  "label_id": "TOM_EARLY_BLIGHT",
  "crop": "tomato",
  "disease_name": "Early Blight",
  "category": "fungal",
  "category_code": "F",
  "severity_range": [0.0, 1.0],
  "description": "Brown spots with concentric rings on lower leaves",
  "treatment_keywords": ["fungicide", "remove_affected", "air_circulation"],
  "prevention_keywords": ["crop_rotation", "mulching", "resistant_varieties"]
}
```
