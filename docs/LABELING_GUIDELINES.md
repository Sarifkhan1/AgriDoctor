# Labeling Guidelines & QA Process

## Overview

These guidelines ensure consistent, accurate labeling of crop disease images. All labelers must complete training before labeling production data.

---

## Label Assignment

### Primary Label

The **primary label** is the main diagnosis - the most significant condition visible.

**Rules:**

1. Choose the single most prominent/severe condition
2. If multiple conditions present, pick the one causing most damage
3. If unsure between two similar labels, choose the more general one
4. Use `*_UNKNOWN` only when truly unclassifiable

### Secondary Labels

**Secondary labels** capture additional conditions present.

**Rules:**

1. List up to 3 secondary labels
2. Only include clearly visible conditions
3. Order by severity (most severe first)
4. Do not repeat the primary label

### Label Decision Tree

```
Is plant clearly healthy?
├── YES → *_HEALTHY
└── NO
    Is main condition identifiable?
    ├── YES → Assign primary label
    │   Are there other visible conditions?
    │   ├── YES → Add secondary labels
    │   └── NO → Primary only
    └── NO
        Is image quality sufficient?
        ├── NO → Flag for re-capture
        └── YES → *_UNKNOWN (add notes)
```

---

## Severity Scoring (0.0 - 1.0)

Severity measures the extent of damage/infection on the visible plant material.

### Severity Rubric

| Score   | Description | Visual Guide                              |
| ------- | ----------- | ----------------------------------------- |
| 0.0     | Healthy     | No symptoms visible                       |
| 0.1-0.2 | Very mild   | 1-5% of visible area affected, <3 lesions |
| 0.3-0.4 | Mild        | 5-15% affected, scattered spots           |
| 0.5-0.6 | Moderate    | 15-35% affected, multiple leaves          |
| 0.7-0.8 | Severe      | 35-60% affected, spreading damage         |
| 0.9-1.0 | Critical    | >60% affected, plant viability threatened |

### Severity Calculation

```
Severity = weighted average of:
  - Lesion coverage (% of leaf surface)    × 0.5
  - Number of affected leaves              × 0.3
  - Symptom progression stage              × 0.2
```

### Severity Examples

| Condition                     | Typical Range |
| ----------------------------- | ------------- |
| Single small spot on one leaf | 0.1 - 0.2     |
| Multiple spots on 2-3 leaves  | 0.3 - 0.4     |
| Half of lower leaves affected | 0.5 - 0.6     |
| Spreading to upper canopy     | 0.7 - 0.8     |
| Defoliation, stem damage      | 0.9 - 1.0     |

---

## Quality Scoring (0.0 - 1.0)

Quality score measures how suitable the image is for model training.

### Quality Rubric

| Factor         | Weight | 0 (Poor)                     | 0.5 (Acceptable)            | 1.0 (Excellent)      |
| -------------- | ------ | ---------------------------- | --------------------------- | -------------------- |
| **Focus**      | 0.35   | Blurry, no detail visible    | Partially sharp             | Crisp, lesions clear |
| **Lighting**   | 0.25   | Too dark/bright, washed out  | Uneven but usable           | Even, natural        |
| **Framing**    | 0.20   | Subject cut off, too distant | Subject visible but awkward | Well-composed        |
| **Background** | 0.10   | Distracting, busy            | Some clutter                | Clean, contrasting   |
| **Resolution** | 0.10   | Pixelated, compressed        | Adequate                    | High detail          |

### Quality Calculation

```
Quality = Focus×0.35 + Lighting×0.25 + Framing×0.20 + Background×0.10 + Resolution×0.10
```

### Quality Thresholds

| Score     | Action                     |
| --------- | -------------------------- |
| ≥ 0.7     | Accept for training        |
| 0.4 - 0.7 | Accept with augmentation   |
| < 0.4     | Reject, request re-capture |

---

## QA Process

### Double-Labeling Protocol

1. **Random Selection**: 10% of samples selected randomly for QA
2. **Independent Labeling**: Second labeler labels without seeing first labels
3. **Comparison**: Automated comparison of labels

### Agreement Metrics

| Metric                  | Target | Action if Below                      |
| ----------------------- | ------ | ------------------------------------ |
| Primary Label Agreement | ≥ 0.75 | Review edge cases, update guidelines |
| Severity MAE            | ≤ 0.10 | Calibration training for labelers    |
| Quality Score MAE       | ≤ 0.15 | Review quality rubric understanding  |

### Disagreement Resolution

```
Agreement?
├── YES → Accept majority label
└── NO
    Difference minor? (e.g., severity within 0.15)
    ├── YES → Take average
    └── NO
        Escalate to expert reviewer
        └── Expert decides final label
```

### Weekly QA Review

1. Calculate inter-labeler agreement metrics
2. Identify systematic disagreements
3. Update guidelines if needed
4. Re-train labelers on problem areas
5. Document edge cases

---

## Labeler Training

### Requirements Before Labeling

- [ ] Read CROP_DISEASE_TAXONOMY.md completely
- [ ] Study visual examples for each disease
- [ ] Complete practice set (20 images)
- [ ] Achieve ≥ 70% agreement with reference labels
- [ ] Review common mistakes

### Ongoing Training

- Weekly calibration sessions
- Review of disagreement cases
- New disease examples as collected

---

## Edge Cases

### Multiple Diseases Present

- Primary = most severe (causing most damage)
- Secondary = others clearly visible
- Note in comments if interaction unclear

### Mixed Healthy/Diseased

- If >70% healthy: label as HEALTHY with low severity
- If significant symptoms: label disease, note in comments

### Unclear Disease Stage

- Use current visible stage
- Note if early/late stage uncertain

### Photo of Multiple Plants

- Label based on focal plant
- Note if multiple plants visible
- Flag if unclear which plant is subject

---

## Annotation Fields Summary

| Field              | Type     | Required | Description           |
| ------------------ | -------- | -------- | --------------------- |
| `encounter_id`     | UUID     | Yes      | Links to encounter    |
| `primary_label`    | String   | Yes      | Main diagnosis        |
| `secondary_labels` | Array    | No       | Additional conditions |
| `severity_score`   | Float    | Yes      | 0.0 - 1.0             |
| `quality_score`    | Float    | Yes      | 0.0 - 1.0             |
| `labeler_id`       | String   | Yes      | Who labeled           |
| `labeled_at`       | DateTime | Yes      | When labeled          |
| `qa_verified`      | Boolean  | No       | Double-checked        |
| `notes`            | Text     | No       | Edge cases, comments  |
