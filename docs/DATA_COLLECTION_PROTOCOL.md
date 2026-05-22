# Data Collection Protocol

## Overview

This protocol ensures consistent, high-quality data collection for AgriDoctor AI. Follow these guidelines when collecting crop images and farmer voice notes.

---

## Image Capture Checklist

### Equipment

- [ ] Smartphone with camera ≥ 8MP
- [ ] Clean camera lens
- [ ] Reference card (optional, for color calibration)

### Lighting Guidelines

| Condition | Recommendation                              |
| --------- | ------------------------------------------- |
| Best      | Overcast daylight (diffused natural light)  |
| Good      | Morning/evening sun (avoid harsh shadows)   |
| Avoid     | Direct midday sun, artificial indoor lights |
| Never     | Backlit subjects, flash photography         |

### Required Shots (5 per encounter)

| Shot # | Description                                       | Distance   | Angle         |
| ------ | ------------------------------------------------- | ---------- | ------------- |
| 1      | **Context** - Full plant in environment           | 1-2 meters | Standing      |
| 2      | **Affected Area** - Focus on symptomatic region   | 30-50 cm   | Perpendicular |
| 3      | **Close-up** - Detailed lesion/damage             | 10-15 cm   | Perpendicular |
| 4      | **Comparison** - Healthy vs affected side-by-side | 30 cm      | Perpendicular |
| 5      | **Underside** - Leaf underside if applicable      | 10-15 cm   | From below    |

### Background Control

- Use plain, contrasting background when possible
- Avoid busy backgrounds (other plants, patterns)
- For field shots, include some context but focus on subject
- Remove debris, hands, and irrelevant objects from frame

### Quality Checklist

- [ ] Image is in focus (not blurry)
- [ ] Subject is well-lit
- [ ] No finger covering lens
- [ ] Correct orientation
- [ ] Symptoms clearly visible

---

## Voice Note Script Template

### Duration

- Minimum: 30 seconds
- Maximum: 2 minutes
- Target: 60-90 seconds

### Script Structure

```
"This is [farmer ID] reporting on [date].

1. CROP: I'm looking at my [crop name], [variety if known].
   The plant is about [age in days/weeks] old, at [growth stage].

2. SYMPTOMS: I noticed [describe what you see]:
   - What does it look like? [color, shape, pattern]
   - Where is it? [which leaves, stem, fruit]
   - How much? [few spots, many leaves, whole plant]

3. TIMELINE: I first saw this [number] days ago.
   It is spreading [slowly / quickly / not spreading].

4. CONDITIONS: The weather has been [describe recent weather].
   [Mention rain, temperature, humidity if known]

5. TREATMENTS: I have tried [list any treatments].
   [Or: I have not tried any treatments yet.]

6. SPREAD: [Are other plants affected? How many?]

7. CONCERNS: [Any other observations or questions]"
```

### Voice Recording Tips

- Speak clearly and at normal pace
- Reduce background noise (no machinery, wind)
- Hold phone 15-20 cm from mouth
- Complete all script sections if possible

---

## Consent & De-identification Rules

### Before Collection

- [ ] Farmer provides verbal or written consent
- [ ] Explain data will be used for AI training
- [ ] Explain privacy protections

### During Collection

- [ ] NO faces in photos
- [ ] NO identifying marks (name signs, addresses)
- [ ] NO exact GPS coordinates (region only)
- [ ] Use assigned farmer_id, not real names

### After Collection

- [ ] Review images for identifying information
- [ ] Blur or crop any inadvertent identifiable content
- [ ] Store data in secure location
- [ ] Log consent in metadata

### Data Rights

- Farmers may request data deletion
- Data used only for stated purpose (model training)
- No data sold to third parties

---

## File Naming Convention

### Images

```
{crop}_{encounter_id}_{shot_number}.jpg
Example: tomato_a1b2c3d4_01.jpg
```

### Audio/Speech

```
{encounter_id}_voice.wav
Example: a1b2c3d4_voice.wav
```

### Folder Structure

```
data/
├── images/
│   └── crops/
│       └── tomato/
│           ├── tomato_a1b2c3d4_01.jpg
│           ├── tomato_a1b2c3d4_02.jpg
│           └── ...
├── speech/
│   └── farmer_notes/
│       └── a1b2c3d4_voice.wav
└── raw/
    └── {date}/
        └── {collector_id}/
```

---

## Collector Training Checklist

- [ ] Read this protocol completely
- [ ] Practice taking 5-shot series on sample plants
- [ ] Record sample voice note following script
- [ ] Review samples with supervisor
- [ ] Understand consent requirements
- [ ] Know how to transfer files securely
