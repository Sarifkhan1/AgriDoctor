"""
Offline treatment advice — the knowledge layer for locally-diagnosed cases.

When the CNN answers a case, the disease is already *known*; generating advice
from a language model would add an API call, a rate limit and a hallucination
surface to restate fixed agronomic guidance. So the advice for every class the
CNN can predict is curated here instead.

Consequences worth stating plainly:
  * a locally-diagnosed case needs zero network calls end to end;
  * the advice is deterministic and reviewable — you can read exactly what a
    farmer will be told for TOM_LATE_BLIGHT, which is not true of generated text;
  * it only covers labels the CNN can produce. Anything escalated to the hosted
    model still gets model-written advice (see analyzer.py).

Guidance is deliberately non-prescriptive about chemical dosages: it names active
ingredients and directs the user to the local label rate, because correct rates
are region- and formulation-specific and getting them wrong damages crops.
"""

from typing import Optional

from .schemas import AdviceBlock, Urgency
from .taxonomy import label_meta

# Per-label guidance. Keys are taxonomy label_ids the CNN is trained to predict.
_ADVICE: dict[str, dict] = {
    # --- Tomato ---------------------------------------------------------------
    "TOM_EARLY_BLIGHT": {
        "severity": 0.55,
        "summary": "Early blight (Alternaria) — a fungal disease that starts on the oldest, lowest leaves and works upward.",
        "now": [
            "Remove and destroy affected lower leaves; do not compost them.",
            "Mulch around the base so rain cannot splash soil spores onto leaves.",
            "Apply a protectant fungicide (chlorothalonil or copper) at the label rate.",
            "Water at the soil line only — keep foliage dry.",
        ],
        "prevention": [
            "Rotate away from tomato/potato ground for 2-3 seasons.",
            "Space plants for airflow and stake them off the soil.",
            "Clear all crop debris at the end of the season.",
        ],
        "help": [
            "Lesions reach the upper third of the plant despite treatment.",
            "Fruit begins to show sunken dark spots.",
        ],
    },
    "TOM_LATE_BLIGHT": {
        "severity": 0.9,
        "urgency": Urgency.high,
        "summary": "Late blight (Phytophthora infestans) — the most destructive tomato disease. In cool, wet weather it can kill a crop in under two weeks.",
        "now": [
            "Act today — this spreads far faster than other blights.",
            "Remove and destroy affected plants entirely; bag them, do not compost.",
            "Treat remaining plants with a late-blight-specific fungicide (mancozeb, or metalaxyl-based).",
            "Warn neighbouring growers — spores travel on wind between fields.",
        ],
        "prevention": [
            "Plant resistant varieties where available.",
            "Never leave cull piles or volunteer potatoes nearby — they harbour the pathogen.",
            "Increase scouting during cool, humid, rainy spells.",
        ],
        "help": [
            "More than a few plants are affected — treat this as urgent.",
            "Weather is forecast cool and wet over the coming week.",
        ],
    },
    "TOM_LEAF_MOLD": {
        "severity": 0.5,
        "summary": "Leaf mold (Passalora fulva) — a humidity-driven fungus, most common under cover.",
        "now": [
            "Increase ventilation; this disease is driven by high humidity.",
            "Remove affected leaves to reduce spore load.",
            "Apply a copper-based or chlorothalonil fungicide at the label rate.",
            "Avoid overhead watering, especially late in the day.",
        ],
        "prevention": [
            "Keep humidity below about 85% in tunnels and greenhouses.",
            "Prune lower foliage to open the canopy.",
            "Choose resistant cultivars for protected cropping.",
        ],
        "help": ["Yellowing spreads through the canopy despite better ventilation."],
    },
    "TOM_SEPTORIA": {
        "severity": 0.55,
        "summary": "Septoria leaf spot — many small circular spots with pale grey centres and dark margins.",
        "now": [
            "Strip affected lower leaves and destroy them.",
            "Apply a protectant fungicide (chlorothalonil or copper) at the label rate.",
            "Mulch to stop rain splashing spores up from the soil.",
        ],
        "prevention": [
            "Rotate crops on a 2-3 year cycle.",
            "Water at soil level; keep foliage dry.",
            "Remove all debris after harvest — the fungus overwinters in it.",
        ],
        "help": ["Defoliation reaches the fruit-bearing part of the plant."],
    },
    "TOM_SPIDER_MITES": {
        "severity": 0.5,
        "summary": "Two-spotted spider mites — a pest, not a disease. Fine webbing and yellow stippling, worst in hot dry conditions.",
        "now": [
            "Spray the UNDERSIDE of leaves forcefully with water to dislodge mites.",
            "Apply insecticidal soap, neem oil, or a labelled miticide — repeat after 5-7 days to catch newly hatched mites.",
            "Raise humidity around the plants; mites thrive in hot, dry air.",
            "Isolate badly affected plants.",
        ],
        "prevention": [
            "Inspect leaf undersides weekly during hot dry spells.",
            "Avoid drought-stressing plants — stressed plants attract mites.",
            "Encourage predatory mites and ladybirds; avoid broad-spectrum insecticides.",
        ],
        "help": ["Webbing covers growing tips, or populations rebound after two treatments."],
    },
    "TOM_MOSAIC": {
        "severity": 0.75,
        "urgency": Urgency.high,
        "summary": "Tomato mosaic virus — mottled light/dark green leaves and distorted growth. There is no cure for an infected plant.",
        "now": [
            "Remove and destroy infected plants — they cannot be treated or cured.",
            "Wash hands and disinfect all tools before touching healthy plants; this virus spreads by contact.",
            "Do not use tobacco products near the crop — TMV transmits from them.",
        ],
        "prevention": [
            "Use certified virus-free seed.",
            "Disinfect stakes, ties and tools between plants.",
            "Control weeds that act as a virus reservoir.",
        ],
        "help": ["Several plants show symptoms — you likely have a transmission route to find."],
    },
    "TOM_YELLOW_LEAF_CURL": {
        "severity": 0.85,
        "urgency": Urgency.high,
        "summary": "Tomato yellow leaf curl virus — spread by whitefly. Leaves cup upward with yellow margins and the plant stunts badly.",
        "now": [
            "Remove and destroy infected plants; the virus itself cannot be cured.",
            "Control whitefly immediately — they are the vector. Use yellow sticky traps and a labelled insecticide or insecticidal soap.",
            "Check leaf undersides on neighbouring plants for whitefly.",
        ],
        "prevention": [
            "Grow resistant (TYLCV-tolerant) varieties in areas with whitefly pressure.",
            "Use fine insect netting on protected crops.",
            "Use reflective mulch to deter incoming whitefly.",
        ],
        "help": ["Whitefly persist after treatment, or new plants show symptoms."],
    },
    "TOM_TARGET_SPOT": {
        "severity": 0.6,
        "summary": "Target spot (Corynespora cassiicola) — brown lesions with concentric rings, on leaves, stems and fruit.",
        "now": [
            "Remove affected leaves and any spotted fruit.",
            "Apply a labelled fungicide (chlorothalonil or azoxystrobin).",
            "Improve airflow and avoid overhead irrigation.",
        ],
        "prevention": [
            "Rotate crops and clear debris thoroughly after harvest.",
            "Avoid dense planting; keep the canopy open.",
        ],
        "help": ["Fruit lesions appear — marketable yield is now at risk."],
    },
    "TOM_BACL_SPOT": {
        "severity": 0.65,
        "summary": "Bacterial spot — small dark water-soaked spots on leaves and fruit. Bacterial, so fungicides alone will not control it.",
        "now": [
            "Remove affected leaves; work only when foliage is dry to limit spread.",
            "Apply a copper-based bactericide at the label rate.",
            "Stop overhead irrigation — splashing water is the main spread route.",
        ],
        "prevention": [
            "Use certified disease-free seed and transplants.",
            "Rotate for at least two years away from tomato and pepper.",
            "Avoid handling plants when wet.",
        ],
        "help": ["Spots appear on fruit, or spread continues after copper treatment."],
    },
    "TOM_HEALTHY": {"healthy": True, "crop": "tomato"},

    # --- Potato ---------------------------------------------------------------
    "POT_EARLY_BLIGHT": {
        "severity": 0.55,
        "summary": "Potato early blight (Alternaria solani) — target-shaped brown lesions on older leaves first.",
        "now": [
            "Remove affected lower foliage.",
            "Apply a protectant fungicide (mancozeb or chlorothalonil) at the label rate.",
            "Keep plants adequately watered and fed — stressed potatoes are hit hardest.",
        ],
        "prevention": [
            "Rotate on a 3-year cycle away from potato and tomato.",
            "Hill soil well over developing tubers.",
            "Destroy volunteer potatoes and crop debris.",
        ],
        "help": ["Lesions spread to the upper canopy before tuber bulking finishes."],
    },
    "POT_LATE_BLIGHT": {
        "severity": 0.9,
        "urgency": Urgency.high,
        "summary": "Potato late blight (Phytophthora infestans) — the disease behind the historic potato famines. It can destroy a field in days.",
        "now": [
            "Act today. Remove and destroy affected plants; bag them, do not compost.",
            "Apply a late-blight fungicide (mancozeb, or a metalaxyl-based product) to remaining plants.",
            "If tubers are near maturity, consider cutting and removing the haulm to protect them.",
            "Alert nearby growers — spores travel on wind.",
        ],
        "prevention": [
            "Plant certified seed potatoes and resistant varieties.",
            "Hill soil deeply so spores cannot reach tubers.",
            "Never leave cull piles — they are the classic overwintering source.",
        ],
        "help": ["Any confirmed late blight during cool wet weather warrants immediate expert contact."],
    },
    "POT_HEALTHY": {"healthy": True, "crop": "potato"},

    # --- Pepper ---------------------------------------------------------------
    "PEP_BACT_SPOT": {
        "severity": 0.65,
        "summary": "Pepper bacterial spot (Xanthomonas) — dark water-soaked lesions on leaves and fruit, worsened by warm wet weather.",
        "now": [
            "Remove affected leaves; work only when plants are dry.",
            "Apply a copper-based bactericide at the label rate.",
            "Switch from overhead irrigation to drip or soil-level watering.",
        ],
        "prevention": [
            "Use certified disease-free seed; hot-water seed treatment helps.",
            "Rotate away from pepper and tomato for two years.",
            "Do not work among wet plants.",
        ],
        "help": ["Fruit lesions appear, or spread continues despite copper sprays."],
    },
    "PEP_HEALTHY": {"healthy": True, "crop": "pepper"},

    # --- Maize ----------------------------------------------------------------
    "MAIZE_NLB": {
        "severity": 0.65,
        "summary": "Northern leaf blight (Exserohilum turcicum) — long grey-green cigar-shaped lesions on the leaves.",
        "now": [
            "Assess how far up the plant lesions have reached — damage above the ear leaf costs the most yield.",
            "Apply a labelled foliar fungicide if lesions are spreading and the crop is pre-tasselling.",
            "Note the affected area for next season's rotation planning.",
        ],
        "prevention": [
            "Plant resistant hybrids — the single most effective control.",
            "Rotate away from maize for at least one season.",
            "Bury or remove residue; the fungus overwinters in it.",
        ],
        "help": ["Lesions pass the ear leaf before grain fill completes."],
    },
    "MAIZE_RUST": {
        "severity": 0.5,
        "summary": "Common rust (Puccinia sorghi) — small reddish-brown powdery pustules on both leaf surfaces.",
        "now": [
            "Most field maize tolerates common rust without treatment — assess severity first.",
            "Apply a labelled fungicide only if pustules are dense before tasselling, or on sweet corn.",
            "Monitor weekly; cool humid weather accelerates it.",
        ],
        "prevention": [
            "Grow resistant hybrids.",
            "Avoid very late plantings, which face higher rust pressure.",
        ],
        "help": ["Pustules cover a large share of leaf area before tasselling."],
    },
    "MAIZE_GLS": {
        "severity": 0.65,
        "summary": "Grey leaf spot (Cercospora zeae-maydis) — rectangular grey lesions bounded by the leaf veins.",
        "now": [
            "Check whether lesions have reached the ear leaf — that is the economic threshold.",
            "Apply a labelled fungicide at tasselling if pressure is high.",
            "Record the field; this disease builds in continuous maize.",
        ],
        "prevention": [
            "Rotate away from maize — residue is the primary inoculum source.",
            "Use tillage to bury residue where appropriate.",
            "Select hybrids rated for grey leaf spot resistance.",
        ],
        "help": ["Lesions reach the ear leaf during grain fill."],
    },
    "MAIZE_HEALTHY": {"healthy": True, "crop": "maize"},
}

_HEALTHY_TEMPLATE = {
    "summary": "No disease symptoms detected — this {crop} leaf looks healthy.",
    "now": [
        "No treatment needed. Continue your normal care routine.",
        "Keep scouting weekly, checking leaf undersides too.",
    ],
    "prevention": [
        "Water at the soil line rather than over the leaves.",
        "Maintain spacing for airflow and remove fallen debris.",
        "Rotate crops between seasons to stop disease building in the soil.",
    ],
    "help": ["Symptoms appear and spread to several plants within a few days."],
}


def has_advice(label_id: str) -> bool:
    return label_id in _ADVICE


def severity_for(label_id: str) -> float:
    """Curated severity for a label (0.0 healthy .. 1.0 crop-threatening)."""
    entry = _ADVICE.get(label_id)
    if not entry or entry.get("healthy"):
        return 0.0
    return float(entry.get("severity", 0.5))


def urgency_for(label_id: str) -> Urgency:
    entry = _ADVICE.get(label_id)
    if not entry or entry.get("healthy"):
        return Urgency.low
    if "urgency" in entry:
        return entry["urgency"]
    return Urgency.high if severity_for(label_id) > 0.7 else Urgency.medium


def advice_for(label_id: str) -> Optional[AdviceBlock]:
    """Curated advice for a CNN-predictable label, or None if not covered."""
    entry = _ADVICE.get(label_id)
    if entry is None:
        return None

    if entry.get("healthy"):
        crop = entry.get("crop") or (label_meta(label_id) or {}).get("crop") or "plant"
        return AdviceBlock(
            summary=_HEALTHY_TEMPLATE["summary"].format(crop=crop),
            what_to_do_now=list(_HEALTHY_TEMPLATE["now"]),
            prevention=list(_HEALTHY_TEMPLATE["prevention"]),
            when_to_get_help=list(_HEALTHY_TEMPLATE["help"]),
        )

    return AdviceBlock(
        summary=entry["summary"],
        what_to_do_now=list(entry["now"]),
        prevention=list(entry["prevention"]),
        when_to_get_help=list(entry["help"]),
    )


def contextual_notes(
    transcript: str = "",
    notes: Optional[str] = None,
    onset_days: Optional[int] = None,
    spread: Optional[str] = None,
) -> list[str]:
    """Extra lines derived from what the farmer reported, not from the image.

    Kept rule-based and additive: these supplement the curated advice rather than
    altering the diagnosis, which comes from the image alone.
    """
    out: list[str] = []
    if onset_days is not None:
        if onset_days <= 3:
            out.append(
                "You noticed this within the last few days — early treatment is "
                "much more effective, so act now."
            )
        elif onset_days > 21:
            out.append(
                "This has been developing for over three weeks; expect treatment "
                "to take longer and check whether nearby plants are affected."
            )
    if spread and spread.strip().lower() in {"fast", "rapid", "spreading"}:
        out.append(
            "You reported it spreading quickly — isolate or remove the worst-affected "
            "plants before treating the rest."
        )
    blob = f"{transcript or ''} {notes or ''}".lower()
    if any(w in blob for w in ("rain", "wet", "humid", "monsoon")):
        out.append(
            "Wet conditions favour most leaf diseases — time any spray for a dry "
            "window so it is not washed off."
        )
    return out
