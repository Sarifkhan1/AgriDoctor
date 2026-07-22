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

    # --- Rice -----------------------------------------------------------------
    "RICE_BLAST": {
        "severity": 0.8,
        "urgency": Urgency.high,
        "summary": "Rice blast (Magnaporthe oryzae) — spindle-shaped lesions with grey centres; the most damaging rice disease worldwide.",
        "now": [
            "Apply a labelled blast fungicide (tricyclazole or a strobilurin) promptly if lesions are spreading.",
            "Drain the field briefly if continuously flooded, and avoid further nitrogen for now — lush growth worsens blast.",
            "Remove and destroy heavily infected tillers.",
        ],
        "prevention": [
            "Grow blast-resistant varieties where available.",
            "Split nitrogen applications instead of one heavy dose.",
            "Avoid dense seeding; keep the canopy open.",
        ],
        "help": ["Neck or panicle lesions appear — neck blast can cause near-total yield loss."],
    },
    "RICE_BROWN_SPOT": {
        "severity": 0.55,
        "summary": "Brown spot (Bipolaris oryzae) — many small oval brown spots; often a sign of nutrient-poor or drought-stressed soil.",
        "now": [
            "Correct underlying nutrient deficiency, especially potassium; a soil test helps.",
            "Apply a labelled fungicide if the flag leaf and panicle are threatened.",
            "Maintain even, adequate water — brown spot worsens under drought stress.",
        ],
        "prevention": [
            "Use certified clean seed; treat seed before sowing.",
            "Keep balanced fertility, particularly potassium and silicon.",
        ],
        "help": ["Spots reach the flag leaf or grains discolour."],
    },
    "RICE_BACT_BLIGHT": {
        "severity": 0.75,
        "urgency": Urgency.high,
        "summary": "Bacterial leaf blight (Xanthomonas oryzae) — yellow-to-white lesions spreading from leaf tips and margins; bacterial, so fungicides do not control it.",
        "now": [
            "Drain the field to reduce spread; the bacterium moves in water.",
            "Stop nitrogen top-dressing until the outbreak subsides.",
            "Remove and destroy the worst-affected plants and nearby weed hosts.",
        ],
        "prevention": [
            "Plant resistant varieties and certified clean seed.",
            "Avoid clipping seedling tips at transplant, which spreads the bacterium.",
            "Manage water so fields are not continuously deeply flooded.",
        ],
        "help": ["A large share of the crop wilts (the 'kresek' seedling stage) or lesions spread rapidly."],
    },
    "RICE_SHEATH_BLIGHT": {
        "severity": 0.6,
        "summary": "Sheath blight (Rhizoctonia solani) — oval greenish-grey lesions on leaf sheaths near the waterline, spreading upward in dense canopies.",
        "now": [
            "Apply a labelled fungicide (a strobilurin or validamycin) if lesions climb toward the upper leaves.",
            "Avoid further nitrogen; dense lush growth drives this disease.",
        ],
        "prevention": [
            "Widen spacing to open the canopy and lower humidity.",
            "Split nitrogen rather than a single heavy application.",
            "Remove infected stubble after harvest.",
        ],
        "help": ["Lesions reach the flag leaf during grain filling."],
    },
    "RICE_HEALTHY": {"healthy": True, "crop": "rice"},

    # --- Chili ----------------------------------------------------------------
    "CHILI_POWDERY": {
        "severity": 0.5,
        "summary": "Powdery mildew (Leveillula taurica) — white powdery patches, often on the underside, with yellow blotches above.",
        "now": [
            "Apply a labelled fungicide (sulphur, potassium bicarbonate, or a systemic) — treat leaf undersides.",
            "Improve airflow and reduce shade around plants.",
        ],
        "prevention": [
            "Avoid dense planting; keep foliage dry.",
            "Scout leaf undersides weekly in warm, humid weather.",
        ],
        "help": ["Defoliation exposes fruit to sunscald."],
    },
    "CHILI_THRIPS": {
        "severity": 0.5,
        "summary": "Thrips / mite damage — silvery scarring and upward leaf curl; thrips also transmit viruses.",
        "now": [
            "Apply insecticidal soap, neem, or a labelled insecticide, repeating after 5-7 days.",
            "Use blue sticky traps to monitor and reduce numbers.",
            "Remove severely curled growing tips.",
        ],
        "prevention": [
            "Control weeds that harbour thrips.",
            "Inspect new growth weekly, especially in hot dry spells.",
        ],
        "help": ["Curling and stunting persist after treatment — check for virus transmission."],
    },
    "CHILI_CERCOSPORA": {
        "severity": 0.5,
        "summary": "Cercospora leaf spot — circular spots with pale grey centres and dark rings ('frog-eye'), causing leaf drop.",
        "now": [
            "Remove affected leaves and apply a labelled fungicide (chlorothalonil or copper).",
            "Avoid overhead watering; keep foliage dry.",
        ],
        "prevention": [
            "Rotate crops and clear debris after harvest.",
            "Space plants for airflow.",
        ],
        "help": ["Heavy leaf drop exposes fruit to sunscald."],
    },
    "CHILI_HEALTHY": {"healthy": True, "crop": "chili"},

    # --- Cucumber -------------------------------------------------------------
    "CUC_DOWNY": {
        "severity": 0.75,
        "urgency": Urgency.high,
        "summary": "Downy mildew (Pseudoperonospora cubensis) — angular yellow patches bounded by leaf veins, grey mould beneath; spreads fast in humid weather.",
        "now": [
            "Apply a labelled downy-mildew fungicide promptly — this disease moves quickly.",
            "Improve airflow and avoid overhead irrigation.",
            "Remove the most affected leaves.",
        ],
        "prevention": [
            "Grow resistant varieties where available.",
            "Water early in the day so foliage dries fast.",
            "Scout daily during warm, humid spells.",
        ],
        "help": ["Yellowing spreads rapidly across the canopy despite treatment."],
    },
    "CUC_ANTHRAC": {
        "severity": 0.6,
        "summary": "Anthracnose (Colletotrichum) — brown-to-tan leaf spots that tear out, and sunken fruit lesions.",
        "now": [
            "Remove affected leaves and fruit; apply a labelled fungicide (chlorothalonil).",
            "Stop overhead watering and avoid working among wet vines.",
        ],
        "prevention": [
            "Use clean seed and rotate away from cucurbits for two years.",
            "Trellis vines to keep foliage and fruit off the soil.",
        ],
        "help": ["Fruit lesions appear — marketable yield is at risk."],
    },
    "CUC_BACT_WILT": {
        "severity": 0.85,
        "urgency": Urgency.high,
        "summary": "Bacterial wilt (Erwinia tracheiphila) — sudden wilting of leaves and runners, spread by cucumber beetles. Infected plants cannot be cured.",
        "now": [
            "Remove and destroy wilting plants — they will not recover and are a source of spread.",
            "Control cucumber beetles immediately (the vector) with a labelled insecticide or row covers.",
            "Confirm by cutting a wilted stem: sticky white strands mean bacterial wilt.",
        ],
        "prevention": [
            "Control cucumber beetles from seedling emergence.",
            "Use row covers until flowering; grow tolerant varieties.",
        ],
        "help": ["Several plants wilt — the beetle vector is active and needs urgent control."],
    },
    "CUC_GUMMY": {
        "severity": 0.65,
        "summary": "Gummy stem blight (Didymella bryoniae) — brown leaf lesions and dark gummy cankers on stems.",
        "now": [
            "Apply a labelled fungicide; remove and destroy affected vines.",
            "Avoid overhead irrigation and reduce leaf wetness.",
        ],
        "prevention": [
            "Use clean seed and a 2-year cucurbit rotation.",
            "Improve airflow; remove crop debris after harvest.",
        ],
        "help": ["Stem cankers girdle the main stem, collapsing the plant."],
    },
    "CUC_HEALTHY": {"healthy": True, "crop": "cucumber"},

    # --- Eggplant -------------------------------------------------------------
    "EGG_LEAF_SPOT": {
        "severity": 0.5,
        "summary": "Cercospora leaf spot — angular to circular grey-brown spots with concentric rings, causing leaf drop.",
        "now": [
            "Remove affected leaves and apply a labelled fungicide (chlorothalonil or copper).",
            "Avoid overhead watering; improve airflow.",
        ],
        "prevention": [
            "Rotate away from eggplant, tomato and pepper for two years.",
            "Clear debris after harvest; stake plants off the soil.",
        ],
        "help": ["Heavy defoliation exposes fruit to sunscald."],
    },
    "EGG_FLEA_BEETLE": {
        "severity": 0.5,
        "summary": "Flea beetle damage — many small round 'shot-hole' punctures; seedlings are most at risk.",
        "now": [
            "Apply a labelled insecticide or spinosad if damage is heavy on young plants.",
            "Use row covers on seedlings; dust with kaolin clay as a deterrent.",
        ],
        "prevention": [
            "Protect transplants early with covers until established.",
            "Clear weeds and debris where beetles overwinter.",
        ],
        "help": ["Seedlings are being defoliated faster than they can grow."],
    },
    "EGG_VERTICILLIUM": {
        "severity": 0.75,
        "urgency": Urgency.high,
        "summary": "Verticillium wilt — yellowing and wilting, often one-sided, from a soil fungus that blocks the plant's water vessels. There is no in-season cure.",
        "now": [
            "Remove and destroy severely wilted plants; the pathogen is soil-borne and persistent.",
            "Keep remaining plants evenly watered to reduce stress; do not overwater.",
        ],
        "prevention": [
            "Use a long rotation (4+ years) away from eggplant, tomato, potato and pepper.",
            "Grow grafted or tolerant varieties on infested land.",
        ],
        "help": ["Wilting spreads to several plants — the soil is likely infested and rotation is needed."],
    },
    "EGG_POWDERY": {
        "severity": 0.5,
        "summary": "Powdery mildew — white powdery coating on the upper leaf surface, reducing vigour and yield.",
        "now": [
            "Apply a labelled fungicide (sulphur or potassium bicarbonate).",
            "Improve airflow and reduce shading.",
        ],
        "prevention": [
            "Space plants adequately; scout weekly in warm weather.",
        ],
        "help": ["Coating covers most of the canopy despite treatment."],
    },
    "EGG_MOSAIC": {
        "severity": 0.7,
        "urgency": Urgency.high,
        "summary": "Mosaic virus — mottled light and dark green, puckered distorted leaves. Infected plants cannot be cured.",
        "now": [
            "Remove and destroy infected plants; wash hands and disinfect tools before touching healthy ones.",
            "Control aphids, which spread many mosaic viruses.",
        ],
        "prevention": [
            "Use clean seed and disinfect tools between plants.",
            "Control weeds that act as a virus reservoir.",
        ],
        "help": ["Several plants show symptoms — find and cut the transmission route."],
    },
    "EGG_HEALTHY": {"healthy": True, "crop": "eggplant"},
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
    if label_id.endswith("_HEALTHY"):
        return 0.0
    entry = _ADVICE.get(label_id)
    if entry is None:
        # A disease we have no curated entry for: treat as a genuine but
        # moderate condition, never as harmless (0.0 is reserved for healthy).
        return 0.5
    if entry.get("healthy"):
        return 0.0
    return float(entry.get("severity", 0.5))


def urgency_for(label_id: str) -> Urgency:
    if label_id.endswith("_HEALTHY"):
        return Urgency.low
    entry = _ADVICE.get(label_id)
    if entry and "urgency" in entry:
        return entry["urgency"]
    return Urgency.high if severity_for(label_id) > 0.7 else Urgency.medium


def _healthy_advice(label_id: str) -> AdviceBlock:
    crop = (label_meta(label_id) or {}).get("crop") or "plant"
    return AdviceBlock(
        summary=_HEALTHY_TEMPLATE["summary"].format(crop=crop),
        what_to_do_now=list(_HEALTHY_TEMPLATE["now"]),
        prevention=list(_HEALTHY_TEMPLATE["prevention"]),
        when_to_get_help=list(_HEALTHY_TEMPLATE["help"]),
    )


def advice_for(label_id: str) -> Optional[AdviceBlock]:
    """Curated advice for a CNN-predictable label, or None if not covered."""
    entry = _ADVICE.get(label_id)
    if entry is None:
        # Any *_HEALTHY label gets the healthy template even without an explicit
        # entry, so newly added crops need no boilerplate healthy record.
        if label_id.endswith("_HEALTHY"):
            return _healthy_advice(label_id)
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
