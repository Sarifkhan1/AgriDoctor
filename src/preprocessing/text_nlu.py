"""
AgriDoctor AI - Text NLU Pipeline
Text cleaning, normalization, and symptom entity extraction from farmer voice notes.
"""

import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Knowledge Base / Dictionaries
# ============================================================================

CROP_SYNONYMS = {
    "tomato": ["tomato", "tomatoes", "tamatar", "tamater"],
    "potato": ["potato", "potatoes", "aloo", "alu"],
    "rice": ["rice", "paddy", "chawal", "dhan"],
    "maize": ["maize", "corn", "makka", "makkai", "bhutta"],
    "chili": ["chili", "chilli", "pepper", "mirch", "mirchi", "capsicum"],
    "cucumber": ["cucumber", "cucumbers", "kheera", "khira", "kakdi"]
}

SYMPTOM_PATTERNS = {
    # Visual symptoms
    "spots": ["spot", "spots", "spotting", "speckle", "speckles", "lesion", "lesions"],
    "yellowing": ["yellow", "yellowing", "yellowed", "chlorosis", "chlorotic"],
    "browning": ["brown", "browning", "browned", "necrosis", "necrotic"],
    "wilting": ["wilt", "wilting", "wilted", "drooping", "droopy", "limp"],
    "curling": ["curl", "curling", "curled", "wrinkled", "puckering", "puckered"],
    "mold": ["mold", "mould", "moldy", "mouldy", "fuzzy", "powdery", "dusty"],
    "rot": ["rot", "rotting", "rotten", "decay", "decaying", "soft"],
    "holes": ["hole", "holes", "eaten", "chewed", "bite", "bites"],
    "webbing": ["web", "webs", "webbing", "cobweb", "silk", "silky"],
    "stunting": ["stunted", "stunting", "dwarf", "small", "not growing"],
    "drying": ["dry", "drying", "dried", "crispy", "burnt", "scorched"],
    "discoloration": ["discolor", "discolored", "discoloration", "pale", "faded"],
    
    # Patterns
    "rings": ["ring", "rings", "concentric", "target", "bullseye"],
    "streaks": ["streak", "streaks", "striped", "lines", "linear"],
    "mosaic": ["mosaic", "mottled", "mottling", "patterned"],
    "margins": ["margin", "margins", "edge", "edges", "tip", "tips"]
}

AFFECTED_PARTS = {
    "leaf": ["leaf", "leaves", "foliage", "patti", "patta"],
    "stem": ["stem", "stalk", "trunk", "branch", "branches"],
    "fruit": ["fruit", "fruits", "berry", "berries", "phal"],
    "root": ["root", "roots", "tuber", "bulb"],
    "flower": ["flower", "flowers", "blossom", "bloom", "phool"],
    "whole_plant": ["plant", "entire", "whole", "all", "everywhere"]
}

DURATION_PATTERNS = [
    (r"(\d+)\s*(?:day|days|din|dino)\s*(?:ago|back|pehle)?", "days"),
    (r"(\d+)\s*(?:week|weeks|hafta|hafte)\s*(?:ago|back|pehle)?", "weeks"),
    (r"(?:yesterday|kal)", "1_day"),
    (r"(?:today|aaj)", "0_days"),
    (r"(?:last week|pichle hafta)", "7_days"),
    (r"(?:few|couple)\s*(?:day|days)", "2-3_days"),
    (r"(?:this morning|subah)", "0_days"),
    (r"(?:since|se)\s*(\d+)", "days")
]

SPREAD_PATTERNS = {
    "fast": ["fast", "rapid", "rapidly", "quick", "quickly", "spreading fast", "tezi se"],
    "slow": ["slow", "slowly", "gradual", "gradually", "dhire"],
    "not_spreading": ["not spreading", "same", "stable", "constant", "nahi badh"]
}

URGENCY_KEYWORDS = {
    "high": ["urgent", "emergency", "dying", "dead", "critical", "severe", "serious", 
             "all plants", "whole field", "spreading fast", "help", "please help"],
    "medium": ["spreading", "getting worse", "worried", "concern", "many plants"],
    "low": ["just checking", "minor", "small", "few spots", "one plant", "healthy"]
}

TREATMENT_KEYWORDS = {
    "neem": ["neem", "neem oil", "neemol", "azadirachtin"],
    "baking_soda": ["baking soda", "sodium bicarbonate", "soda"],
    "copper": ["copper", "bordeaux", "copper sulfate"],
    "fungicide": ["fungicide", "antifungal", "dawaai"],
    "pesticide": ["pesticide", "insecticide", "spray", "chemical"],
    "organic": ["organic", "natural", "homemade", "gharelu"],
    "nothing": ["nothing", "no treatment", "haven't tried", "kuch nahi"]
}

WEATHER_KEYWORDS = {
    "rain": ["rain", "rainy", "raining", "rained", "monsoon", "barish", "baarish"],
    "humidity": ["humid", "humidity", "moist", "damp", "moisture"],
    "hot": ["hot", "heat", "warm", "sunny", "garmi", "dhoop"],
    "cold": ["cold", "cool", "winter", "frost", "thanda", "sardi"]
}


@dataclass
class ExtractedEntities:
    """Structured entities extracted from text."""
    encounter_id: str
    raw_text: str
    cleaned_text: str
    
    # Extracted fields
    crop_name: Optional[str] = None
    crop_confidence: float = 0.0
    
    symptoms: List[str] = field(default_factory=list)
    affected_parts: List[str] = field(default_factory=list)
    
    duration_text: Optional[str] = None
    duration_days: Optional[int] = None
    
    spread_speed: Optional[str] = None
    urgency_level: str = "medium"
    
    treatments_tried: List[str] = field(default_factory=list)
    weather_conditions: List[str] = field(default_factory=list)
    
    # Confidence scores
    extraction_quality: float = 0.0


class TextNLU:
    """Natural Language Understanding for farmer symptom reports."""
    
    def __init__(self):
        """Initialize NLU processor."""
        self._build_reverse_lookups()
    
    def _build_reverse_lookups(self):
        """Build reverse lookup dictionaries for fast matching."""
        self.crop_lookup = {}
        for crop, synonyms in CROP_SYNONYMS.items():
            for syn in synonyms:
                self.crop_lookup[syn.lower()] = crop
        
        self.symptom_lookup = {}
        for symptom, keywords in SYMPTOM_PATTERNS.items():
            for kw in keywords:
                self.symptom_lookup[kw.lower()] = symptom
        
        self.part_lookup = {}
        for part, keywords in AFFECTED_PARTS.items():
            for kw in keywords:
                self.part_lookup[kw.lower()] = part
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep periods and commas
        text = re.sub(r'[^\w\s.,\'-]', '', text)
        
        # Fix common transcription errors
        replacements = {
            "gonna": "going to",
            "wanna": "want to",
            "kinda": "kind of",
            "sorta": "sort of",
            "dunno": "don't know",
            "lemme": "let me",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def extract_crop(self, text: str) -> Tuple[Optional[str], float]:
        """
        Extract crop name from text.
        
        Returns:
            Tuple of (crop_name, confidence)
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check each word
        for word in words:
            # Clean word
            word = re.sub(r'[^\w]', '', word)
            if word in self.crop_lookup:
                return self.crop_lookup[word], 0.95
        
        # Check for partial matches
        for crop, synonyms in CROP_SYNONYMS.items():
            for syn in synonyms:
                if syn in text_lower:
                    return crop, 0.85
        
        return None, 0.0
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptom mentions from text."""
        text_lower = text.lower()
        found_symptoms = set()
        
        # Check for keywords
        for keyword, symptom in self.symptom_lookup.items():
            if keyword in text_lower:
                found_symptoms.add(symptom)
        
        return list(found_symptoms)
    
    def extract_affected_parts(self, text: str) -> List[str]:
        """Extract mentioned plant parts."""
        text_lower = text.lower()
        found_parts = set()
        
        for keyword, part in self.part_lookup.items():
            if keyword in text_lower:
                found_parts.add(part)
        
        return list(found_parts)
    
    def extract_duration(self, text: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Extract duration information.
        
        Returns:
            Tuple of (duration_text, estimated_days)
        """
        text_lower = text.lower()
        
        for pattern, unit in DURATION_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                if unit == "1_day":
                    return "yesterday", 1
                elif unit == "0_days":
                    return "today", 0
                elif unit == "7_days":
                    return "last week", 7
                elif unit == "2-3_days":
                    return "few days", 2
                else:
                    try:
                        num = int(match.group(1))
                        if unit == "weeks":
                            return f"{num} weeks", num * 7
                        else:
                            return f"{num} days", num
                    except (ValueError, IndexError):
                        pass
        
        return None, None
    
    def extract_spread_speed(self, text: str) -> Optional[str]:
        """Extract spread speed information."""
        text_lower = text.lower()
        
        for speed, keywords in SPREAD_PATTERNS.items():
            for kw in keywords:
                if kw in text_lower:
                    return speed
        
        return None
    
    def extract_urgency(self, text: str) -> str:
        """Determine urgency level from text."""
        text_lower = text.lower()
        
        for level in ["high", "medium", "low"]:
            for keyword in URGENCY_KEYWORDS[level]:
                if keyword in text_lower:
                    return level
        
        return "medium"
    
    def extract_treatments(self, text: str) -> List[str]:
        """Extract mentioned treatments."""
        text_lower = text.lower()
        found_treatments = set()
        
        for treatment, keywords in TREATMENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    found_treatments.add(treatment)
        
        return list(found_treatments)
    
    def extract_weather(self, text: str) -> List[str]:
        """Extract weather condition mentions."""
        text_lower = text.lower()
        found_weather = set()
        
        for condition, keywords in WEATHER_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    found_weather.add(condition)
        
        return list(found_weather)
    
    def calculate_quality(self, entities: ExtractedEntities) -> float:
        """Calculate extraction quality score."""
        score = 0.0
        max_score = 0.0
        
        # Crop identified (important)
        max_score += 0.25
        if entities.crop_name:
            score += 0.25 * entities.crop_confidence
        
        # Symptoms found
        max_score += 0.25
        if entities.symptoms:
            score += min(0.25, len(entities.symptoms) * 0.08)
        
        # Duration mentioned
        max_score += 0.15
        if entities.duration_days is not None:
            score += 0.15
        
        # Affected parts mentioned
        max_score += 0.15
        if entities.affected_parts:
            score += min(0.15, len(entities.affected_parts) * 0.05)
        
        # Spread speed mentioned
        max_score += 0.10
        if entities.spread_speed:
            score += 0.10
        
        # Weather mentioned
        max_score += 0.10
        if entities.weather_conditions:
            score += 0.10
        
        return round(score / max_score if max_score > 0 else 0, 3)
    
    def extract(
        self,
        text: str,
        encounter_id: str = ""
    ) -> ExtractedEntities:
        """
        Extract all entities from text.
        
        Args:
            text: Input text (transcript or direct input)
            encounter_id: Optional encounter identifier
            
        Returns:
            ExtractedEntities with all extracted information
        """
        cleaned = self.clean_text(text)
        
        # Extract crop
        crop_name, crop_conf = self.extract_crop(cleaned)
        
        # Extract other entities
        symptoms = self.extract_symptoms(cleaned)
        parts = self.extract_affected_parts(cleaned)
        duration_text, duration_days = self.extract_duration(cleaned)
        spread = self.extract_spread_speed(cleaned)
        urgency = self.extract_urgency(cleaned)
        treatments = self.extract_treatments(cleaned)
        weather = self.extract_weather(cleaned)
        
        entities = ExtractedEntities(
            encounter_id=encounter_id,
            raw_text=text,
            cleaned_text=cleaned,
            crop_name=crop_name,
            crop_confidence=crop_conf,
            symptoms=symptoms,
            affected_parts=parts,
            duration_text=duration_text,
            duration_days=duration_days,
            spread_speed=spread,
            urgency_level=urgency,
            treatments_tried=treatments,
            weather_conditions=weather
        )
        
        entities.extraction_quality = self.calculate_quality(entities)
        
        return entities
    
    def save_extraction(
        self,
        entities: ExtractedEntities,
        output_path: str
    ) -> Path:
        """Save extracted entities to JSON."""
        output_path = Path(output_path)
        
        if output_path.is_dir():
            output_path = output_path / f"{entities.encounter_id}_entities.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = asdict(entities)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved entities: {output_path}")
        return output_path
    
    def process_transcript(
        self,
        transcript_path: str,
        output_dir: str
    ) -> Path:
        """
        Process a transcript JSON file.
        
        Args:
            transcript_path: Path to transcript JSON
            output_dir: Output directory for entities
            
        Returns:
            Path to saved entities file
        """
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        text = transcript.get('full_text', '')
        encounter_id = transcript.get('encounter_id', Path(transcript_path).stem)
        
        entities = self.extract(text, encounter_id)
        
        return self.save_extraction(entities, output_dir)
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str
    ) -> List[Path]:
        """Process all transcript files in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        transcript_files = list(input_path.glob("*_transcript.json"))
        logger.info(f"Found {len(transcript_files)} transcript files")
        
        outputs = []
        for tfile in transcript_files:
            try:
                out = self.process_transcript(str(tfile), str(output_path))
                outputs.append(out)
            except Exception as e:
                logger.error(f"Error processing {tfile}: {e}")
        
        logger.info(f"Processed {len(outputs)} transcripts")
        return outputs


def main():
    """CLI for text NLU."""
    parser = argparse.ArgumentParser(description="Extract symptoms from farmer reports")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input text, transcript JSON, or directory")
    parser.add_argument("--output", "-o", default="./entities", 
                       help="Output directory")
    parser.add_argument("--text", "-t", action="store_true",
                       help="Treat input as direct text instead of file")
    
    args = parser.parse_args()
    
    nlu = TextNLU()
    
    if args.text:
        # Direct text input
        entities = nlu.extract(args.input)
        print(json.dumps(asdict(entities), indent=2))
        
    else:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single transcript file
            out = nlu.process_transcript(str(input_path), args.output)
            print(f"Saved: {out}")
            
        elif input_path.is_dir():
            # Directory of transcripts
            outputs = nlu.process_directory(str(input_path), args.output)
            print(f"Processed {len(outputs)} files")
        
        else:
            print(f"Error: {args.input} not found")


if __name__ == "__main__":
    main()
