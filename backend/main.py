"""
AgriDoctor AI - FastAPI Backend
Main application with auth, cases, and inference endpoints.
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import json
import os
import shutil

# Security
import hashlib
import secrets
from jose import JWTError, jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Database
import sqlite3
from contextlib import contextmanager

# App configuration
APP_NAME = "AgriDoctor AI"
VERSION = "1.0.0"
SECRET_KEY = os.getenv("SECRET_KEY", "agridoctor-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "agridoctor.db"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Database Setup
# ============================================================================

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db():
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Initialize database tables."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                category TEXT CHECK (category IN ('crop', 'livestock')),
                crop_name TEXT,
                animal_type TEXT,
                status TEXT DEFAULT 'created',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            CREATE TABLE IF NOT EXISTS media (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                type TEXT CHECK (type IN ('image', 'speech', 'audio', 'video')),
                uri TEXT NOT NULL,
                content_type TEXT,
                duration_sec REAL,
                quality_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            );
            
            CREATE TABLE IF NOT EXISTS metadata (
                case_id TEXT PRIMARY KEY,
                onset_days INTEGER,
                spread_speed TEXT,
                weather_json TEXT,
                treatments_json TEXT,
                notes_text TEXT,
                language TEXT,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            );
            
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                model_version TEXT,
                primary_label TEXT,
                secondary_labels_json TEXT,
                severity_score REAL,
                urgency_level TEXT,
                confidence REAL,
                advice_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            );
            
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                heatmap_uri TEXT,
                top_text_spans_json TEXT,
                audio_events_json TEXT,
                explainability_notes TEXT,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            );
            
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                status TEXT DEFAULT 'queued',
                progress INTEGER DEFAULT 0,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_cases_user ON cases(user_id);
            CREATE INDEX IF NOT EXISTS idx_media_case ON media(case_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_case ON predictions(case_id);
        """)


# ============================================================================
# Security
# ============================================================================

# Password Hashing using PBKDF2 (Python 3.13 compatible)
security = HTTPBearer()

HASH_ITERATIONS = 100000
HASH_ALGORITHM = "sha256"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a stored hash."""
    try:
        salt, stored_hash = hashed_password.split(":")
        computed_hash = hashlib.pbkdf2_hmac(
            HASH_ALGORITHM,
            plain_password.encode("utf-8"),
            bytes.fromhex(salt),
            HASH_ITERATIONS
        ).hex()
        return secrets.compare_digest(stored_hash, computed_hash)
    except (ValueError, AttributeError):
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using PBKDF2."""
    salt = secrets.token_bytes(16)
    hash_bytes = hashlib.pbkdf2_hmac(
        HASH_ALGORITHM,
        password.encode("utf-8"),
        salt,
        HASH_ITERATIONS
    )
    return f"{salt.hex()}:{hash_bytes.hex()}"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Validate JWT token and return user."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        with get_db() as conn:
            user = conn.execute(
                "SELECT id, email, full_name, role FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            
            return dict(user)
            
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ============================================================================
# Pydantic Models
# ============================================================================

class UserRegister(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    full_name: Optional[str] = None


class CaseCreate(BaseModel):
    category: str = Field(..., pattern="^(crop|livestock)$")
    crop_name: Optional[str] = None
    animal_type: Optional[str] = None


class CaseMetadata(BaseModel):
    onset_days: Optional[int] = None
    spread_speed: Optional[str] = None
    weather: Optional[Dict[str, Any]] = None
    treatments: Optional[List[str]] = None
    notes: Optional[str] = None
    language: Optional[str] = "en"


class CaseResponse(BaseModel):
    id: str
    user_id: str
    category: str
    crop_name: Optional[str]
    status: str
    created_at: str


class PredictionResponse(BaseModel):
    primary_label: str
    confidence: float
    severity_score: float
    urgency_level: str
    advice: Dict[str, Any]
    secondary_labels: List[str]


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description="Multimodal agricultural health assistant API"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    init_db()


# ============================================================================
# Auth Endpoints
# ============================================================================

@app.post("/api/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(user: UserRegister):
    """Register a new user."""
    with get_db() as conn:
        # Check if user exists
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (user.email,)
        ).fetchone()
        
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        user_id = str(uuid.uuid4())
        password_hash = get_password_hash(user.password)
        
        conn.execute(
            "INSERT INTO users (id, email, full_name, password_hash) VALUES (?, ?, ?, ?)",
            (user_id, user.email, user.full_name, password_hash)
        )
        
        token = create_access_token({"sub": user_id})
        
        return TokenResponse(
            access_token=token,
            user_id=user_id,
            email=user.email,
            full_name=user.full_name
        )


@app.post("/api/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(user: UserLogin):
    """Login user."""
    with get_db() as conn:
        db_user = conn.execute(
            "SELECT id, email, full_name, password_hash FROM users WHERE email = ?",
            (user.email,)
        ).fetchone()
        
        if not db_user or not verify_password(user.password, db_user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = create_access_token({"sub": db_user["id"]})
        
        return TokenResponse(
            access_token=token,
            user_id=db_user["id"],
            email=db_user["email"],
            full_name=db_user["full_name"]
        )


@app.get("/api/auth/me", tags=["Auth"])
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info."""
    return current_user


# ============================================================================
# Cases Endpoints
# ============================================================================

@app.post("/api/cases", response_model=CaseResponse, tags=["Cases"])
async def create_case(
    case: CaseCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new case."""
    case_id = str(uuid.uuid4())
    
    with get_db() as conn:
        conn.execute(
            """INSERT INTO cases (id, user_id, category, crop_name, animal_type, status)
               VALUES (?, ?, ?, ?, ?, 'created')""",
            (case_id, current_user["id"], case.category, case.crop_name, case.animal_type)
        )
        
        case_data = conn.execute(
            "SELECT * FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
    
    return CaseResponse(**dict(case_data))


@app.get("/api/cases", tags=["Cases"])
async def list_cases(
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """List user's cases."""
    with get_db() as conn:
        cases = conn.execute(
            """SELECT * FROM cases 
               WHERE user_id = ? 
               ORDER BY created_at DESC 
               LIMIT ? OFFSET ?""",
            (current_user["id"], limit, offset)
        ).fetchall()
    
    return [dict(c) for c in cases]


@app.get("/api/cases/{case_id}", tags=["Cases"])
async def get_case(
    case_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get case details."""
    with get_db() as conn:
        case = conn.execute(
            "SELECT * FROM cases WHERE id = ? AND user_id = ?",
            (case_id, current_user["id"])
        ).fetchone()
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get media
        media = conn.execute(
            "SELECT * FROM media WHERE case_id = ?", (case_id,)
        ).fetchall()
        
        # Get metadata
        metadata = conn.execute(
            "SELECT * FROM metadata WHERE case_id = ?", (case_id,)
        ).fetchone()
        
        # Get predictions
        predictions = conn.execute(
            "SELECT * FROM predictions WHERE case_id = ? ORDER BY created_at DESC LIMIT 1",
            (case_id,)
        ).fetchone()
    
    result = dict(case)
    result["media"] = [dict(m) for m in media]
    result["metadata"] = dict(metadata) if metadata else None
    result["prediction"] = dict(predictions) if predictions else None
    
    return result


@app.post("/api/cases/{case_id}/media/image", tags=["Cases"])
async def upload_image(
    case_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload image for case."""
    # Verify case ownership
    with get_db() as conn:
        case = conn.execute(
            "SELECT id FROM cases WHERE id = ? AND user_id = ?",
            (case_id, current_user["id"])
        ).fetchone()
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    
    # Save file
    media_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".jpg"
    file_path = UPLOADS_DIR / "images" / f"{media_id}{file_ext}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Store in database
    with get_db() as conn:
        conn.execute(
            """INSERT INTO media (id, case_id, type, uri, content_type)
               VALUES (?, ?, 'image', ?, ?)""",
            (media_id, case_id, str(file_path), file.content_type)
        )
        
        # Update case status
        conn.execute(
            "UPDATE cases SET status = 'uploaded', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (case_id,)
        )
    
    return {"media_id": media_id, "uri": str(file_path)}


@app.post("/api/cases/{case_id}/media/speech", tags=["Cases"])
async def upload_speech(
    case_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload voice note for case."""
    with get_db() as conn:
        case = conn.execute(
            "SELECT id FROM cases WHERE id = ? AND user_id = ?",
            (case_id, current_user["id"])
        ).fetchone()
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    
    # Save file
    media_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".wav"
    file_path = UPLOADS_DIR / "speech" / f"{media_id}{file_ext}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    with get_db() as conn:
        conn.execute(
            """INSERT INTO media (id, case_id, type, uri, content_type)
               VALUES (?, ?, 'speech', ?, ?)""",
            (media_id, case_id, str(file_path), file.content_type)
        )
    
    return {"media_id": media_id, "uri": str(file_path)}


@app.post("/api/cases/{case_id}/metadata", tags=["Cases"])
async def update_metadata(
    case_id: str,
    metadata: CaseMetadata,
    current_user: dict = Depends(get_current_user)
):
    """Update case metadata."""
    with get_db() as conn:
        case = conn.execute(
            "SELECT id FROM cases WHERE id = ? AND user_id = ?",
            (case_id, current_user["id"])
        ).fetchone()
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        conn.execute(
            """INSERT OR REPLACE INTO metadata 
               (case_id, onset_days, spread_speed, weather_json, treatments_json, notes_text, language)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                case_id,
                metadata.onset_days,
                metadata.spread_speed,
                json.dumps(metadata.weather) if metadata.weather else None,
                json.dumps(metadata.treatments) if metadata.treatments else None,
                metadata.notes,
                metadata.language
            )
        )
    
    return {"status": "updated"}


# ============================================================================
# Inference Endpoints
# ============================================================================

@app.post("/api/cases/{case_id}/run", tags=["Inference"])
async def run_inference(
    case_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Start inference job for case."""
    with get_db() as conn:
        case = conn.execute(
            "SELECT * FROM cases WHERE id = ? AND user_id = ?",
            (case_id, current_user["id"])
        ).fetchone()
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Create job
        job_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO jobs (id, case_id, status) VALUES (?, ?, 'queued')",
            (job_id, case_id)
        )
        
        # Update case status
        conn.execute(
            "UPDATE cases SET status = 'running', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (case_id,)
        )
    
    # Run inference in background
    background_tasks.add_task(process_case, case_id, job_id)
    
    return {"job_id": job_id, "status": "queued"}


async def process_case(case_id: str, job_id: str):
    """Background task to process a case."""
    try:
        with get_db() as conn:
            # Update job status
            conn.execute(
                "UPDATE jobs SET status = 'running', progress = 10 WHERE id = ?",
                (job_id,)
            )
            
            # Get case data
            case = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            media = conn.execute(
                "SELECT * FROM media WHERE case_id = ? AND type = 'image'", (case_id,)
            ).fetchall()
            metadata = conn.execute(
                "SELECT * FROM metadata WHERE case_id = ?", (case_id,)
            ).fetchone()
        
        # Simulate inference (replace with actual model inference)
        import time
        time.sleep(2)  # Simulate processing time
        
        # Generate mock prediction
        prediction = generate_mock_prediction(case, media, metadata)
        
        with get_db() as conn:
            # Store prediction
            pred_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO predictions 
                   (id, case_id, model_version, primary_label, secondary_labels_json,
                    severity_score, urgency_level, confidence, advice_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pred_id, case_id, "v1.0.0",
                    prediction["primary_label"],
                    json.dumps(prediction["secondary_labels"]),
                    prediction["severity_score"],
                    prediction["urgency_level"],
                    prediction["confidence"],
                    json.dumps(prediction["advice"])
                )
            )
            
            # Update job
            conn.execute(
                "UPDATE jobs SET status = 'done', progress = 100 WHERE id = ?",
                (job_id,)
            )
            
            # Update case
            conn.execute(
                "UPDATE cases SET status = 'done', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (case_id,)
            )
            
    except Exception as e:
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = ? WHERE id = ?",
                (str(e), job_id)
            )
            conn.execute(
                "UPDATE cases SET status = 'failed', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (case_id,)
            )


def generate_mock_prediction(case, media, metadata) -> dict:
    """Generate a mock prediction for demo purposes."""
    crop = case["crop_name"] or "tomato"
    
    # Detailed disease knowledge base
    knowledge_base = {
        "tomato": {
            "primary": "TOM_EARLY_BLIGHT",
            "secondary": ["TOM_SEPTORIA", "TOM_BACTERIAL_SPOT"],
            "confidence": 0.89,
            "severity": 0.45,
            "urgency": "medium",
            "title": "Early Blight (Alternaria solani)",
            "summary": "The analysis detects **Early Blight**, a common fungal disease caused by *Alternaria solani*. It typically starts on lower leaves as circular brown spots with concentric rings ('bullseye' pattern). If left untreated, it causes yellowing of leaves (chlorosis), defoliation, and sunscald on the fruit due to loss of leaf cover.",
            "what_to_do_now": [
                "**Step 1: Pruning & Sanitation**: Immediately prune off all infected lower leaves using sterilized shears. Dip tools in 10% bleach solution between cuts. Dispose of debris far from the garden (do not compost).",
                "**Step 2: Fungicide Application**: Apply a copper-based fungicide or products containing Mancozeb or Chlorothalonil. For organic options, use Bacillus subtilis biofungicide. Repeat every 7–10 days.",
                "**Step 3: Modify Watering**: Stop overhead irrigation immediately. Water only at the base of the plant to keep foliage dry, as the fungus requires moisture to spread."
            ],
            "prevention": [
                "**Crop Rotation**: Do not plant tomatoes, potatoes, or eggplants in the same soil for at least 3 years.",
                "**Mulching**: maintain a 2-3 inch layer of organic mulch (straw/leaves) at the base to prevent soil spores from splashing onto lower leaves.",
                "**Stake & Air Flow**: Stake plants and prune suckers to improve air circulation, reducing humidity around the leaves."
            ],
            "when_to_get_help": [
                "If lesions appear on the stems (collar rot) which can kill the plant.",
                "If more than 50% of the foliage is yellowing despite treatment."
            ]
        },
        "potato": {
            "primary": "POT_LATE_BLIGHT",
            "secondary": ["POT_EARLY_BLIGHT", "POT_VIRUS_Y"],
            "confidence": 0.92,
            "severity": 0.85,
            "urgency": "high",
            "title": "Late Blight (Phytophthora infestans)",
            "summary": "This is a **critical detection** of Late Blight, the devastating disease responsible for the Irish Potato Famine. It appears as large, dark/oily lesions on leaves and stems, often with white fungal growth on the underside in humid weather. It can destroy an entire crop in days.",
            "what_to_do_now": [
                "**Step 1: Immediate Removal**: Verify if only a few plants are affected. If so, pull them out entirely and bag them on site to prevent spore release. Destroy by burning.",
                "**Step 2: Chemical Control**: For remaining healthy plants, apply a protective fungicide immediately (e.g., Chlorothalonil, Mancozeb). Late blight is difficult to cure once established.",
                "**Step 3: Desiccation**: If the crop is near harvest, kill the vines (leaves/stems) immediately to prevent the fungus from travelling down the stem and infecting the tubers.",
                "**Step 4: Harvest Care**: Wait 2-3 weeks after vine death before harvesting to ensure spore death."
            ],
            "prevention": [
                "**Certified Seed**: Only plant certified disease-free seed potatoes.",
                "**Eliminate Culls**: Destroy any volunteer potatoes or piles of culled potatoes near the field as they harbor the pathogen over winter.",
                "**Resistant Varieties**: Choose resistant varieties like 'Sarpo Mira' or 'Defender'."
            ],
            "when_to_get_help": [
                "**IMMEDIATELY**: Contact your local agricultural extension office as Late Blight outbreaks often require community warnings."
            ]
        },
        "rice": {
            "primary": "RICE_BLAST",
            "secondary": ["RICE_BROWN_SPOT", "RICE_SHEATH_BLIGHT"],
            "confidence": 0.85,
            "severity": 0.60,
            "urgency": "medium",
            "title": "Rice Blast (Magnaporthe oryzae)",
            "summary": "The symptoms indicate **Rice Blast**, one of the most destructive rice diseases. It forms spindle-shaped lesions with whitish/gray centers and reddish-brown borders. It can affect leaves, nodes, and the neck of the panicle (Neck Blast), which causes severe yield loss.",
            "what_to_do_now": [
                "**Step 1: Nitrogen Management**: Stop applying nitrogen fertilizer immediately. High nitrogen levels exacerbate the disease.",
                "**Step 2: Water Management**: Maintain a continuous deep flood (water level) in the paddy, as the disease is worse in dry soil conditions.",
                "**Step 3: Chemical Treatment**: Apply systemic fungicides such as Tricyclazole, Isoprothiolane, or Azoxystrobin immediately if the blast is at the booting or heading stage."
            ],
            "prevention": [
                "**Resistant Varieties**: This is the most effective method. Use locally recommended resistant strains.",
                "**Planting Time**: Adjust planting dates to avoid heading stages during periods of high humidity/rainfall.",
                "**Seed Treatment**: Treat seeds with Pseudomonas fluorescens before sowing."
            ],
            "when_to_get_help": [
                "If 'Neck Blast' is observed (lesions at the base of the panicle) as this leads to total grain failure.",
                "If the field has a history of severe outbreaks."
            ]
        },
        "maize": {
            "primary": "MAIZE_RUST",
            "secondary": ["MAIZE_NLB", "MAIZE_GREY_LEAF_SPOT"],
            "confidence": 0.78,
            "severity": 0.35,
            "urgency": "low",
            "title": "Common Rust (Puccinia sorghi)",
            "summary": "Detected **Common Rust**, characterized by small, reddish-brown pustules on both upper and lower leaf surfaces. While spectacular in appearance, it often does not require chemical control unless the infection happens very early in the season.",
            "what_to_do_now": [
                "**Step 1: Field Monitoring**: Check if the pustules are covering more than 30-50% of the leaf surface. If less, the economic impact may be minimal.",
                "**Step 2: Fungicide**: In sweet corn or seed corn (high value), apply foliar fungicides (Strobilurins or Triazoles) if pustules appear before tasseling.",
                "**Step 3: Cultural**: Remove alternate hosts (Oxalis species weeds) from around the field."
            ],
            "prevention": [
                "**Resistant Hybrids**: Plant resistant maize hybrids.",
                "**Planting Date**: Early planting helps the crop mature before rust spore loads become high in the atmosphere.",
                "**Crop Rotation**: Effective but less critical than for soil-borne diseases as rust blows in via wind."
            ],
            "when_to_get_help": [
                "If pustules turn black (teliospore stage) and infection is severe on ear leaves.",
                "If you suspect 'Southern Rust' (pustules only on top of leaves) which is more aggressive."
            ]
        },
        "chili": {
            "primary": "CHILI_ANTHRAC",
            "secondary": ["CHILI_LEAF_CURL", "CHILI_WILT"],
            "confidence": 0.81,
            "severity": 0.55,
            "urgency": "medium",
            "title": "Anthracnose (Colletotrichum spp.)",
            "summary": "The fruit/leaf lesions suggest **Anthracnose**. On fruits, it causes sunken, circular, excessively wet spots with concentric rings of salmon-colored spores. It makes the fruit unmarketable.",
            "what_to_do_now": [
                "**Step 1: Harvest & Destruction**: Pick ALL infected fruits immediately and bury them. They are the primary source of new infection.",
                "**Step 2: Spraying**: Apply fungicides like Mancozeb, Copper Oxychloride, or Azoxystrobin. Ensure thorough coverage of the fruits, not just leaves.",
                "**Step 3: Seed Treatment**: If saving seeds, treat them with hot water (52°C for 30 mins) or Thiram to kill seed-borne infection."
            ],
            "prevention": [
                "**Wider Spacing**: Increase space between plants to ensure rapid drying after rain.",
                "**Weed Control**: Keep the field weed-free as weeds can host the fungus.",
                "**Drainage**: Ensure fields drain well to avoid high humidity microclimates."
            ],
            "when_to_get_help": [
                "If 'Dieback' symptoms occur (branches drying from tip downwards)."
            ]
        },
        "cucumber": {
            "primary": "CUC_POWDERY",
            "secondary": ["CUC_DOWNY", "CUC_MOSAIC"],
            "confidence": 0.88,
            "severity": 0.40,
            "urgency": "medium",
            "title": "Powdery Mildew (Podosphaera xanthii)",
            "summary": "Identified **Powdery Mildew**, appearing as a white, dusty flour-like coating on leaves and stems. Unlike many fungi, it thrives in dry conditions with high humidity, even without rain.",
            "what_to_do_now": [
                "**Step 1: Bio-Control**: Spray a solution of Baking Soda (1%) or Potassium Bicarbonate. Milk solution (1 part milk : 9 parts water) is also effective.",
                "**Step 2: Chemical Control**: If severe, use sulfur-based fungicides or Mycobutanil. Do not apply sulfur if temperatures are above 30°C (85°F) to avoid burn.",
                "**Step 3: Pruning**: Remove lower leaves that are heavily coated to improve airflow."
            ],
            "prevention": [
                "**Resistant Cultivars**: Use PM-resistant cucumber varieties.",
                "**Air Circulation**: Trellis cucumbers to maximize airflow and light exposure.",
                "**Weeding**: Remove wild cucurbits and weeds nearby."
            ],
            "when_to_get_help": [
                "If you see yellow, angular spots bounded by leaf veins; this might be Downy Mildew instead, which requires different treatment."
            ]
        }
    }
    
    # Get crop data or default
    data = knowledge_base.get(crop.lower(), knowledge_base["tomato"])
    
    return {
        "primary_label": data["primary"],
        "secondary_labels": data["secondary"],
        "confidence": data["confidence"],
        "severity_score": data["severity"],
        "urgency_level": data["urgency"],
        "advice": {
            "summary": data["summary"],
            "what_to_do_now": data["what_to_do_now"],
            "prevention": data["prevention"],
            "when_to_get_help": data["when_to_get_help"],
            "safety_note": "This diagnosis is generated by AgriDoctor AI. While highly accurate, environmental factors can mimic diseases. Please confirm with a local agronomist before applying chemical treatments."
        }
    }


@app.get("/api/jobs/{job_id}", tags=["Inference"])
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get job status."""
    with get_db() as conn:
        job = conn.execute(
            """SELECT j.* FROM jobs j
               JOIN cases c ON j.case_id = c.id
               WHERE j.id = ? AND c.user_id = ?""",
            (job_id, current_user["id"])
        ).fetchone()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
    
    return dict(job)


@app.get("/api/cases/{case_id}/result", tags=["Inference"])
async def get_result(
    case_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get prediction result for case."""
    with get_db() as conn:
        case = conn.execute(
            "SELECT * FROM cases WHERE id = ? AND user_id = ?",
            (case_id, current_user["id"])
        ).fetchone()
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        prediction = conn.execute(
            "SELECT * FROM predictions WHERE case_id = ? ORDER BY created_at DESC LIMIT 1",
            (case_id,)
        ).fetchone()
        
        if not prediction:
            raise HTTPException(status_code=404, detail="No prediction available")
        
        evidence = conn.execute(
            "SELECT * FROM evidence WHERE case_id = ?", (case_id,)
        ).fetchone()
    
    result = dict(prediction)
    result["secondary_labels"] = json.loads(result.get("secondary_labels_json") or "[]")
    result["advice"] = json.loads(result.get("advice_json") or "{}")
    result["evidence"] = dict(evidence) if evidence else None
    
    return result


# ============================================================================
# Models Endpoints
# ============================================================================

@app.get("/api/models", tags=["Models"])
async def list_models(current_user: dict = Depends(get_current_user)):
    """List available models."""
    return [
        {
            "id": "crop-vit-v1",
            "name": "Crop Disease ViT",
            "version": "1.0.0",
            "type": "image",
            "crops": ["tomato", "potato", "rice", "maize", "chili", "cucumber"]
        },
        {
            "id": "multimodal-v1",
            "name": "Multimodal Fusion",
            "version": "1.0.0",
            "type": "multimodal",
            "crops": ["tomato", "potato", "rice", "maize", "chili", "cucumber"]
        }
    ]


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": VERSION}


@app.get("/", tags=["System"])
async def root():
    """Root endpoint."""
    return {
        "name": APP_NAME,
        "version": VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
