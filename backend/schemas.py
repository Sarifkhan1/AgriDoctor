"""
AgriDoctor AI - Request/response models for auth and cases.
(AI analysis models live in backend/ai/schemas.py.)
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    password: str = Field(..., min_length=8, max_length=128)


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
