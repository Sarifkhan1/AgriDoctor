"""Auth endpoints: register, login, me."""

import uuid

from fastapi import APIRouter, Depends, HTTPException

from ..db import get_db
from ..schemas import TokenResponse, UserLogin, UserRegister
from ..security import (
    create_access_token,
    get_current_user,
    get_password_hash,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["Auth"])


@router.post("/register", response_model=TokenResponse)
async def register(user: UserRegister):
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (user.email,)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        user_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO users (id, email, full_name, password_hash) VALUES (?, ?, ?, ?)",
            (user_id, user.email, user.full_name, get_password_hash(user.password)),
        )

    token = create_access_token({"sub": user_id})
    return TokenResponse(
        access_token=token, user_id=user_id, email=user.email, full_name=user.full_name
    )


@router.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    with get_db() as conn:
        db_user = conn.execute(
            "SELECT id, email, full_name, password_hash FROM users WHERE email = ?",
            (user.email,),
        ).fetchone()

    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": db_user["id"]})
    return TokenResponse(
        access_token=token,
        user_id=db_user["id"],
        email=db_user["email"],
        full_name=db_user["full_name"],
    )


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user
