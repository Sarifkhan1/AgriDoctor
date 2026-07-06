# 06 — Security Hardening (Phase 0 — do this first)

Every item here maps to a finding from the code audit. **Phase 0 is a prerequisite** because live credentials are currently committed to the repository — no feature work should proceed until they are rotated and removed.

---

## 6.1 🔴 Rotate and purge committed credentials (CRITICAL)

Two secrets are in git today:

1. **Root SSH password** in [`ssh_run.py`](../../ssh_run.py): `root@69.62.78.148:2222` / `64X1i7jkQg1@RO--y)L7`.
2. **The same string reused as the default JWT `SECRET_KEY`** in [`docker-compose.yml`](../../docker-compose.yml) — anyone with the repo can forge auth tokens for the running server.

Actions (in order):
- [ ] **Rotate the VPS root password now** (or better, disable password login and switch to SSH keys).
- [ ] **Generate a fresh JWT secret**: `python -c "import secrets; print(secrets.token_urlsafe(48))"` and set it only via env.
- [ ] **Delete `ssh_run.py`** from the repo. Deployment auth belongs in SSH keys / CI secrets, never in code.
- [ ] **Scrub git history** so the secrets aren't recoverable from old commits (`git filter-repo --path ssh_run.py --invert-paths`, or BFG). Since history currently has a single commit, the cheapest safe option may be to re-init history. Force-push after coordinating.
- [ ] Treat both leaked strings as permanently burned — never reuse them anywhere.

## 6.2 🔴 Mandatory SECRET_KEY, no insecure default (CRITICAL)

Today [`main.py`](../../backend/main.py) falls back to `"agridoctor-secret-key-change-in-production"`. Remove the fallback; the app must refuse to start without a real key.

- Handled by `config.py` (03 §3.2): `secret_key: str` with no default → `Settings()` raises at import if missing.
- Remove the `SECRET_KEY=${SECRET_KEY:-...}` default from `docker-compose.yml`; require it from the host environment / secrets manager.

## 6.3 🔴 Fix stored/reflected XSS (CRITICAL)

`displayResults` injects user `notes` and model text into `innerHTML`, and `parseMarkdown` runs regex on *unescaped* input → script injection.

- [ ] Escape before render (`esc`) and only apply bolding on already-escaped text (`mdSafe`) — see [05 §5.5](./05-FRONTEND-REDESIGN.md).
- [ ] Build result DOM with `textContent`/nodes, not raw string concatenation of untrusted data.
- [ ] Add a Content-Security-Policy header (below) as defense-in-depth.

## 6.4 🔴 Restrict CORS (CRITICAL)

`allow_origins=["*"]` **with** `allow_credentials=True` is invalid/unsafe.

- [ ] Drive origins from `ALLOWED_ORIGINS` env (03 §3.2 / 04 §4.7). List the real frontend origins only.
- [ ] Limit methods/headers to what's used (`GET`, `POST`; `Authorization`, `Content-Type`).

## 6.5 🟠 Server-side upload validation (MAJOR)

Uploads are currently trusted (`content_type` is client-declared; filename suffix used raw). Add `backend/services/validation.py`:

```python
import magic
from io import BytesIO
from PIL import Image
from fastapi import HTTPException
from ..config import settings

ALLOWED_IMAGE = {"image/jpeg","image/png","image/webp"}
ALLOWED_AUDIO = {"audio/webm","audio/wav","audio/x-wav","audio/mpeg","audio/mp4","audio/ogg"}

def validate_image(data: bytes, filename: str, declared_ct: str) -> str:
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"Image exceeds {settings.max_upload_mb} MB limit.")
    sniffed = magic.from_buffer(data, mime=True)          # magic bytes, not the header
    if sniffed not in ALLOWED_IMAGE:
        raise HTTPException(400, "Please upload a JPG, PNG, or WebP image.")
    try:
        with Image.open(BytesIO(data)) as im:
            im.verify()
            w, h = im.size
    except Exception:
        raise HTTPException(400, "That file isn't a valid image.")
    if w < 64 or h < 64:
        raise HTTPException(400, "Image is too small to analyze.")
    return sniffed

def validate_audio(audio: dict):
    if len(audio["bytes"]) > 15 * 1024 * 1024:
        raise HTTPException(400, "Voice note is too large.")
    if magic.from_buffer(audio["bytes"], mime=True) not in ALLOWED_AUDIO:
        raise HTTPException(400, "Unsupported audio format.")
```

- Never build server paths from the client filename; store under `{uuid}{safe_ext}` (existing pattern is OK, keep the uuid).

## 6.6 🟠 Auth robustness (MAJOR)

- [ ] Password minimum → 8 chars + basic strength/deny-common-passwords check.
- [ ] Rate limit `/api/auth/login` (5/min/IP) and `/api/analyze` (10/min/IP) — see [04 §4.5](./04-BACKEND-API.md).
- [ ] Use `datetime.now(timezone.utc)` for token expiry (deprecation fix).
- [ ] Consider short access-token TTL + refresh later; not required for MVP.

## 6.7 🟠 Real `.gitignore` (MAJOR)

The repo's `.gitignore` is **empty (0 bytes)** — the SQLite DB (user emails + password hashes), logs, and uploads are one `git add .` from being published.

Create it with at least:

```gitignore
# Secrets / env
.env
.env.*
*.pem
*.key

# Data & artifacts (contain PII / large binaries)
data/agridoctor.db
data/uploads/
models/
*.log

# Python
__pycache__/
*.pyc
.venv/
venv/

# OS / editor
.DS_Store
.idea/
.vscode/
```

- [ ] Confirm `data/agridoctor.db` is **not already tracked** (`git ls-files | grep agridoctor.db`). It currently isn't — keep it that way.

## 6.8 Security headers & CSP

Nginx already sets `X-Frame-Options`, `X-Content-Type-Options`, etc. Add a CSP (tighten as the frontend stabilizes). Because current `index.html` loads Google Fonts, allow that host or self-host the font:

```
add_header Content-Security-Policy "default-src 'self';
  img-src 'self' data: blob:;
  media-src 'self' blob:;
  style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;
  font-src 'self' https://fonts.gstatic.com;
  script-src 'self';
  connect-src 'self';" always;
```

> Self-hosting the Inter font lets you drop the two font hosts and use a stricter CSP. Recommended.

## 6.9 Don't leak secrets in logs

- Never log `GROQ_API_KEY`, `SECRET_KEY`, raw passwords, or Authorization headers.
- The AI `raw_response` may be stored for calibration — it contains no user credentials, but treat the DB as sensitive (contains emails + hashes) and keep it out of git (6.7).

## 6.10 Phase 0 acceptance checklist

- [ ] VPS root password rotated; `ssh_run.py` deleted; history scrubbed; force-pushed.
- [ ] Fresh `SECRET_KEY` and `GROQ_API_KEY` only in `.env` / host env; app fails to boot without them.
- [ ] `.gitignore` populated; DB/logs/uploads/.env confirmed untracked.
- [ ] CORS restricted; CSP + security headers live.
- [ ] XSS fix merged (frontend renders untrusted text safely).
- [ ] Upload validation rejects oversized / non-image / spoofed-type files.
