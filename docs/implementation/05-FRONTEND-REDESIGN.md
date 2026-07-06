# 05 — Frontend Redesign

Goal: a modern, fully responsive, **app-like** experience with a bottom button navbar, installable as a PWA, that consumes the new instant `/api/analyze` endpoint and renders every result state safely (no XSS).

---

## 5.1 Navigation: app-style bottom bar

Replace the top text nav with a **fixed bottom navigation bar** on mobile (the primary form factor) and a slim top header on desktop.

```
Mobile (default)                         Desktop (≥ 900px)
┌───────────────────────┐               ┌──────────────────────────────┐
│  🌿 AgriDoctor    👤   │  top header   │ 🌿 AgriDoctor  Home Diagnose  │
│                       │               │                History  👤   │
│     [ page content ]  │               ├──────────────────────────────┤
│                       │               │        [ page content ]       │
│                       │               │                              │
├───────────────────────┤               └──────────────────────────────┘
│ 🏠      📷       📋    │  bottom nav   (bottom bar hidden; top nav used)
│ Home  Diagnose History│
└───────────────────────┘
```

Markup (add to `index.html`, replace `.nav-menu`):

```html
<nav class="bottom-nav" id="bottom-nav" aria-label="Primary">
  <button class="bottom-nav__item active" data-page="home">
    <span class="bottom-nav__icon">🏠</span><span class="bottom-nav__label">Home</span>
  </button>
  <button class="bottom-nav__item bottom-nav__cta" data-page="diagnose">
    <span class="bottom-nav__icon">📷</span><span class="bottom-nav__label">Diagnose</span>
  </button>
  <button class="bottom-nav__item" data-page="history">
    <span class="bottom-nav__icon">📋</span><span class="bottom-nav__label">History</span>
  </button>
</nav>
```

Key CSS (mobile-first; hide top nav links on small screens, hide bottom bar on desktop):

```css
.bottom-nav{
  position:fixed; inset:auto 0 0 0; z-index:50;
  display:flex; justify-content:space-around;
  background:var(--surface); border-top:1px solid var(--border);
  padding:.4rem .5rem calc(.4rem + env(safe-area-inset-bottom));
  box-shadow:0 -4px 20px rgba(0,0,0,.06);
}
.bottom-nav__item{ flex:1; display:flex; flex-direction:column; align-items:center;
  gap:2px; border:0; background:none; color:var(--text-muted);
  font-size:.72rem; padding:.35rem; border-radius:12px; }
.bottom-nav__item.active{ color:var(--brand); }
.bottom-nav__cta .bottom-nav__icon{ /* raised center action button */
  transform:translateY(-10px); background:var(--brand); color:#fff;
  width:52px; height:52px; border-radius:50%; display:grid; place-items:center;
  font-size:1.4rem; box-shadow:0 6px 16px rgba(16,185,129,.45); }
main{ padding-bottom:84px; }               /* clear the bar */
@media (min-width:900px){
  .bottom-nav{ display:none; }
  .nav-menu{ display:flex; }               /* desktop top nav */
}
@media (max-width:899px){ .nav-menu{ display:none; } }
```

The center **Diagnose** action is a raised circular button (classic mobile-app pattern) — the primary CTA.

## 5.2 The instant diagnose flow (simplified)

Because the AI now auto-detects the crop, the flow gets shorter and image-first:

```
Step 1 (optional): "Which crop? (or skip — we'll detect it)"  → sets crop_hint
Step 2: Capture / upload leaf photo            (required)
Step 3: Optional voice note + onset/spread/notes
   [ Analyze Now ] → single POST /api/analyze → spinner → Results
```

Keep the existing 4-step stepper visuals, but:
- Add a **"Skip — auto-detect"** option to Step 1.
- Step 2 becomes the emphasized step (image is the signal).
- On **Analyze**, call the new client method (5.4) and render by `result.kind` (5.5).

## 5.3 Fix the silent-fallback bug (audit + 02 §2.7)

Current `runAnalysis` falls back to `simulateAnalysis()` on **any** error, showing fake results when the backend is down. **Remove `simulateAnalysis` and the fallback entirely.** On error, show an honest error card with a Retry button. No fabricated diagnoses, ever.

## 5.4 API client (`frontend/js/api.js`)

Add one method; it sends multipart and returns the result:

```js
async analyze({ image, audio, cropHint, onsetDays, spread, notes }) {
  const fd = new FormData();
  fd.append('image', image);
  if (audio)     fd.append('audio', audio);
  if (cropHint)  fd.append('crop_hint', cropHint);
  if (onsetDays != null) fd.append('onset_days', onsetDays);
  if (spread)    fd.append('spread', spread);
  if (notes)     fd.append('notes', notes);

  const headers = {};
  if (this.token) headers['Authorization'] = `Bearer ${this.token}`;

  const res = await fetch(`${API_BASE}/analyze`, { method:'POST', headers, body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Analysis failed (HTTP ${res.status})`);
  }
  return res.json();   // AnalysisResult + case_id
}
```

## 5.5 Rendering result states — **XSS-safe** (fixes audit critical)

The current code injects user `notes` and model text into `innerHTML` with only a naive markdown regex → stored/reflected XSS. **Rule: never interpolate untrusted strings into `innerHTML`.** Build DOM nodes and use `textContent`, or escape first.

```js
function esc(s){ const d=document.createElement('div'); d.textContent=s??''; return d.innerHTML; }

// Minimal, SAFE markdown (escape FIRST, then apply bold/italic on the escaped text)
function mdSafe(s){
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,'<em>$1</em>');
}
```

Then branch on `kind`:

```js
function renderResult(r){
  switch(r.kind){
    case 'not_a_leaf':       return renderRejection('🌱','Not a plant leaf', r.message);
    case 'unsupported_crop': return renderUnsupported(r);     // names detected_crop + supported list
    case 'low_confidence':   return renderRetake(r.message);  // "retake a clear photo"
    case 'healthy':          return renderHealthy(r);         // green, reassuring
    case 'diagnosis':        return renderDiagnosis(r);       // full report (image + advice)
    default:                 return renderError('Unexpected response');
  }
}
```

Each `render*` builds nodes with `textContent`/`mdSafe`, never raw `innerHTML` of user/model data. The rich diagnosis card (image, confidence/severity/urgency meters, action plan, prevention, escalation, safety note) reuses the existing visual design — just rendered safely and driven by the real `AnalysisResult`.

### The unsupported-crop card (the requested behavior)

```js
function renderUnsupported(r){
  const detected = r.detected_crop ? esc(r.detected_crop) : 'this plant';
  return card('🚫', 'Crop not supported yet',
    `This looks like <strong>${detected}</strong>. AgriDoctor currently supports
     Tomato 🍅, Potato 🥔, Rice 🌾, Maize 🌽, Chili 🌶️, and Cucumber 🥒.
     Please upload a leaf from one of these crops.`);
}
```

### Crop-mismatch nudge

If `r.matches_hint === false`, show a small banner on a valid diagnosis: *"You selected Tomato, but this looks like a Potato leaf — we diagnosed it as Potato."*

## 5.6 Responsiveness checklist

- Mobile-first CSS; test at **360px, 390px, 768px, 1024px, 1440px**.
- Fluid type with `clamp()`; touch targets ≥ 44px; the record button large and thumb-reachable.
- Results layout: single column on mobile, two-column (image left / advice right) on ≥ 900px.
- Respect `env(safe-area-inset-*)` for notched phones.
- `prefers-color-scheme` dark mode via CSS variables.
- Images: `max-width:100%`, lazy-load history thumbnails.

## 5.7 PWA (install "as an app")

Add so users can install to the home screen and get an app shell:

1. `frontend/manifest.webmanifest`:
```json
{
  "name":"AgriDoctor AI","short_name":"AgriDoctor","start_url":"/",
  "display":"standalone","background_color":"#0b0f0d","theme_color":"#10b981",
  "icons":[{"src":"/icons/icon-192.png","sizes":"192x192","type":"image/png"},
           {"src":"/icons/icon-512.png","sizes":"512x512","type":"image/png"}]
}
```
2. Link it in `index.html`: `<link rel="manifest" href="/manifest.webmanifest">`.
3. `frontend/sw.js`: cache the app shell (html/css/js/icons) for offline load of the UI. **Do not** cache `/api/*` responses. Register it in `app.js`.

> A service worker requires HTTPS (already terminated at the VPS edge per [agridoctor.cloud.conf](../../agridoctor.cloud.conf)) or `localhost`.

## 5.8 Loading & error UX

- Analyze button → inline spinner + disabled state; overlay copy "Reading the leaf…" then "Consulting knowledge base…".
- Network/503 → error card with **Retry**.
- 429 → "You're going fast! Please wait a moment and try again."
- Blurry/dark (`low_confidence`) → retake guidance with an example of a good photo.

## 5.9 File structure after redesign

```
frontend/
├── index.html              # bottom-nav markup, manifest link, PWA meta
├── manifest.webmanifest
├── sw.js
├── icons/ (192, 512, maskable)
├── css/styles.css          # + bottom-nav, result-state cards, dark mode, responsive
└── js/
    ├── api.js              # + analyze()
    ├── ui.js               # NEW: esc/mdSafe + render* result builders (XSS-safe)
    └── app.js              # flow/state; calls api.analyze(); no simulateAnalysis()
```
