AgriDoctor: A Vision–Language Assistant for Crop Disease Diagnosis
Mohammad Sarif Khan
2238572
Link to git project repo: https://github.com/Sarifkhan1/AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor

## Introduction

AgriDoctor is an interactive web application that helps farmers diagnose crop-leaf
diseases from a photograph. A user uploads (or captures) an image of an affected
leaf, optionally adds a short spoken or typed description of the symptoms, and
receives a structured diagnosis: the identified crop, the most likely disease, a
confidence and severity estimate, and an actionable treatment and prevention plan.

The motivation is concrete and evidence-based. The Food and Agriculture
Organization of the United Nations (FAO) estimates that **between 20% and 40% of
global crop production is lost every year to pests and diseases**, with plant
diseases alone costing the world economy around **US$220 billion** annually [1],
[2]. These losses fall hardest on smallholder farmers in regions where trained
agronomists are scarce and a single misdiagnosis can mean a wasted pesticide
purchase or a lost harvest. AgriDoctor aims to put a "first-opinion" diagnostic
tool in the hands of any farmer with a phone.

This report describes what was **actually built and deployed**, is explicit about
what remains a **prototype or future work**, and situates the project against
existing products and research in the field. The design principle throughout was
honesty about capability: the system is engineered so that when it does *not* know
(a blurry image, an unsupported crop, a photo that is not a leaf at all), it says
so rather than inventing a confident but false answer.

**Scope actually delivered (this submission):**

- **Crop-leaf disease diagnosis for six crops**: Tomato, Potato, Rice, Maize,
  Chili, and Cucumber, covering ~40 disease/pest/health labels across fungal,
  bacterial, viral, pest, and nutrient categories (see `data/taxonomy.json`).
- **A live vision–language diagnostic engine** (not a mock) that analyses the
  uploaded image on every request and generates the diagnosis and advice at
  run time.
- **Out-of-scope rejection**: non-leaf images, unsupported crops, and
  low-quality images are detected and refused rather than force-fit to a disease.
- **Optional voice input** transcribed to text and fused into the diagnosis.
- **A responsive Progressive Web App** front end and a secured FastAPI back end
  with optional user accounts for saving case history.

**Explicitly NOT delivered (and therefore not claimed as working):** livestock
diagnosis, and a custom trained neural network serving predictions. Both are
discussed honestly in the *Critical reflection* and *Future work* sections.

## Background and related work

**The problem domain.** Automated plant-disease diagnosis from leaf images is a
well-established research area. The public **PlantVillage** dataset [3] — ~54,000
labelled leaf images across 14 crops and 26 diseases — catalysed a wave of deep
learning work, most of it based on Convolutional Neural Networks and, more
recently, **Vision Transformers (ViT)** [4], which treat an image as a sequence of
patches and have become a strong baseline for fine-grained classification. A known
and important limitation of models trained on PlantVillage is that they generalise
poorly to real "in-the-field" photos, because the dataset images are captured on
plain backgrounds under controlled lighting [3]. This directly informed a core
design decision in AgriDoctor (discussed below).

**Existing products (market analysis).** AgriDoctor is not being built in a
vacuum, and it is useful to be honest about where it sits relative to mature tools:

| Product | Approach | Scale / strengths | Relevance to AgriDoctor |
| :-- | :-- | :-- | :-- |
| **Plantix** [5], [6] | Deep-learning image classifier + agronomist community, mobile-first | The most-downloaded ag-tech app: 10M+ downloads across 150+ countries, 100M+ crop questions answered; covers ~30–60 crops and ~800 symptoms; multilingual | The direct benchmark: mature, data-rich, and specialised. AgriDoctor cannot match its dataset, but can match its *interaction model* and honesty-of-uncertainty. |
| **PlantNet** | CNN species identification from a very large citizen-science image base | Excellent at *species* ID, not disease diagnosis | Shows the value of visual species identification, which AgriDoctor performs as a gating step. |
| **Google Lens / general VLMs** | General-purpose vision–language models | Broad, zero training, conversational | Closest in *technique* to AgriDoctor's engine, but not agriculture-constrained and will happily hallucinate a crop disease for any image. |
| **Gamaya, Taranis** | Drone / hyperspectral, enterprise remote sensing | Field-scale analytics for large farms | Different market (enterprise, not the individual smallholder AgriDoctor targets). |

Two lessons shaped this project. First, the incumbents' advantage is **data**, not
cleverness — a solo student project cannot out-collect Plantix. Second, general
vision–language models are powerful but **unsafe by default** for this task: they
will confidently diagnose a disease on a photo of a car. AgriDoctor's contribution
is therefore not a new model, but a **safety-constrained application layer** around
a strong general model, plus an honest UX. This is the gap the project targets.

## Implementation of the interactive application

The deployed system is a decoupled client–server application.

**Front end (`frontend/`).** A Progressive Web App built with HTML5, CSS3, and
vanilla JavaScript so it runs on low-end phones without a build step. It uses the
**MediaDevices API** for camera capture and the **MediaRecorder API** for voice
notes, packages them into a `FormData` request, and renders the returned diagnosis.
All model/user text is rendered through an escaping layer (`frontend/js/ui.js`) to
prevent cross-site scripting. Five distinct result states are handled explicitly:
*diagnosis*, *healthy*, *unsupported crop*, *not a leaf*, and *low confidence*.

**Back end (`backend/`).** A **FastAPI** service exposing a single synchronous
endpoint, `POST /api/analyze`, which accepts the image (and optional audio and
context) as multipart form data and returns the full diagnosis in one response —
there is no fake job queue or simulated latency. Accounts (JWT with PBKDF2 password
hashing) are optional and only gate saved case history; anonymous diagnosis works
without login. Uploads are validated by byte-signature sniffing and re-decoding
with Pillow (not by trusting the file extension), and an in-process rate limiter
protects the endpoint.

**The diagnostic engine (`backend/ai/`).** This is the core of the system and the
part most worth explaining precisely.

1. **Vision–language analysis.** The validated image is sent, together with a
   carefully constrained system prompt, to a hosted multimodal model
   (**Qwen3.6-27B**, a vision-capable model served on Groq for low latency). The
   prompt forces the model to reason *only from what is visible*, to choose a
   disease label *only* from the project's controlled taxonomy, and to return a
   single strict JSON object. Because the model is a *reasoning* model, reasoning
   output is suppressed (`reasoning_effort="none"`) so the response is valid JSON.
2. **Voice fusion.** If a voice note is supplied it is transcribed with
   **Whisper-large-v3** and the transcript is added to the analysis context, so the
   farmer's spoken description ("spots started three days ago after heavy rain")
   informs the diagnosis and advice.
3. **A server-side safety-invariant layer** (`analyzer.py::_enforce_invariants`).
   This is what separates AgriDoctor from simply calling a chatbot. The model's
   output is never trusted blindly. The server *independently* enforces that:
   a diagnosis is only allowed for a supported crop with a label that exists in the
   taxonomy; any diagnosis claimed for a non-leaf or an unsupported crop is
   **stripped and converted into a polite rejection**; confidence and severity are
   clamped to valid ranges; and a `*_HEALTHY` label is reclassified as "healthy",
   not a disease. A single model hallucination therefore cannot leak a fabricated
   diagnosis to the user.
4. **Advice generation.** For a confirmed diagnosis, a structured treatment plan
   (immediate actions, long-term prevention, and escalation guidance, preferring
   cultural controls before chemical ones) is generated at run time and tailored to
   the crop, disease, and any farmer-supplied context.

**A key, deliberate design decision — and an honest correction of an earlier
mistake.** An early version of this project shipped with *hard-coded* predictions:
regardless of the image, it returned a fixed result per crop button. That is not
AI, and it is removed. The current system was rebuilt around a real model. During
that rebuild a subtle but important bug was found and fixed: when the farmer's
selected crop was passed to the vision model as text, the model **deferred to that
guess** and would, for example, label a *grape* leaf as "tomato early blight." The
crop button is now used only as a post-hoc "you selected X but this looks like Y"
hint; **species identification is performed purely from the image**. After this
fix, a grape leaf submitted under a "tomato" selection is correctly rejected as an
unsupported crop. This is exactly the failure mode (every image → the same
diagnosis) that must not occur, and it is now verified not to.

## Additional experimentation and design exploration

Beyond the deployed application, the repository contains experimentation with
custom trained models. Their status is stated precisely below — none of the
deployed diagnoses come from them; the hosted vision–language model handles every
real request.

- **A real image-classifier training run (actually performed).** Using
  `scripts/train_real.py`, a compact MobileNetV3-small model was fine-tuned (transfer
  learning, Apple-Silicon MPS) on 17 disease/healthy classes drawn from a public
  PlantVillage-derived dataset and mapped to AgriDoctor's taxonomy across four crops
  (Tomato, Potato, Pepper, Maize) — 4,080 training and 1,020 validation images. It
  reached a **genuinely measured 99.3% validation accuracy and 0.9931 macro-F1** on
  the held-out split (full per-class scores in `data/models/metrics.json`).
  **Honest caveat:** PlantVillage images have plain backgrounds and controlled
  lighting, on which even small models score very high [3]; this is *not* evidence
  of field accuracy, which would be markedly lower on messy real-world photos. The
  value of the run is a real, reproducible pipeline and honest numbers — not a claim
  that it beats the deployed engine.
- **A multimodal fusion architecture** (`src/models/train_multimodal.py`): a ViT
  image encoder and a DistilBERT [7] text encoder joined by a cross-modal attention
  module, with disease and severity heads. This code is implemented but **was not
  trained** (no labelled multimodal image+text dataset was assembled); therefore no
  fusion accuracy numbers are claimed. It remains a documented design, not a result.
- **A serving path for the trained classifier** (`backend/ai/cnn_predictor.py`) is
  wired into the analyzer: when a checkpoint is present it runs first and defers to
  the hosted model on low confidence. It is available as an optional local fast-path
  for the crop classes it covers; the hosted model remains the default so all 12
  subjects (including livestock, absent from the training data) are covered.
- **A Streamlit annotation tool** (`tools/annotator_app.py`) was built to add the
  severity and image-quality labels that public datasets like PlantVillage lack —
  the enabling step for the fusion model above, had data collection continued.

The honest lesson from this exploration is *why the deployed engine is a hosted
vision–language model rather than the custom network*: with no labelled in-the-wild
dataset and no training compute, a from-scratch model could not have been made
accurate or safe in the project timeframe, and a model trained only on
plain-background PlantVillage images would have failed on real farmer photos [3].
Reframing the contribution as a **safety layer over a strong general model** was
the change that made a genuinely working product possible.

## Critical reflection

**What worked well.** The synchronous, single-endpoint design made the app feel
instant and simplified the front end. The server-side invariant layer proved to be
the most valuable idea: it is what lets the app refuse a photo of a car instead of
diagnosing it, which is the single biggest correctness and trust win. Testing with
real leaf images (tomato early blight, healthy tomato, an unsupported grape leaf,
and non-plant photos) confirmed the system now returns *different, appropriate*
results per image and rejects out-of-scope inputs — the behaviour a diagnostic tool
must have.

**Shortcomings.** (1) Diagnostic accuracy is bounded by a general-purpose model
rather than a model trained on a curated crop-disease dataset; it is a credible
first opinion, not a validated clinical instrument, and its accuracy has not been
measured against a labelled test set (doing so is the top future priority). (2) The
custom multimodal model remains untrained. (3) The free hosting tier imposes a
tokens-per-minute limit, so under load the app must ask the user to retry; the
back end now detects this and returns a clear "service busy" message rather than a
generic error. (4) Livestock diagnosis, an early ambition, is **not implemented**
and is listed only as future scope.

**Ethical considerations.** A diagnostic tool that suggests pesticides carries real
risk of harm from misdiagnosis. Mitigations are built in: low-confidence and
out-of-scope results are surfaced honestly rather than hidden; every diagnosis
carries a visible disclaimer that AI advice does not replace a qualified agronomist;
treatment advice prefers cultural and organic controls before naming chemical
options with safety caveats; and the system is deliberately biased toward *not*
diagnosing when uncertain. Following the ACM Code of Ethics' principle of avoiding
harm [8], refusing to answer is treated as a valid and often preferable outcome.

## Future work

- Assemble a labelled, in-the-wild crop-disease dataset (using the annotation tool)
  and train and **measure** the multimodal model, reporting real accuracy/F1 against
  a held-out test set.
- Add the livestock diagnostic scope as a genuinely implemented second phase.
- Multilingual advice output for non-English-speaking farmers.
- On-device or quantised inference to remove the hosting rate limit and work offline.

## LLM disclaimer

Large Language Models were used as an assistive tool during development for:
(1) generating boilerplate (Pydantic models, CSS scaffolding); (2) explaining
library errors during debugging; and (3) drafting and copy-editing prose in this
report. All generated code and text were reviewed, tested, and edited by the author.
Separately — and distinct from the above — a hosted vision–language model
(Qwen3.6-27B) is an intentional *runtime component* of the deployed product, as
described in the *Implementation* section; this is a design choice, not undisclosed
assistance.

## Bibliography

[1] Food and Agriculture Organization of the United Nations (2021) *New standards to
curb the global spread of plant pests and diseases*. Available at:
https://www.fao.org/newsroom/detail/New-standards-to-curb-the-global-spread-of-plant-pests-and-diseases/en
(Accessed: 18 July 2026).

[2] Food and Agriculture Organization of the United Nations, *Understanding the
context — Pest and Pesticide Management*. Available at:
https://www.fao.org/pest-and-pesticide-management/about/understanding-the-context/en/
(Accessed: 18 July 2026).

[3] Hughes, D. and Salathé, M. (2015) 'An open access repository of images on plant
health to enable the development of mobile disease diagnostics', *arXiv preprint
arXiv:1511.08060*. Available at: https://arxiv.org/abs/1511.08060.

[4] Dosovitskiy, A. et al. (2020) 'An Image is Worth 16x16 Words: Transformers for
Image Recognition at Scale', *International Conference on Learning Representations
(ICLR)*. Available at: https://arxiv.org/abs/2010.11929.

[5] Plantix (2025) *Plantix — the #1 free app for crop diagnosis and treatments*.
Available at: https://plantix.net/en/ (Accessed: 18 July 2026).

[6] GSMA (2020) *Detecting and managing crop pests and diseases with AI: Insights
from Plantix*. Available at:
https://www.gsma.com/solutions-and-impact/connectivity-for-good/mobile-for-development/programme/agritech/detecting-and-managing-crop-pests-and-diseases-with-ai-insights-from-plantix/
(Accessed: 18 July 2026).

[7] Sanh, V. et al. (2019) 'DistilBERT, a distilled version of BERT: smaller,
faster, cheaper and lighter', *arXiv preprint arXiv:1910.01108*.

[8] Association for Computing Machinery (2018) *ACM Code of Ethics and Professional
Conduct*. Available at: https://www.acm.org/code-of-ethics (Accessed: 18 July 2026).
