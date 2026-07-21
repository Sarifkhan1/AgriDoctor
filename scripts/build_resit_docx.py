import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_border(cell, **kwargs):
    """
    Set cell borders
    kwargs: top, bottom, left, right, insideH, insideV
    value is a dictionary like: {'sz': 12, 'val': 'single', 'color': 'FF0000', 'space': '0'}
    """
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)

    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = 'w:{}'.format(edge)
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, val in edge_data.items():
                element.set(qn('w:{}'.format(key)), str(val))

def create_document():
    doc = Document()

    # Define Colors
    COLOR_PRIMARY = RGBColor(46, 117, 89)      # Forest Green
    COLOR_SECONDARY = RGBColor(33, 37, 41)    # Off Black
    COLOR_MUTED = RGBColor(108, 117, 125)     # Gray
    COLOR_CODE = RGBColor(220, 53, 69)        # Crimson for code highlight

    # Set standard margins (1 inch on all sides)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Style definitions
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Arial'
    font.size = Pt(11)
    font.color.rgb = COLOR_SECONDARY

    # ==========================================
    # TITLE PAGE
    # ==========================================
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("AGRIDOCTOR AI\n")
    title_run.font.name = 'Arial'
    title_run.font.size = Pt(28)
    title_run.font.bold = True
    title_run.font.color.rgb = COLOR_PRIMARY

    subtitle_run = title_p.add_run("A Safety-Constrained Vision-Language and Multimodal Framework for Mobile Crop-Leaf Pathology and Livestock Health Diagnostics\n\n\n")
    subtitle_run.font.name = 'Arial'
    subtitle_run.font.size = Pt(14)
    subtitle_run.font.color.rgb = COLOR_MUTED

    details_p = doc.add_paragraph()
    details_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    details_p.paragraph_format.line_spacing = 1.3
    
    details_text = (
        "Module: AI-4-Creativity Project (Final Major Project)\n"
        "Student Name: Mohammad Sarif Khan\n"
        "Student ID: 2238572\n"
        "Submission Category: Resit Submission Report\n"
        "Evaluation Note: Capped at 40% Record / Evaluated at Full Quality Standard\n\n"
        "Academic Year: 2026\n"
    )
    run_details = details_p.add_run(details_text)
    run_details.font.size = Pt(11)
    run_details.font.color.rgb = COLOR_SECONDARY

    doc.add_page_break()

    # ==========================================
    # HELPER FUNCTIONS FOR HEADINGS
    # ==========================================
    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(18)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        return p

    def add_paragraph(text, bold_prefix="", space_after=6):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            prefix_run = p.add_run(bold_prefix)
            prefix_run.font.bold = True
        p.add_run(text)
        return p

    def add_code_block(code_text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.line_spacing = 1.0
        
        # Add background shading via XML manipulation
        pPr = p._p.get_or_add_pPr()
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8F9FA"/>')
        pPr.append(shading)
        
        run = p.add_run(code_text)
        run.font.name = 'Consolas'
        run.font.size = Pt(9.5)
        run.font.color.rgb = COLOR_SECONDARY
        return p

    # ==========================================
    # SECTION 1: EXECUTIVE SUMMARY
    # ==========================================
    add_heading_1("1. EXECUTIVE SUMMARY / ABSTRACT")
    
    add_paragraph(
        "AgriDoctor AI is an interactive, full-stack, mobile-first web application designed to act as a first-opinion diagnostic assistant for smallholder farmers dealing with crop pathologies and visible livestock health anomalies. The system allows farmers to capture leaf or animal skin images, supply optional vocal descriptions of symptoms, and receive immediate, structured botanical or veterinary recommendations."
    )
    
    add_paragraph(
        "Traditional deep neural network classifiers suffer from severe out-of-distribution generalizability issues when deployed in complex field environments. Conversely, unconstrained Vision-Language Models (VLMs) suffer from visual hallucinations, frequently mapping irrelevant noise to confident, incorrect agricultural advice. To bridge this gap, AgriDoctor implements a server-side safety-invariant layer that evaluates VLM responses against a rigid taxonomy, converting out-of-scope or low-confidence outputs into polite rejections."
    )
    
    add_paragraph(
        "This portfolio documents the full research, development, and validation process of AgriDoctor. It details two local machine learning pipelines (a MobileNetV3 classifier achieving a 99.31% validation F1-score on clinical datasets, and a ViT+DistilBERT cross-modal attention fusion prototype showing an 8.3-point macro-F1 gain via speech integration). Finally, it outlines the production deployment of the safety-gated system on a secure VPS server at https://agridoctor.cloud."
    )

    # ==========================================
    # SECTION 2: DEVELOPMENT PROCESS & RESEARCH METHODOLOGY
    # ==========================================
    add_heading_1("2. DEVELOPMENT PROCESS & RESEARCH METHODOLOGY (FMP PART 1)")
    
    add_heading_2("2.1 Introduction & The Socio-Economic Problem Space")
    add_paragraph(
        "Smallholder farming is the economic backbone of developing agricultural regions, yet it is continuously threatened by crop blights, viral pathotypes, and livestock epidemics. The Food and Agriculture Organization of the United Nations (FAO) estimates that plant pests and diseases account for annual losses of 20% to 40% of global crop yields, costing the global agricultural economy roughly US$220 billion [7]. In many developing rural regions, the scarcity of certified agronomists and veterinarians means that farmers must travel long distances to seek consultation or buy expensive, unverified chemicals based on guesswork. A single misdiagnosis can lead to crop devastation, wasted pesticide expenses, and soil contamination."
    )
    add_paragraph(
        "AgriDoctor AI was conceived to address this diagnostic barrier by placing a high-quality, safety-gated diagnostic tool directly into the hands of farmers via low-cost mobile devices."
    )

    add_heading_2("2.2 Theoretical Research: Computer Vision in Agriculture & The Generalization Gap")
    add_paragraph(
        "An extensive review of computer vision literature in agricultural pathology reveals that deep convolutional neural networks (CNNs) and Vision Transformers (ViTs) can achieve near-perfect classification scores in laboratory settings. The standard reference database is the public PlantVillage dataset [5], which consists of approximately 54,000 images of leaves taken against uniform gray backgrounds under controlled studio lighting."
    )
    add_paragraph(
        "However, literature emphasizes the 'generalization gap' or 'domain shift' associated with models trained on PlantVillage. When a CNN trained on plain-background leaves is presented with a real field photo containing cluttered backgrounds (such as soil, weeds, shadows, or varying camera angles), its classification performance drops significantly. This degradation is due to the network learning the clinical context of the training set rather than the actual biological symptoms of the leaf."
    )
    add_paragraph(
        "Therefore, a primary engineering requirement for AgriDoctor was to design a system that does not rely blindly on raw classification outputs. Instead, it must combine broad zero-shot multi-modal capability with strict boundary checks to filter out out-of-distribution noise."
    )

    add_heading_2("2.3 Market Analysis: Competitive Critique of Existing Solutions")
    add_paragraph(
        "To understand where AgriDoctor fits, we conducted a competitive review of existing agricultural technologies:"
    )
    
    add_paragraph(
        " Approach: Proprietary deep learning image classifier + social community.\n"
        "Scale & Scope: 10M+ downloads, 60+ crops, 800+ symptoms, multilingual.\n"
        "Strengths: Largest curated database of real field images, excellent commercial scale.\n"
        "Limitation/Relevance: Highly specialized and capital-intensive. AgriDoctor does not seek to compete with Plantix's massive dataset, but adopts its clean interaction model while introducing a robust, open safety-invariant layer.",
        bold_prefix="* Plantix [9]:"
    )
    
    add_paragraph(
        " Approach: Crowd-sourced species classification using CNNs.\n"
        "Scale & Scope: Global wild flora identification.\n"
        "Strengths: Highly accurate at botanical species and genus identification.\n"
        "Limitation/Relevance: Focuses on botanical species identification rather than specific disease pathologies. It highlights the value of species gating to ensure the leaf is actually a supported crop before attempting disease classification.",
        bold_prefix="* PlantNet:"
    )

    add_paragraph(
        " Approach: Large-scale vision-language models.\n"
        "Scale & Scope: General multi-modal reasoning.\n"
        "Strengths: Zero-shot performance on diverse visual data, context synthesis, conversational advice.\n"
        "Limitation/Relevance: No safety constraints. These models will confidently diagnose a disease on an image of a leaf-shaped car or a dog. AgriDoctor wraps these engines in a deterministic server-side safety filter.",
        bold_prefix="* General VLMs (e.g., Qwen-Vision, LLaVA, GPT-4o):"
    )

    add_heading_2("2.4 Core Evolution: From Hard-Coded Monolith to Safety-Gated Runtime")
    add_paragraph(
        "The development of AgriDoctor went through three primary design iterations:"
    )
    add_paragraph(
        "An early layout that returned static, hard-coded advice based purely on the selected crop button. If the user clicked \"Tomato,\" it returned early blight regardless of the uploaded image. This was an honest limitation that failed to represent real AI.",
        bold_prefix="1. Version 1.0 (Mock Prototype): "
    )
    add_paragraph(
        "The backend was rewritten to pipe images to a hosted vision-language model. While this added real inference, testing revealed that the model was overly cooperative: if a user selected \"Tomato\" but uploaded a picture of a grape leaf, the VLM forced a diagnosis of tomato early blight, exhibiting severe confirmation bias.",
        bold_prefix="2. Version 2.0 (Unconstrained VLM): "
    )
    add_paragraph(
        "The current version implements a dual-stage architecture. The frontend crop buttons are used only as hints. The backend VLM is queried without locking the crop type, and the VLM's JSON response is filtered through a deterministic server-side validator. This validator checks the predicted crop and disease label against a strict local taxonomy, rejecting out-of-scope images and correcting hallucinated combinations.",
        bold_prefix="3. Version 3.0 (Safety-Gated Multimodal Framework - Deployed): "
    )

    # ==========================================
    # SECTION 3: SYSTEM ARCHITECTURE & TECHNICAL DESCRIPTION
    # ==========================================
    doc.add_page_break()
    add_heading_1("3. SYSTEM ARCHITECTURE & TECHNICAL DESCRIPTION (FMP PART 2)")
    
    add_heading_2("3.1 Decoupled Client-Server Stack & Mobile PWA Features")
    add_paragraph(
        "AgriDoctor is structured as a decoupled client-server architecture:"
    )
    add_paragraph(
        "Buildless vanilla HTML5, CSS3, and JavaScript, ensuring rapid rendering and compatibility with low-end mobile devices in rural areas.",
        bold_prefix="* Frontend (Progressive Web App): "
    )
    add_paragraph(
        "Uses the browser's native MediaDevices API to capture high-resolution leaf images directly from the phone's back camera. It integrates the MediaRecorder API to capture audio symptom notes, packaging them into a standard WebM container.",
        bold_prefix="  - Media API Integration: "
    )
    add_paragraph(
        "Built with dynamic CSS variables to support light and dark modes, utilizing a fluid grid system optimized for small mobile screens.",
        bold_prefix="  - Responsive Styling: "
    )
    add_paragraph(
        "FastAPI running on Uvicorn. Exposes a unified POST /api/analyze endpoint.",
        bold_prefix="* Backend Service (FastAPI & SQLite): "
    )
    add_paragraph(
        "Uses JWT tokens with PBKDF2 password hashing for optional user accounts. If a user is authenticated, their case history is written to an SQLite database; otherwise, the endpoint allows anonymous, rate-limited public diagnostic requests.",
        bold_prefix="  - State Management: "
    )

    add_heading_2("3.2 End-to-End Processing Mechanism & Flow Diagram")
    add_paragraph(
        "When a farmer submits a diagnosis request, the data travels through a highly optimized dual-stage hybrid routing pipeline. The system coordinates local edge-inference models and hosted cloud APIs to achieve zero-latency results for common crops, while preserving broad multi-modal diagnostic reasoning for advanced or rare cases:"
    )
    add_paragraph("The farmer captures/uploads a leaf or skin image and optionally records a voice note. The PWA client packages these binary payloads along with metadata (user selection, onset days, spread notes) into a standard multipart/form-data request, transmitting it via an encrypted HTTPS POST call to the backend API.", bold_prefix="1. Request Submission (Client): ")
    add_paragraph("The FastAPI server validates the file upload headers and byte structures. Python's Pillow library re-decodes and re-saves the image. This sanitizes the file against potential steganographic malware or exploit vectors before any model inference begins.", bold_prefix="2. Input Verification & Sanitization (Backend): ")
    add_paragraph("If a WebM audio stream is attached, it is routed to the Whisper-large-v3-turbo model. Whisper is a state-of-the-art automatic speech recognition (ASR) system based on an encoder-decoder Transformer architecture. It processes the raw audio waveform, outputs textual symptom transcripts ('dark spots on leaves', 'spreading fast'), and fuses this text context into the analysis payload.", bold_prefix="3. Asynchronous Speech Recognition & Transcription (ASR): ")
    add_paragraph("The sanitized image and the text notes/ASR transcripts are combined into a system prompt. The system queries Qwen3.6-27B (or a fallback like meta-llama/llama-4-scout-17b). This open-weights vision-language model is hosted on Groq's LPU (Language Processing Unit) architecture. LPUs use a static scheduling chip configuration that avoids typical GPU memory-bus bottlenecks, allowing the VLM to return structured JSON responses within milliseconds. We suppress reasoning tokens (reasoning_effort=\"none\") to avoid extra latency.", bold_prefix="4. Prompt Engineering & Vision-Language Model (VLM) Inference: ")
    add_paragraph("The VLM's raw output is never returned directly to the client. The backend passes it to a deterministic validation function. The server checks the detected subject and disease label against a strict local taxonomy, maps healthy states, clamps numeric fields (confidence, severity), and translates advice strings into the user's chosen language.", bold_prefix="5. Invariant Safety Layer Gating (Server-side): ")
    add_paragraph("The validated, formatted JSON payload (consisting of the diagnosis, confidence, severity, and structured cultural, organic, and prevention steps) is returned to the frontend. The JavaScript rendering engine builds the dynamic result card and injects the advice safely into the DOM.", bold_prefix="6. UI Rendering & Advice Delivery (Client): ")

    add_heading_2("3.3 Asynchronous API Verification & Image Spoofing Countermeasures")
    add_paragraph(
        "To protect the system from denial-of-service attempts and model manipulation, the backend implements security and rate-limiting patterns:"
    )
    add_paragraph(
        "To prevent buffer-overflow exploits or hidden shellcodes embedded inside image metadata, the Pillow library strips all EXIF tags and re-saves the pixel buffers. If the re-decoding fails, the request is terminated immediately with an HTTP 400 Bad Request.",
        bold_prefix="* Metadata Stripping & Re-decoding: "
    )
    add_paragraph(
        "Groq's hosted vision APIs operate on rate limits. To protect the user experience from transient limits (HTTP 429), the GroqProvider class implements an exponential backoff retry mechanism (using custom parsing of the 'retry-after' header). If the API limits are fully exhausted, the backend catches the error and returns a friendly 503 service message rather than failing with a raw stack trace.",
        bold_prefix="* Rate Limit Backoff & Catching: "
    )

    add_heading_2("3.4 The Safety-Invariant Gating Layer: Logical & Mathematical Formulation")
    add_paragraph(
        "The safety-invariant layer is a deterministic validation system implemented on the server."
    )
    add_paragraph(
        "Let S be the set of supported subjects:\n"
        "  S = { Tomato, Potato, Rice, Maize, Chili, Cucumber, Pepper, Eggplant, Cattle, Goat, Sheep, Poultry }\n"
        "Let L(s) be the set of valid taxonomic labels for a given subject s in S.\n"
        "Let R be the structured output dictionary generated by the VLM, where R_kind is the type of result, R_subject is the subject, R_label is the condition code, and R_conf and R_sev represent confidence and severity."
    )
    
    add_code_block(
        "# Server-side safety invariant verification loop\n"
        "def enforce_invariants(R, user_selected_subject=None):\n"
        "    # Step 1: Handle immediate rejections from VLM\n"
        "    if R.kind in [\"not_a_leaf\", \"unsupported_crop\", \"low_confidence\"]:\n"
        "        return rejection_response(R.kind, \"The uploaded image is out of scope.\")\n"
        "\n"
        "    # Step 2: Validate subject against the supported set\n"
        "    detected_subject = R.subject.strip().title()\n"
        "    if detected_subject not in S:\n"
        "        return rejection_response(\"unsupported_crop\", f\"Subject '{detected_subject}' is not supported.\")\n"
        "\n"
        "    # Step 3: Validate label against the subject's taxonomy\n"
        "    predicted_label = R.label.strip().upper()\n"
        "    if predicted_label not in L(detected_subject):\n"
        "        return rejection_response(\"low_confidence\", \"Diagnosis label mismatch.\")\n"
        "\n"
        "    # Step 4: Prevent user-selection bias overrides\n"
        "    if user_selected_subject and user_selected_subject.title() != detected_subject:\n"
        "        R.warning = f\"Note: Selected {user_selected_subject}, but detected {detected_subject}.\"\n"
        "\n"
        "    # Step 5: Normalize and clamp numerical ranges\n"
        "    R.confidence = max(0.0, min(1.0, float(R.conf)))\n"
        "    R.severity_score = max(0.0, min(1.0, float(R.sev)))\n"
        "\n"
        "    # Step 6: Map healthy taxonomy codes to healthy kind\n"
        "    if predicted_label.endswith(\"_HEALTHY\"):\n"
        "        R.kind = \"healthy\"\n"
        "\n"
        "    return R"
    )

    # ==========================================
    # SECTION 4: EMPIRICAL MACHINE LEARNING EXPERIMENTS
    # ==========================================
    doc.add_page_break()
    add_heading_1("4. EMPIRICAL MACHINE LEARNING EXPERIMENTS & PROTOTYPES")
    
    add_heading_2("4.1 Experiment A: Fine-Tuning a MobileNetV3 Crop-Leaf Classifier")
    add_paragraph(
        "To explore the viability of a locally hosted classifier, an offline training run was executed using the script scripts/train_real.py."
    )
    add_paragraph("Pretrained MobileNetV3-small, chosen for its low parameter count and fast mobile execution.", bold_prefix="* Backbone Architecture: ")
    add_paragraph("Mapped from minhhungg/plant-disease-dataset (17 classes, 4 crops: Tomato, Potato, Pepper, Maize). Mapped 80% training (4,080 images) and 20% validation (1,020 images).", bold_prefix="* Dataset: ")
    add_paragraph("Input resolution: 224x224. Augmentations: RandomHorizontalFlip, RandomRotation(15), ColorJitter. AdamW optimizer (learning rate 1e-4, weight decay 1e-2), batch size 32, 8 epochs on Apple Silicon MPS.", bold_prefix="* Training Pipeline: ")
    add_paragraph("Validation Accuracy: 99.31%, Macro F1-score: 0.9931. Lowest Class F1: 0.966 (Maize Gray Leaf Spot). Highest Class F1: 1.000.", bold_prefix="* Measured Results: ")
    
    add_paragraph(
        "While a validation F1-score of 99.31% suggests the classification problem is solved, this score is inflated by the nature of the PlantVillage dataset. The images are captured against uniform, plain backgrounds with optimal lighting. This dataset lacks background clutter, weed interference, and shadow variations. When tested on real field photos, this model's accuracy dropped to around 60%. This performance gap is why the deployed project uses a hosted VLM for visual inference, reserving the local CNN as an optional fast-path for high-confidence crop images.",
        bold_prefix="Academic Critique of Clinical Datasets: "
    )

    add_heading_2("4.2 Experiment B: Multimodal Fusion Ablation Study (ViT & DistilBERT)")
    add_paragraph(
        "To verify if combining textual symptom descriptions with images improves classification accuracy, we evaluated a multimodal fusion network (scripts/train_fusion_real.py)."
    )
    add_paragraph("Frozen Vision Transformer (ViT-B/16) outputs a 768-d visual vector. Frozen DistilBERT outputs a 768-d textual token vector. These features feed into a cross-modal attention module (queries from visual path attend to keys/values from textual path) and a joint classifier.", bold_prefix="* Architecture: ")
    add_paragraph("17 crop classes, 2,040 samples. The symptom text notes were programmatically generated from taxonomic symptom lists (incorporating random word dropout and noise to simulate user transcription errors).", bold_prefix="* Dataset: ")
    add_paragraph(
        " - Image-only Modality (ViT-B/16): 0.907 Macro F1-score\n"
        " - Text-only Modality (DistilBERT): 0.778 Macro F1-score\n"
        " - Combined Multimodal Fusion: 0.990 Macro F1-score",
        bold_prefix="* Ablation Study Results: "
    )
    add_paragraph(
        "The multimodal fusion approach improved the macro F1-score by +8.3 points over the image-only model. This indicates that adding symptom context allows the model to resolve visual ambiguities (such as early-stage lesions that look like general nutrient deficiencies)."
    )

    add_heading_2("4.3 Custom Data Annotation Infrastructure (Streamlit Framework)")
    add_paragraph(
        "To address the lack of severity and image-quality labels in public datasets, we developed a custom annotation tool (tools/annotator_app.py). Built with Streamlit, the tool allows annotators to load raw images, assign primary crop/disease labels, adjust a continuous severity slider (0.0 to 1.0), flag focus/blur quality issues, and write natural language symptom summaries. This tool helps build the specialized datasets needed to train future multimodal models."
    )

    # ==========================================
    # SECTION 5: VALIDATION, USER TESTING & DEMO CASES
    # ==========================================
    doc.add_page_break()
    add_heading_1("5. SYSTEM VALIDATION, USER TESTING, & DEMO CASES")
    
    add_paragraph(
        "To validate the safety-invariant layer and the diagnostic pipeline, the following four test cases were executed on the deployed VPS application:"
    )

    add_heading_2("5.1 Scenario 1: Target Crop Leaf Pathology (Concentric Tomato Blight)")
    add_paragraph("A leaf showing dark brown lesions with concentric target-board rings.", bold_prefix="* Test Image: ")
    add_paragraph("\"spots on lower tomato leaves spreading upwards after heavy rain\"", bold_prefix="* Spoken Symptom Note: ")
    add_paragraph("Whisper transcribed the audio successfully. The Qwen3.6-27B model analyzed the image and transcript, mapping it to the taxonomic label TOM_EARLY_BLIGHT. The safety-invariant layer verified that TOM_EARLY_BLIGHT is a valid label for Tomato.", bold_prefix="* System Processing: ")
    add_paragraph("Crop Detected: Tomato; Diagnosis: Tomato Early Blight (Alternaria solani); Confidence: 94%; Severity: 55%; Actionable Advice: Remove lower diseased leaves, apply copper-based organic fungicides, and avoid overhead watering.", bold_prefix="* Client Response: ")

    add_heading_2("5.2 Scenario 2: Target Crop Healthy State (Healthy Potato Leaf)")
    add_paragraph("A clean, green potato leaf with no visual spots or insect damage.", bold_prefix="* Test Image: ")
    add_paragraph("The VLM mapped the image to the taxonomic label POT_HEALTHY. The safety-invariant layer identified the healthy label, confirmed the crop classification, and mapped the kind to healthy.", bold_prefix="* System Processing: ")
    add_paragraph("Crop Detected: Potato; Diagnosis: Healthy; Confidence: 98%; Severity: 0%; Actionable Advice: Maintain regular watering schedules and inspect lower leaves weekly for early signs of blight.", bold_prefix="* Client Response: ")

    add_heading_2("5.3 Scenario 3: Out-of-Scope Leaf Rejection (Grape Leaf)")
    add_paragraph("A grape leaf showing signs of downy mildew.", bold_prefix="* Test Image: ")
    add_paragraph("Selected \"Tomato\" in the frontend.", bold_prefix="* User Selection: ")
    add_paragraph("The VLM identified that the leaf is Grape, mapping the kind to unsupported_crop. The safety-invariant layer ran the crop check: Grape is not in the supported crop set S. The safety layer stripped the disease details and returned a polite rejection.", bold_prefix="* System Processing: ")
    add_paragraph("Classification: Rejected (Unsupported Crop); Response Message: \"The uploaded leaf appears to be Grape, which is currently unsupported. We support Tomato, Potato, Rice, Maize, Chili, Cucumber, Pepper, and Eggplant.\"", bold_prefix="* Client Response: ")

    add_heading_2("5.4 Scenario 4: Irrelevant Non-Plant Visual Noise Rejection (Automobile)")
    add_paragraph("A photograph of a sedan car.", bold_prefix="* Test Image: ")
    add_paragraph("Selected \"Cucumber\" in the frontend.", bold_prefix="* User Selection: ")
    add_paragraph("The VLM analyzed the image and mapped the kind to not_a_leaf. The safety-invariant layer identified the not_a_leaf classification, bypassed the taxonomy checks, and issued an immediate rejection.", bold_prefix="* System Processing: ")
    add_paragraph("Classification: Rejected (Invalid Subject); Response Message: \"No plant leaf or supported animal was detected in this photo. Please upload a clear, close-up image of a supported crop leaf or animal.\"", bold_prefix="* Client Response: ")

    # ==========================================
    # SECTION 6: ETHICAL DISCLOSURES
    # ==========================================
    add_heading_1("6. ETHICAL DISCLOSURES & PROFESSIONAL ALIGNMENT (ACM CODE OF ETHICS)")
    add_paragraph(
        "Providing automated agricultural and veterinary diagnostics carries real safety risks. A false diagnosis could lead a farmer to apply incorrect chemicals, resulting in crop damage, wasted expenses, or environmental harm. AgriDoctor is aligned with the ACM Code of Ethics and Professional Conduct [6]:"
    )
    add_paragraph(
        "Livestock diagnostics are restricted to visible conditions, and the system always prompts the user to contact a certified veterinarian before administering medication. Suggested treatments prioritize cultural controls (crop rotation, watering adjustments) and organic solutions (neem oil, copper soap) before recommending synthetic chemical pesticides.",
        bold_prefix="* Section 1.2 (Avoid Harm): "
    )
    add_paragraph(
        "Designed as an installable PWA with vanilla JavaScript and no build step. This keeps the application lightweight, fast, and accessible on low-end mobile devices and slower cellular networks in rural areas.",
        bold_prefix="* Section 1.4 (Be Fair and Take Action Not to Discriminate): "
    )
    add_paragraph(
        "The system is biased toward uncertainty. If the VLM is not confident in the diagnosis, the server rejects the request with a polite explanation rather than return an uncertain guess.",
        bold_prefix="* Section 2.5 (Give Evaluations of Limitations): "
    )

    # ==========================================
    # SECTION 7: REQUIRED DIGITAL ASSETS
    # ==========================================
    add_heading_1("7. REQUIRED DIGITAL ASSETS & HOSTING LINKS")
    add_paragraph("https://agridoctor.cloud (Hosted on a secure VPS instance with Let's Encrypt SSL. Active and hosted beyond September 1st, 2026.)", bold_prefix="* Production Web Deployment: ")
    add_paragraph("https://github.com/Sarifkhan1/AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor (Public GitHub repository containing FastAPI backend, PWA frontend, deployment scripts, and PyTorch models.)", bold_prefix="* Git Repository: ")
    add_paragraph("https://youtu.be/SOlaeAC4mLw (A complete video walkthrough showing the application workflow, voice integrations, and out-of-scope rejection scenarios.)", bold_prefix="* Video Demonstration: ")

    # ==========================================
    # SECTION 8: BIBLIOGRAPHY
    # ==========================================
    add_heading_1("8. BIBLIOGRAPHY & REFERENCES (IEEE STYLE)")
    
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(4)
    p.add_run("[1] A. Dosovitskiy et al., \"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,\" in International Conference on Learning Representations (ICLR), 2021. Available: https://arxiv.org/abs/2010.11929\n\n")
    p.add_run("[2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, \"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,\" arXiv preprint arXiv:1910.01108, 2019. Available: https://arxiv.org/abs/1910.01108\n\n")
    p.add_run("[3] S. Antol et al., \"VQA: Visual Question Answering,\" in IEEE International Conference on Computer Vision (ICCV), 2015, pp. 2425-2433. Available: https://arxiv.org/abs/1505.00468\n\n")
    p.add_run("[4] A. Radford et al., \"Robust Speech Recognition via Large-Scale Weak Supervision,\" arXiv preprint arXiv:2212.04356, 2022. Available: https://arxiv.org/abs/2212.04356\n\n")
    p.add_run("[5] D. P. Hughes and M. Salathé, \"An open access repository of images on plant health to enable the development of mobile disease diagnostics,\" arXiv preprint arXiv:1511.08060, 2015. Available: https://arxiv.org/abs/1511.08060\n\n")
    p.add_run("[6] Association for Computing Machinery, \"ACM Code of Ethics and Professional Conduct,\" ACM, 2018. Available: https://www.acm.org/code-of-ethics\n\n")
    p.add_run("[7] Food and Agriculture Organization of the United Nations (FAO), \"New standards to curb the global spread of plant pests and diseases,\" FAO Newsroom, 2021. Available: https://www.fao.org/newsroom/detail/New-standards-to-curb-the-global-spread-of-plant-pests-and-diseases/en\n\n")
    p.add_run("[8] Food and Agriculture Organization of the United Nations (FAO), \"Understanding the context — Pest and Pesticide Management,\" FAO, 2021. Available: https://www.fao.org/pest-and-pesticide-management/about/understanding-the-context/en/\n\n")
    p.add_run("[9] GSMA, \"Detecting and managing crop pests and diseases with AI: Insights from Plantix,\" GSMA Agritech, 2020. Available: https://www.gsma.com/solutions-and-impact/connectivity-for-good/mobile-for-development/programme/agritech/detecting-and-managing-crop-pests-and-diseases-with-ai-insights-from-plantix/")

    # Save the document
    doc.save("/Users/santosh/Documents/agri-doctor/AgriDoctor_Resit_Submission_Report.docx")
    print("Document created successfully at /Users/santosh/Documents/agri-doctor/AgriDoctor_Resit_Submission_Report.docx")

if __name__ == "__main__":
    create_document()
