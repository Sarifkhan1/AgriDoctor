AgriDoctor: Multimodal Crop + Livestock Health Assistant
Mohammad Sarif Khan
2238572
01/01/2000
Link to git project repo: https://github.com/Sarifkhan1/AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor

## Introduction

AgriDoctor AI is a comprehensive, multimodal artificial intelligence system designed to assist farmers and agricultural professionals in the early diagnosis of crop diseases and livestock health issues. The motivation behind this project stems from the critical need for accessible, accurate, and timely agricultural expertise in remote or underserved farming communities. By leveraging modern deep learning techniques, the application aims to bridge the gap between complex veterinary or botanical knowledge and the day-to-day realities of farming, potentially reducing crop loss and improving animal welfare.

This project implements a full-stack interactive application that accepts both visual (images of crops/animals) and auditory (spoken descriptions of symptoms) inputs. It processes these inputs using a suite of advanced AI models to provide diagnostic predictions, severity assessments, and actionable treatment advice. The system is designed to be intuitive, supporting features like voice interaction to accommodate users with varying levels of literacy.

**Features of the project:**

- **Multimodal Diagnosis**: Combines computer vision (image analysis) and natural language processing (symptom descriptions) for higher accuracy.
- **Voice-First Interface**: Utilizes OpenAI Whisper for automatic speech recognition (ASR), allowing users to describe symptoms verbally in multiple languages.
- **Specialized Disease Detection**: Supports diagnosis for 6 major crops (Tomato, Potato, Rice, Maize, Chili, Cucumber) and 4 livestock types (Cattle, Goat, Sheep, Poultry).
- **Severity Estimation**: A unified regression head in the model architecture providing a severity score (0-1) alongside classification.
- **Interactive Web Application**: A responsive frontend built with vanilla JavaScript for broad device compatibility, featuring camera integration and audio recording.
- **Secure Case Management**: User authentication system to track and manage historical diagnosis cases securely.
- **Custom Data Annotation Tool**: A Streamlit-based application developed to facilitate the efficient labeling of agricultural datasets.

## Background

The project builds upon state-of-the-art advancements in computer vision and natural language processing. The core visual analysis utilizes **Vision Transformers (ViT)** [1], specifically the `vit_b_16` architecture, establishing a strong baseline for image classification tasks compared to traditional Convolutional Neural Networks (CNNs). For phenotypic feature extraction from text, the system leverages **DistilBERT** [2], a distilled version of BERT that offers a balance between performance and computational efficiency suitable for web deployment.

The multimodal fusion approach draws inspiration from recent research in **Visual Question Answering (VQA)** and multimodal learning, where cross-attention mechanisms are employed to align features from different modalities [3]. The application also integrates **OpenAI Whisper** [4] for robust speech-to-text capabilities, enabling the system to handle noisy agricultural environments and diverse accents.

For dataset construction, publicly available datasets such as **PlantVillage** [5] served as the foundation for crop disease images, supplemented by synthetic data augmentation techniques. The project allows for comparison against standard benchmarks in agricultural disease classification.

## Implementation details for interactive application

The application is architected as a decoupled system driven by a content-neutral API layer, ensuring flexibility and scalability.

**Frontend (Client Layer)**:
The user interface is built using standard HTML5, CSS3, and Vanilla JavaScript, ensuring it runs efficiently on low-end mobile devices common in agricultural settings. The `app.js` file manages the application state (e.g., `currentStep`, `selectedCrop`, `audioBlob`). It utilizes the **MediaDevices API** for capturing real-time images from the device camera and the **MediaRecorder API** for recording voice notes. These inputs are converted to Blobs and transmitted to the backend via `FormData` objects.

**Backend (API Layer)**:
The server is implemented using **FastAPI**, a high-performance Python framework. The `main.py` entry point orchestrates the flow:

1.  **Authentication**: Uses JWT (JSON Web Tokens) with PBKDF2 password hashing to secure user sessions.
2.  **Request Handling**: Endpoints like `/api/cases/{id}/run` accept multimodal inputs. Images are stored in the filesystem, while metadata (symptoms, weather) is serialized into the SQLite database (`agridoctor.db`).
3.  **Inference Pipeline**: The system uses a background task worker management approach. When a user requests analysis, a job is queued. The backend simulates the inference latency (or calls the trained PyTorch models) and generates a JSON response containing the diagnosis, confidence score, and structured advice/treatment plans.

**AI Logic & Models**:
The core intelligence resides in `src/models/`. The **Multimodal Fusion Transformer** (`train_multimodal.py`) is the centerpiece:

- **Image Encoder**: A `vit_b_16` model extracts a 768-dimensional feature vector from the uploaded image.
- **Text Encoder**: A `DistilBERT` model processes the transcribed text (from the Whisper ASR module) to generate text feature embeddings.
- **Cross-Modal Attention**: A custom `CrossModalAttention` module allows image features to "attend" to relevant text features and vice-versa, fusing them into a meaningful joint representation.
- **Heads**: The fused features are passed to two parallel heads: a classifier for the specific disease label and a regressor for the severity score.

_(Logic Flow: User Input -> APIs -> Background Worker -> Model Inference -> JSON Response -> UI Rendering)_

## Details of additional experimentation

Extensive experimentation was conducted outside the main application deployment to validate the model architectures and improve data quality.

- **Ablation Studies**: A systematic ablation study was implemented in `train_multimodal.py` (function `run_ablation`). This experiment compared the F1-scores of the **Multimodal model** against **Image-only** and **Text-only** baselines. This experimentation was crucial to prove that adding symptom descriptions actually improved diagnostic accuracy over simple image classification.
- **Custom Data Annotation Tool**: To address the lack of high-quality metadata in public datasets, a custom labeling tool was built using **Streamlit** (`tools/annotator_app.py`). This tool allows annotators to review images, assign primary/secondary disease labels, and drastically improves the dataset quality by adding "severity scores" (0.0 to 1.0) and "image quality scores" which are not present in standard datasets like PlantVillage.
- **Model Architecture Search**: The training script (`train_image_model.py`) was designed to support multiple backbones, specifically **ViT-B/16** vs **Swin Transformer (Tiny)**. Experiments showed that while Swin Transformers offer hierarchical feature extraction, ViT provided a more stable convergence for this specific medical/agricultural domain during the limited training runs.
- **Severity Regression Task**: A unique experimental addition was the "Severity Head". Unlike standard classification, this experiment attempted to quantify _how_ diseased a plant is. This involved modifying the loss function to be a weighted sum of CrossEntropy (for class) and MSE (Mean Squared Error for severity), optimizing for both objectives simultaneously.

## Critical reflection

**What worked well**:
The integration of different modalities proved highly effective. The user interface design, focusing on a "wizard" style step-by-step process, was successful in simplifying the complex data entry required for multimodal diagnosis. The decision to use **FastAPI** coupled with **SQLite** allowed for rapid prototyping without the overhead of heavy database management, which smoothed the development of the interactive application components. The **Streamlit** annotation tool was a significant productivity booster, allowing for rapid dataset curation that would have been tedious with manual CSV editing.

**Shortcomings & Future Improvements**:
A primary shortcoming is the reliance on mock inference in the final deployment for demonstration purposes, due to the computational constraints of running distinct heavy Transformers (ViT + BERT) alongside the web server. If I were to start again, I would implement **model quantization** or **knowledge distillation** to create smaller, faster models suitable for real-time CPU inference on the server. Additionally, the current system processes inputs sequentially; a true real-time streaming approach for voice and video could provide a more fluid user experience.

**Ethical Considerations**:
The deployment of medical diagnostic tools (even for agriculture) carries risk. Misdiagnosis could lead to the misuse of pesticides or the unnecessary culling of livestock. To mitigate this, an explicit **confidence threshold** (e.g., <50%) was implemented to flag uncertain results. Furthermore, the application includes prominent "Safety Notes" and disclaimers emphasizing that AI suggestions do not replace professional veterinary or agronomic advice. Adhering to the ACM Code of Ethics [6], the system prioritizes "Recall" (avoiding missed dangerous diseases) over precision in its internal weighting during training.

## LLM disclaimer

Large Language Models (LLMs) were used as an assistive tool during the development of this project. specifically for:

1.  **Boilerplate Code Generation**: Generating the initial Pydantic models for the FastAPI backend to match the database schema.
2.  **CSS Styling**: Assisting in generating the CSS for the responsive grid layout in the frontend.
3.  **Debugging**: Explaining obscure PyTorch error messages related to tensor dimension mismatches in the Cross-Attention layer.

All code generated was reviewed, tested, and significantly modified to fit the specific architecture of AgriDoctor. The core logic for the multimodal fusion and the custom training loops were written manually to ensure correctness and understanding.

## Bibliography

[1] Dosovitskiy, A. et al. (2020) 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', _International Conference on Learning Representations (ICLR)_. Available at: https://arxiv.org/abs/2010.11929.

[2] Sanh, V. et al. (2019) 'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter', _arXiv preprint arXiv:1910.01108_.

[3] Antol, S. et al. (2015) 'VQA: Visual Question Answering', _Proceedings of the IEEE International Conference on Computer Vision (ICCV)_, pp. 2425-2433.

[4] Radford, A. et al. (2022) 'Robust Speech Recognition via Large-Scale Weak Supervision', _OpenAI Training_. Available at: https://arxiv.org/abs/2212.04356.

[5] Hughes, D. and Salathé, M. (2015) 'An open access repository of images on plant health to enable the development of mobile disease diagnostics', _arXiv preprint arXiv:1511.08060_.

[6] Association for Computing Machinery (2018) _ACM Code of Ethics and Professional Conduct_. Available at: https://www.acm.org/code-of-ethics (Accessed: 25 February 2025).
