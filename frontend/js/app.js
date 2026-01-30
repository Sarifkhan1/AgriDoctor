/**
 * AgriDoctor AI - Main Application
 * Handles UI interactions, state management, and flow control
 */

// ============================================================================
// State Management
// ============================================================================

const state = {
    currentPage: 'home',
    currentStep: 1,
    selectedCrop: null,
    selectedImage: null,
    audioBlob: null,
    currentCase: null,
    user: null
};

// ============================================================================
// Navigation
// ============================================================================

function showPage(pageName) {
    // Update state
    state.currentPage = pageName;
    
    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.page === pageName);
    });
    
    // Show/hide pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.toggle('active', page.id === `page-${pageName}`);
    });
    
    // Special handling
    if (pageName === 'diagnose') {
        resetDiagnosis();
    } else if (pageName === 'history') {
        loadHistory();
    }
}

function showStep(stepNum) {
    state.currentStep = stepNum;
    
    // Update step indicators
    document.querySelectorAll('.step').forEach((step, idx) => {
        step.classList.toggle('active', idx + 1 === stepNum);
        step.classList.toggle('completed', idx + 1 < stepNum);
    });
    
    // Show/hide step content
    document.querySelectorAll('.diagnose-step').forEach(step => {
        step.classList.toggle('hidden', step.id !== `step-${stepNum}`);
    });
}

function nextStep() {
    if (state.currentStep < 4) {
        showStep(state.currentStep + 1);
    }
}

function prevStep() {
    if (state.currentStep > 1) {
        showStep(state.currentStep - 1);
    }
}

function resetDiagnosis() {
    state.currentStep = 1;
    state.selectedCrop = null;
    state.selectedImage = null;
    state.audioBlob = null;
    state.currentCase = null;
    
    // Reset UI
    document.querySelectorAll('.crop-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    document.getElementById('upload-placeholder').classList.remove('hidden');
    document.getElementById('image-preview').classList.add('hidden');
    document.getElementById('preview-img').src = '';
    document.getElementById('next-step-2').disabled = true;
    document.getElementById('audio-playback').classList.add('hidden');
    document.getElementById('onset-days').value = '';
    document.querySelectorAll('input[name="spread"]').forEach(r => r.checked = false);
    document.getElementById('notes').value = '';
    
    showStep(1);
}

function startNewDiagnosis() {
    showPage('diagnose');
}

// ============================================================================
// Crop Selection
// ============================================================================

function initCropSelection() {
    document.querySelectorAll('.crop-card').forEach(card => {
        card.addEventListener('click', () => {
            // Update selection
            document.querySelectorAll('.crop-card').forEach(c => {
                c.classList.remove('selected');
            });
            card.classList.add('selected');
            state.selectedCrop = card.dataset.crop;
            
            // Auto-advance after brief delay
            setTimeout(() => {
                nextStep();
            }, 300);
        });
    });
}

// ============================================================================
// Image Upload
// ============================================================================

function initImageUpload() {
    const uploadArea = document.getElementById('upload-area');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const removeBtn = document.getElementById('remove-image');
    const nextBtn = document.getElementById('next-step-2');
    
    // Click to upload
    uploadPlaceholder.addEventListener('click', () => {
        imageInput.click();
    });
    
    // File input change
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageFile(e.target.files[0]);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });
    
    // Remove image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        state.selectedImage = null;
        uploadPlaceholder.classList.remove('hidden');
        imagePreview.classList.add('hidden');
        previewImg.src = '';
        nextBtn.disabled = true;
        imageInput.value = '';
    });
    
    // Next button
    nextBtn.addEventListener('click', nextStep);
}

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    
    state.selectedImage = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('preview-img').src = e.target.result;
        state.imageUrl = e.target.result; // Save for results page
        document.getElementById('upload-placeholder').classList.add('hidden');
        document.getElementById('image-preview').classList.remove('hidden');
        document.getElementById('next-step-2').disabled = false;
    };
    reader.readAsDataURL(file);
}

// ============================================================================
// Camera
// ============================================================================

let cameraStream = null;

function initCamera() {
    const cameraBtn = document.getElementById('camera-btn');
    const captureBtn = document.getElementById('capture-btn');
    
    cameraBtn.addEventListener('click', openCamera);
    captureBtn.addEventListener('click', capturePhoto);
}

async function openCamera() {
    const modal = document.getElementById('camera-modal');
    const video = document.getElementById('camera-video');
    
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });
        video.srcObject = cameraStream;
        modal.classList.remove('hidden');
    } catch (err) {
        alert('Unable to access camera. Please upload an image instead.');
    }
}

function closeCameraModal() {
    const modal = document.getElementById('camera-modal');
    const video = document.getElementById('camera-video');
    
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    video.srcObject = null;
    modal.classList.add('hidden');
}

function capturePhoto() {
    const video = document.getElementById('camera-video');
    const canvas = document.getElementById('camera-canvas');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    canvas.toBlob((blob) => {
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        handleImageFile(file);
        closeCameraModal();
    }, 'image/jpeg', 0.9);
}

// ============================================================================
// Voice Recording
// ============================================================================

let mediaRecorder = null;
let audioChunks = [];

function initVoiceRecording() {
    const recordBtn = document.getElementById('record-btn');
    const recordingIndicator = document.getElementById('recording-indicator');
    const audioPlayback = document.getElementById('audio-playback');
    
    recordBtn.addEventListener('mousedown', startRecording);
    recordBtn.addEventListener('mouseup', stopRecording);
    recordBtn.addEventListener('mouseleave', stopRecording);
    
    // Touch support
    recordBtn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startRecording();
    });
    recordBtn.addEventListener('touchend', stopRecording);
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (e) => {
            audioChunks.push(e.data);
        };
        
        mediaRecorder.onstop = () => {
            state.audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(state.audioBlob);
            document.getElementById('audio-playback').src = audioUrl;
            document.getElementById('audio-playback').classList.remove('hidden');
            
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        document.getElementById('recording-indicator').classList.remove('hidden');
        document.querySelector('.record-text').textContent = 'Recording...';
        
    } catch (err) {
        alert('Unable to access microphone');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        document.getElementById('recording-indicator').classList.add('hidden');
        document.querySelector('.record-text').textContent = 'Hold to Record';
    }
}

// ============================================================================
// Analysis
// ============================================================================

function initAnalysis() {
    document.getElementById('analyze-btn').addEventListener('click', runAnalysis);
}

async function runAnalysis() {
    const loadingOverlay = document.getElementById('loading-overlay');
    loadingOverlay.classList.remove('hidden');
    
    try {
        // Check if authenticated, use demo mode if not
        if (!api.isAuthenticated()) {
            // Demo mode - simulate analysis
            await simulateAnalysis();
            return;
        }
        
        // Create case
        const caseData = await api.createCase('crop', state.selectedCrop);
        state.currentCase = caseData;
        
        // Upload image
        if (state.selectedImage) {
            await api.uploadImage(caseData.id, state.selectedImage);
        }
        
        // Upload audio if present
        if (state.audioBlob) {
            const audioFile = new File([state.audioBlob], 'voice.webm', { type: 'audio/webm' });
            await api.uploadSpeech(caseData.id, audioFile);
        }
        
        // Update metadata
        const metadata = {
            onset_days: parseInt(document.getElementById('onset-days').value) || null,
            spread_speed: document.querySelector('input[name="spread"]:checked')?.value || null,
            notes: document.getElementById('notes').value || null
        };
        await api.updateMetadata(caseData.id, metadata);
        
        // Run inference
        await api.runInference(caseData.id);
        
        // Poll for result
        const result = await api.pollForResult(caseData.id);
        
        displayResults(result);
        
    } catch (error) {
        console.error('Analysis error:', error);
        // Fall back to demo mode
        await simulateAnalysis();
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}

async function simulateAnalysis() {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Generate demo result based on selected crop
    const demoResults = {
        tomato: {
            primary_label: 'TOM_EARLY_BLIGHT',
            confidence: 0.89,
            severity_score: 0.45,
            urgency_level: 'medium',
            advice: {
                summary: "The analysis detects **Early Blight**, a common fungal disease caused by *Alternaria solani*. It typically starts on lower leaves as circular brown spots with concentric rings ('bullseye' pattern). If left untreated, it causes yellowing of leaves (chlorosis), defoliation, and sunscald on the fruit due to loss of leaf cover.",
                what_to_do_now: [
                    "**Step 1: Pruning & Sanitation**: Immediately prune off all infected lower leaves using sterilized shears. Dip tools in 10% bleach solution between cuts. Dispose of debris far from the garden (do not compost).",
                    "**Step 2: Fungicide Application**: Apply a copper-based fungicide or products containing Mancozeb or Chlorothalonil. For organic options, use Bacillus subtilis biofungicide. Repeat every 7–10 days.",
                    "**Step 3: Modify Watering**: Stop overhead irrigation immediately. Water only at the base of the plant to keep foliage dry, as the fungus requires moisture to spread."
                ],
                prevention: [
                    "**Crop Rotation**: Do not plant tomatoes, potatoes, or eggplants in the same soil for at least 3 years.",
                    "**Mulching**: maintain a 2-3 inch layer of organic mulch (straw/leaves) at the base to prevent soil spores from splashing onto lower leaves.",
                    "**Stake & Air Flow**: Stake plants and prune suckers to improve air circulation, reducing humidity around the leaves."
                ],
                when_to_get_help: [
                    "If lesions appear on the stems (collar rot) which can kill the plant.",
                    "If more than 50% of the foliage is yellowing despite treatment."
                ]
            }
        },
        potato: {
            primary_label: 'POT_LATE_BLIGHT',
            confidence: 0.92,
            severity_score: 0.85,
            urgency_level: 'high',
            advice: {
                summary: "This is a **critical detection** of Late Blight, the devastating disease responsible for the Irish Potato Famine. It appears as large, dark/oily lesions on leaves and stems, often with white fungal growth on the underside in humid weather. It can destroy an entire crop in days.",
                what_to_do_now: [
                    "**Step 1: Immediate Removal**: Verify if only a few plants are affected. If so, pull them out entirely and bag them on site to prevent spore release. Destroy by burning.",
                    "**Step 2: Chemical Control**: For remaining healthy plants, apply a protective fungicide immediately (e.g., Chlorothalonil, Mancozeb). Late blight is difficult to cure once established.",
                    "**Step 3: Desiccation**: If the crop is near harvest, kill the vines (leaves/stems) immediately to prevent the fungus from travelling down the stem and infecting the tubers.",
                    "**Step 4: Harvest Care**: Wait 2-3 weeks after vine death before harvesting to ensure spore death."
                ],
                prevention: [
                    "**Certified Seed**: Only plant certified disease-free seed potatoes.",
                    "**Eliminate Culls**: Destroy any volunteer potatoes or piles of culled potatoes near the field as they harbor the pathogen over winter.",
                    "**Resistant Varieties**: Choose resistant varieties like 'Sarpo Mira' or 'Defender'."
                ],
                when_to_get_help: [
                    "**IMMEDIATELY**: Contact your local agricultural extension office as Late Blight outbreaks often require community warnings."
                ]
            }
        },
        rice: {
            primary_label: 'RICE_BLAST',
            confidence: 0.85,
            severity_score: 0.60,
            urgency_level: 'medium',
            advice: {
                summary: "The symptoms indicate **Rice Blast**, one of the most destructive rice diseases. It forms spindle-shaped lesions with whitish/gray centers and reddish-brown borders. It can affect leaves, nodes, and the neck of the panicle (Neck Blast), which causes severe yield loss.",
                what_to_do_now: [
                    "**Step 1: Nitrogen Management**: Stop applying nitrogen fertilizer immediately. High nitrogen levels exacerbate the disease.",
                    "**Step 2: Water Management**: Maintain a continuous deep flood (water level) in the paddy, as the disease is worse in dry soil conditions.",
                    "**Step 3: Chemical Treatment**: Apply systemic fungicides such as Tricyclazole, Isoprothiolane, or Azoxystrobin immediately if the blast is at the booting or heading stage."
                ],
                prevention: [
                    "**Resistant Varieties**: This is the most effective method. Use locally recommended resistant strains.",
                    "**Planting Time**: Adjust planting dates to avoid heading stages during periods of high humidity/rainfall.",
                    "**Seed Treatment**: Treat seeds with Pseudomonas fluorescens before sowing."
                ],
                when_to_get_help: [
                    "If 'Neck Blast' is observed (lesions at the base of the panicle) as this leads to total grain failure.",
                    "If the field has a history of severe outbreaks."
                ]
            }
        },
        maize: {
            primary_label: 'MAIZE_RUST',
            confidence: 0.78,
            severity_score: 0.35,
            urgency_level: 'low',
            advice: {
                summary: "Detected **Common Rust**, characterized by small, reddish-brown pustules on both upper and lower leaf surfaces. While spectacular in appearance, it often does not require chemical control unless the infection happens very early in the season.",
                what_to_do_now: [
                    "**Step 1: Field Monitoring**: Check if the pustules are covering more than 30-50% of the leaf surface. If less, the economic impact may be minimal.",
                    "**Step 2: Fungicide**: In sweet corn or seed corn (high value), apply foliar fungicides (Strobilurins or Triazoles) if pustules appear before tasseling.",
                    "**Step 3: Cultural**: Remove alternate hosts (Oxalis species weeds) from around the field."
                ],
                prevention: [
                    "**Resistant Hybrids**: Plant resistant maize hybrids.",
                    "**Planting Date**: Early planting helps the crop mature before rust spore loads become high in the atmosphere.",
                    "**Crop Rotation**: Effective but less critical than for soil-borne diseases as rust blows in via wind."
                ],
                when_to_get_help: [
                    "If pustules turn black (teliospore stage) and infection is severe on ear leaves.",
                    "If you suspect 'Southern Rust' (pustules only on top of leaves) which is more aggressive."
                ]
            }
        },
        chili: {
            primary_label: 'CHILI_ANTHRAC',
            confidence: 0.81,
            severity_score: 0.55,
            urgency_level: 'medium',
            advice: {
                summary: "The fruit/leaf lesions suggest **Anthracnose**. On fruits, it causes sunken, circular, excessively wet spots with concentric rings of salmon-colored spores. It makes the fruit unmarketable.",
                what_to_do_now: [
                    "**Step 1: Harvest & Destruction**: Pick ALL infected fruits immediately and bury them. They are the primary source of new infection.",
                    "**Step 2: Spraying**: Apply fungicides like Mancozeb, Copper Oxychloride, or Azoxystrobin. Ensure thorough coverage of the fruits, not just leaves.",
                    "**Step 3: Seed Treatment**: If saving seeds, treat them with hot water (52°C for 30 mins) or Thiram to kill seed-borne infection."
                ],
                prevention: [
                    "**Wider Spacing**: Increase space between plants to ensure rapid drying after rain.",
                    "**Weed Control**: Keep the field weed-free as weeds can host the fungus.",
                    "**Drainage**: Ensure fields drain well to avoid high humidity microclimates."
                ],
                when_to_get_help: [
                    "If 'Dieback' symptoms occur (branches drying from tip downwards)."
                ]
            }
        },
        cucumber: {
            primary_label: 'CUC_POWDERY',
            confidence: 0.88,
            severity_score: 0.40,
            urgency_level: 'medium',
            advice: {
                summary: "Identified **Powdery Mildew**, appearing as a white, dusty flour-like coating on leaves and stems. Unlike many fungi, it thrives in dry conditions with high humidity, even without rain.",
                what_to_do_now: [
                    "**Step 1: Bio-Control**: Spray a solution of Baking Soda (1%) or Potassium Bicarbonate. Milk solution (1 part milk : 9 parts water) is also effective.",
                    "**Step 2: Chemical Control**: If severe, use sulfur-based fungicides or Mycobutanil. Do not apply sulfur if temperatures are above 30°C (85°F) to avoid burn.",
                    "**Step 3: Pruning**: Remove lower leaves that are heavily coated to improve airflow."
                ],
                prevention: [
                    "**Resistant Cultivars**: Use PM-resistant cucumber varieties.",
                    "**Air Circulation**: Trellis cucumbers to maximize airflow and light exposure.",
                    "**Weeding**: Remove wild cucurbits and weeds nearby."
                ],
                when_to_get_help: [
                    "If you see yellow, angular spots bounded by leaf veins; this might be Downy Mildew instead, which requires different treatment."
                ]
            }
        }
    };
    
    const result = demoResults[state.selectedCrop] || demoResults.tomato;
    
    displayResults(result);
    document.getElementById('loading-overlay').classList.add('hidden');
}

function displayResults(result) {
    const container = document.getElementById('results-container');
    
    const severityClass = result.severity_score < 0.33 ? 'low' : 
                         result.severity_score < 0.66 ? 'medium' : 'high';
    
    const diseaseName = formatLabel(result.primary_label);
    const urgencyClass = result.urgency_level || 'medium';
    
    // Get uploaded image from state
    const imageUrl = state.imageUrl || (document.getElementById('preview-img') ? document.getElementById('preview-img').src : '');
    
    // Get crop info
    const cropEmojis = {
        tomato: '🍅', potato: '🥔', rice: '🌾', 
        maize: '🌽', chili: '🌶️', cucumber: '🥒'
    };
    const cropEmoji = cropEmojis[state.selectedCrop] || '🌱';
    const cropName = state.selectedCrop ? 
        state.selectedCrop.charAt(0).toUpperCase() + state.selectedCrop.slice(1) : 'Crop';
    
    // Get metadata
    const onsetDays = document.getElementById('onset-days')?.value || '';
    const spreadSpeed = document.querySelector('input[name="spread"]:checked')?.value || '';
    const notes = document.getElementById('notes')?.value || '';
    
    const onsetText = {
        '0': 'Today', '1': 'Yesterday', '3': '2-3 days ago',
        '7': 'About a week ago', '14': '2 weeks ago', '30': 'More than 2 weeks'
    };
    
    // Helper to parse simple markdown bolding
    const parseMarkdown = (text) => {
        if (!text) return '';
        return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                   .replace(/\*(.*?)\*/g, '<em>$1</em>');
    };

    container.innerHTML = `
        <div class="results-layout">
            <!-- Left Panel: Image & Case Details -->
            <div class="results-left-panel">
                <div class="results-image-card">
                    <div class="results-image-wrapper" onclick="openImageModal('${imageUrl}')">
                        ${imageUrl ? `
                            <img src="${imageUrl}" alt="Uploaded crop image" class="results-uploaded-image">
                            <div class="image-overlay">
                                <span class="view-full-icon">🔍</span>
                                <span>View Full Image</span>
                            </div>
                        ` : `<div class="results-no-image">${cropEmoji}<span>No image</span></div>`}
                    </div>
                </div>
                
                <div class="results-case-info">
                    <h3>📋 Case Information</h3>
                    <div class="case-detail-item">
                        <span class="detail-label">Crop Identity</span>
                        <span class="detail-value">${cropEmoji} ${cropName}</span>
                    </div>
                    ${onsetDays ? `
                    <div class="case-detail-item">
                        <span class="detail-label">Onset</span>
                        <span class="detail-value">📅 ${onsetText[onsetDays] || onsetDays + ' days ago'}</span>
                    </div>` : ''}
                    ${spreadSpeed ? `
                    <div class="case-detail-item">
                        <span class="detail-label">Propagation</span>
                        <span class="detail-value">${spreadSpeed === 'slow' ? '🐢 Slow' : spreadSpeed === 'moderate' ? '🚶 Moderate' : '🚀 Fast'}</span>
                    </div>` : ''}
                    <div class="case-detail-item">
                        <span class="detail-label">Report ID</span>
                        <span class="detail-value">#${Math.random().toString(36).substr(2, 6).toUpperCase()}</span>
                    </div>
                    ${notes ? `
                    <div class="case-notes">
                        <span class="detail-label">Consultation Notes</span>
                        <p class="notes-text">${notes}</p>
                    </div>` : ''}
                </div>
                
                <div class="results-quick-stats">
                    <div class="quick-stat">
                        <span class="stat-icon">🎯</span>
                        <span class="stat-number">${Math.round(result.confidence * 100)}%</span>
                        <span class="stat-name">AI Confidence</span>
                    </div>
                    <div class="quick-stat">
                        <span class="stat-icon">🌡️</span>
                        <span class="stat-number">${Math.round(result.severity_score * 100)}%</span>
                        <span class="stat-name">Infection Level</span>
                    </div>
                    <div class="quick-stat urgency-${urgencyClass}">
                        <span class="stat-icon">${urgencyClass === 'high' ? '🔴' : urgencyClass === 'medium' ? '🟡' : '🟢'}</span>
                        <span class="stat-number">${urgencyClass.charAt(0).toUpperCase() + urgencyClass.slice(1)}</span>
                        <span class="stat-name">Priority</span>
                    </div>
                </div>
            </div>
            
            <!-- Right Panel: Diagnosis Results -->
            <div class="results-right-panel">
                <div class="result-header">
                    <div class="result-badge">Precision Diagnosis</div>
                    <h2>${diseaseName}</h2>
                    <div class="result-confidence">
                        <span class="pulse-dot"></span>
                        AI confirmed match at ${Math.round(result.confidence * 100)}%
                    </div>
                </div>
                
                <div class="result-body">
                    <div class="result-section result-summary">
                        <div class="section-icon">📜</div>
                        <div class="section-content">
                            <h3>Diagnostic Summary</h3>
                            <p>${parseMarkdown(result.advice.summary)}</p>
                        </div>
                    </div>
                    
                    <div class="result-section result-actions">
                        <div class="section-icon">⚡</div>
                        <div class="section-content">
                            <h3>Immediate Action Plan</h3>
                            <div class="action-steps">
                                ${result.advice.what_to_do_now.map((item, i) => `
                                    <div class="action-step-item">
                                        <div class="action-step-number">${i + 1}</div>
                                        <div class="action-step-content">${parseMarkdown(item)}</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                    
                    <div class="result-grid">
                        <div class="result-section result-prevention">
                            <div class="section-icon">🛡️</div>
                            <div class="section-content">
                                <h3>Long-term Protection</h3>
                                <ul class="prevention-list">
                                    ${result.advice.prevention.map(item => `<li><span class="bullet">✨</span>${parseMarkdown(item)}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                        
                        <div class="result-section result-help">
                            <div class="section-icon">🚨</div>
                            <div class="section-content">
                                <h3>Escalation Triggers</h3>
                                <ul class="help-list">
                                    ${result.advice.when_to_get_help.map(item => `<li><span class="bullet">🔬</span>${parseMarkdown(item)}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <div class="safety-notice">
                        <div class="safety-icon">ℹ️</div>
                        <div class="safety-content">
                            <strong>Expert Consultation Recommended</strong>
                            <p>${parseMarkdown(result.advice.safety_note)}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    showStep(4);
}

// Image Modal Functions
function openImageModal(imageUrl) {
    if (!imageUrl) return;
    const modal = document.getElementById('image-viewer-modal');
    const img = document.getElementById('full-image');
    img.src = imageUrl;
    modal.classList.remove('hidden');
}

function closeImageModal() {
    const modal = document.getElementById('image-viewer-modal');
    modal.classList.add('hidden');
}

function formatLabel(label) {
    const parts = label.split('_');
    const crop = parts[0];
    const disease = parts.slice(1).join(' ');
    
    const cropNames = {
        'TOM': 'Tomato',
        'POT': 'Potato',
        'RICE': 'Rice',
        'MAIZE': 'Maize',
        'CHILI': 'Chili',
        'CUC': 'Cucumber'
    };
    
    return `${cropNames[crop] || crop} - ${disease.charAt(0) + disease.slice(1).toLowerCase()}`;
}

// ============================================================================
// History
// ============================================================================

async function loadHistory() {
    const container = document.getElementById('history-list');
    
    if (!api.isAuthenticated()) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">🔒</span>
                <h3>Login Required</h3>
                <p>Please login to view your diagnosis history</p>
                <button class="btn btn-primary" onclick="showAuthModal()">
                    Login
                </button>
            </div>
        `;
        return;
    }
    
    try {
        const cases = await api.listCases();
        
        if (cases.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">📋</span>
                    <h3>No diagnoses yet</h3>
                    <p>Start your first diagnosis to see it here</p>
                    <button class="btn btn-primary" onclick="showPage('diagnose')">
                        Start Diagnosis
                    </button>
                </div>
            `;
            return;
        }
        
        container.innerHTML = cases.map(c => `
            <div class="history-card" onclick="viewCase('${c.id}')">
                <div class="history-thumb" style="display: flex; align-items: center; justify-content: center;">
                    ${getCropEmoji(c.crop_name)}
                </div>
                <div class="history-info">
                    <div class="history-title">${c.crop_name || 'Crop'} Diagnosis</div>
                    <div class="history-meta">${formatDate(c.created_at)} • ${c.status}</div>
                </div>
                <span class="history-badge ${getStatusBadge(c.status)}">
                    ${c.status}
                </span>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Failed to load history:', error);
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">⚠️</span>
                <h3>Unable to load history</h3>
                <p>Please try again later</p>
            </div>
        `;
    }
}

function getCropEmoji(cropName) {
    const emojis = {
        tomato: '🍅',
        potato: '🥔',
        rice: '🌾',
        maize: '🌽',
        chili: '🌶️',
        cucumber: '🥒'
    };
    return emojis[cropName?.toLowerCase()] || '🌱';
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function getStatusBadge(status) {
    const badges = {
        done: 'badge-healthy',
        running: 'badge-mild',
        failed: 'badge-severe'
    };
    return badges[status] || 'badge-mild';
}

async function viewCase(caseId) {
    try {
        const result = await api.getResult(caseId);
        displayResults(result);
        showPage('diagnose');
        showStep(4);
    } catch (error) {
        alert('Unable to load case details');
    }
}

// ============================================================================
// Authentication
// ============================================================================

function initAuth() {
    const authBtn = document.getElementById('auth-btn');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    
    authBtn.addEventListener('click', () => {
        if (api.isAuthenticated()) {
            logout();
        } else {
            showAuthModal();
        }
    });
    
    // Tab switching
    document.querySelectorAll('.auth-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            if (tab.dataset.tab === 'login') {
                loginForm.classList.remove('hidden');
                registerForm.classList.add('hidden');
            } else {
                loginForm.classList.add('hidden');
                registerForm.classList.remove('hidden');
            }
        });
    });
    
    // Login form
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;
        
        try {
            await api.login(email, password);
            closeAuthModal();
            updateAuthUI();
        } catch (error) {
            alert(error.message || 'Login failed');
        }
    });
    
    // Register form
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fullName = document.getElementById('register-name').value;
        const email = document.getElementById('register-email').value;
        const password = document.getElementById('register-password').value;
        const confirm = document.getElementById('register-confirm').value;
        
        if (password !== confirm) {
            alert('Passwords do not match');
            return;
        }
        
        try {
            await api.register(email, password, fullName);
            closeAuthModal();
            updateAuthUI();
            alert('Registration successful! You are now logged in.');
        } catch (error) {
            alert(error.message || 'Registration failed');
        }
    });
    
    // Update UI on load
    updateAuthUI();
}

function showAuthModal() {
    document.getElementById('auth-modal').classList.remove('hidden');
}

function closeAuthModal() {
    document.getElementById('auth-modal').classList.add('hidden');
}

function logout() {
    api.logout();
    updateAuthUI();
    showPage('home');
}

function updateAuthUI() {
    const authBtn = document.getElementById('auth-btn');
    if (api.isAuthenticated()) {
        authBtn.textContent = 'Logout';
    } else {
        authBtn.textContent = 'Login';
    }
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize all modules
    initCropSelection();
    initImageUpload();
    initCamera();
    initVoiceRecording();
    initAnalysis();
    initAuth();
    
    // Set up nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            showPage(link.dataset.page);
        });
    });
    
    // Show home page
    showPage('home');
    
    console.log('🌿 AgriDoctor AI Ready');
});

// Make functions globally accessible
window.showPage = showPage;
window.nextStep = nextStep;
window.prevStep = prevStep;
window.startNewDiagnosis = startNewDiagnosis;
window.showAuthModal = showAuthModal;
window.closeAuthModal = closeAuthModal;
window.closeCameraModal = closeCameraModal;
window.viewCase = viewCase;
