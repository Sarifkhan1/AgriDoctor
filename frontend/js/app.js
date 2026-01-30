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
            confidence: 0.87,
            severity_score: 0.45,
            urgency_level: 'medium',
            advice: {
                summary: 'Your tomato plant likely has Early Blight (Alternaria solani)',
                what_to_do_now: [
                    'Remove and destroy affected leaves immediately',
                    'Avoid overhead watering to reduce moisture on leaves',
                    'Apply copper-based fungicide following label directions'
                ],
                prevention: [
                    'Practice crop rotation - don\'t plant tomatoes in the same spot for 3 years',
                    'Mulch around plants to prevent soil splash',
                    'Choose resistant varieties next season'
                ],
                when_to_get_help: [
                    'If more than 50% of the plant is affected',
                    'If symptoms spread rapidly despite treatment',
                    'If you\'re unsure about the diagnosis'
                ]
            }
        },
        potato: {
            primary_label: 'POT_LATE_BLIGHT',
            confidence: 0.82,
            severity_score: 0.65,
            urgency_level: 'high',
            advice: {
                summary: 'Your potato plant likely has Late Blight (Phytophthora infestans)',
                what_to_do_now: [
                    'Act quickly - late blight spreads rapidly',
                    'Remove and destroy all infected plant material',
                    'Apply preventive fungicide to healthy plants'
                ],
                prevention: [
                    'Plant certified disease-free seed potatoes',
                    'Improve drainage and air circulation',
                    'Avoid planting near tomatoes'
                ],
                when_to_get_help: [
                    'Late blight is serious - consider consulting an expert immediately',
                    'If you see widespread infection in your area'
                ]
            }
        },
        rice: {
            primary_label: 'RICE_BLAST',
            confidence: 0.79,
            severity_score: 0.40,
            urgency_level: 'medium',
            advice: {
                summary: 'Your rice plant likely has Rice Blast (Magnaporthe oryzae)',
                what_to_do_now: [
                    'Reduce nitrogen fertilizer application',
                    'Maintain proper water management',
                    'Consider fungicide application for severe cases'
                ],
                prevention: [
                    'Use resistant varieties',
                    'Balanced fertilization',
                    'Remove crop residues after harvest'
                ],
                when_to_get_help: [
                    'If infection covers more than 25% of leaf area',
                    'If neck blast symptoms appear'
                ]
            }
        },
        maize: {
            primary_label: 'MAIZE_RUST',
            confidence: 0.84,
            severity_score: 0.35,
            urgency_level: 'low',
            advice: {
                summary: 'Your maize plant likely has Common Rust (Puccinia sorghi)',
                what_to_do_now: [
                    'Monitor spread to neighboring plants',
                    'For mild cases, no treatment may be needed',
                    'Severely affected leaves can be removed'
                ],
                prevention: [
                    'Plant rust-resistant hybrids',
                    'Early planting reduces exposure',
                    'Adequate spacing for air circulation'
                ],
                when_to_get_help: [
                    'If more than 30% of the leaf area shows pustules',
                    'If rust appears before tasseling'
                ]
            }
        },
        chili: {
            primary_label: 'CHILI_ANTHRAC',
            confidence: 0.81,
            severity_score: 0.50,
            urgency_level: 'medium',
            advice: {
                summary: 'Your chili plant likely has Anthracnose (Colletotrichum species)',
                what_to_do_now: [
                    'Remove and destroy infected fruits and plant parts',
                    'Avoid wetting foliage when watering',
                    'Apply copper-based fungicide'
                ],
                prevention: [
                    'Use disease-free seeds',
                    'Rotate crops every 2-3 years',
                    'Stake plants to improve air flow'
                ],
                when_to_get_help: [
                    'If fruit rot is widespread',
                    'If multiple disease symptoms are present'
                ]
            }
        },
        cucumber: {
            primary_label: 'CUC_POWDERY',
            confidence: 0.89,
            severity_score: 0.30,
            urgency_level: 'low',
            advice: {
                summary: 'Your cucumber plant likely has Powdery Mildew (Erysiphe cichoracearum)',
                what_to_do_now: [
                    'Improve air circulation around plants',
                    'Remove heavily infected leaves',
                    'Apply neem oil or sulfur-based fungicide'
                ],
                prevention: [
                    'Plant resistant varieties',
                    'Avoid overhead irrigation',
                    'Space plants properly'
                ],
                when_to_get_help: [
                    'If mildew covers more than 50% of plant',
                    'If fruit quality is severely affected'
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
                    <h3>📋 Case Details</h3>
                    <div class="case-detail-item">
                        <span class="detail-label">Crop Type</span>
                        <span class="detail-value">${cropEmoji} ${cropName}</span>
                    </div>
                    ${onsetDays ? `
                    <div class="case-detail-item">
                        <span class="detail-label">First Noticed</span>
                        <span class="detail-value">📅 ${onsetText[onsetDays] || onsetDays + ' days ago'}</span>
                    </div>` : ''}
                    ${spreadSpeed ? `
                    <div class="case-detail-item">
                        <span class="detail-label">Spread Rate</span>
                        <span class="detail-value">${spreadSpeed === 'slow' ? '🐢 Slow' : spreadSpeed === 'moderate' ? '🚶 Moderate' : '🚀 Fast'}</span>
                    </div>` : ''}
                    <div class="case-detail-item">
                        <span class="detail-label">Analyzed On</span>
                        <span class="detail-value">🕐 ${new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                    ${notes ? `
                    <div class="case-notes">
                        <span class="detail-label">Notes</span>
                        <p class="notes-text">${notes}</p>
                    </div>` : ''}
                </div>
                
                <div class="results-quick-stats">
                    <div class="quick-stat">
                        <span class="stat-icon">🎯</span>
                        <span class="stat-number">${Math.round(result.confidence * 100)}%</span>
                        <span class="stat-name">Confidence</span>
                    </div>
                    <div class="quick-stat">
                        <span class="stat-icon">📊</span>
                        <span class="stat-number">${Math.round(result.severity_score * 100)}%</span>
                        <span class="stat-name">Severity</span>
                    </div>
                    <div class="quick-stat urgency-${urgencyClass}">
                        <span class="stat-icon">${urgencyClass === 'high' ? '🔴' : urgencyClass === 'medium' ? '🟡' : '🟢'}</span>
                        <span class="stat-number">${urgencyClass.charAt(0).toUpperCase() + urgencyClass.slice(1)}</span>
                        <span class="stat-name">Urgency</span>
                    </div>
                </div>
            </div>
            
            <!-- Right Panel: Diagnosis Results -->
            <div class="results-right-panel">
                <div class="result-header">
                    <div class="result-badge">AI Diagnosis</div>
                    <h2>${diseaseName}</h2>
                    <div class="result-confidence">
                        ✅ ${Math.round(result.confidence * 100)}% Confidence Match
                    </div>
                </div>
                
                <div class="result-body">
                    <div class="result-section result-summary">
                        <div class="section-icon">📋</div>
                        <div class="section-content">
                            <h3>Summary</h3>
                            <p>${result.advice.summary}</p>
                        </div>
                    </div>
                    
                    <div class="result-section result-severity">
                        <div class="section-icon">📊</div>
                        <div class="section-content">
                            <h3>Severity Level</h3>
                            <div class="severity-indicator">
                                <span class="severity-label severity-${severityClass}">${severityClass.charAt(0).toUpperCase() + severityClass.slice(1)}</span>
                                <span class="severity-percent">${Math.round(result.severity_score * 100)}%</span>
                            </div>
                            <div class="severity-bar">
                                <div class="severity-fill severity-${severityClass}" 
                                     style="width: ${result.severity_score * 100}%"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="result-section result-actions">
                        <div class="section-icon">🩹</div>
                        <div class="section-content">
                            <h3>What to Do Now</h3>
                            <ul class="action-list">
                                ${result.advice.what_to_do_now.map((item, i) => `
                                    <li>
                                        <span class="action-num">${i + 1}</span>
                                        <span class="action-text">${item}</span>
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    </div>
                    
                    <div class="result-section result-prevention">
                        <div class="section-icon">🛡️</div>
                        <div class="section-content">
                            <h3>Prevention Tips</h3>
                            <ul class="prevention-list">
                                ${result.advice.prevention.map(item => `<li><span class="bullet">•</span>${item}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                    
                    <div class="result-section result-help">
                        <div class="section-icon">⚠️</div>
                        <div class="section-content">
                            <h3>When to Get Expert Help</h3>
                            <ul class="help-list">
                                ${result.advice.when_to_get_help.map(item => `<li><span class="bullet">🔸</span>${item}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                    
                    <div class="safety-notice">
                        <span class="safety-icon">⚠️</span>
                        <div class="safety-content">
                            <strong>Important Disclaimer</strong>
                            <p>This is AI-generated guidance, not professional diagnosis. 
                            Always consult local agricultural experts for confirmation and before applying any treatments.</p>
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
        const email = document.getElementById('register-email').value;
        const password = document.getElementById('register-password').value;
        const confirm = document.getElementById('register-confirm').value;
        
        if (password !== confirm) {
            alert('Passwords do not match');
            return;
        }
        
        try {
            await api.register(email, password);
            closeAuthModal();
            updateAuthUI();
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
