/**
 * AgriDoctor AI - Main Application
 * Flow control + state. Diagnosis uses the real AI via api.analyze().
 * Result rendering is delegated to UI (ui.js) which is XSS-safe.
 */

const state = {
  currentPage: 'home',
  currentStep: 1,
  selectedCrop: null, // null = auto-detect
  selectedImage: null,
  imageUrl: null,
  audioBlob: null,
  lastResult: null,
};

const ONSET_TEXT = {
  '0': 'Today', '1': 'Yesterday', '3': '2-3 days ago',
  '7': 'About a week ago', '14': '2 weeks ago', '30': 'More than 2 weeks',
};

// ============================================================================
// Navigation
// ============================================================================
function showPage(pageName) {
  state.currentPage = pageName;
  document.querySelectorAll('[data-page]').forEach((link) => {
    link.classList.toggle('active', link.dataset.page === pageName);
  });
  document.querySelectorAll('.page').forEach((page) => {
    page.classList.toggle('active', page.id === `page-${pageName}`);
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (pageName === 'diagnose') resetDiagnosis();
  else if (pageName === 'history') loadHistory();
}

function showStep(stepNum) {
  state.currentStep = stepNum;
  document.querySelectorAll('.step').forEach((step, idx) => {
    step.classList.toggle('active', idx + 1 === stepNum);
    step.classList.toggle('completed', idx + 1 < stepNum);
  });
  document.querySelectorAll('.diagnose-step').forEach((step) => {
    step.classList.toggle('hidden', step.id !== `step-${stepNum}`);
  });
}

function nextStep() { if (state.currentStep < 4) showStep(state.currentStep + 1); }
function prevStep() { if (state.currentStep > 1) showStep(state.currentStep - 1); }

function resetDiagnosis() {
  state.currentStep = 1;
  state.selectedCrop = null;
  state.selectedImage = null;
  state.imageUrl = null;
  state.audioBlob = null;

  document.querySelectorAll('.crop-card').forEach((c) => c.classList.remove('selected'));
  document.getElementById('upload-placeholder')?.classList.remove('hidden');
  document.getElementById('image-preview')?.classList.add('hidden');
  const prev = document.getElementById('preview-img'); if (prev) prev.src = '';
  const nb = document.getElementById('next-step-2'); if (nb) nb.disabled = true;
  document.getElementById('audio-playback')?.classList.add('hidden');
  const onset = document.getElementById('onset-days'); if (onset) onset.value = '';
  document.querySelectorAll('input[name="spread"]').forEach((r) => (r.checked = false));
  const notes = document.getElementById('notes'); if (notes) notes.value = '';
  showStep(1);
}

function startNewDiagnosis() { showPage('diagnose'); }

// ============================================================================
// Crop selection (optional — AI auto-detects)
// ============================================================================
function initCropSelection() {
  document.querySelectorAll('.crop-card').forEach((card) => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.crop-card').forEach((c) => c.classList.remove('selected'));
      card.classList.add('selected');
      state.selectedCrop = card.dataset.crop;
      setTimeout(nextStep, 250);
    });
  });
  const skip = document.getElementById('skip-crop');
  if (skip) skip.addEventListener('click', () => { state.selectedCrop = null; nextStep(); });
}

// ============================================================================
// Image upload
// ============================================================================
function initImageUpload() {
  const uploadArea = document.getElementById('upload-area');
  const uploadPlaceholder = document.getElementById('upload-placeholder');
  const imageInput = document.getElementById('image-input');
  const imagePreview = document.getElementById('image-preview');
  const removeBtn = document.getElementById('remove-image');
  const nextBtn = document.getElementById('next-step-2');
  if (!uploadArea) return;

  uploadPlaceholder.addEventListener('click', () => imageInput.click());
  imageInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleImageFile(e.target.files[0]);
  });
  uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault(); uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleImageFile(e.dataTransfer.files[0]);
  });
  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    state.selectedImage = null; state.imageUrl = null;
    uploadPlaceholder.classList.remove('hidden');
    imagePreview.classList.add('hidden');
    document.getElementById('preview-img').src = '';
    nextBtn.disabled = true; imageInput.value = '';
  });
  nextBtn.addEventListener('click', nextStep);
}

function handleImageFile(file) {
  if (!file.type.startsWith('image/')) { alert('Please choose an image file.'); return; }
  if (file.size > 10 * 1024 * 1024) { alert('Image is larger than 10 MB. Please choose a smaller photo.'); return; }
  state.selectedImage = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    state.imageUrl = e.target.result;
    document.getElementById('preview-img').src = e.target.result;
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
  document.getElementById('camera-btn')?.addEventListener('click', openCamera);
  document.getElementById('capture-btn')?.addEventListener('click', capturePhoto);
}
async function openCamera() {
  const modal = document.getElementById('camera-modal');
  const video = document.getElementById('camera-video');
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = cameraStream;
    modal.classList.remove('hidden');
  } catch (err) { alert('Unable to access camera. Please upload an image instead.'); }
}
function closeCameraModal() {
  const modal = document.getElementById('camera-modal');
  const video = document.getElementById('camera-video');
  if (cameraStream) { cameraStream.getTracks().forEach((t) => t.stop()); cameraStream = null; }
  video.srcObject = null; modal.classList.add('hidden');
}
function capturePhoto() {
  const video = document.getElementById('camera-video');
  const canvas = document.getElementById('camera-canvas');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    handleImageFile(new File([blob], 'capture.jpg', { type: 'image/jpeg' }));
    closeCameraModal();
  }, 'image/jpeg', 0.9);
}

// ============================================================================
// Voice recording
// ============================================================================
let mediaRecorder = null;
let audioChunks = [];
function initVoiceRecording() {
  const recordBtn = document.getElementById('record-btn');
  if (!recordBtn) return;
  recordBtn.addEventListener('mousedown', startRecording);
  recordBtn.addEventListener('mouseup', stopRecording);
  recordBtn.addEventListener('mouseleave', stopRecording);
  recordBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
  recordBtn.addEventListener('touchend', (e) => { e.preventDefault(); stopRecording(); });
}
async function startRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      state.audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const audio = document.getElementById('audio-playback');
      audio.src = URL.createObjectURL(state.audioBlob);
      audio.classList.remove('hidden');
      stream.getTracks().forEach((t) => t.stop());
    };
    mediaRecorder.start();
    document.getElementById('recording-indicator')?.classList.remove('hidden');
    const rt = document.querySelector('.record-text'); if (rt) rt.textContent = 'Recording…';
  } catch (err) { alert('Unable to access microphone.'); }
}
function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    document.getElementById('recording-indicator')?.classList.add('hidden');
    const rt = document.querySelector('.record-text'); if (rt) rt.textContent = 'Hold to Record';
  }
}

// ============================================================================
// Analysis (the real AI call)
// ============================================================================
function initAnalysis() {
  document.getElementById('analyze-btn')?.addEventListener('click', runAnalysis);
}

const LOADING_STAGES = [
  'Uploading your photo…',
  'Checking the image is a plant leaf…',
  'Identifying the crop species…',
  'Looking for disease symptoms…',
  'Assessing severity…',
  'Writing your treatment plan…',
  'Almost there — finalising the diagnosis…',
];
let _loadingTimer = null;
function startLoadingStages() {
  const el = document.getElementById('loading-stage');
  if (!el) return;
  let i = 0;
  el.textContent = LOADING_STAGES[0];
  _loadingTimer = setInterval(() => {
    i = Math.min(i + 1, LOADING_STAGES.length - 1);
    el.textContent = LOADING_STAGES[i];
  }, 3500);
}
function stopLoadingStages() {
  if (_loadingTimer) { clearInterval(_loadingTimer); _loadingTimer = null; }
}

async function runAnalysis() {
  if (!state.selectedImage) { alert('Please add a leaf photo first.'); showStep(2); return; }

  const overlay = document.getElementById('loading-overlay');
  overlay?.classList.remove('hidden');
  startLoadingStages();

  const payload = {
    image: state.selectedImage,
    cropHint: state.selectedCrop || undefined,
    onsetDays: document.getElementById('onset-days')?.value || undefined,
    spread: document.querySelector('input[name="spread"]:checked')?.value || undefined,
    notes: document.getElementById('notes')?.value || undefined,
  };
  if (state.audioBlob) {
    payload.audio = new File([state.audioBlob], 'voice.webm', { type: 'audio/webm' });
  }

  const container = document.getElementById('results-container');
  try {
    const result = await api.analyze(payload);
    state.lastResult = result;
    UI.renderResult(container, result, buildCtx(result));
    showStep(4);
  } catch (error) {
    console.error('Analysis error:', error);
    let msg = error.message || 'Please try again.';
    if (error.status === 429) msg = "You're going too fast — wait a moment and try again.";
    if (error.status === 503) msg = 'The AI service is busy right now. Please retry shortly.';
    UI.renderError(container, msg);
    showStep(4);
  } finally {
    stopLoadingStages();
    overlay?.classList.add('hidden');
  }
}

function buildCtx(result) {
  const onset = document.getElementById('onset-days')?.value || '';
  return {
    imageUrl: state.imageUrl,
    onsetText: onset ? (ONSET_TEXT[onset] || `${onset} days ago`) : '',
    spread: document.querySelector('input[name="spread"]:checked')?.value || '',
  };
}

// ============================================================================
// Image modal
// ============================================================================
function openImageModal(url) {
  if (!url) return;
  document.getElementById('full-image').src = url;
  document.getElementById('image-viewer-modal').classList.remove('hidden');
}
function closeImageModal() {
  document.getElementById('image-viewer-modal').classList.add('hidden');
}

// ============================================================================
// History
// ============================================================================
async function loadHistory() {
  const container = document.getElementById('history-list');
  if (!api.isAuthenticated()) {
    container.innerHTML = '';
    const card = document.createElement('div');
    card.className = 'empty-state';
    card.innerHTML = `<span class="empty-icon">🔒</span><h3>Login required</h3>
      <p>Log in to save and view your diagnosis history.</p>`;
    const btn = document.createElement('button');
    btn.className = 'btn btn-primary'; btn.textContent = 'Login';
    btn.addEventListener('click', showAuthModal);
    card.appendChild(btn); container.appendChild(card);
    return;
  }
  try {
    const cases = await api.listCases();
    if (!cases.length) {
      container.innerHTML = `<div class="empty-state"><span class="empty-icon">📋</span>
        <h3>No diagnoses yet</h3><p>Start your first diagnosis to see it here.</p></div>`;
      return;
    }
    container.innerHTML = '';
    cases.forEach((c) => {
      const emoji = UI.CROP_EMOJI[(c.crop_name || c.detected_crop || '').toLowerCase()] || '🌱';
      const title = c.disease_name || UI.titleCase(c.detected_crop || c.crop_name || 'Crop');
      const row = document.createElement('div');
      row.className = 'history-card';
      row.appendChild(elText('div', 'history-thumb', emoji));
      const info = document.createElement('div'); info.className = 'history-info';
      info.appendChild(elText('div', 'history-title', title));
      info.appendChild(elText('div', 'history-meta', `${formatDate(c.created_at)} · ${c.kind || c.status || ''}`));
      row.appendChild(info);
      row.appendChild(elText('span', `history-badge ${badgeClass(c.kind)}`, c.kind || c.status || ''));
      row.addEventListener('click', () => viewCase(c.id));
      container.appendChild(row);
    });
  } catch (error) {
    container.innerHTML = `<div class="empty-state"><span class="empty-icon">⚠️</span>
      <h3>Unable to load history</h3><p>Please try again later.</p></div>`;
  }
}

function elText(tag, cls, text) {
  const n = document.createElement(tag); n.className = cls; n.textContent = text; return n;
}
function formatDate(s) {
  const d = new Date(s + (s && s.endsWith('Z') ? '' : 'Z'));
  return isNaN(d) ? '' : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
function badgeClass(kind) {
  return { diagnosis: 'badge-mild', healthy: 'badge-healthy', unsupported_crop: 'badge-severe' }[kind] || 'badge-mild';
}

async function viewCase(caseId) {
  try {
    const data = await api.getCase(caseId);
    const p = data.prediction;
    if (!p) { alert('No result stored for this case.'); return; }
    const media = (data.media || [])[0];
    let imageUrl = null;
    if (media) { try { imageUrl = await api.fetchMediaObjectURL(media.id); } catch (_) {} }
    showPage('diagnose');
    UI.renderResult(document.getElementById('results-container'), p, { imageUrl });
    showStep(4);
  } catch (e) { alert('Unable to load this case.'); }
}

// ============================================================================
// Auth
// ============================================================================
function initAuth() {
  const authBtn = document.getElementById('auth-btn');
  const loginForm = document.getElementById('login-form');
  const registerForm = document.getElementById('register-form');

  authBtn.addEventListener('click', () => (api.isAuthenticated() ? logout() : showAuthModal()));

  document.querySelectorAll('.auth-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.auth-tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const isLogin = tab.dataset.tab === 'login';
      loginForm.classList.toggle('hidden', !isLogin);
      registerForm.classList.toggle('hidden', isLogin);
    });
  });

  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      await api.login(document.getElementById('login-email').value, document.getElementById('login-password').value);
      closeAuthModal(); updateAuthUI();
    } catch (error) { alert(error.message || 'Login failed'); }
  });

  registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const pw = document.getElementById('register-password').value;
    if (pw !== document.getElementById('register-confirm').value) { alert('Passwords do not match'); return; }
    if (pw.length < 8) { alert('Password must be at least 8 characters'); return; }
    try {
      await api.register(document.getElementById('register-email').value, pw, document.getElementById('register-name').value);
      closeAuthModal(); updateAuthUI();
    } catch (error) { alert(error.message || 'Registration failed'); }
  });

  updateAuthUI();
}
function showAuthModal() { document.getElementById('auth-modal').classList.remove('hidden'); }
function closeAuthModal() { document.getElementById('auth-modal').classList.add('hidden'); }
function logout() { api.logout(); updateAuthUI(); showPage('home'); }
function updateAuthUI() {
  const authBtn = document.getElementById('auth-btn');
  if (authBtn) authBtn.textContent = api.isAuthenticated() ? 'Logout' : 'Login';
}

// ============================================================================
// Theme Toggle
// ============================================================================
function initThemeToggle() {
  const themeToggle = document.getElementById('theme-toggle');
  if (!themeToggle) return;

  themeToggle.addEventListener('click', () => {
    const nextTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', nextTheme);
    localStorage.setItem('theme', nextTheme);
  });
}

// ============================================================================
// Init
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initCropSelection();
  initImageUpload();
  initCamera();
  initVoiceRecording();
  initAnalysis();
  initAuth();

  document.querySelectorAll('[data-page]').forEach((link) => {
    link.addEventListener('click', (e) => { e.preventDefault(); showPage(link.dataset.page); });
  });

  showPage('home');

  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js').catch(() => {}));
  }
  console.log('🌿 AgriDoctor AI ready');
});

// Expose handlers used by inline onclick in index.html
Object.assign(window, {
  showPage, nextStep, prevStep, startNewDiagnosis, showAuthModal,
  closeAuthModal, closeCameraModal, closeImageModal, openImageModal, viewCase,
});
