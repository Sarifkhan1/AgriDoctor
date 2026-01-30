/**
 * AgriDoctor AI - API Client
 * Handles all API communication with the backend
 */

const API_BASE = 'http://localhost:8000/api';

class APIClient {
    constructor() {
        this.token = localStorage.getItem('auth_token');
    }

    // ========================================================================
    // Request Helpers
    // ========================================================================

    async request(endpoint, options = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }

        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return response.json();
    }

    async uploadFile(endpoint, file, additionalData = {}) {
        const formData = new FormData();
        formData.append('file', file);

        for (const [key, value] of Object.entries(additionalData)) {
            formData.append(key, value);
        }

        const headers = {};
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }

        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers,
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `Upload failed: HTTP ${response.status}`);
        }

        return response.json();
    }

    setToken(token) {
        this.token = token;
        if (token) {
            localStorage.setItem('auth_token', token);
        } else {
            localStorage.removeItem('auth_token');
        }
    }

    // ========================================================================
    // Auth
    // ========================================================================

    async register(email, password) {
        const data = await this.request('/auth/register', {
            method: 'POST',
            body: JSON.stringify({ email, password })
        });
        this.setToken(data.access_token);
        return data;
    }

    async login(email, password) {
        const data = await this.request('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ email, password })
        });
        this.setToken(data.access_token);
        return data;
    }

    async getMe() {
        return this.request('/auth/me');
    }

    logout() {
        this.setToken(null);
    }

    isAuthenticated() {
        return !!this.token;
    }

    // ========================================================================
    // Cases
    // ========================================================================

    async createCase(category, cropName = null, animalType = null) {
        return this.request('/cases', {
            method: 'POST',
            body: JSON.stringify({
                category,
                crop_name: cropName,
                animal_type: animalType
            })
        });
    }

    async listCases(limit = 20, offset = 0) {
        return this.request(`/cases?limit=${limit}&offset=${offset}`);
    }

    async getCase(caseId) {
        return this.request(`/cases/${caseId}`);
    }

    async uploadImage(caseId, file) {
        return this.uploadFile(`/cases/${caseId}/media/image`, file);
    }

    async uploadSpeech(caseId, file) {
        return this.uploadFile(`/cases/${caseId}/media/speech`, file);
    }

    async updateMetadata(caseId, metadata) {
        return this.request(`/cases/${caseId}/metadata`, {
            method: 'POST',
            body: JSON.stringify(metadata)
        });
    }

    // ========================================================================
    // Inference
    // ========================================================================

    async runInference(caseId) {
        return this.request(`/cases/${caseId}/run`, {
            method: 'POST'
        });
    }

    async getJobStatus(jobId) {
        return this.request(`/jobs/${jobId}`);
    }

    async getResult(caseId) {
        return this.request(`/cases/${caseId}/result`);
    }

    async pollForResult(caseId, maxAttempts = 30, intervalMs = 1000) {
        for (let i = 0; i < maxAttempts; i++) {
            try {
                const caseData = await this.getCase(caseId);
                if (caseData.status === 'done') {
                    return this.getResult(caseId);
                }
                if (caseData.status === 'failed') {
                    throw new Error('Analysis failed');
                }
            } catch (e) {
                // Continue polling
            }
            await new Promise(resolve => setTimeout(resolve, intervalMs));
        }
        throw new Error('Analysis timed out');
    }

    // ========================================================================
    // Models
    // ========================================================================

    async listModels() {
        return this.request('/models');
    }
}

// Create global instance
window.api = new APIClient();
