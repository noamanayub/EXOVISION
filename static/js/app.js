// NASA Exoplanet Detective - 3D Enhanced JavaScript Application

class ExoplanetDetector {
    constructor() {
        this.initializeEventListeners();
        this.loadModelInfo();
        this.init3DBackground();
        this.initNavbarScroll();
        this.checkServerHealth();
    }
    
    async checkServerHealth() {
        try {
            const response = await fetch('/sample_data');
            if (response.ok) {
                console.log('Server is healthy and responding');
                // Enable sample data button
                const sampleBtn = document.querySelector('button[onclick="loadSampleData()"]');
                if (sampleBtn) {
                    sampleBtn.disabled = false;
                    sampleBtn.title = 'Load sample exoplanet data';
                }
            }
        } catch (error) {
            console.warn('Server health check failed:', error);
            // Disable sample data button with explanation
            const sampleBtn = document.querySelector('button[onclick="loadSampleData()"]');
            if (sampleBtn) {
                sampleBtn.disabled = true;
                sampleBtn.title = 'Server unavailable - please start the Flask app';
                sampleBtn.innerHTML = '<i class="fas fa-flask me-2"></i>Server Offline';
            }
        }
    }

    init3DBackground() {
        // Create 3D space background with Three.js
        const canvas = document.getElementById('space-canvas');
        if (!canvas) {
            console.log('Canvas element not found');
            return;
        }
        
        if (typeof THREE === 'undefined') {
            console.log('Three.js not loaded, using CSS background fallback');
            // Add fallback CSS animation class
            document.body.classList.add('no-threejs-fallback');
            return;
        }

        // Scene setup
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000, 0);

        // Create stars
        this.createStarField();
        
        // Create floating particles
        this.createParticles();
        
        // Position camera
        this.camera.position.z = 5;
        
        // Start animation
        this.animate3D();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    createStarField() {
        const starGeometry = new THREE.BufferGeometry();
        const starCount = 2000;
        const positions = new Float32Array(starCount * 3);
        const colors = new Float32Array(starCount * 3);
        
        for (let i = 0; i < starCount * 3; i += 3) {
            // Random positions in a sphere
            positions[i] = (Math.random() - 0.5) * 2000;
            positions[i + 1] = (Math.random() - 0.5) * 2000;
            positions[i + 2] = (Math.random() - 0.5) * 2000;
            
            // NASA-themed star colors (white to NASA blue)
            const intensity = Math.random() * 0.5 + 0.5;
            const nasaBlue = Math.random() * 0.3 + 0.2; // NASA Blue component
            colors[i] = intensity; // Red
            colors[i + 1] = intensity; // Green  
            colors[i + 2] = intensity + nasaBlue; // Blue (NASA Blue tint)
        }
        
        starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        starGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const starMaterial = new THREE.PointsMaterial({
            size: 2,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });
        
        this.stars = new THREE.Points(starGeometry, starMaterial);
        this.scene.add(this.stars);
    }

    createParticles() {
        const particleCount = 500;
        const particles = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 100;
            positions[i + 1] = (Math.random() - 0.5) * 100;
            positions[i + 2] = (Math.random() - 0.5) * 100;
        }
        
        particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const particleMaterial = new THREE.PointsMaterial({
            color: 0xFC3D21, // NASA Red
            size: 1,
            transparent: true,
            opacity: 0.6
        });
        
        this.particles = new THREE.Points(particles, particleMaterial);
        this.scene.add(this.particles);
    }

    animate3D() {
        requestAnimationFrame(() => this.animate3D());
        
        // Rotate stars slowly
        if (this.stars) {
            this.stars.rotation.x += 0.0005;
            this.stars.rotation.y += 0.0002;
        }
        
        // Animate particles
        if (this.particles) {
            this.particles.rotation.x += 0.001;
            this.particles.rotation.y += 0.002;
        }
        
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    initNavbarScroll() {
        // Add scroll effect to navbar
        window.addEventListener('scroll', () => {
            const navbar = document.querySelector('.navbar');
            if (window.scrollY > 100) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }

    initializeEventListeners() {
        // File upload event listeners
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // Drag and drop events
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });

        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    async loadModelInfo() {
        try {
            const response = await fetch('/model_info');
            const data = await response.json();
            
            if (data.success) {
                this.updateModelCount(data.total_models);
                this.updateAverageAccuracy(data.average_accuracy);
                this.displayModelCards(data.models);
                this.populateModelSelector(data.models, data.best_model);
            }
        } catch (error) {
            console.error('Error loading model info:', error);
            // Fallback to the /models endpoint if /model_info fails
            try {
                const fallbackResponse = await fetch('/models');
                const fallbackData = await fallbackResponse.json();
                this.updateModelCount(fallbackData.total_models);
                if (fallbackData.average_accuracy) {
                    this.updateAverageAccuracy(fallbackData.average_accuracy);
                }
                this.displayModelCards(fallbackData.models);
                this.populateModelSelector(fallbackData.models, fallbackData.best_model);
            } catch (fallbackError) {
                console.error('Error loading fallback model info:', fallbackError);
            }
        }
    }

    populateModelSelector(models, bestModel) {
        const modelSelect = document.getElementById('modelSelect');
        if (!modelSelect) return;

        // Clear existing options except the first one (All Models)
        while (modelSelect.children.length > 1) {
            modelSelect.removeChild(modelSelect.lastChild);
        }

        // Add individual model options
        Object.entries(models).forEach(([name, info]) => {
            if (info.loaded) {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name + (info.is_best ? ' (Best)' : '');
                if (info.is_best) {
                    option.style.fontWeight = 'bold';
                }
                modelSelect.appendChild(option);
            }
        });

        // Set default to best model
        if (bestModel) {
            modelSelect.value = bestModel;
        }
    }

    updateModelCount(count) {
        const modelsCountElement = document.getElementById('models-count');
        if (modelsCountElement) {
            modelsCountElement.textContent = count;
        }
    }

    updateAverageAccuracy(accuracy) {
        // Find the accuracy stat element and update it
        const statElements = document.querySelectorAll('.stat-number');
        statElements.forEach(element => {
            const label = element.nextElementSibling;
            if (label && (label.textContent.includes('Accuracy') || label.textContent.includes('Avg Accuracy'))) {
                element.textContent = Math.round(accuracy * 100) + '%';
            }
        });
    }

    displayModelCards(models) {
        const container = document.getElementById('modelsContainer');
        if (!container) return;

        container.innerHTML = '';

        const modelConfigs = {
            'SVM Fast': {
                icon: 'fas fa-vector-square',
                color: 'text-success',
                description: 'Support Vector Machine - Fast Training (Best Model for Exoplanet Detection)'
            },
            'LightGBM Fast': {
                icon: 'fas fa-bolt',
                color: 'text-warning',
                description: 'Light Gradient Boosting - Fast Training with high accuracy'
            },
            'Random Forest Fast': {
                icon: 'fas fa-tree',
                color: 'text-info',
                description: 'Random Forest - Fast Training ensemble method'
            },
            'Gradient Boosting Fast': {
                icon: 'fas fa-chart-line',
                color: 'text-primary',
                description: 'Gradient Boosting - Fast Training for sequential learning'
            },
            'Ensemble Fast': {
                icon: 'fas fa-users',
                color: 'text-secondary',
                description: 'Ensemble Voting - Fast Training combining multiple models'
            },
            'Best Model Fast': {
                icon: 'fas fa-crown',
                color: 'text-warning',
                description: 'Best Performing Model (SVM Fast) - Optimized for Exoplanet Detection'
            },
            // Legacy model configs for backward compatibility
            'Random Forest': {
                icon: 'fas fa-tree',
                color: 'text-success',
                description: 'Ensemble method using multiple decision trees for robust predictions with feature importance analysis.'
            },
            'LightGBM': {
                icon: 'fas fa-bolt',
                color: 'text-warning',
                description: 'Gradient boosting framework optimized for speed and accuracy with minimal memory usage.'
            },
            'Neural Network': {
                icon: 'fas fa-brain',
                color: 'text-info',
                description: 'Deep feedforward network with multiple hidden layers and dropout regularization.'
            },
            'CNN': {
                icon: 'fas fa-wave-square',
                color: 'text-primary',
                description: 'Convolutional Neural Network designed for time-series pattern recognition in light curves.'
            },
            'LSTM': {
                icon: 'fas fa-project-diagram',
                color: 'text-purple',
                description: 'Long Short-Term Memory network for capturing temporal dependencies in sequential data.'
            }
        };

        Object.entries(models).forEach(([name, info]) => {
            const config = modelConfigs[name] || {
                icon: 'fas fa-robot',
                color: 'text-secondary',
                description: 'Advanced machine learning model for exoplanet detection.'
            };

            const bestBadge = info.is_best ? '<span class="badge bg-warning text-dark ms-2"><i class="fas fa-crown me-1"></i>Best</span>' : '';
            
            // Performance metrics display
            const metricsHtml = info.accuracy ? `
                <div class="model-metrics">
                    <div class="metric">
                        <span class="metric-label">Accuracy:</span>
                        <span class="metric-value">${Math.round(info.accuracy * 100)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">F1-Score:</span>
                        <span class="metric-value">${info.f1_score ? info.f1_score.toFixed(3) : 'N/A'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Training:</span>
                        <span class="metric-value">${info.training_time || 'N/A'}</span>
                    </div>
                </div>
            ` : '';

            const cardHtml = `
                <div class="col-lg-4 col-md-6">
                    <div class="model-card ${info.is_best ? 'best-model' : ''}">
                        <div class="model-icon ${config.color}">
                            <i class="${config.icon}"></i>
                        </div>
                        <div class="model-title">${name}${bestBadge}</div>
                        <div class="model-description">${config.description}</div>
                        ${metricsHtml}
                        <div class="model-status ${info.loaded !== false ? 'status-loaded' : 'status-error'}">
                            <i class="fas ${info.loaded !== false ? 'fa-check-circle' : 'fa-exclamation-circle'} me-1"></i>
                            ${info.loaded !== false ? 'Loaded' : 'Error'}
                        </div>
                    </div>
                </div>
            `;
            container.innerHTML += cardHtml;
        });
    }

    async handleFileUpload(file) {
        // Validate file
        if (!this.validateFile(file)) {
            return;
        }

        // Show loading state
        this.showLoadingModal();
        
        try {
            // Create FormData
            const formData = new FormData();
            formData.append('file', file);
            
            // Add selected model
            const selectedModel = document.getElementById('modelSelect').value;
            formData.append('selected_model', selectedModel);

            // Upload and analyze
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.displayResults(data);
                this.scrollToResults();
            } else {
                this.showError(data.error || 'Upload failed');
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.showError('Network error occurred');
        } finally {
            this.hideLoadingModal();
        }
    }

    validateFile(file) {
        const allowedTypes = ['text/csv', 'text/plain', 'application/vnd.ms-excel'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.csv')) {
            this.showError('Please upload a CSV or TXT file');
            return false;
        }

        if (file.size > maxSize) {
            this.showError('File size must be less than 10MB');
            return false;
        }

        return true;
    }

    displayResults(data) {
        // Update file info
        this.updateFileInfo(data);
        
        // Display model results
        this.displayModelResults(data.results);
        
        // Show visualization
        if (data.plot_url) {
            this.displayVisualization(data.plot_url);
        }
        
        // Show consensus
        this.displayConsensus(data.results);
        
        // Show results section
        document.getElementById('results-section').style.display = 'block';
    }

    updateFileInfo(data) {
        document.getElementById('fileName').textContent = data.filename;
        document.getElementById('dataPoints').textContent = data.data_points.toLocaleString();
        document.getElementById('timeSpan').textContent = `${data.time_span.toFixed(1)} days`;
        
        // Count successful models
        const modelCount = Object.keys(data.results).filter(key => 
            key !== 'features' && key !== 'feature_count' && !data.results[key].error
        ).length;
        document.getElementById('modelCount').textContent = modelCount;
    }

    displayModelResults(results) {
        const container = document.getElementById('modelResults');
        container.innerHTML = '';

        // Filter out non-model keys
        const modelResults = Object.entries(results).filter(([key]) => 
            !['features', 'feature_count'].includes(key)
        );

        modelResults.forEach(([modelName, result]) => {
            if (result.error) {
                return; // Skip error models
            }

            const isExoplanet = result.prediction === 1;
            const confidence = (result.confidence * 100).toFixed(1);
            
            const cardHtml = `
                <div class="model-result-card ${isExoplanet ? 'exoplanet' : 'no-exoplanet'} fade-in-up">
                    <div class="row align-items-center">
                        <div class="col-md-8">
                            <div class="model-name">
                                <i class="fas ${this.getModelIcon(modelName)} me-2"></i>
                                ${modelName}
                            </div>
                            <div class="model-prediction ${isExoplanet ? 'text-success' : 'text-danger'}">
                                <i class="fas ${isExoplanet ? 'fa-check-circle' : 'fa-times-circle'} me-2"></i>
                                ${result.prediction_text}
                            </div>
                            <div class="model-confidence">
                                Confidence: ${confidence}%
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="confidence-bar">
                                <div class="confidence-fill ${isExoplanet ? 'bg-success' : 'bg-danger'}" 
                                     style="width: ${confidence}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            container.innerHTML += cardHtml;
        });
    }

    getModelIcon(modelName) {
        const icons = {
            'Random Forest': 'fa-tree',
            'LightGBM': 'fa-bolt',
            'Neural Network': 'fa-brain',
            'CNN': 'fa-wave-square',
            'LSTM': 'fa-project-diagram'
        };
        return icons[modelName] || 'fa-robot';
    }

    displayVisualization(plotUrl) {
        const container = document.getElementById('visualizationContainer');
        container.innerHTML = `
            <img src="${plotUrl}" alt="Analysis Visualization" class="img-fluid fade-in-up">
        `;
    }

    displayConsensus(results) {
        const container = document.getElementById('consensusCard');
        
        // Calculate consensus
        const modelResults = Object.entries(results).filter(([key]) => 
            !['features', 'feature_count'].includes(key)
        );
        
        const exoplanetVotes = modelResults.filter(([_, result]) => 
            !result.error && result.prediction === 1
        ).length;
        
        const totalModels = modelResults.filter(([_, result]) => !result.error).length;
        const isConsensusExoplanet = exoplanetVotes >= totalModels / 2;
        
        const avgConfidence = modelResults
            .filter(([_, result]) => !result.error)
            .reduce((sum, [_, result]) => sum + result.confidence, 0) / totalModels;

        const consensusHtml = `
            <div class="consensus-result">
                <div class="consensus-icon ${isConsensusExoplanet ? 'consensus-exoplanet' : 'consensus-no-exoplanet'}">
                    <i class="fas ${isConsensusExoplanet ? 'fa-globe' : 'fa-star'}"></i>
                </div>
                <div class="consensus-text ${isConsensusExoplanet ? 'consensus-exoplanet' : 'consensus-no-exoplanet'}">
                    ${isConsensusExoplanet ? 'EXOPLANET DETECTED' : 'NO EXOPLANET DETECTED'}
                </div>
                <div class="consensus-details">
                    ${exoplanetVotes}/${totalModels} models detected an exoplanet<br>
                    Average confidence: ${(avgConfidence * 100).toFixed(1)}%
                </div>
            </div>
        `;
        
        container.innerHTML = consensusHtml;
    }

    async loadSampleData() {
        this.showLoadingModal();
        
        try {
            console.log('Loading sample data...');
            const response = await fetch('/sample_data');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Sample data response:', data);
            
            if (data.success) {
                // Get selected model
                const selectedModel = document.getElementById('modelSelect').value;
                
                // Predict on sample data
                console.log('Sending prediction request...');
                const predictResponse = await fetch('/predict_sample', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        time: data.time,
                        flux: data.flux,
                        selected_model: selectedModel
                    })
                });
                
                if (!predictResponse.ok) {
                    throw new Error(`Prediction HTTP error! status: ${predictResponse.status}`);
                }
                
                const predictData = await predictResponse.json();
                console.log('Prediction response:', predictData);
                
                if (predictData.success) {
                    // Add sample info to results
                    predictData.filename = `${data.star_id} (Sample Data)`;
                    predictData.data_points = data.time.length;
                    predictData.time_span = data.time[data.time.length - 1] - data.time[0];
                    predictData.selected_model = selectedModel;
                    
                    this.displayResults(predictData);
                    this.scrollToResults();
                    
                    // Show true label if available
                    if (data.true_label !== undefined) {
                        this.showSampleTrueLabel(data.true_label);
                    }
                } else {
                    this.showError(predictData.error || 'Prediction failed');
                }
            } else {
                this.showError(data.error || 'Failed to load sample data');
            }
        } catch (error) {
            console.error('Sample data error:', error);
            
            // Provide more specific error messages
            if (error.message.includes('Failed to fetch')) {
                this.showError('Unable to connect to server. Please ensure the Flask app is running.');
            } else if (error.message.includes('HTTP error')) {
                this.showError(`Server error: ${error.message}. Check server logs for details.`);
            } else {
                this.showError(`Failed to load sample data: ${error.message}`);
            }
        } finally {
            this.hideLoadingModal();
        }
    }

    showSampleTrueLabel(trueLabel) {
        const consensusCard = document.getElementById('consensusCard');
        const trueLabelHtml = `
            <div class="mt-3 p-3 rounded" style="background: rgba(255, 193, 7, 0.1); border: 1px solid #ffc107;">
                <h6 class="text-warning mb-2">
                    <i class="fas fa-info-circle me-2"></i>True Label (Sample Data)
                </h6>
                <p class="mb-0">
                    This star ${trueLabel === 1 ? 'has' : 'does not have'} an exoplanet according to the training data.
                </p>
            </div>
        `;
        consensusCard.innerHTML += trueLabelHtml;
    }

    generateTestData() {
        // Generate synthetic light curve data for testing
        const timePoints = 1000;
        const time = Array.from({length: timePoints}, (_, i) => i * 0.365);
        
        // Create a light curve with a transit
        const flux = time.map(t => {
            let baseFlux = 1.0 + 0.01 * (Math.random() - 0.5);
            
            // Add transit every ~50 days
            const period = 50;
            const transitDuration = 3; // 3 days
            const transitDepth = 0.01; // 1% depth
            
            const phase = (t % period);
            if (phase < transitDuration) {
                baseFlux *= (1 - transitDepth);
            }
            
            // Add noise
            baseFlux += 0.001 * (Math.random() - 0.5);
            
            return baseFlux;
        });

        this.showLoadingModal();
        
        // Simulate API call
        setTimeout(async () => {
            try {
                const response = await fetch('/predict_sample', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        time: time,
                        flux: flux
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    data.filename = 'Generated Test Data (with simulated exoplanet)';
                    data.data_points = time.length;
                    data.time_span = time[time.length - 1] - time[0];
                    
                    this.displayResults(data);
                    this.scrollToResults();
                } else {
                    this.showError(data.error || 'Prediction failed');
                }
            } catch (error) {
                console.error('Test data error:', error);
                this.showError('Failed to generate test data');
            } finally {
                this.hideLoadingModal();
            }
        }, 1000);
    }

    showLoadingModal() {
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }

    hideLoadingModal() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
    }

    scrollToResults() {
        setTimeout(() => {
            document.getElementById('results-section').scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }, 500);
    }

    showError(message) {
        // Create error alert
        const alertHtml = `
            <div class="alert alert-danger alert-dismissible fade show position-fixed" 
                 style="top: 100px; right: 20px; z-index: 9999; max-width: 400px;" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Error:</strong> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alert = document.querySelector('.alert-danger');
            if (alert) {
                alert.remove();
            }
        }, 5000);
    }

    showSuccess(message) {
        const alertHtml = `
            <div class="alert alert-success alert-dismissible fade show position-fixed" 
                 style="top: 100px; right: 20px; z-index: 9999; max-width: 400px;" role="alert">
                <i class="fas fa-check-circle me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', alertHtml);
        
        setTimeout(() => {
            const alert = document.querySelector('.alert-success');
            if (alert) {
                alert.remove();
            }
        }, 3000);
    }
}

// Global functions for HTML onclick events
function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

function loadSampleData() {
    window.detector.loadSampleData();
}

function generateTestData() {
    window.detector.generateTestData();
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.detector = new ExoplanetDetector();
    
    // Add some interactive animations
    addScrollAnimations();
    addParallaxEffect();
});

function addScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
            }
        });
    }, {
        threshold: 0.1
    });

    // Observe elements that should animate on scroll
    document.querySelectorAll('.feature-card, .model-card, .info-card').forEach(el => {
        observer.observe(el);
    });
}

function addParallaxEffect() {
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const parallax = document.querySelector('.hero-background');
        
        if (parallax) {
            const speed = scrolled * 0.5;
            parallax.style.transform = `translateY(${speed}px)`;
        }
    });
}

// Add some easter eggs and interactive features
document.addEventListener('keydown', (e) => {
    // Konami code for fun
    const konamiCode = [38, 38, 40, 40, 37, 39, 37, 39, 66, 65];
    window.konamiProgress = window.konamiProgress || 0;
    
    if (e.keyCode === konamiCode[window.konamiProgress]) {
        window.konamiProgress++;
        if (window.konamiProgress === konamiCode.length) {
            // Easter egg activated
            document.body.style.filter = 'hue-rotate(180deg)';
            setTimeout(() => {
                document.body.style.filter = '';
            }, 3000);
            window.konamiProgress = 0;
        }
    } else {
        window.konamiProgress = 0;
    }
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch(e.key) {
            case 'u':
                e.preventDefault();
                scrollToSection('upload-section');
                break;
            case 'r':
                e.preventDefault();
                if (document.getElementById('results-section').style.display !== 'none') {
                    scrollToSection('results-section');
                }
                break;
        }
    }
});

// Add dynamic background stars
function createStars() {
    const starsContainer = document.createElement('div');
    starsContainer.className = 'stars-background';
    starsContainer.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    `;
    
    for (let i = 0; i < 100; i++) {
        const star = document.createElement('div');
        star.style.cssText = `
            position: absolute;
            width: 2px;
            height: 2px;
            background: white;
            border-radius: 50%;
            top: ${Math.random() * 100}%;
            left: ${Math.random() * 100}%;
            opacity: ${Math.random() * 0.8 + 0.2};
            animation: twinkle ${Math.random() * 3 + 2}s infinite;
        `;
        starsContainer.appendChild(star);
    }
    
    document.body.appendChild(starsContainer);
}

// Add twinkle animation
const style = document.createElement('style');
style.textContent = `
    @keyframes twinkle {
        0%, 100% { opacity: 0.2; }
        50% { opacity: 1; }
    }
`;
document.head.appendChild(style);

// Create stars on load
setTimeout(createStars, 1000);