// Main JavaScript for RD Sharma Question Extractor

document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in animation to main content
    const mainContent = document.querySelector('main');
    if (mainContent) {
        mainContent.classList.add('fade-in');
    }

    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // File upload enhancement
    const fileInput = document.getElementById('pdf_file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const fileSize = (file.size / (1024 * 1024)).toFixed(2);
                const fileName = file.name;
                
                // Create file info display
                const fileInfo = document.createElement('div');
                fileInfo.className = 'alert alert-info mt-2';
                fileInfo.innerHTML = `
                    <i class="fas fa-file-pdf me-2"></i>
                    <strong>${fileName}</strong> (${fileSize} MB)
                `;
                
                // Remove previous file info if exists
                const existingInfo = fileInput.parentNode.querySelector('.alert');
                if (existingInfo) {
                    existingInfo.remove();
                }
                
                fileInput.parentNode.appendChild(fileInfo);
            }
        });
    }

    // Form validation enhancement
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const chapter = document.getElementById('chapter').value;
            const topic = document.getElementById('topic').value;
            const file = document.getElementById('pdf_file').files[0];
            
            let isValid = true;
            let errorMessage = '';
            
            if (!file) {
                errorMessage = 'Please select a PDF file.';
                isValid = false;
            } else if (!file.name.toLowerCase().endsWith('.pdf')) {
                errorMessage = 'Please select a valid PDF file.';
                isValid = false;
            } else if (file.size > 50 * 1024 * 1024) { // 50MB limit
                errorMessage = 'File size must be less than 50MB.';
                isValid = false;
            }
            
            if (!chapter || chapter < 1 || chapter > 100) {
                errorMessage = 'Please enter a valid chapter number (1-100).';
                isValid = false;
            }
            
            if (!topic || topic.trim().length < 3) {
                errorMessage = 'Please enter a valid topic name (at least 3 characters).';
                isValid = false;
            }
            
            if (!isValid) {
                e.preventDefault();
                showAlert(errorMessage, 'danger');
            }
        });
    }

    // Alert system
    function showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            <i class="fas fa-${type === 'danger' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at the top of the main content
        const mainContent = document.querySelector('main');
        if (mainContent) {
            mainContent.insertBefore(alertDiv, mainContent.firstChild);
            
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, 5000);
        }
    }

    // Smooth scrolling for anchor links
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

    // Copy to clipboard functionality
    document.querySelectorAll('.copy-btn').forEach(button => {
        button.addEventListener('click', function() {
            const textToCopy = this.getAttribute('data-clipboard-text');
            if (textToCopy) {
                navigator.clipboard.writeText(textToCopy).then(() => {
                    // Show success feedback
                    const originalText = this.innerHTML;
                    this.innerHTML = '<i class="fas fa-check me-2"></i>Copied!';
                    this.classList.add('btn-success');
                    this.classList.remove('btn-outline-secondary');
                    
                    setTimeout(() => {
                        this.innerHTML = originalText;
                        this.classList.remove('btn-success');
                        this.classList.add('btn-outline-secondary');
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy text: ', err);
                    showAlert('Failed to copy text to clipboard', 'danger');
                });
            }
        });
    });

    // Question item interactions
    document.querySelectorAll('.question-item').forEach(item => {
        item.addEventListener('click', function() {
            // Toggle LaTeX view if available
            const latexToggle = this.querySelector('[data-bs-toggle="collapse"]');
            if (latexToggle) {
                latexToggle.click();
            }
        });
    });

    // Progress bar animation
    function animateProgressBar(progressBar, targetValue) {
        let currentValue = 0;
        const increment = targetValue / 100;
        
        const timer = setInterval(() => {
            currentValue += increment;
            progressBar.style.width = Math.min(currentValue, targetValue) + '%';
            
            if (currentValue >= targetValue) {
                clearInterval(timer);
            }
        }, 20);
    }

    // Initialize progress bars if they exist
    document.querySelectorAll('.progress-bar').forEach(bar => {
        const targetValue = bar.getAttribute('aria-valuenow');
        if (targetValue) {
            animateProgressBar(bar, parseInt(targetValue));
        }
    });

    // Search functionality for questions
    const searchInput = document.getElementById('questionSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const questions = document.querySelectorAll('.question-item');
            
            questions.forEach(question => {
                const questionText = question.querySelector('.question-text').textContent.toLowerCase();
                const confidence = question.querySelector('.badge').textContent.toLowerCase();
                
                if (questionText.includes(searchTerm) || confidence.includes(searchTerm)) {
                    question.style.display = 'block';
                } else {
                    question.style.display = 'none';
                }
            });
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const submitBtn = document.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                submitBtn.click();
            }
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) {
                    modalInstance.hide();
                }
            });
        }
    });

    // Performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', function() {
            const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
            console.log(`Page load time: ${loadTime}ms`);
        });
    }

    // Accessibility enhancements
    document.querySelectorAll('button, a, input, select, textarea').forEach(element => {
        if (!element.hasAttribute('aria-label') && !element.textContent.trim()) {
            const icon = element.querySelector('i');
            if (icon) {
                const iconClass = icon.className;
                if (iconClass.includes('upload')) {
                    element.setAttribute('aria-label', 'Upload file');
                } else if (iconClass.includes('download')) {
                    element.setAttribute('aria-label', 'Download file');
                } else if (iconClass.includes('eye')) {
                    element.setAttribute('aria-label', 'View details');
                }
            }
        }
    });

    // Console welcome message
    console.log(`
    🚀 RD Sharma Question Extractor Web Interface
    📱 Version: 1.0.0
    🔧 Built with Flask, Bootstrap, and JavaScript
    💡 Check out the demo at /demo
    📚 GitHub: https://github.com/Ani-2003-HD/rd-sharma-question-extraction
    `);
}); 