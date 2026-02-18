/**
 * COPD Care AI - Main JavaScript File
 * Technovation 2026 Project
 * Contains all core functionality, utilities, and global functions
 */

// ==================== GLOBAL CONFIGURATION ====================

const CONFIG = {
    APP_NAME: 'COPD Care AI',
    VERSION: '2.0.0',
    API_BASE_URL: 'https://api.copdcareai.com/v1',
    WHATSAPP_NUMBER: '+14155238886', // Twilio Sandbox number
    EMERGENCY_NUMBER: '108',
    DEFAULT_LOCATION: {
        city: 'Kota',
        state: 'Rajasthan',
        lat: 25.1765,
        lng: 75.8451
    },
    RISK_LEVELS: {
        LOW: { min: 0, max: 30, color: '#10b981', label: 'Low Risk' },
        MEDIUM: { min: 31, max: 60, color: '#f59e0b', label: 'Moderate Risk' },
        HIGH: { min: 61, max: 100, color: '#ef4444', label: 'High Risk' }
    },
    SYMPTOMS: [
        'Chronic Cough',
        'Excessive Phlegm',
        'Wheezing',
        'Shortness of Breath',
        'Chest Tightness',
        'Fatigue',
        'Frequent Respiratory Infections',
        'Bluish Lips or Fingernails'
    ]
};

// ==================== UTILITY FUNCTIONS ====================

/**
 * Format date to readable string
 * @param {Date} date - Date object
 * @param {string} format - Format type (full, short, time)
 * @returns {string} Formatted date string
 */
function formatDate(date = new Date(), format = 'full') {
    const options = {
        full: { year: 'numeric', month: 'long', day: 'numeric' },
        short: { year: 'numeric', month: 'short', day: 'numeric' },
        time: { hour: '2-digit', minute: '2-digit' },
        datetime: { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' }
    };
    
    return date.toLocaleDateString('en-US', options[format] || options.full);
}

/**
 * Show toast notification
 * @param {string} message - Message to display
 * @param {string} type - success, error, info, warning
 * @param {number} duration - Duration in milliseconds
 */
function showToast(message, type = 'info', duration = 3000) {
    // Remove existing toast
    const existingToast = document.querySelector('.toast-notification');
    if (existingToast) {
        existingToast.remove();
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast-notification fixed bottom-4 right-4 px-6 py-4 rounded-lg shadow-xl text-white z-50 animate-slide-in ${getToastColor(type)}`;
    toast.innerHTML = `
        <div class="flex items-center space-x-3">
            <i class="fas ${getToastIcon(type)} text-xl"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(toast);
    
    // Auto remove
    setTimeout(() => {
        toast.classList.add('animate-fade-out');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * Get toast color based on type
 * @param {string} type - Toast type
 * @returns {string} CSS class
 */
function getToastColor(type) {
    const colors = {
        success: 'bg-green-600',
        error: 'bg-red-600',
        warning: 'bg-yellow-600',
        info: 'bg-blue-600'
    };
    return colors[type] || colors.info;
}

/**
 * Get toast icon based on type
 * @param {string} type - Toast type
 * @returns {string} Font Awesome icon class
 */
function getToastIcon(type) {
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    return icons[type] || icons.info;
}

/**
 * Show loading spinner
 * @param {string} message - Loading message
 */
function showLoading(message = 'Loading...') {
    const loader = document.createElement('div');
    loader.className = 'loading-overlay fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    loader.id = 'globalLoader';
    loader.innerHTML = `
        <div class="bg-white rounded-xl p-8 text-center">
            <div class="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p class="text-gray-700">${message}</p>
        </div>
    `;
    document.body.appendChild(loader);
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    const loader = document.getElementById('globalLoader');
    if (loader) {
        loader.remove();
    }
}

/**
 * Debounce function for search inputs
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait = 300) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Validate email format
 * @param {string} email - Email to validate
 * @returns {boolean} Is valid email
 */
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

/**
 * Validate phone number (Indian format)
 * @param {string} phone - Phone number to validate
 * @returns {boolean} Is valid phone number
 */
function validatePhone(phone) {
    const re = /^[6-9]\d{9}$/;
    return re.test(phone.replace(/[^0-9]/g, ''));
}

/**
 * Format phone number to Indian format
 * @param {string} phone - Phone number
 * @returns {string} Formatted phone number
 */
function formatPhone(phone) {
    const cleaned = phone.replace(/[^0-9]/g, '');
    if (cleaned.length === 10) {
        return `+91 ${cleaned.slice(0, 5)} ${cleaned.slice(5)}`;
    }
    return phone;
}

/**
 * Calculate BMI
 * @param {number} weight - Weight in kg
 * @param {number} height - Height in cm
 * @returns {number} BMI value
 */
function calculateBMI(weight, height) {
    const heightInMeters = height / 100;
    return (weight / (heightInMeters * heightInMeters)).toFixed(1);
}

/**
 * Get BMI category
 * @param {number} bmi - BMI value
 * @returns {string} BMI category
 */
function getBMICategory(bmi) {
    if (bmi < 18.5) return 'Underweight';
    if (bmi < 25) return 'Normal weight';
    if (bmi < 30) return 'Overweight';
    return 'Obese';
}

// ==================== LOCAL STORAGE MANAGEMENT ====================

/**
 * Save data to localStorage
 * @param {string} key - Storage key
 * @param {any} data - Data to store
 */
function saveToStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
        return true;
    } catch (error) {
        console.error('Error saving to localStorage:', error);
        return false;
    }
}

/**
 * Get data from localStorage
 * @param {string} key - Storage key
 * @returns {any} Stored data or null
 */
function getFromStorage(key) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : null;
    } catch (error) {
        console.error('Error reading from localStorage:', error);
        return null;
    }
}

/**
 * Remove data from localStorage
 * @param {string} key - Storage key
 */
function removeFromStorage(key) {
    try {
        localStorage.removeItem(key);
        return true;
    } catch (error) {
        console.error('Error removing from localStorage:', error);
        return false;
    }
}

/**
 * Clear all app data from localStorage
 */
function clearAppData() {
    const keysToKeep = ['user']; // Keep user logged in
    const allKeys = Object.keys(localStorage);
    
    allKeys.forEach(key => {
        if (!keysToKeep.includes(key)) {
            localStorage.removeItem(key);
        }
    });
    
    showToast('App data cleared successfully', 'success');
}

// ==================== USER MANAGEMENT ====================

/**
 * Get current logged in user
 * @returns {object|null} User object or null
 */
function getCurrentUser() {
    return getFromStorage('user');
}

/**
 * Check if user is logged in
 * @returns {boolean} Login status
 */
function isLoggedIn() {
    const user = getCurrentUser();
    return user && user.isLoggedIn;
}

/**
 * Require authentication - redirect to login if not authenticated
 */
function requireAuth() {
    if (!isLoggedIn()) {
        window.location.href = '/pages/login.html';
        return false;
    }
    return true;
}

/**
 * Logout user
 */
function logout() {
    const user = getCurrentUser();
    if (user) {
        user.isLoggedIn = false;
        saveToStorage('user', user);
    }
    showToast('Logged out successfully', 'success');
    setTimeout(() => {
        window.location.href = '/index.html';
    }, 1000);
}

/**
 * Update user profile
 * @param {object} userData - Updated user data
 */
function updateUserProfile(userData) {
    const currentUser = getCurrentUser();
    if (currentUser) {
        const updatedUser = { ...currentUser, ...userData };
        saveToStorage('user', updatedUser);
        showToast('Profile updated successfully', 'success');
        return true;
    }
    return false;
}

// ==================== API SIMULATION / MOCK DATA ====================

/**
 * Mock API call for COPD prediction
 * @param {object} formData - Prediction form data
 * @returns {Promise} Prediction result
 */
async function predictCOPD(formData) {
    showLoading('Analyzing your health data...');
    
    return new Promise((resolve) => {
        setTimeout(() => {
            hideLoading();
            
            // Calculate mock risk score based on inputs
            let riskScore = 0;
            
            // Age factor
            if (formData.age > 60) riskScore += 25;
            else if (formData.age > 50) riskScore += 20;
            else if (formData.age > 40) riskScore += 15;
            
            // Smoking factor
            if (formData.smokingStatus === 'current') riskScore += 30;
            else if (formData.smokingStatus === 'former') riskScore += 15;
            
            // Symptoms count
            if (formData.symptoms) {
                riskScore += formData.symptoms.length * 5;
            }
            
            // Oxygen level
            if (formData.oxygen < 90) riskScore += 25;
            else if (formData.oxygen < 95) riskScore += 15;
            
            // Cap at 100
            riskScore = Math.min(riskScore, 100);
            
            // Determine risk level
            let riskLevel = 'Low';
            if (riskScore > 60) riskLevel = 'High';
            else if (riskScore > 30) riskLevel = 'Moderate';
            
            const result = {
                success: true,
                riskScore,
                riskLevel,
                recommendations: getRecommendations(riskLevel),
                timestamp: new Date().toISOString()
            };
            
            // Save prediction history
            savePredictionHistory(result);
            
            resolve(result);
        }, 2000);
    });
}

/**
 * Get recommendations based on risk level
 * @param {string} riskLevel - Low/Moderate/High
 * @returns {Array} Recommendations list
 */
function getRecommendations(riskLevel) {
    const recommendations = {
        Low: [
            'Maintain a healthy lifestyle with regular exercise',
            'Eat a balanced diet rich in antioxidants',
            'Avoid smoking and secondhand smoke',
            'Get annual check-ups',
            'Practice breathing exercises daily'
        ],
        Moderate: [
            'Consult a pulmonologist for a check-up',
            'Follow an anti-inflammatory diet plan',
            'Monitor your symptoms daily',
            'Avoid exposure to pollutants',
            'Take prescribed medications regularly',
            'Practice pursed-lip breathing'
        ],
        High: [
            'IMMEDIATE CONSULTATION WITH PULMONOLOGIST REQUIRED',
            'Follow prescribed treatment plan strictly',
            'Use oxygen therapy if prescribed',
            'Avoid all respiratory irritants',
            'Keep emergency contact handy',
            'Monitor oxygen levels regularly',
            'Consider pulmonary rehabilitation'
        ]
    };
    
    return recommendations[riskLevel] || recommendations.Low;
}

/**
 * Save prediction to history
 * @param {object} prediction - Prediction result
 */
function savePredictionHistory(prediction) {
    const user = getCurrentUser();
    if (user) {
        const history = getFromStorage(`predictions_${user.email}`) || [];
        history.unshift(prediction);
        saveToStorage(`predictions_${user.email}`, history.slice(0, 10)); // Keep last 10
    }
}

/**
 * Get user's prediction history
 * @returns {Array} Prediction history
 */
function getPredictionHistory() {
    const user = getCurrentUser();
    if (user) {
        return getFromStorage(`predictions_${user.email}`) || [];
    }
    return [];
}

// ==================== WHATSAPP INTEGRATION ====================

/**
 * Open WhatsApp chat with predefined message
 * @param {string} message - Message to send
 */
function openWhatsApp(message = 'Hello, I need help with COPD management') {
    const phoneNumber = CONFIG.WHATSAPP_NUMBER;
    const encodedMessage = encodeURIComponent(message);
    const url = `https://wa.me/${phoneNumber}?text=${encodedMessage}`;
    window.open(url, '_blank');
}

/**
 * Share diet plan on WhatsApp
 * @param {object} dietPlan - Diet plan data
 */
function shareDietPlanOnWhatsApp(dietPlan) {
    const message = `*My COPD Diet Plan*\n\n` +
        `*Breakfast:* ${dietPlan.breakfast}\n` +
        `*Lunch:* ${dietPlan.lunch}\n` +
        `*Dinner:* ${dietPlan.dinner}\n` +
        `*Calories:* ${dietPlan.calories} kcal\n\n` +
        `Sent from COPD Care AI`;
    
    openWhatsApp(message);
}

/**
 * Share health report on WhatsApp
 * @param {object} report - Health report data
 */
function shareHealthReport(report) {
    const message = `*My COPD Health Report*\n\n` +
        `*Risk Level:* ${report.riskLevel} (${report.riskScore}%)\n` +
        `*Date:* ${formatDate(new Date(report.timestamp), 'full')}\n` +
        `*Recommendations:*\n${report.recommendations.map(r => `• ${r}`).join('\n')}\n\n` +
        `Sent from COPD Care AI`;
    
    openWhatsApp(message);
}

// ==================== DOCTOR SEARCH ====================

/**
 * Search for doctors near location
 * @param {string} location - Location to search
 * @param {number} radius - Search radius in km
 * @returns {Promise} List of doctors
 */
async function searchDoctors(location, radius = 10) {
    showLoading('Searching for doctors...');
    
    return new Promise((resolve) => {
        setTimeout(() => {
            hideLoading();
            
            // Mock doctor data
            const doctors = [
                {
                    id: 1,
                    name: 'Dr. Rakesh Gupta',
                    rating: 4.8,
                    reviews: 150,
                    qualification: 'MD, FCCP',
                    specialty: 'Senior Pulmonologist',
                    hospital: 'Apex Lung Centre',
                    area: 'Talwandi, Kota',
                    experience: 18,
                    timing: 'Mon-Sat, 10:00 AM - 7:00 PM',
                    fee: 800,
                    available: true,
                    distance: '1.2 km'
                },
                {
                    id: 2,
                    name: 'Dr. Suman Verma',
                    rating: 4.7,
                    reviews: 88,
                    qualification: 'DNB',
                    specialty: 'Chest Physician',
                    hospital: 'Maitri Hospital',
                    area: 'Vigan Nagar, Kota',
                    experience: 10,
                    timing: 'Mon-Sat, 11:00 AM - 5:00 PM',
                    fee: 600,
                    available: true,
                    distance: '2.5 km'
                },
                {
                    id: 3,
                    name: 'Dr. Amit Jain',
                    rating: 4.5,
                    reviews: 65,
                    qualification: 'MD',
                    specialty: 'Pulmonologist',
                    hospital: 'Kota Heart & Lung Institute',
                    area: 'Civil Lines, Kota',
                    experience: 8,
                    timing: 'Mon-Fri, 8:00 AM - 2:00 PM',
                    fee: 500,
                    available: false,
                    distance: '3.1 km'
                },
                {
                    id: 4,
                    name: 'Dr. Priya Sharma',
                    rating: 4.9,
                    reviews: 200,
                    qualification: 'MD, FCCP',
                    specialty: 'Respiratory Specialist',
                    hospital: 'Shree Hospital',
                    area: 'Kunadi, Kota',
                    experience: 15,
                    timing: 'Mon-Sat, 9:00 AM - 6:00 PM',
                    fee: 900,
                    available: true,
                    distance: '4.0 km'
                },
                {
                    id: 5,
                    name: 'Dr. Vikas Mehta',
                    rating: 4.6,
                    reviews: 120,
                    qualification: 'DM',
                    specialty: 'Critical Care',
                    hospital: 'MBS Hospital',
                    area: 'Nayapura, Kota',
                    experience: 12,
                    timing: 'Mon-Sat, 10:00 AM - 8:00 PM',
                    fee: 700,
                    available: true,
                    distance: '2.8 km'
                }
            ];
            
            resolve({
                success: true,
                count: doctors.length,
                doctors: doctors.filter(d => d.distance <= `${radius} km` || true)
            });
        }, 1500);
    });
}

/**
 * Book appointment with doctor
 * @param {number} doctorId - Doctor ID
 * @param {object} appointmentData - Appointment details
 */
function bookAppointment(doctorId, appointmentData) {
    showLoading('Booking appointment...');
    
    return new Promise((resolve) => {
        setTimeout(() => {
            hideLoading();
            
            const success = Math.random() > 0.2; // 80% success rate
            
            if (success) {
                const bookingId = 'APT' + Date.now();
                showToast('Appointment booked successfully!', 'success');
                
                // Save to appointments history
                const user = getCurrentUser();
                if (user) {
                    const appointments = getFromStorage(`appointments_${user.email}`) || [];
                    appointments.unshift({
                        id: bookingId,
                        doctorId,
                        ...appointmentData,
                        timestamp: new Date().toISOString()
                    });
                    saveToStorage(`appointments_${user.email}`, appointments.slice(0, 20));
                }
                
                resolve({
                    success: true,
                    bookingId,
                    message: 'Appointment confirmed'
                });
            } else {
                showToast('Booking failed. Please try again.', 'error');
                resolve({
                    success: false,
                    message: 'Booking failed'
                });
            }
        }, 2000);
    });
}

// ==================== WEATHER & LOCATION ====================

/**
 * Get user's current location
 * @returns {Promise} Location coordinates
 */
function getCurrentLocation() {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject(new Error('Geolocation not supported'));
            return;
        }
        
        navigator.geolocation.getCurrentPosition(
            (position) => {
                resolve({
                    lat: position.coords.latitude,
                    lng: position.coords.longitude
                });
            },
            (error) => {
                reject(error);
            },
            {
                enableHighAccuracy: true,
                timeout: 5000,
                maximumAge: 0
            }
        );
    });
}

/**
 * Get weather data for location
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 * @returns {Promise} Weather data
 */
async function getWeatherData(lat, lng) {
    // Mock weather data (in real app, call weather API)
    const weatherConditions = ['Sunny', 'Cloudy', 'Partly Cloudy', 'Clear', 'Rainy'];
    const randomCondition = weatherConditions[Math.floor(Math.random() * weatherConditions.length)];
    const randomTemp = Math.floor(Math.random() * (35 - 20) + 20); // 20-35°C
    
    return {
        temp: randomTemp,
        condition: randomCondition,
        humidity: Math.floor(Math.random() * (80 - 40) + 40),
        airQuality: Math.floor(Math.random() * (150 - 50) + 50),
        icon: getWeatherIcon(randomCondition)
    };
}

/**
 * Get weather icon based on condition
 * @param {string} condition - Weather condition
 * @returns {string} Font Awesome icon class
 */
function getWeatherIcon(condition) {
    const icons = {
        'Sunny': 'fa-sun text-yellow-500',
        'Cloudy': 'fa-cloud text-gray-500',
        'Partly Cloudy': 'fa-cloud-sun text-gray-500',
        'Clear': 'fa-moon text-indigo-500',
        'Rainy': 'fa-cloud-rain text-blue-500'
    };
    return icons[condition] || 'fa-cloud-sun';
}

// ==================== BREATHING EXERCISES ====================

/**
 * Breathing exercises library
 */
const breathingExercises = [
    {
        id: 1,
        name: 'Pursed Lip Breathing',
        duration: '5 minutes',
        description: 'Breathe in slowly through your nose, then breathe out slowly through pursed lips.',
        benefits: 'Helps slow breathing pace, keeps airways open longer',
        steps: [
            'Relax your neck and shoulder muscles',
            'Breathe in slowly through your nose for 2 counts',
            'Purse your lips as if whistling',
            'Breathe out slowly through pursed lips for 4 counts',
            'Repeat for 5-10 minutes'
        ],
        videoUrl: '/assets/videos/pursed-lip.mp4',
        icon: 'fa-lungs'
    },
    {
        id: 2,
        name: 'Diaphragmatic Breathing',
        duration: '5-10 minutes',
        description: 'Breathe deeply using your diaphragm to strengthen this muscle.',
        benefits: 'Strengthens diaphragm, decreases breathing effort',
        steps: [
            'Lie on your back with knees bent',
            'Place one hand on your chest, one on your belly',
            'Breathe in slowly through your nose, feeling belly rise',
            'Tighten stomach muscles, let them fall inward as you exhale',
            'Practice for 5-10 minutes, 3-4 times daily'
        ],
        videoUrl: '/assets/videos/diaphragmatic.mp4',
        icon: 'fa-lungs'
    },
    {
        id: 3,
        name: 'Deep Breathing',
        duration: '5 minutes',
        description: 'Simple deep breathing to increase oxygen flow.',
        benefits: 'Increases oxygen levels, reduces stress',
        steps: [
            'Sit comfortably with straight back',
            'Breathe in deeply through your nose for 4 seconds',
            'Hold breath for 2 seconds',
            'Exhale slowly through your mouth for 6 seconds',
            'Repeat 10 times'
        ],
        videoUrl: '/assets/videos/deep-breathing.mp4',
        icon: 'fa-lungs'
    },
    {
        id: 4,
        name: 'Coordinated Breathing',
        duration: '5 minutes',
        description: 'Coordinate breathing with activities that require effort.',
        benefits: 'Prevents breathlessness during activities',
        steps: [
            'Breathe in before starting an activity',
            'Breathe out during the hardest part of activity',
            'Practice during walking or climbing stairs',
            'Use pursed lips when exhaling'
        ],
        videoUrl: '/assets/videos/coordinated.mp4',
        icon: 'fa-person-walking'
    }
];

/**
 * Get all breathing exercises
 * @returns {Array} Breathing exercises
 */
function getBreathingExercises() {
    return breathingExercises;
}

/**
 * Get exercise by ID
 * @param {number} id - Exercise ID
 * @returns {object} Exercise object
 */
function getExerciseById(id) {
    return breathingExercises.find(ex => ex.id === id);
}

// ==================== ANALYTICS & TRACKING ====================

/**
 * Track user activity
 * @param {string} action - Action performed
 * @param {object} data - Additional data
 */
function trackActivity(action, data = {}) {
    const user = getCurrentUser();
    if (user) {
        const activities = getFromStorage(`activities_${user.email}`) || [];
        activities.unshift({
            action,
            data,
            timestamp: new Date().toISOString()
        });
        saveToStorage(`activities_${user.email}`, activities.slice(0, 50));
    }
}

/**
 * Get user activity history
 * @returns {Array} Activity history
 */
function getActivityHistory() {
    const user = getCurrentUser();
    if (user) {
        return getFromStorage(`activities_${user.email}`) || [];
    }
    return [];
}

/**
 * Calculate health score based on various factors
 * @param {object} healthData - Health data
 * @returns {number} Health score (0-100)
 */
function calculateHealthScore(healthData) {
    let score = 70; // Base score
    
    // Adjust based on risk level
    if (healthData.riskLevel === 'Low') score += 20;
    else if (healthData.riskLevel === 'Moderate') score -= 10;
    else if (healthData.riskLevel === 'High') score -= 30;
    
    // Adjust based on exercise frequency
    if (healthData.exerciseFrequency === 'daily') score += 10;
    else if (healthData.exerciseFrequency === 'weekly') score += 5;
    
    // Adjust based on diet adherence
    if (healthData.dietAdherence > 80) score += 10;
    else if (healthData.dietAdherence < 50) score -= 10;
    
    // Cap between 0-100
    return Math.max(0, Math.min(100, score));
}

// ==================== INITIALIZATION ====================

/**
 * Initialize app on page load
 */
document.addEventListener('DOMContentLoaded', function() {
    // Add global styles
    addGlobalStyles();
    
    // Check authentication on protected pages
    const protectedPages = [
        'dashboard.html',
        'prediction.html',
        'diet-plan.html',
        'doctor-search.html',
        'whatsapp-agent.html',
        'settings.html'
    ];
    
    const currentPage = window.location.pathname.split('/').pop();
    if (protectedPages.includes(currentPage)) {
        requireAuth();
    }
    
    // Update user name in sidebar if exists
    updateSidebarUser();
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add active class to current nav item
    highlightCurrentNav();
});

/**
 * Add global styles to document
 */
function addGlobalStyles() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes fadeOut {
            from {
                opacity: 1;
            }
            to {
                opacity: 0;
            }
        }
        
        .animate-slide-in {
            animation: slideIn 0.3s ease-out;
        }
        
        .animate-fade-out {
            animation: fadeOut 0.3s ease-out;
        }
        
        .toast-notification {
            z-index: 9999;
            min-width: 300px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .loading-overlay {
            z-index: 10000;
        }
        
        .gradient-text {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .hover-scale {
            transition: transform 0.3s ease;
        }
        
        .hover-scale:hover {
            transform: scale(1.05);
        }
        
        .card-hover {
            transition: all 0.3s ease;
        }
        
        .card-hover:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
    `;
    document.head.appendChild(style);
}

/**
 * Update sidebar user name
 */
function updateSidebarUser() {
    const userNameElements = document.querySelectorAll('#userName, .user-name');
    const user = getCurrentUser();
    
    if (user && userNameElements.length) {
        const displayName = user.firstName || user.name || 'User';
        userNameElements.forEach(el => {
            el.textContent = displayName;
        });
    }
}

/**
 * Highlight current navigation item
 */
function highlightCurrentNav() {
    const currentPage = window.location.pathname.split('/').pop();
    const navLinks = document.querySelectorAll('.sidebar-item, nav a');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href && href.includes(currentPage)) {
            link.classList.add('active');
        }
    });
}

// ==================== EXPORT FUNCTIONS ====================

// Make functions globally available
window.COPDApp = {
    // Core
    formatDate,
    showToast,
    showLoading,
    hideLoading,
    debounce,
    
    // User
    getCurrentUser,
    isLoggedIn,
    requireAuth,
    logout,
    updateUserProfile,
    
    // Storage
    saveToStorage,
    getFromStorage,
    removeFromStorage,
    clearAppData,
    
    // Prediction
    predictCOPD,
    getPredictionHistory,
    
    // WhatsApp
    openWhatsApp,
    shareDietPlanOnWhatsApp,
    shareHealthReport,
    
    // Doctors
    searchDoctors,
    bookAppointment,
    
    // Location
    getCurrentLocation,
    getWeatherData,
    
    // Exercises
    getBreathingExercises,
    getExerciseById,
    
    // Analytics
    trackActivity,
    getActivityHistory,
    calculateHealthScore,
    
    // Validation
    validateEmail,
    validatePhone,
    formatPhone,
    calculateBMI,
    getBMICategory,
    
    // Config
    CONFIG
};

// Export individual functions for direct use
window.formatDate = formatDate;
window.showToast = showToast;
window.showLoading = showLoading;
window.hideLoading = hideLoading;
window.isLoggedIn = isLoggedIn;
window.logout = logout;
window.openWhatsApp = openWhatsApp;
window.getCurrentLocation = getCurrentLocation;
window.validateEmail = validateEmail;
window.calculateBMI = calculateBMI;