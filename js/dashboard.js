/**
 * COPD Care AI - Dashboard JavaScript File
 * Technovation 2026 Project
 * Handles all dashboard-specific functionality
 */

// ==================== DASHBOARD CONFIGURATION ====================

const DASHBOARD_CONFIG = {
    REFRESH_INTERVAL: 300000, // 5 minutes in milliseconds
    MAX_RECENT_ACTIVITIES: 10,
    CHART_COLORS: {
        primary: '#3b82f6',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        info: '#8b5cf6'
    },
    HEALTH_TIPS: [
        "Stay hydrated! Proper hydration helps thin mucus and makes breathing easier. Aim for 8-10 glasses of water daily.",
        "Practice pursed-lip breathing when you feel short of breath. Inhale through nose, exhale slowly through pursed lips.",
        "Avoid smoking and secondhand smoke. It's the most important step to prevent COPD progression.",
        "Eat small, frequent meals rather than large meals to prevent bloating and make breathing easier.",
        "Get your annual flu shot and pneumonia vaccine to prevent respiratory infections.",
        "Exercise regularly but listen to your body. Start slow and gradually increase intensity.",
        "Keep your home clean and dust-free. Use air purifiers if needed.",
        "Monitor your oxygen levels if prescribed. Contact doctor if levels drop below 92%.",
        "Join a pulmonary rehabilitation program for supervised exercise and education.",
        "Take medications exactly as prescribed. Don't skip doses even if you feel well.",
        "Use a humidifier in dry environments to keep airways moist.",
        "Practice deep breathing exercises daily to strengthen your lungs.",
        "Avoid extreme temperatures and air pollution when possible.",
        "Keep emergency numbers handy and inform family members about your condition.",
        "Track your symptoms daily to identify triggers and patterns."
    ],
    BREATHING_EXERCISES: {
        1: {
            id: 1,
            name: 'Pursed Lip Breathing',
            duration: 300, // 5 minutes in seconds
            description: 'This exercise helps slow your breathing pace and keeps airways open longer.',
            steps: [
                'Relax your neck and shoulder muscles',
                'Breathe in slowly through your nose for 2 counts',
                'Purse your lips as if whistling',
                'Breathe out slowly through pursed lips for 4 counts',
                'Repeat for 5-10 minutes'
            ],
            benefits: ['Slows breathing pace', 'Keeps airways open longer', 'Reduces shortness of breath'],
            icon: 'fa-lungs',
            color: 'blue'
        },
        2: {
            id: 2,
            name: 'Diaphragmatic Breathing',
            duration: 300,
            description: 'This exercise strengthens your diaphragm and decreases breathing effort.',
            steps: [
                'Lie on your back with knees bent',
                'Place one hand on your chest, one on your belly',
                'Breathe in slowly through your nose, feeling belly rise',
                'Tighten stomach muscles as you exhale',
                'Practice for 5-10 minutes, 3-4 times daily'
            ],
            benefits: ['Strengthens diaphragm', 'Decreases breathing effort', 'Improves oxygen flow'],
            icon: 'fa-lungs',
            color: 'green'
        },
        3: {
            id: 3,
            name: 'Deep Breathing',
            duration: 300,
            description: 'Simple deep breathing to increase oxygen flow and reduce stress.',
            steps: [
                'Sit comfortably with straight back',
                'Breathe in deeply through your nose for 4 seconds',
                'Hold breath for 2 seconds',
                'Exhale slowly through your mouth for 6 seconds',
                'Repeat 10 times'
            ],
            benefits: ['Increases oxygen levels', 'Reduces stress', 'Calms nervous system'],
            icon: 'fa-lungs',
            color: 'purple'
        },
        4: {
            id: 4,
            name: 'Coordinated Breathing',
            duration: 300,
            description: 'Coordinate breathing with activities that require effort.',
            steps: [
                'Breathe in before starting an activity',
                'Breathe out during the hardest part of activity',
                'Practice during walking or climbing stairs',
                'Use pursed lips when exhaling'
            ],
            benefits: ['Prevents breathlessness during activities', 'Improves exercise tolerance'],
            icon: 'fa-person-walking',
            color: 'orange'
        }
    }
};

// ==================== STATE MANAGEMENT ====================

let dashboardState = {
    user: null,
    currentRisk: 'medium',
    mealProgress: {
        breakfast: false,
        morningSnack: false,
        lunch: false,
        eveningSnack: false,
        dinner: false
    },
    waterIntake: 0,
    waterTarget: 8,
    healthTipIndex: 0,
    notifications: [],
    recentActivities: [],
    weather: {
        location: 'Kota, Rajasthan',
        temp: 25,
        condition: 'Sunny',
        humidity: 60,
        airQuality: 85
    },
    charts: {},
    timers: {}
};

// ==================== INITIALIZATION ====================

/**
 * Initialize dashboard
 */
async function initDashboard() {
    console.log('Initializing dashboard...');
    
    // Check authentication
    if (!window.COPDApp || !window.COPDApp.isLoggedIn()) {
        window.location.href = 'login.html';
        return;
    }
    
    // Get current user
    dashboardState.user = window.COPDApp.getCurrentUser();
    
    // Update UI with user info
    updateUserInfo();
    
    // Set current date
    updateCurrentDate();
    
    // Load dashboard data
    await loadDashboardData();
    
    // Load weather
    await loadWeatherData();
    
    // Load local doctors
    loadLocalDoctors();
    
    // Load notifications
    loadNotifications();
    
    // Load recent activities
    loadRecentActivities();
    
    // Initialize charts
    initCharts();
    
    // Set up auto-refresh
    setupAutoRefresh();
    
    // Track page view
    if (window.COPDApp && window.COPDApp.trackActivity) {
        window.COPDApp.trackActivity('dashboard_view', { timestamp: new Date().toISOString() });
    }
    
    console.log('Dashboard initialized successfully');
}

// ==================== USER INFO FUNCTIONS ====================

/**
 * Update user information in UI
 */
function updateUserInfo() {
    if (!dashboardState.user) return;
    
    const user = dashboardState.user;
    
    // Set user name
    const userNameElements = document.querySelectorAll('#userName, .user-name-display');
    const fullName = `${user.firstName || 'Bharat'} ${user.lastName || 'Goswami'}`;
    userNameElements.forEach(el => {
        if (el) el.textContent = fullName;
    });
    
    // Set welcome name
    const welcomeElements = document.querySelectorAll('#welcomeName');
    welcomeElements.forEach(el => {
        if (el) el.textContent = user.firstName || 'Bharat';
    });
    
    // Set initials
    const initials = (user.firstName?.[0] || 'B') + (user.lastName?.[0] || 'G');
    const initialsElements = document.querySelectorAll('#userInitials, .user-initials');
    initialsElements.forEach(el => {
        if (el) el.textContent = initials;
    });
}

/**
 * Update current date display
 */
function updateCurrentDate() {
    const dateElement = document.getElementById('currentDate');
    if (dateElement) {
        const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
        dateElement.textContent = new Date().toLocaleDateString('en-US', options);
    }
}

// ==================== DASHBOARD DATA LOADING ====================

/**
 * Load all dashboard data
 */
async function loadDashboardData() {
    await loadPredictionData();
    await loadDietPlan();
    loadHealthTip();
}

/**
 * Load prediction data from localStorage
 */
function loadPredictionData() {
    // Check if there's saved prediction
    const savedPrediction = localStorage.getItem('lastPrediction');
    
    if (savedPrediction) {
        try {
            const prediction = JSON.parse(savedPrediction);
            
            // Update UI with prediction data
            updateRiskDisplay(prediction);
            
            // Set risk level for diet plan
            if (prediction.riskLevel) {
                const riskLower = prediction.riskLevel.toLowerCase();
                if (riskLower.includes('low')) dashboardState.currentRisk = 'low';
                else if (riskLower.includes('moderate') || riskLower.includes('medium')) dashboardState.currentRisk = 'medium';
                else if (riskLower.includes('high')) dashboardState.currentRisk = 'high';
                
                localStorage.setItem('userRiskLevel', dashboardState.currentRisk);
            }
            
            return prediction;
        } catch (error) {
            console.error('Error parsing prediction data:', error);
        }
    }
    
    // Mock data if no prediction exists
    const mockPrediction = {
        riskLevel: 'Mild Risk',
        riskScore: 72,
        recommendations: ['Monitor symptoms regularly', 'Follow diet plan', 'Practice breathing exercises'],
        timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString() // 2 days ago
    };
    
    updateRiskDisplay(mockPrediction);
    dashboardState.currentRisk = 'medium';
    localStorage.setItem('userRiskLevel', 'medium');
    
    return mockPrediction;
}

/**
 * Update risk display in UI
 * @param {object} prediction - Prediction data
 */
function updateRiskDisplay(prediction) {
    // Recent prediction status
    const recentRiskEl = document.getElementById('recentRisk');
    if (recentRiskEl) {
        recentRiskEl.textContent = `${prediction.riskLevel} (${prediction.riskScore}%)`;
    }
    
    // Risk level card
    const riskLevelEl = document.getElementById('riskLevel');
    if (riskLevelEl) {
        riskLevelEl.textContent = prediction.riskLevel;
    }
    
    const riskPercentageEl = document.getElementById('riskPercentage');
    if (riskPercentageEl) {
        riskPercentageEl.textContent = `${prediction.riskScore}% Risk Level`;
    }
    
    // Recommendation
    const recommendationEl = document.getElementById('recommendation');
    if (recommendationEl && prediction.recommendations && prediction.recommendations.length > 0) {
        recommendationEl.innerHTML = `
            <i class="fas fa-info-circle text-blue-600 mr-2"></i>
            ${prediction.recommendations[0]}
        `;
    }
    
    // Prediction date
    const predictionDateEl = document.getElementById('predictionDate');
    if (predictionDateEl && prediction.timestamp) {
        const predDate = new Date(prediction.timestamp);
        predictionDateEl.textContent = predDate.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    // Last update
    const lastUpdateEl = document.getElementById('lastUpdate');
    if (lastUpdateEl && prediction.timestamp) {
        const predDate = new Date(prediction.timestamp);
        lastUpdateEl.textContent = `Last updated ${predDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;
    }
}

/**
 * Load diet plan based on risk level
 */
function loadDietPlan() {
    const dietPlans = {
        low: {
            name: 'Maintenance Diet',
            breakfast: 'Fruit bowl with yogurt',
            breakfastCal: 350,
            lunch: 'Quinoa salad with veggies',
            lunchCal: 450,
            dinner: 'Grilled fish with vegetables',
            dinnerCal: 400,
            total: 1200,
            hasSnacks: false,
            times: {
                breakfast: '8:00 AM',
                lunch: '1:00 PM',
                dinner: '7:00 PM'
            }
        },
        medium: {
            name: 'Respiratory Support Diet',
            breakfast: 'Oatmeal with berries',
            breakfastCal: 380,
            morningSnack: 'Green smoothie',
            morningSnackCal: 150,
            lunch: 'Grilled chicken with quinoa',
            lunchCal: 520,
            eveningSnack: 'Fruit and nuts',
            eveningSnackCal: 200,
            dinner: 'Salmon with vegetables',
            dinnerCal: 450,
            total: 1700,
            hasSnacks: true,
            times: {
                breakfast: '8:00 AM',
                morningSnack: '11:00 AM',
                lunch: '1:30 PM',
                eveningSnack: '4:30 PM',
                dinner: '7:30 PM'
            }
        },
        high: {
            name: 'Clinical Recovery Diet',
            breakfast: 'Protein smoothie',
            breakfastCal: 380,
            morningSnack: 'Greek yogurt',
            morningSnackCal: 200,
            lunch: 'Therapeutic chicken soup',
            lunchCal: 550,
            eveningSnack: 'Protein shake',
            eveningSnackCal: 250,
            dinner: 'Poached fish with mash',
            dinnerCal: 480,
            total: 1860,
            hasSnacks: true,
            times: {
                breakfast: '8:00 AM',
                morningSnack: '11:00 AM',
                lunch: '1:30 PM',
                eveningSnack: '4:30 PM',
                dinner: '7:00 PM'
            }
        }
    };
    
    const plan = dietPlans[dashboardState.currentRisk] || dietPlans.medium;
    
    // Update diet name
    const dietNameEl = document.getElementById('dietName');
    if (dietNameEl) dietNameEl.textContent = plan.name;
    
    // Update meals
    updateMealDisplay('breakfast', plan.breakfast, plan.breakfastCal, plan.times.breakfast);
    updateMealDisplay('lunch', plan.lunch, plan.lunchCal, plan.times.lunch);
    updateMealDisplay('dinner', plan.dinner, plan.dinnerCal, plan.times.dinner);
    
    // Update total calories
    const totalCaloriesEl = document.getElementById('totalCalories');
    if (totalCaloriesEl) totalCaloriesEl.textContent = plan.total;
    
    // Handle snacks
    const snackSection = document.getElementById('snackSection');
    if (snackSection) {
        if (plan.hasSnacks) {
            snackSection.classList.remove('hidden');
            updateMealDisplay('morningSnack', plan.morningSnack, plan.morningSnackCal, plan.times.morningSnack);
            updateMealDisplay('eveningSnack', plan.eveningSnack, plan.eveningSnackCal, plan.times.eveningSnack);
        } else {
            snackSection.classList.add('hidden');
        }
    }
    
    // Update next meal
    updateNextMeal(plan.times);
    
    // Save water target based on risk
    dashboardState.waterTarget = dashboardState.currentRisk === 'high' ? 10 : 8;
    
    return plan;
}

/**
 * Update individual meal display
 * @param {string} meal - Meal identifier
 * @param {string} name - Meal name
 * @param {number} calories - Calories
 * @param {string} time - Meal time
 */
function updateMealDisplay(meal, name, calories, time) {
    const mealEl = document.getElementById(meal + 'Meal');
    if (mealEl) mealEl.textContent = name || 'Meal';
    
    const caloriesEl = document.getElementById(meal + 'Calories');
    if (caloriesEl) caloriesEl.textContent = (calories || 0) + ' kcal';
    
    const timeEl = document.getElementById(meal + 'Time');
    if (timeEl) timeEl.textContent = time || '';
}

/**
 * Update next meal based on current time
 * @param {object} times - Meal times object
 */
function updateNextMeal(times) {
    const now = new Date();
    const currentHour = now.getHours();
    const currentMinute = now.getMinutes();
    const currentTime = currentHour * 60 + currentMinute;
    
    // Convert meal times to minutes
    const mealTimes = {};
    for (const [meal, timeStr] of Object.entries(times)) {
        if (timeStr) {
            const [hour, minutePart] = timeStr.split(':');
            const minute = parseInt(minutePart.split(' ')[0]);
            const isPM = timeStr.includes('PM');
            let hourNum = parseInt(hour);
            if (isPM && hourNum !== 12) hourNum += 12;
            if (!isPM && hourNum === 12) hourNum = 0;
            
            mealTimes[meal] = hourNum * 60 + minute;
        }
    }
    
    // Find next meal
    let nextMeal = null;
    let minDiff = Infinity;
    
    for (const [meal, time] of Object.entries(mealTimes)) {
        if (time > currentTime) {
            const diff = time - currentTime;
            if (diff < minDiff) {
                minDiff = diff;
                nextMeal = meal;
            }
        }
    }
    
    // If no meal today, first meal tomorrow
    if (!nextMeal) {
        const firstMeal = Object.entries(mealTimes).sort((a, b) => a[1] - b[1])[0];
        nextMeal = firstMeal ? firstMeal[0] : 'breakfast';
    }
    
    // Format next meal text
    const mealNames = {
        breakfast: 'Breakfast',
        morningSnack: 'Morning Snack',
        lunch: 'Lunch',
        eveningSnack: 'Evening Snack',
        dinner: 'Dinner'
    };
    
    const nextMealEl = document.getElementById('nextMeal');
    if (nextMealEl && nextMeal) {
        if (minDiff < 60) {
            nextMealEl.textContent = `${mealNames[nextMeal]} in ${minDiff} minutes`;
        } else if (minDiff < 120) {
            nextMealEl.textContent = `${mealNames[nextMeal]} in 1 hour`;
        } else {
            nextMealEl.textContent = `${mealNames[nextMeal]} soon`;
        }
    }
}

// ==================== MEAL TRACKING ====================

/**
 * Update meal progress when checkbox is toggled
 */
function updateMealProgress() {
    const meals = ['breakfast', 'lunch', 'dinner'];
    let completed = 0;
    let total = 3;
    
    // Check if snacks are visible
    const snackSection = document.getElementById('snackSection');
    if (snackSection && !snackSection.classList.contains('hidden')) {
        meals.push('morningSnack', 'eveningSnack');
        total = 5;
    }
    
    meals.forEach(meal => {
        const checkbox = document.getElementById(meal + 'Check');
        if (checkbox && checkbox.checked) {
            completed++;
            dashboardState.mealProgress[meal] = true;
        } else if (checkbox) {
            dashboardState.mealProgress[meal] = false;
        }
    });
    
    const percentage = (completed / total) * 100;
    
    // Update progress bar
    const progressBar = document.getElementById('mealProgressBar');
    if (progressBar) progressBar.style.width = percentage + '%';
    
    // Update progress text
    const progressText = document.getElementById('mealProgressText');
    if (progressText) progressText.textContent = `${completed}/${total} completed`;
    
    // Save progress
    localStorage.setItem('mealProgress', JSON.stringify({
        completed,
        total,
        percentage,
        date: new Date().toDateString(),
        meals: dashboardState.mealProgress
    }));
    
    // Track activity
    if (window.COPDApp && window.COPDApp.trackActivity) {
        window.COPDApp.trackActivity('meal_progress_updated', { 
            completed, 
            total,
            percentage 
        });
    }
    
    return { completed, total, percentage };
}

/**
 * Reset meal progress for new day
 */
function resetMealProgress() {
    const meals = ['breakfast', 'morningSnack', 'lunch', 'eveningSnack', 'dinner'];
    
    meals.forEach(meal => {
        const checkbox = document.getElementById(meal + 'Check');
        if (checkbox) {
            checkbox.checked = false;
        }
        dashboardState.mealProgress[meal] = false;
    });
    
    updateMealProgress();
    
    if (window.COPDApp) {
        window.COPDApp.showToast('Meal progress reset for new day', 'info');
    }
}

// ==================== WATER TRACKING ====================

/**
 * Add water intake
 */
function addWater() {
    if (dashboardState.waterIntake < dashboardState.waterTarget) {
        dashboardState.waterIntake++;
        updateWaterDisplay();
        
        // Track activity
        if (window.COPDApp && window.COPDApp.trackActivity) {
            window.COPDApp.trackActivity('water_added', { 
                current: dashboardState.waterIntake,
                target: dashboardState.waterTarget
            });
        }
    } else {
        if (window.COPDApp) {
            window.COPDApp.showToast('Daily water target reached! Great job!', 'success');
        }
    }
}

/**
 * Reset water intake
 */
function resetWater() {
    dashboardState.waterIntake = 0;
    updateWaterDisplay();
    
    if (window.COPDApp) {
        window.COPDApp.trackActivity('water_reset', {});
    }
}

/**
 * Update water intake display
 */
function updateWaterDisplay() {
    const waterIntakeEl = document.getElementById('waterIntake');
    const waterProgressEl = document.getElementById('waterProgress');
    const currentWaterEl = document.getElementById('currentWater');
    const waterTargetEl = document.getElementById('waterTarget');
    
    if (waterIntakeEl) waterIntakeEl.textContent = dashboardState.waterIntake;
    if (currentWaterEl) currentWaterEl.textContent = dashboardState.waterIntake;
    if (waterTargetEl) waterTargetEl.textContent = dashboardState.waterTarget;
    
    if (waterProgressEl) {
        const percentage = (dashboardState.waterIntake / dashboardState.waterTarget) * 100;
        waterProgressEl.style.width = percentage + '%';
    }
}

// ==================== HEALTH TIPS ====================

/**
 * Load health tip
 */
function loadHealthTip() {
    const healthTipEl = document.getElementById('healthTip');
    if (healthTipEl) {
        healthTipEl.textContent = DASHBOARD_CONFIG.HEALTH_TIPS[dashboardState.healthTipIndex];
    }
}

/**
 * Show next health tip
 */
function nextTip() {
    dashboardState.healthTipIndex = (dashboardState.healthTipIndex + 1) % DASHBOARD_CONFIG.HEALTH_TIPS.length;
    loadHealthTip();
    
    if (window.COPDApp) {
        window.COPDApp.trackActivity('health_tip_viewed', { 
            tipIndex: dashboardState.healthTipIndex 
        });
    }
}

// ==================== WEATHER FUNCTIONS ====================

/**
 * Load weather data
 */
async function loadWeatherData() {
    try {
        // Try to get user location for accurate weather
        if (window.COPDApp && window.COPDApp.getCurrentLocation) {
            try {
                const location = await window.COPDApp.getCurrentLocation();
                // In production, fetch real weather data using coordinates
                // For now, use mock data
                updateWeatherDisplay(dashboardState.weather);
            } catch (error) {
                console.log('Using default location for weather');
                updateWeatherDisplay(dashboardState.weather);
            }
        } else {
            // Mock weather data
            setTimeout(() => {
                updateWeatherDisplay(dashboardState.weather);
            }, 1000);
        }
    } catch (error) {
        console.error('Error loading weather:', error);
    }
}

/**
 * Update weather display
 * @param {object} weather - Weather data
 */
function updateWeatherDisplay(weather) {
    const weatherWidget = document.getElementById('weatherWidget');
    const weatherLocation = document.getElementById('weatherLocation');
    const weatherInfo = document.getElementById('weatherInfo');
    
    if (weatherWidget) {
        weatherWidget.innerHTML = `
            <i class="fas ${getWeatherIcon(weather.condition)} text-2xl"></i>
            <div>
                <div class="text-sm">${weather.location}</div>
                <div class="font-semibold">${weather.temp}°C ${weather.condition}</div>
            </div>
        `;
    }
    
    if (weatherLocation) weatherLocation.textContent = weather.location;
    if (weatherInfo) {
        weatherInfo.innerHTML = `
            <i class="fas ${getWeatherIcon(weather.condition)} mr-1"></i> 
            ${weather.temp}°C ${weather.condition}
        `;
    }
}

/**
 * Get weather icon based on condition
 * @param {string} condition - Weather condition
 * @returns {string} Font Awesome icon class
 */
function getWeatherIcon(condition) {
    const icons = {
        'Sunny': 'fa-sun',
        'Clear': 'fa-moon',
        'Cloudy': 'fa-cloud',
        'Partly Cloudy': 'fa-cloud-sun',
        'Rainy': 'fa-cloud-rain',
        'Stormy': 'fa-cloud-bolt',
        'Snowy': 'fa-snowflake',
        'Windy': 'fa-wind'
    };
    
    return icons[condition] || 'fa-cloud-sun';
}

// ==================== DOCTORS FUNCTIONS ====================

/**
 * Load local doctors
 */
function loadLocalDoctors() {
    const doctors = [
        {
            name: 'Dr. Rakesh Gupta',
            specialty: 'Senior Pulmonologist',
            hospital: 'Apex Lung Centre',
            area: 'Talwandi',
            available: true,
            rating: 4.8,
            distance: '1.2 km'
        },
        {
            name: 'Dr. Suman Verma',
            specialty: 'Chest Physician',
            hospital: 'Maitri Hospital',
            area: 'Vigan Nagar',
            available: true,
            rating: 4.7,
            distance: '2.5 km'
        },
        {
            name: 'Dr. Amit Jain',
            specialty: 'Pulmonologist',
            hospital: 'Kota Heart & Lung',
            area: 'Civil Lines',
            available: false,
            rating: 4.5,
            distance: '3.1 km'
        },
        {
            name: 'Dr. Priya Sharma',
            specialty: 'Respiratory Specialist',
            hospital: 'Shree Hospital',
            area: 'Kunadi',
            available: true,
            rating: 4.9,
            distance: '4.0 km'
        }
    ];
    
    const container = document.getElementById('doctorsList');
    if (!container) return;
    
    container.innerHTML = '';
    
    doctors.forEach(doctor => {
        const doctorEl = createDoctorElement(doctor);
        container.appendChild(doctorEl);
    });
    
    // Update doctor count
    const doctorCountEl = document.getElementById('doctorCount');
    if (doctorCountEl) {
        doctorCountEl.textContent = doctors.length + ' doctors';
    }
    
    // Update nearest doctor
    const nearestDoctorEl = document.getElementById('nearestDoctor');
    if (nearestDoctorEl) {
        nearestDoctorEl.textContent = 'Nearest: ' + doctors[0].name;
    }
}

/**
 * Create doctor element
 * @param {object} doctor - Doctor data
 * @returns {HTMLElement} Doctor element
 */
function createDoctorElement(doctor) {
    const div = document.createElement('div');
    div.className = 'flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg cursor-pointer transition';
    div.onclick = () => window.location.href = 'doctor-search.html';
    
    div.innerHTML = `
        <div class="flex items-center space-x-3">
            <div class="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                <i class="fas fa-user-md"></i>
            </div>
            <div>
                <div class="font-semibold text-sm">${doctor.name}</div>
                <div class="text-xs text-gray-500">${doctor.specialty} • ${doctor.hospital}</div>
                <div class="flex items-center mt-1">
                    <span class="text-xs text-yellow-500 mr-2">
                        <i class="fas fa-star"></i> ${doctor.rating}
                    </span>
                    <span class="text-xs text-gray-400">${doctor.distance}</span>
                </div>
            </div>
        </div>
        ${doctor.available ? 
            '<span class="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">Available</span>' : 
            '<span class="bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded-full">Busy</span>'}
    `;
    
    return div;
}

// ==================== NOTIFICATIONS ====================

/**
 * Load notifications
 */
function loadNotifications() {
    dashboardState.notifications = [
        {
            id: 1,
            title: 'Medicine Reminder',
            message: 'Time to take your inhaler',
            time: '5 min ago',
            icon: 'fa-clock',
            color: 'blue',
            read: false
        },
        {
            id: 2,
            title: 'Diet Plan Updated',
            message: 'New diet plan based on your risk level',
            time: '1 hour ago',
            icon: 'fa-utensils',
            color: 'green',
            read: false
        },
        {
            id: 3,
            title: 'Appointment Reminder',
            message: 'You have an appointment tomorrow at 10:00 AM',
            time: '2 hours ago',
            icon: 'fa-calendar',
            color: 'purple',
            read: false
        },
        {
            id: 4,
            title: 'Breathing Exercise',
            message: 'Don\'t forget your daily breathing exercises',
            time: '3 hours ago',
            icon: 'fa-lungs',
            color: 'blue',
            read: true
        },
        {
            id: 5,
            title: 'Health Tip',
            message: 'Stay hydrated! Drink 8 glasses of water today',
            time: '5 hours ago',
            icon: 'fa-droplet',
            color: 'blue',
            read: true
        }
    ];
    
    // Update notification badge
    updateNotificationBadge();
}

/**
 * Update notification badge
 */
function updateNotificationBadge() {
    const unreadCount = dashboardState.notifications.filter(n => !n.read).length;
    const badge = document.getElementById('notificationBadge');
    if (badge) {
        badge.textContent = unreadCount;
        badge.style.display = unreadCount > 0 ? 'flex' : 'none';
    }
}

/**
 * Show notifications modal
 */
function showNotifications() {
    const modal = document.getElementById('notificationModal');
    if (!modal) return;
    
    // Populate notifications
    const notificationsList = document.getElementById('notificationsList');
    if (notificationsList) {
        notificationsList.innerHTML = '';
        
        dashboardState.notifications.forEach(notif => {
            const notifEl = createNotificationElement(notif);
            notificationsList.appendChild(notifEl);
        });
    }
    
    modal.classList.remove('hidden');
}

/**
 * Create notification element
 * @param {object} notification - Notification data
 * @returns {HTMLElement} Notification element
 */
function createNotificationElement(notification) {
    const div = document.createElement('div');
    div.className = `p-3 ${notification.read ? 'bg-gray-50' : 'bg-blue-50'} rounded-lg hover:bg-gray-100 transition cursor-pointer`;
    div.onclick = () => markNotificationRead(notification.id);
    
    div.innerHTML = `
        <div class="flex items-start space-x-3">
            <div class="w-8 h-8 bg-${notification.color}-100 rounded-full flex items-center justify-center flex-shrink-0">
                <i class="fas ${notification.icon} text-${notification.color}-600"></i>
            </div>
            <div class="flex-1">
                <div class="flex justify-between items-start">
                    <h4 class="font-semibold text-sm">${notification.title}</h4>
                    ${!notification.read ? '<span class="w-2 h-2 bg-blue-600 rounded-full"></span>' : ''}
                </div>
                <p class="text-xs text-gray-600 mt-1">${notification.message}</p>
                <p class="text-xs text-gray-400 mt-1">${notification.time}</p>
            </div>
        </div>
    `;
    
    return div;
}

/**
 * Mark notification as read
 * @param {number} notificationId - Notification ID
 */
function markNotificationRead(notificationId) {
    const notification = dashboardState.notifications.find(n => n.id === notificationId);
    if (notification) {
        notification.read = true;
        updateNotificationBadge();
        showNotifications(); // Refresh modal
    }
}

/**
 * Mark all notifications as read
 */
function markAllNotificationsRead() {
    dashboardState.notifications.forEach(n => n.read = true);
    updateNotificationBadge();
    showNotifications();
    
    if (window.COPDApp) {
        window.COPDApp.showToast('All notifications marked as read', 'success');
    }
}

/**
 * Close notifications modal
 */
function closeNotifications() {
    const modal = document.getElementById('notificationModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

// ==================== RECENT ACTIVITIES ====================

/**
 * Load recent activities
 */
function loadRecentActivities() {
    // Try to load from localStorage
    const user = dashboardState.user;
    if (user && window.COPDApp && window.COPDApp.getFromStorage) {
        const activities = window.COPDApp.getFromStorage(`activities_${user.email}`);
        if (activities && activities.length > 0) {
            dashboardState.recentActivities = activities.slice(0, DASHBOARD_CONFIG.MAX_RECENT_ACTIVITIES);
        } else {
            // Mock activities
            dashboardState.recentActivities = [
                { action: 'Logged in', timestamp: new Date(Date.now() - 5 * 60000).toISOString() },
                { action: 'Viewed prediction', timestamp: new Date(Date.now() - 30 * 60000).toISOString() },
                { action: 'Updated diet plan', timestamp: new Date(Date.now() - 2 * 60 * 60000).toISOString() }
            ];
        }
    }
    
    // Display recent activities (if element exists)
    const activitiesContainer = document.getElementById('recentActivities');
    if (activitiesContainer) {
        displayRecentActivities();
    }
}

/**
 * Display recent activities
 */
function displayRecentActivities() {
    const container = document.getElementById('recentActivities');
    if (!container) return;
    
    container.innerHTML = '';
    
    dashboardState.recentActivities.forEach(activity => {
        const date = new Date(activity.timestamp);
        const timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        
        const div = document.createElement('div');
        div.className = 'flex items-center justify-between p-2 hover:bg-gray-50 rounded-lg';
        div.innerHTML = `
            <span class="text-sm text-gray-700">${activity.action}</span>
            <span class="text-xs text-gray-400">${timeStr}</span>
        `;
        container.appendChild(div);
    });
}

// ==================== CHARTS ====================

/**
 * Initialize charts
 */
function initCharts() {
    // Only initialize if Chart.js is available
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded');
        return;
    }
    
    // Risk trend chart
    const riskCtx = document.getElementById('riskTrendChart');
    if (riskCtx) {
        dashboardState.charts.riskTrend = new Chart(riskCtx, {
            type: 'line',
            data: {
                labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                datasets: [{
                    label: 'Risk Score',
                    data: [65, 68, 72, 70],
                    borderColor: DASHBOARD_CONFIG.CHART_COLORS.primary,
                    backgroundColor: DASHBOARD_CONFIG.CHART_COLORS.primary + '20',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
    
    // Symptom tracking chart
    const symptomCtx = document.getElementById('symptomChart');
    if (symptomCtx) {
        dashboardState.charts.symptom = new Chart(symptomCtx, {
            type: 'radar',
            data: {
                labels: ['Cough', 'Breathlessness', 'Wheezing', 'Fatigue', 'Chest Tightness'],
                datasets: [{
                    label: 'Symptom Severity',
                    data: [4, 3, 2, 4, 2],
                    backgroundColor: DASHBOARD_CONFIG.CHART_COLORS.warning + '40',
                    borderColor: DASHBOARD_CONFIG.CHART_COLORS.warning,
                    pointBackgroundColor: DASHBOARD_CONFIG.CHART_COLORS.warning
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 5
                    }
                }
            }
        });
    }
}

// ==================== AUTO-REFRESH ====================

/**
 * Set up auto-refresh
 */
function setupAutoRefresh() {
    // Clear any existing timer
    if (dashboardState.timers.refresh) {
        clearInterval(dashboardState.timers.refresh);
    }
    
    // Set new timer
    dashboardState.timers.refresh = setInterval(() => {
        refreshDashboardData();
    }, DASHBOARD_CONFIG.REFRESH_INTERVAL);
}

/**
 * Refresh dashboard data
 */
async function refreshDashboardData() {
    console.log('Refreshing dashboard data...');
    
    // Reload weather
    await loadWeatherData();
    
    // Update next meal
    const dietPlan = loadDietPlan();
    updateNextMeal(dietPlan.times);
    
    // Reload notifications
    loadNotifications();
    
    // Show toast notification
    if (window.COPDApp) {
        window.COPDApp.showToast('Dashboard updated', 'info');
    }
}

// ==================== EXERCISE FUNCTIONS ====================

/**
 * Start exercise
 * @param {number} exerciseId - Exercise ID
 */
function startExercise(exerciseId) {
    const exercise = DASHBOARD_CONFIG.BREATHING_EXERCISES[exerciseId];
    if (!exercise) return;
    
    const modal = document.getElementById('exerciseModal');
    if (!modal) return;
    
    // Set exercise title
    const titleEl = document.getElementById('exerciseTitle');
    if (titleEl) titleEl.textContent = exercise.name;
    
    // Set exercise content
    const contentEl = document.getElementById('exerciseContent');
    if (contentEl) {
        contentEl.innerHTML = `
            <p class="text-gray-700 mb-4">${exercise.description}</p>
            
            <h4 class="font-semibold mb-2 text-sm">Steps:</h4>
            <ol class="list-decimal list-inside space-y-2 text-sm text-gray-600 mb-4">
                ${exercise.steps.map(step => `<li>${step}</li>`).join('')}
            </ol>
            
            <h4 class="font-semibold mb-2 text-sm">Benefits:</h4>
            <ul class="list-disc list-inside space-y-1 text-sm text-gray-600">
                ${exercise.benefits.map(benefit => `<li>${benefit}</li>`).join('')}
            </ul>
        `;
    }
    
    // Store exercise ID for timer
    modal.dataset.exerciseId = exerciseId;
    modal.dataset.duration = exercise.duration;
    
    modal.classList.remove('hidden');
    
    // Track activity
    if (window.COPDApp && window.COPDApp.trackActivity) {
        window.COPDApp.trackActivity('exercise_started', { 
            exerciseId, 
            exerciseName: exercise.name 
        });
    }
}

/**
 * Close exercise modal
 */
function closeExercise() {
    const modal = document.getElementById('exerciseModal');
    if (modal) {
        modal.classList.add('hidden');
        resetTimer();
    }
}

/**
 * Start exercise timer
 */
function startTimer() {
    const modal = document.getElementById('exerciseModal');
    if (!modal) return;
    
    const duration = parseInt(modal.dataset.duration) || 300;
    const timerDisplay = document.getElementById('timerDisplay');
    const startBtn = event.target;
    
    if (!timerDisplay) return;
    
    timerDisplay.classList.remove('hidden');
    startBtn.disabled = true;
    startBtn.classList.add('opacity-50', 'cursor-not-allowed');
    
    let seconds = duration;
    updateTimerDisplay(seconds);
    
    dashboardState.timers.exercise = setInterval(() => {
        seconds--;
        updateTimerDisplay(seconds);
        
        if (seconds <= 0) {
            clearInterval(dashboardState.timers.exercise);
            timerDisplay.textContent = "00:00";
            
            if (window.COPDApp) {
                window.COPDApp.showToast('Exercise completed! Great job!', 'success');
            }
            
            // Track completion
            if (window.COPDApp && window.COPDApp.trackActivity) {
                window.COPDApp.trackActivity('exercise_completed', { 
                    exerciseId: modal.dataset.exerciseId,
                    duration: duration
                });
            }
            
            setTimeout(() => {
                closeExercise();
            }, 2000);
        }
    }, 1000);
}

/**
 * Update timer display
 * @param {number} seconds - Seconds remaining
 */
function updateTimerDisplay(seconds) {
    const timerDisplay = document.getElementById('timerDisplay');
    if (!timerDisplay) return;
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

/**
 * Reset timer
 */
function resetTimer() {
    if (dashboardState.timers.exercise) {
        clearInterval(dashboardState.timers.exercise);
        dashboardState.timers.exercise = null;
    }
    
    const timerDisplay = document.getElementById('timerDisplay');
    if (timerDisplay) {
        timerDisplay.classList.add('hidden');
        timerDisplay.textContent = '05:00';
    }
    
    // Re-enable start button
    const startBtn = document.querySelector('#exerciseModal button.bg-blue-600');
    if (startBtn) {
        startBtn.disabled = false;
        startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
}

// ==================== AI CHAT FUNCTIONS ====================

/**
 * Send quick message to AI
 */
function sendQuickMessage() {
    const input = document.getElementById('quickMessage');
    if (!input || !input.value.trim()) return;
    
    const message = input.value.trim();
    const chatPreview = document.getElementById('chatPreview');
    
    if (!chatPreview) return;
    
    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'mb-2 text-right';
    userMsg.innerHTML = `
        <div class="bg-blue-600 text-white p-2 rounded-lg inline-block max-w-[80%] text-sm">
            ${message}
        </div>
        <div class="text-xs text-gray-500 mt-1">You • Just now</div>
    `;
    chatPreview.appendChild(userMsg);
    
    // Clear input
    input.value = '';
    
    // Scroll to bottom
    chatPreview.scrollTop = chatPreview.scrollHeight;
    
    // Simulate AI response
    setTimeout(() => {
        const aiResponse = getAIResponse(message);
        const aiMsg = document.createElement('div');
        aiMsg.className = 'mb-2';
        aiMsg.innerHTML = `
            <div class="bg-gray-100 text-gray-800 p-2 rounded-lg inline-block max-w-[80%] text-sm">
                ${aiResponse}
            </div>
            <div class="text-xs text-gray-500 mt-1">AI Assistant • Just now</div>
        `;
        chatPreview.appendChild(aiMsg);
        chatPreview.scrollTop = chatPreview.scrollHeight;
        
        // Track activity
        if (window.COPDApp && window.COPDApp.trackActivity) {
            window.COPDApp.trackActivity('ai_chat', { 
                message: message.substring(0, 50) 
            });
        }
    }, 1000);
}

/**
 * Quick question button
 * @param {string} question - Question text
 */
function quickQuestion(question) {
    const input = document.getElementById('quickMessage');
    if (input) {
        input.value = question;
        sendQuickMessage();
    }
}

/**
 * Get AI response based on message
 * @param {string} message - User message
 * @returns {string} AI response
 */
function getAIResponse(message) {
    const lowerMsg = message.toLowerCase();
    
    if (lowerMsg.includes('risk') || lowerMsg.includes('level')) {
        return `Your current risk level is ${document.getElementById('riskLevel')?.textContent || 'Mild Risk'}. Would you like to do a new assessment?`;
    }
    else if (lowerMsg.includes('diet') || lowerMsg.includes('food') || lowerMsg.includes('eat')) {
        return `Today's diet: ${document.getElementById('breakfastMeal')?.textContent} for breakfast, ${document.getElementById('lunchMeal')?.textContent} for lunch, and ${document.getElementById('dinnerMeal')?.textContent} for dinner.`;
    }
    else if (lowerMsg.includes('doctor') || lowerMsg.includes('specialist')) {
        return "I found several pulmonologists near you. Visit the Doctor Search page to view and book appointments.";
    }
    else if (lowerMsg.includes('hello') || lowerMsg.includes('hi')) {
        return "Hello! How can I help you with your COPD management today?";
    }
    else if (lowerMsg.includes('exercise') || lowerMsg.includes('breathe')) {
        return "Try pursed-lip breathing or diaphragmatic breathing. Click on the Breathing Exercises section to start a guided session.";
    }
    else if (lowerMsg.includes('medicine') || lowerMsg.includes('medication')) {
        return "Don't forget to take your medications as prescribed. Set reminders in the WhatsApp Agent for medicine alerts.";
    }
    else {
        return "I'll help you with that! For detailed assistance, please visit the WhatsApp Agent page or ask me something specific about your COPD management.";
    }
}

// ==================== REPORT FUNCTIONS ====================

/**
 * Export health report
 */
function exportHealthReport() {
    const report = {
        user: dashboardState.user,
        riskLevel: document.getElementById('riskLevel')?.textContent || 'Unknown',
        riskPercentage: document.getElementById('riskPercentage')?.textContent || 'Unknown',
        diet: {
            breakfast: document.getElementById('breakfastMeal')?.textContent || 'Unknown',
            lunch: document.getElementById('lunchMeal')?.textContent || 'Unknown',
            dinner: document.getElementById('dinnerMeal')?.textContent || 'Unknown'
        },
        mealProgress: dashboardState.mealProgress,
        waterIntake: dashboardState.waterIntake,
        waterTarget: dashboardState.waterTarget,
        recentActivities: dashboardState.recentActivities,
        timestamp: new Date().toISOString()
    };
    
    const dataStr = JSON.stringify(report, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `health-report-${new Date().toISOString().split('T')[0]}.json`;
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    if (window.COPDApp) {
        window.COPDApp.showToast('Health report downloaded!', 'success');
        window.COPDApp.trackActivity('report_exported', {});
    }
}

/**
 * Share health report on WhatsApp
 */
function shareHealthReport() {
    const phoneNumber = "+14155238886";
    const riskLevel = document.getElementById('riskLevel')?.textContent || 'Unknown';
    const breakfast = document.getElementById('breakfastMeal')?.textContent || 'Unknown';
    const lunch = document.getElementById('lunchMeal')?.textContent || 'Unknown';
    const dinner = document.getElementById('dinnerMeal')?.textContent || 'Unknown';
    
    const message = encodeURIComponent(
        `*My Health Report - COPD Care AI*\n\n` +
        `📊 *Risk Level:* ${riskLevel}\n` +
        `🥗 *Today's Diet:*\n` +
        `• Breakfast: ${breakfast}\n` +
        `• Lunch: ${lunch}\n` +
        `• Dinner: ${dinner}\n` +
        `💧 *Water Intake:* ${dashboardState.waterIntake}/${dashboardState.waterTarget} cups\n\n` +
        `Sent from COPD Care AI`
    );
    
    window.open(`https://wa.me/${phoneNumber}?text=${message}`, '_blank');
    
    if (window.COPDApp) {
        window.COPDApp.trackActivity('report_shared', {});
    }
}

// ==================== CLEANUP ====================

/**
 * Clean up dashboard resources
 */
function cleanupDashboard() {
    // Clear all timers
    Object.values(dashboardState.timers).forEach(timer => {
        if (timer) clearInterval(timer);
    });
    
    // Destroy charts
    Object.values(dashboardState.charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    
    console.log('Dashboard cleaned up');
}

// ==================== INITIALIZE ON PAGE LOAD ====================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard
    initDashboard();
});

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    cleanupDashboard();
});

// ==================== EXPORT FUNCTIONS ====================

// Make functions globally available
window.initDashboard = initDashboard;
window.updateMealProgress = updateMealProgress;
window.resetMealProgress = resetMealProgress;
window.addWater = addWater;
window.resetWater = resetWater;
window.nextTip = nextTip;
window.showNotifications = showNotifications;
window.closeNotifications = closeNotifications;
window.markAllNotificationsRead = markAllNotificationsRead;
window.startExercise = startExercise;
window.closeExercise = closeExercise;
window.startTimer = startTimer;
window.sendQuickMessage = sendQuickMessage;
window.quickQuestion = quickQuestion;
window.exportHealthReport = exportHealthReport;
window.shareHealthReport = shareHealthReport;
window.logout = function() {
    if (window.COPDApp && window.COPDApp.logout) {
        window.COPDApp.logout();
    } else {
        window.location.href = 'login.html';
    }
    
};
// Emergency call function
const emergencyNumber = "102";  // 108 या 102 जो भी चाहें

function confirmEmergencyCall() {
    if (confirm(`Are you sure you want to call emergency services (${emergencyNumber})?`)) {
        window.location.href = `tel:${emergencyNumber}`;
    }
}

// ==================== EXPORT FOR MODULE USE ====================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initDashboard,
        updateMealProgress,
        addWater,
        resetWater,
        nextTip,
        DASHBOARD_CONFIG
    };
}