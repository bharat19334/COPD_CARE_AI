/**
 * COPD Care AI - Doctor Search & Management JavaScript File
 * Technovation 2026 Project
 * Handles all doctor-related functionality, map integration, and appointment booking
 */

// ==================== DOCTOR CONFIGURATION ====================

const DOCTOR_CONFIG = {
    DEFAULT_LOCATION: {
        city: 'Kota',
        state: 'Rajasthan',
        lat: 25.1765,
        lng: 75.8451
    },
    SEARCH_RADIUS: 10, // km
    SORT_OPTIONS: {
        DISTANCE: 'distance',
        RATING: 'rating',
        EXPERIENCE: 'experience',
        PRICE_LOW: 'price_low',
        PRICE_HIGH: 'price_high'
    },
    SPECIALTIES: [
        'Pulmonology',
        'Respiratory Medicine',
        'Critical Care',
        'Allergy & Immunology',
        'Sleep Medicine',
        'Thoracic Surgery'
    ],
    AVAILABILITY: [
        'Today',
        'Tomorrow',
        'This Week',
        'Next Week',
        'Any Time'
    ]
};

// ==================== COMPLETE DOCTORS DATABASE ====================

const DOCTORS_DATABASE = [
    {
        id: 1,
        name: 'Dr. Rakesh Gupta',
        rating: 4.8,
        reviews: 150,
        qualification: 'MD, FCCP',
        specialty: 'Senior Pulmonologist',
        hospital: 'Apex Lung Centre',
        address: 'Talwandi, Kota, Rajasthan 324005',
        area: 'Talwandi',
        city: 'Kota',
        state: 'Rajasthan',
        experience: 18,
        timing: {
            monday: '10:00 AM - 7:00 PM',
            tuesday: '10:00 AM - 7:00 PM',
            wednesday: '10:00 AM - 7:00 PM',
            thursday: '10:00 AM - 7:00 PM',
            friday: '10:00 AM - 7:00 PM',
            saturday: '10:00 AM - 7:00 PM',
            sunday: 'Closed'
        },
        fee: {
            initial: 800,
            followup: 500
        },
        available: true,
        availableToday: true,
        coordinates: { lat: 25.1765, lng: 75.8451 },
        phone: '+91 98765 43210',
        email: 'dr.rakesh.gupta@apexlung.com',
        languages: ['English', 'Hindi', 'Rajasthani'],
        education: [
            'MD - Pulmonary Medicine, AIIMS Delhi',
            'FCCP - American College of Chest Physicians',
            'MBBS - SMS Medical College, Jaipur'
        ],
        expertise: [
            'COPD Management',
            'Asthma',
            'Interstitial Lung Disease',
            'Pulmonary Rehabilitation',
            'Bronchoscopy',
            'Sleep Apnea'
        ],
        about: 'Dr. Rakesh Gupta is a highly experienced pulmonologist with over 18 years of expertise in treating respiratory conditions including COPD, asthma, and interstitial lung disease. He is known for his patient-centric approach and commitment to providing comprehensive care.',
        awards: [
            'Best Pulmonologist Award - Kota Medical Association 2023',
            'Excellence in Respiratory Care - 2022',
            'Research Publication in International Journal of COPD'
        ],
        insurance: ['ICICI Lombard', 'Star Health', 'New India Assurance', 'HDFC Ergo'],
        videoConsultation: true,
        emergencyAvailable: true,
        distance: '1.2 km',
        image: '/assets/images/doctors/doctor1.jpg'
    },
    {
        id: 2,
        name: 'Dr. Suman Verma',
        rating: 4.7,
        reviews: 88,
        qualification: 'DNB',
        specialty: 'Chest Physician & Allergist',
        hospital: 'Maitri Hospital',
        address: 'Vigan Nagar, Kota, Rajasthan 324005',
        area: 'Vigan Nagar',
        city: 'Kota',
        state: 'Rajasthan',
        experience: 10,
        timing: {
            monday: '11:00 AM - 5:00 PM',
            tuesday: '11:00 AM - 5:00 PM',
            wednesday: '11:00 AM - 5:00 PM',
            thursday: '11:00 AM - 5:00 PM',
            friday: '11:00 AM - 5:00 PM',
            saturday: '11:00 AM - 5:00 PM',
            sunday: 'Closed'
        },
        fee: {
            initial: 600,
            followup: 400
        },
        available: true,
        availableToday: true,
        coordinates: { lat: 25.1698, lng: 75.8423 },
        phone: '+91 98765 43211',
        email: 'dr.suman.verma@maitrihospital.com',
        languages: ['English', 'Hindi'],
        education: [
            'DNB - Respiratory Medicine',
            'MBBS - Government Medical College, Kota'
        ],
        expertise: [
            'Allergic Asthma',
            'COPD',
            'Respiratory Allergies',
            'Pulmonary Function Testing',
            'Immunotherapy'
        ],
        about: 'Dr. Suman Verma is a dedicated chest physician specializing in allergic respiratory conditions. With 10 years of experience, she provides comprehensive care for asthma, allergies, and COPD patients.',
        awards: [
            'Young Achiever Award - 2021',
            'Best Allergist - Kota Health Summit 2022'
        ],
        insurance: ['Star Health', 'New India Assurance', 'Bajaj Allianz'],
        videoConsultation: true,
        emergencyAvailable: false,
        distance: '2.5 km',
        image: '/assets/images/doctors/doctor2.jpg'
    },
    {
        id: 3,
        name: 'Dr. Amit Jain',
        rating: 4.5,
        reviews: 65,
        qualification: 'MD',
        specialty: 'Pulmonologist',
        hospital: 'Kota Heart & Lung Institute',
        address: 'Civil Lines, Kota, Rajasthan 324001',
        area: 'Civil Lines',
        city: 'Kota',
        state: 'Rajasthan',
        experience: 8,
        timing: {
            monday: '8:00 AM - 2:00 PM',
            tuesday: '8:00 AM - 2:00 PM',
            wednesday: '8:00 AM - 2:00 PM',
            thursday: '8:00 AM - 2:00 PM',
            friday: '8:00 AM - 2:00 PM',
            saturday: '9:00 AM - 1:00 PM',
            sunday: 'Closed'
        },
        fee: {
            initial: 500,
            followup: 300
        },
        available: false,
        availableToday: false,
        coordinates: { lat: 25.1723, lng: 75.8489 },
        phone: '+91 98765 43212',
        email: 'dr.amit.jain@khl.in',
        languages: ['English', 'Hindi'],
        education: [
            'MD - Pulmonary Medicine',
            'MBBS - RNT Medical College, Udaipur'
        ],
        expertise: [
            'COPD',
            'Asthma',
            'Respiratory Infections',
            'Tuberculosis',
            'Sleep Disorders'
        ],
        about: 'Dr. Amit Jain is a young and dynamic pulmonologist with expertise in managing various respiratory conditions. He focuses on preventive care and patient education.',
        awards: [],
        insurance: ['ICICI Lombard', 'HDFC Ergo'],
        videoConsultation: false,
        emergencyAvailable: true,
        distance: '3.1 km',
        image: '/assets/images/doctors/doctor3.jpg'
    },
    {
        id: 4,
        name: 'Dr. Priya Sharma',
        rating: 4.9,
        reviews: 200,
        qualification: 'MD, FCCP',
        specialty: 'Respiratory Specialist',
        hospital: 'Shree Hospital',
        address: 'Kunadi, Kota, Rajasthan 324008',
        area: 'Kunadi',
        city: 'Kota',
        state: 'Rajasthan',
        experience: 15,
        timing: {
            monday: '9:00 AM - 6:00 PM',
            tuesday: '9:00 AM - 6:00 PM',
            wednesday: '9:00 AM - 6:00 PM',
            thursday: '9:00 AM - 6:00 PM',
            friday: '9:00 AM - 6:00 PM',
            saturday: '9:00 AM - 4:00 PM',
            sunday: 'Closed'
        },
        fee: {
            initial: 900,
            followup: 600
        },
        available: true,
        availableToday: true,
        coordinates: { lat: 25.1634, lng: 75.8312 },
        phone: '+91 98765 43213',
        email: 'dr.priya.sharma@shreehospital.com',
        languages: ['English', 'Hindi', 'Punjabi'],
        education: [
            'MD - Pulmonary Medicine, PGI Chandigarh',
            'FCCP - Royal College of Physicians',
            'MBBS - Lady Hardinge Medical College, Delhi'
        ],
        expertise: [
            'Interventional Pulmonology',
            'Lung Cancer Screening',
            'COPD',
            'Pulmonary Hypertension',
            'Critical Care',
            'Bronchoscopy'
        ],
        about: 'Dr. Priya Sharma is one of Kota\'s most respected respiratory specialists. With training from top institutions, she offers advanced care for complex respiratory conditions.',
        awards: [
            'Excellence in Pulmonology - 2023',
            'Best Woman Doctor Award - 2022',
            'Research Excellence Award - 2021'
        ],
        insurance: ['All Major Insurances Accepted'],
        videoConsultation: true,
        emergencyAvailable: true,
        distance: '4.0 km',
        image: '/assets/images/doctors/doctor4.jpg'
    },
    {
        id: 5,
        name: 'Dr. Vikas Mehta',
        rating: 4.6,
        reviews: 120,
        qualification: 'DM',
        specialty: 'Critical Care Specialist',
        hospital: 'MBS Hospital',
        address: 'Nayapura, Kota, Rajasthan 324001',
        area: 'Nayapura',
        city: 'Kota',
        state: 'Rajasthan',
        experience: 12,
        timing: {
            monday: '10:00 AM - 8:00 PM',
            tuesday: '10:00 AM - 8:00 PM',
            wednesday: '10:00 AM - 8:00 PM',
            thursday: '10:00 AM - 8:00 PM',
            friday: '10:00 AM - 8:00 PM',
            saturday: '10:00 AM - 6:00 PM',
            sunday: 'Emergency Only'
        },
        fee: {
            initial: 700,
            followup: 400
        },
        available: true,
        availableToday: false,
        coordinates: { lat: 25.1689, lng: 75.8398 },
        phone: '+91 98765 43214',
        email: 'dr.vikas.mehta@mbshospital.com',
        languages: ['English', 'Hindi'],
        education: [
            'DM - Critical Care Medicine',
            'MD - Internal Medicine',
            'MBBS - SMS Medical College, Jaipur'
        ],
        expertise: [
            'Critical Care',
            'Ventilator Management',
            'Sepsis',
            'ARDS',
            'COPD Exacerbations',
            'Post-ICU Rehabilitation'
        ],
        about: 'Dr. Vikas Mehta specializes in critical care management of respiratory patients. He has extensive experience in managing severe COPD exacerbations and ventilator-dependent patients.',
        awards: [
            'Critical Care Excellence Award - 2022'
        ],
        insurance: ['Star Health', 'New India Assurance', 'ICICI Lombard'],
        videoConsultation: true,
        emergencyAvailable: true,
        distance: '2.8 km',
        image: '/assets/images/doctors/doctor5.jpg'
    },
    {
        id: 6,
        name: 'Dr. Neha Gupta',
        rating: 4.8,
        reviews: 95,
        qualification: 'MD',
        specialty: 'Pediatric Pulmonologist',
        hospital: 'Children\'s Lung Center',
        address: 'Dadabari, Kota, Rajasthan 324009',
        area: 'Dadabari',
        city: 'Kota',
        state: 'Rajasthan',
        experience: 9,
        timing: {
            monday: '9:00 AM - 5:00 PM',
            tuesday: '9:00 AM - 5:00 PM',
            wednesday: '9:00 AM - 5:00 PM',
            thursday: '9:00 AM - 5:00 PM',
            friday: '9:00 AM - 5:00 PM',
            saturday: '9:00 AM - 2:00 PM',
            sunday: 'Closed'
        },
        fee: {
            initial: 650,
            followup: 400
        },
        available: true,
        availableToday: true,
        coordinates: { lat: 25.1587, lng: 75.8523 },
        phone: '+91 98765 43215',
        email: 'dr.neha.gupta@childrenslung.in',
        languages: ['English', 'Hindi'],
        education: [
            'MD - Pediatric Pulmonology',
            'MBBS - AIIMS Delhi'
        ],
        expertise: [
            'Pediatric Asthma',
            'Childhood COPD',
            'Respiratory Allergies in Children',
            'Cystic Fibrosis',
            'Neonatal Respiratory Care'
        ],
        about: 'Dr. Neha Gupta is a specialist in pediatric respiratory conditions, providing compassionate care for children with asthma, allergies, and other breathing problems.',
        awards: [
            'Best Pediatric Pulmonologist - 2023'
        ],
        insurance: ['Star Health', 'New India Assurance'],
        videoConsultation: true,
        emergencyAvailable: true,
        distance: '3.5 km',
        image: '/assets/images/doctors/doctor6.jpg'
    },
    {
        id: 7,
        name: 'Dr. Rajesh Kumar',
        rating: 4.4,
        reviews: 78,
        qualification: 'MD',
        specialty: 'Sleep Medicine Specialist',
        hospital: 'Kota Sleep Clinic',
        address: 'Mahaveer Nagar, Kota, Rajasthan 324010',
        area: 'Mahaveer Nagar',
        city: 'Kota',
        state: 'Rajasthan',
        experience: 7,
        timing: {
            monday: '4:00 PM - 9:00 PM',
            tuesday: '4:00 PM - 9:00 PM',
            wednesday: '4:00 PM - 9:00 PM',
            thursday: '4:00 PM - 9:00 PM',
            friday: '4:00 PM - 9:00 PM',
            saturday: '10:00 AM - 2:00 PM',
            sunday: 'Closed'
        },
        fee: {
            initial: 550,
            followup: 350
        },
        available: true,
        availableToday: true,
        coordinates: { lat: 25.1812, lng: 75.8621 },
        phone: '+91 98765 43216',
        email: 'dr.rajesh.kumar@kotasleep.in',
        languages: ['English', 'Hindi'],
        education: [
            'MD - Pulmonary Medicine',
            'Fellowship in Sleep Medicine'
        ],
        expertise: [
            'Sleep Apnea',
            'Insomnia',
            'CPAP Therapy',
            'Sleep Studies',
            'COPD-Related Sleep Disorders'
        ],
        about: 'Dr. Rajesh Kumar specializes in sleep-related breathing disorders. He helps patients with sleep apnea and other conditions that affect breathing during sleep.',
        awards: [],
        insurance: ['ICICI Lombard', 'HDFC Ergo'],
        videoConsultation: false,
        emergencyAvailable: false,
        distance: '4.2 km',
        image: '/assets/images/doctors/doctor7.jpg'
    }
];

// ==================== STATE MANAGEMENT ====================

let currentDoctors = [...DOCTORS_DATABASE];
let selectedDoctor = null;
let mapInstance = null;
let mapMarkers = [];
let userMarker = null;
let searchRadius = 10;
let currentLocation = DOCTOR_CONFIG.DEFAULT_LOCATION;

// ==================== INITIALIZATION ====================

/**
 * Initialize doctor search page
 */
function initDoctorSearch() {
    // Check authentication
    if (!window.COPDApp || !window.COPDApp.isLoggedIn()) {
        window.location.href = 'login.html';
        return;
    }
    
    // Set current date
    updateCurrentDate();
    
    // Load specialties dropdown
    loadSpecialties();
    
    // Load availability dropdown
    loadAvailability();
    
    // Initialize map
    initMap();
    
    // Load initial doctors
    displayDoctors(currentDoctors);
    
    // Setup event listeners
    setupSearchListeners();
}

/**
 * Update current date display
 */
function updateCurrentDate() {
    const dateElement = document.getElementById('currentDate');
    if (dateElement) {
        const options = { year: 'numeric', month: 'long', day: 'numeric' };
        dateElement.textContent = new Date().toLocaleDateString('en-US', options);
    }
}

/**
 * Load specialties into dropdown
 */
function loadSpecialties() {
    const specialtySelect = document.getElementById('specialty');
    if (specialtySelect) {
        specialtySelect.innerHTML = '<option value="">All Specialties</option>' +
            DOCTOR_CONFIG.SPECIALTIES.map(s => `<option value="${s}">${s}</option>`).join('');
    }
}

/**
 * Load availability options into dropdown
 */
function loadAvailability() {
    const availabilitySelect = document.getElementById('availability');
    if (availabilitySelect) {
        availabilitySelect.innerHTML = DOCTOR_CONFIG.AVAILABILITY.map(a => 
            `<option value="${a}">${a}</option>`
        ).join('');
    }
}

/**
 * Setup search event listeners
 */
function setupSearchListeners() {
    const searchBtn = document.getElementById('searchBtn');
    if (searchBtn) {
        searchBtn.addEventListener('click', performSearch);
    }
    
    const locationInput = document.getElementById('location');
    if (locationInput) {
        locationInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performSearch();
        });
    }
    
    const radiusSelect = document.getElementById('radius');
    if (radiusSelect) {
        radiusSelect.addEventListener('change', (e) => {
            searchRadius = parseInt(e.target.value);
            performSearch();
        });
    }
    
    const sortSelect = document.getElementById('sortBy');
    if (sortSelect) {
        sortSelect.addEventListener('change', sortDoctors);
    }
}

// ==================== MAP FUNCTIONS ====================

/**
 * Initialize Leaflet map
 */
function initMap() {
    // Check if map container exists
    const mapContainer = document.getElementById('doctorMap');
    if (!mapContainer) return;
    
    // Initialize map
    mapInstance = L.map('doctorMap').setView(
        [currentLocation.lat, currentLocation.lng], 
        13
    );
    
    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(mapInstance);
    
    // Add user location marker
    addUserMarker(currentLocation.lat, currentLocation.lng);
    
    // Add doctor markers
    addDoctorMarkers(currentDoctors);
    
    // Add radius circle
    addRadiusCircle(currentLocation.lat, currentLocation.lng, searchRadius * 1000);
}

/**
 * Add user location marker to map
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 */
function addUserMarker(lat, lng) {
    if (!mapInstance) return;
    
    // Remove existing user marker
    if (userMarker) {
        mapInstance.removeLayer(userMarker);
    }
    
    // Create custom icon for user location
    const userIcon = L.divIcon({
        className: 'user-location-marker',
        html: `<div style="background-color: #2563eb; width: 20px; height: 20px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 10px rgba(0,0,0,0.3);"></div>`,
        iconSize: [20, 20],
        popupAnchor: [0, -10]
    });
    
    userMarker = L.marker([lat, lng], { icon: userIcon }).addTo(mapInstance);
    userMarker.bindPopup('Your Location');
}

/**
 * Add doctor markers to map
 * @param {Array} doctors - List of doctors
 */
function addDoctorMarkers(doctors) {
    if (!mapInstance) return;
    
    // Clear existing markers
    mapMarkers.forEach(marker => mapInstance.removeLayer(marker));
    mapMarkers = [];
    
    // Create custom icon for doctors
    const doctorIcon = L.divIcon({
        className: 'doctor-marker',
        html: `<div style="background-color: white; width: 30px; height: 30px; border-radius: 50%; border: 2px solid #2563eb; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                <i class="fas fa-user-md" style="color: #2563eb; font-size: 14px;"></i>
               </div>`,
        iconSize: [30, 30],
        popupAnchor: [0, -15]
    });
    
    doctors.forEach(doctor => {
        if (doctor.coordinates) {
            const marker = L.marker(
                [doctor.coordinates.lat, doctor.coordinates.lng], 
                { icon: doctorIcon }
            ).addTo(mapInstance);
            
            marker.bindPopup(createDoctorPopup(doctor));
            
            marker.on('click', () => {
                selectDoctor(doctor.id);
            });
            
            mapMarkers.push(marker);
        }
    });
}

/**
 * Create HTML for doctor popup
 * @param {object} doctor - Doctor object
 * @returns {string} HTML string
 */
function createDoctorPopup(doctor) {
    return `
        <div style="min-width: 250px; padding: 10px;">
            <h3 style="font-weight: bold; font-size: 16px; margin-bottom: 5px;">${doctor.name}</h3>
            <p style="margin: 2px 0; font-size: 13px; color: #4b5563;">${doctor.specialty}</p>
            <p style="margin: 2px 0; font-size: 13px;">
                <span style="color: #fbbf24;">★</span> ${doctor.rating} (${doctor.reviews} reviews)
            </p>
            <p style="margin: 2px 0; font-size: 13px;">${doctor.hospital}</p>
            <p style="margin: 2px 0; font-size: 13px;">💰 ₹${doctor.fee.initial}</p>
            <p style="margin: 5px 0; font-size: 12px; color: #6b7280;">📍 ${doctor.distance}</p>
            <button onclick="selectDoctor(${doctor.id})" 
                    style="background-color: #2563eb; color: white; border: none; padding: 8px 16px; border-radius: 6px; margin-top: 8px; width: 100%; cursor: pointer; font-size: 13px;">
                View Profile & Book
            </button>
        </div>
    `;
}

/**
 * Add radius circle to map
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 * @param {number} radius - Radius in meters
 */
function addRadiusCircle(lat, lng, radius) {
    if (!mapInstance) return;
    
    // Remove existing circles
    mapInstance.eachLayer(layer => {
        if (layer instanceof L.Circle) {
            mapInstance.removeLayer(layer);
        }
    });
    
    // Add new circle
    L.circle([lat, lng], {
        color: '#2563eb',
        fillColor: '#2563eb',
        fillOpacity: 0.1,
        radius: radius
    }).addTo(mapInstance);
}

// ==================== DOCTOR DISPLAY FUNCTIONS ====================

/**
 * Display doctors in list view
 * @param {Array} doctors - List of doctors to display
 */
function displayDoctors(doctors) {
    const doctorList = document.getElementById('doctorList');
    const resultCount = document.getElementById('resultCount');
    
    if (!doctorList) return;
    
    // Update result count
    if (resultCount) {
        resultCount.textContent = `${doctors.length} doctors found`;
    }
    
    if (doctors.length === 0) {
        doctorList.innerHTML = `
            <div class="text-center py-12">
                <div class="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i class="fas fa-user-md text-4xl text-gray-400"></i>
                </div>
                <h3 class="text-lg font-semibold text-gray-800 mb-2">No Doctors Found</h3>
                <p class="text-gray-600 mb-4">No doctors match your search criteria</p>
                <button onclick="resetSearch()" class="text-blue-600 hover:text-blue-700">
                    <i class="fas fa-redo-alt mr-2"></i>Reset Search
                </button>
            </div>
        `;
        return;
    }
    
    let html = '';
    doctors.forEach(doctor => {
        html += createDoctorCard(doctor);
    });
    
    doctorList.innerHTML = html;
}

/**
 * Create HTML for doctor card
 * @param {object} doctor - Doctor object
 * @returns {string} HTML string
 */
function createDoctorCard(doctor) {
    const availabilityClass = doctor.availableToday ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800';
    const availabilityText = doctor.availableToday ? 'Available Today' : 'Next Available Tomorrow';
    
    return `
        <div class="doctor-card bg-white rounded-xl shadow-sm p-6 hover:shadow-lg transition cursor-pointer" onclick="selectDoctor(${doctor.id})">
            <div class="flex items-start space-x-4">
                <div class="w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-user-md text-white text-2xl"></i>
                </div>
                
                <div class="flex-1">
                    <div class="flex items-start justify-between">
                        <div>
                            <h3 class="text-lg font-bold text-gray-800">${doctor.name}</h3>
                            <p class="text-sm text-gray-600">${doctor.qualification} - ${doctor.specialty}</p>
                        </div>
                        <span class="${availabilityClass} text-xs px-2 py-1 rounded-full">${availabilityText}</span>
                    </div>
                    
                    <div class="flex items-center mt-2">
                        <div class="flex text-yellow-400">
                            ${generateStarRating(doctor.rating)}
                        </div>
                        <span class="text-sm text-gray-600 ml-2">${doctor.rating} (${doctor.reviews} reviews)</span>
                        <span class="mx-2 text-gray-300">|</span>
                        <span class="text-sm text-gray-600">${doctor.experience} yrs exp</span>
                    </div>
                    
                    <div class="mt-3 space-y-1">
                        <p class="text-sm text-gray-700">
                            <i class="fas fa-hospital text-blue-600 w-4 mr-2"></i>
                            ${doctor.hospital}, ${doctor.area}
                        </p>
                        <p class="text-sm text-gray-700">
                            <i class="fas fa-clock text-blue-600 w-4 mr-2"></i>
                            ${doctor.timing.monday}
                        </p>
                        <p class="text-sm text-gray-700">
                            <i class="fas fa-rupee-sign text-blue-600 w-4 mr-2"></i>
                            ₹${doctor.fee.initial} (Initial) / ₹${doctor.fee.followup} (Follow-up)
                        </p>
                        <p class="text-sm text-gray-700">
                            <i class="fas fa-map-marker-alt text-blue-600 w-4 mr-2"></i>
                            ${doctor.distance} away
                        </p>
                    </div>
                    
                    <div class="mt-4 flex flex-wrap gap-2">
                        ${doctor.videoConsultation ? 
                            '<span class="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full"><i class="fas fa-video mr-1"></i>Video Available</span>' : ''}
                        ${doctor.emergencyAvailable ? 
                            '<span class="bg-red-100 text-red-800 text-xs px-2 py-1 rounded-full"><i class="fas fa-ambulance mr-1"></i>Emergency</span>' : ''}
                    </div>
                    
                    <button onclick="event.stopPropagation(); selectDoctor(${doctor.id})" 
                            class="mt-4 w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition">
                        <i class="fas fa-calendar-check mr-2"></i>Book Appointment
                    </button>
                </div>
            </div>
        </div>
    `;
}

/**
 * Generate star rating HTML
 * @param {number} rating - Rating value
 * @returns {string} Star HTML
 */
function generateStarRating(rating) {
    const fullStars = Math.floor(rating);
    const halfStar = rating % 1 >= 0.5;
    let stars = '';
    
    for (let i = 0; i < fullStars; i++) {
        stars += '<i class="fas fa-star"></i>';
    }
    
    if (halfStar) {
        stars += '<i class="fas fa-star-half-alt"></i>';
    }
    
    const emptyStars = 5 - Math.ceil(rating);
    for (let i = 0; i < emptyStars; i++) {
        stars += '<i class="far fa-star"></i>';
    }
    
    return stars;
}

// ==================== SEARCH FUNCTIONS ====================

/**
 * Perform doctor search
 */
function performSearch() {
    const location = document.getElementById('location')?.value || 'Kota';
    const specialty = document.getElementById('specialty')?.value || '';
    const availability = document.getElementById('availability')?.value || 'Any Time';
    
    // Show loading
    if (window.COPDApp) {
        window.COPDApp.showLoading('Searching for doctors...');
    }
    
    // Simulate API call
    setTimeout(() => {
        if (window.COPDApp) {
            window.COPDApp.hideLoading();
        }
        
        // Filter doctors
        let filtered = DOCTORS_DATABASE;
        
        // Filter by specialty
        if (specialty) {
            filtered = filtered.filter(d => d.specialty.includes(specialty));
        }
        
        // Filter by availability
        if (availability === 'Today') {
            filtered = filtered.filter(d => d.availableToday);
        }
        
        // Update display
        currentDoctors = filtered;
        displayDoctors(filtered);
        
        // Update map
        if (mapInstance) {
            addDoctorMarkers(filtered);
        }
        
        // Update result info
        updateSearchInfo(location, filtered.length);
        
        if (window.COPDApp) {
            window.COPDApp.showToast(`Found ${filtered.length} doctors`, 'success');
        }
    }, 1000);
}

/**
 * Update search info display
 * @param {string} location - Search location
 * @param {number} count - Number of doctors found
 */
function updateSearchInfo(location, count) {
    const searchInfo = document.getElementById('searchInfo');
    if (searchInfo) {
        searchInfo.innerHTML = `
            <div class="bg-blue-50 p-4 rounded-lg">
                <p class="text-sm text-blue-800">
                    <i class="fas fa-search mr-2"></i>
                    Searching in: <span class="font-semibold">${location}</span><br>
                    <span class="text-xs">${count} doctors found within ${searchRadius} km radius</span>
                </p>
            </div>
        `;
    }
}

/**
 * Reset search filters
 */
function resetSearch() {
    document.getElementById('location').value = 'Kota';
    document.getElementById('specialty').value = '';
    document.getElementById('availability').value = 'Any Time';
    document.getElementById('radius').value = '10';
    
    searchRadius = 10;
    currentDoctors = [...DOCTORS_DATABASE];
    
    displayDoctors(currentDoctors);
    addDoctorMarkers(currentDoctors);
    
    if (window.COPDApp) {
        window.COPDApp.showToast('Search filters reset', 'info');
    }
}

/**
 * Sort doctors based on selected option
 */
function sortDoctors() {
    const sortBy = document.getElementById('sortBy')?.value;
    if (!sortBy) return;
    
    let sorted = [...currentDoctors];
    
    switch(sortBy) {
        case 'rating':
            sorted.sort((a, b) => b.rating - a.rating);
            break;
        case 'experience':
            sorted.sort((a, b) => b.experience - a.experience);
            break;
        case 'price_low':
            sorted.sort((a, b) => a.fee.initial - b.fee.initial);
            break;
        case 'price_high':
            sorted.sort((a, b) => b.fee.initial - a.fee.initial);
            break;
        default:
            // Default sort by distance
            sorted.sort((a, b) => parseFloat(a.distance) - parseFloat(b.distance));
    }
    
    currentDoctors = sorted;
    displayDoctors(sorted);
}

// ==================== DOCTOR SELECTION & DETAILS ====================

/**
 * Select doctor and show details
 * @param {number} doctorId - Doctor ID
 */
function selectDoctor(doctorId) {
    const doctor = DOCTORS_DATABASE.find(d => d.id === doctorId);
    if (!doctor) return;
    
    selectedDoctor = doctor;
    
    // Store selected doctor in session
    sessionStorage.setItem('selectedDoctor', JSON.stringify(doctor));
    
    // Redirect to doctor details page
    window.location.href = `doctor-details.html?id=${doctorId}`;
}

/**
 * Load doctor details on details page
 */
function loadDoctorDetails() {
    // Get doctor ID from URL
    const urlParams = new URLSearchParams(window.location.search);
    const doctorId = parseInt(urlParams.get('id'));
    
    if (!doctorId) {
        window.location.href = 'doctor-search.html';
        return;
    }
    
    const doctor = DOCTORS_DATABASE.find(d => d.id === doctorId);
    if (!doctor) {
        window.location.href = 'doctor-search.html';
        return;
    }
    
    selectedDoctor = doctor;
    displayDoctorDetails(doctor);
}

/**
 * Display doctor details on details page
 * @param {object} doctor - Doctor object
 */
function displayDoctorDetails(doctor) {
    // Update page title
    document.title = `${doctor.name} - COPD Care AI`;
    
    // Populate doctor details
    const elements = {
        doctorName: doctor.name,
        doctorQualification: doctor.qualification,
        doctorSpecialty: doctor.specialty,
        doctorRating: doctor.rating,
        doctorReviews: doctor.reviews,
        doctorExperience: doctor.experience,
        doctorHospital: doctor.hospital,
        doctorAddress: doctor.address,
        doctorAbout: doctor.about,
        doctorPhone: doctor.phone,
        doctorEmail: doctor.email,
        doctorFee: `₹${doctor.fee.initial} (Initial) / ₹${doctor.fee.followup} (Follow-up)`
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    });
    
    // Display expertise
    const expertiseList = document.getElementById('doctorExpertise');
    if (expertiseList && doctor.expertise) {
        expertiseList.innerHTML = doctor.expertise.map(exp => 
            `<span class="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">${exp}</span>`
        ).join('');
    }
    
    // Display education
    const educationList = document.getElementById('doctorEducation');
    if (educationList && doctor.education) {
        educationList.innerHTML = doctor.education.map(edu => 
            `<li class="flex items-start"><i class="fas fa-graduation-cap text-blue-600 mt-1 mr-3"></i>${edu}</li>`
        ).join('');
    }
    
    // Display languages
    const languagesList = document.getElementById('doctorLanguages');
    if (languagesList && doctor.languages) {
        languagesList.innerHTML = doctor.languages.map(lang => 
            `<span class="bg-gray-100 text-gray-800 px-3 py-1 rounded-full text-sm">${lang}</span>`
        ).join('');
    }
    
    // Display timing
    displayTimingTable(doctor.timing);
    
    // Display similar doctors
    displaySimilarDoctors(doctor);
}

/**
 * Display timing table
 * @param {object} timing - Timing object
 */
function displayTimingTable(timing) {
    const timingBody = document.getElementById('timingBody');
    if (!timingBody) return;
    
    const days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];
    const dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    
    let html = '';
    days.forEach((day, index) => {
        html += `
            <tr class="border-b">
                <td class="py-2 font-medium">${dayNames[index]}</td>
                <td class="py-2 text-gray-600">${timing[day] || 'Closed'}</td>
            </tr>
        `;
    });
    
    timingBody.innerHTML = html;
}

/**
 * Display similar doctors
 * @param {object} currentDoctor - Current doctor object
 */
function displaySimilarDoctors(currentDoctor) {
    const similarContainer = document.getElementById('similarDoctors');
    if (!similarContainer) return;
    
    const similar = DOCTORS_DATABASE
        .filter(d => d.id !== currentDoctor.id && d.city === currentDoctor.city)
        .slice(0, 3);
    
    if (similar.length === 0) {
        similarContainer.innerHTML = '<p class="text-gray-500 text-center py-4">No similar doctors found</p>';
        return;
    }
    
    let html = '';
    similar.forEach(doctor => {
        html += `
            <div class="border rounded-lg p-4 hover:shadow-lg transition cursor-pointer" onclick="selectDoctor(${doctor.id})">
                <div class="flex items-center space-x-3 mb-3">
                    <div class="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center">
                        <i class="fas fa-user-md text-white"></i>
                    </div>
                    <div>
                        <h3 class="font-semibold">${doctor.name}</h3>
                        <p class="text-sm text-gray-600">${doctor.specialty}</p>
                    </div>
                </div>
                <p class="text-sm text-gray-600 mb-2">${doctor.hospital}</p>
                <div class="flex items-center text-sm">
                    <i class="fas fa-star text-yellow-400 mr-1"></i>
                    <span>${doctor.rating} (${doctor.reviews} reviews)</span>
                </div>
                <button onclick="event.stopPropagation(); selectDoctor(${doctor.id})" 
                        class="mt-3 text-blue-600 text-sm block hover:text-blue-700">
                    View Profile →
                </button>
            </div>
        `;
    });
    
    similarContainer.innerHTML = html;
}

// ==================== APPOINTMENT BOOKING ====================

/**
 * Book appointment with doctor
 */
async function bookAppointment() {
    // Get appointment details
    const date = document.getElementById('appointmentDate')?.value;
    const time = document.getElementById('appointmentTime')?.value;
    const type = document.querySelector('input[name="appointmentType"]:checked')?.value;
    const reason = document.getElementById('appointmentReason')?.value;
    
    if (!date || !time || !type) {
        if (window.COPDApp) {
            window.COPDApp.showToast('Please fill in all required fields', 'error');
        }
        return;
    }
    
    if (!selectedDoctor) {
        if (window.COPDApp) {
            window.COPDApp.showToast('No doctor selected', 'error');
        }
        return;
    }
    
    // Show loading
    if (window.COPDApp) {
        window.COPDApp.showLoading('Booking your appointment...');
    }
    
    // Simulate API call
    setTimeout(() => {
        if (window.COPDApp) {
            window.COPDApp.hideLoading();
        }
        
        // Generate booking ID
        const bookingId = 'APT' + Date.now();
        
        // Save to localStorage
        const user = JSON.parse(localStorage.getItem('user'));
        if (user) {
            const appointments = JSON.parse(localStorage.getItem(`appointments_${user.email}`)) || [];
            appointments.unshift({
                id: bookingId,
                doctorId: selectedDoctor.id,
                doctorName: selectedDoctor.name,
                hospital: selectedDoctor.hospital,
                date: date,
                time: time,
                type: type,
                reason: reason,
                status: 'confirmed',
                timestamp: new Date().toISOString()
            });
            localStorage.setItem(`appointments_${user.email}`, JSON.stringify(appointments.slice(0, 20)));
        }
        
        // Show success message
        if (window.COPDApp) {
            window.COPDApp.showToast('Appointment booked successfully!', 'success');
        }
        
        // Show confirmation modal
        showBookingConfirmation(bookingId, selectedDoctor, date, time, type);
    }, 2000);
}

/**
 * Show booking confirmation modal
 * @param {string} bookingId - Booking ID
 * @param {object} doctor - Doctor object
 * @param {string} date - Appointment date
 * @param {string} time - Appointment time
 * @param {string} type - Appointment type
 */
function showBookingConfirmation(bookingId, doctor, date, time, type) {
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-white rounded-xl p-8 max-w-md w-full mx-4 animate-slide-in">
            <div class="text-center mb-6">
                <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i class="fas fa-check-circle text-green-600 text-3xl"></i>
                </div>
                <h2 class="text-2xl font-bold text-gray-800">Appointment Confirmed!</h2>
                <p class="text-gray-600 mt-2">Your appointment has been booked successfully</p>
            </div>
            
            <div class="bg-gray-50 rounded-lg p-4 mb-6">
                <p class="text-sm text-gray-600 mb-2">Booking ID: <span class="font-mono font-bold">${bookingId}</span></p>
                <div class="space-y-2">
                    <p><span class="font-medium">Doctor:</span> ${doctor.name}</p>
                    <p><span class="font-medium">Hospital:</span> ${doctor.hospital}</p>
                    <p><span class="font-medium">Date:</span> ${new Date(date).toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
                    <p><span class="font-medium">Time:</span> ${time}</p>
                    <p><span class="font-medium">Type:</span> ${type === 'clinic' ? 'In-Clinic' : 'Video'} Consultation</p>
                </div>
            </div>
            
            <div class="flex space-x-3">
                <button onclick="window.location.href='dashboard.html'" 
                        class="flex-1 bg-gray-600 text-white py-3 rounded-lg hover:bg-gray-700">
                    Go to Dashboard
                </button>
                <button onclick="window.open('https://wa.me/${CONFIG.WHATSAPP_NUMBER}?text=${encodeURIComponent(`I have an appointment on ${date} at ${time} with Dr. ${doctor.name}. Booking ID: ${bookingId}`)}', '_blank')" 
                        class="flex-1 bg-green-600 text-white py-3 rounded-lg hover:bg-green-700">
                    <i class="fab fa-whatsapp mr-2"></i>Share on WhatsApp
                </button>
            </div>
            
            <button onclick="this.closest('.fixed').remove()" 
                    class="absolute top-4 right-4 text-gray-400 hover:text-gray-600">
                <i class="fas fa-times text-xl"></i>
            </button>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (modal.parentNode) {
            modal.remove();
        }
    }, 5000);
}

// ==================== LOCATION FUNCTIONS ====================

/**
 * Detect user's current location
 */
function detectLocation() {
    if (!navigator.geolocation) {
        if (window.COPDApp) {
            window.COPDApp.showToast('Geolocation not supported', 'error');
        }
        return;
    }
    
    if (window.COPDApp) {
        window.COPDApp.showLoading('Detecting your location...');
    }
    
    navigator.geolocation.getCurrentPosition(
        (position) => {
            if (window.COPDApp) {
                window.COPDApp.hideLoading();
            }
            
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;
            
            currentLocation = { lat, lng };
            
            // Update map
            if (mapInstance) {
                mapInstance.setView([lat, lng], 13);
                addUserMarker(lat, lng);
                addRadiusCircle(lat, lng, searchRadius * 1000);
            }
            
            // Reverse geocode to get address
            fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`)
                .then(response => response.json())
                .then(data => {
                    const locationInput = document.getElementById('location');
                    if (locationInput) {
                        const city = data.address.city || data.address.town || data.address.village || 'Unknown';
                        locationInput.value = city;
                    }
                });
            
            if (window.COPDApp) {
                window.COPDApp.showToast('Location detected', 'success');
            }
        },
        (error) => {
            if (window.COPDApp) {
                window.COPDApp.hideLoading();
                window.COPDApp.showToast('Could not detect location', 'error');
            }
        }
    );
}

/**
 * Expand search radius
 */
function expandRadius() {
    const radiusSelect = document.getElementById('radius');
    if (radiusSelect) {
        const currentValue = parseInt(radiusSelect.value);
        if (currentValue < 50) {
            radiusSelect.value = currentValue + 10;
            searchRadius = currentValue + 10;
            performSearch();
        }
    }
}

// ==================== APPOINTMENT HISTORY ====================

/**
 * Load user's appointment history
 */
function loadAppointmentHistory() {
    const user = JSON.parse(localStorage.getItem('user'));
    if (!user) return;
    
    const appointments = JSON.parse(localStorage.getItem(`appointments_${user.email}`)) || [];
    const historyContainer = document.getElementById('appointmentHistory');
    
    if (!historyContainer) return;
    
    if (appointments.length === 0) {
        historyContainer.innerHTML = `
            <div class="text-center py-8">
                <i class="fas fa-calendar-times text-4xl text-gray-400 mb-3"></i>
                <p class="text-gray-600">No appointments yet</p>
                <a href="doctor-search.html" class="text-blue-600 text-sm mt-2 inline-block">Find a Doctor →</a>
            </div>
        `;
        return;
    }
    
    let html = '';
    appointments.forEach(app => {
        const date = new Date(app.date);
        const formattedDate = date.toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' });
        
        html += `
            <div class="border rounded-lg p-4 hover:shadow-md transition">
                <div class="flex justify-between items-start">
                    <div>
                        <h3 class="font-semibold">${app.doctorName}</h3>
                        <p class="text-sm text-gray-600">${app.hospital}</p>
                    </div>
                    <span class="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">${app.status}</span>
                </div>
                <div class="mt-3 grid grid-cols-2 gap-2 text-sm">
                    <div>
                        <span class="text-gray-500">Date:</span>
                        <span class="ml-1 font-medium">${formattedDate}</span>
                    </div>
                    <div>
                        <span class="text-gray-500">Time:</span>
                        <span class="ml-1 font-medium">${app.time}</span>
                    </div>
                    <div>
                        <span class="text-gray-500">Type:</span>
                        <span class="ml-1 font-medium">${app.type === 'clinic' ? 'In-Clinic' : 'Video'}</span>
                    </div>
                    <div>
                        <span class="text-gray-500">ID:</span>
                        <span class="ml-1 font-mono text-xs">${app.id}</span>
                    </div>
                </div>
                <div class="mt-3 flex justify-end space-x-2">
                    <button onclick="rescheduleAppointment('${app.id}')" class="text-sm text-blue-600 hover:text-blue-700">
                        <i class="fas fa-calendar-alt mr-1"></i>Reschedule
                    </button>
                    <button onclick="cancelAppointment('${app.id}')" class="text-sm text-red-600 hover:text-red-700">
                        <i class="fas fa-times mr-1"></i>Cancel
                    </button>
                </div>
            </div>
        `;
    });
    
    historyContainer.innerHTML = html;
}

/**
 * Reschedule appointment
 * @param {string} appointmentId - Appointment ID
 */
function rescheduleAppointment(appointmentId) {
    if (window.COPDApp) {
        window.COPDApp.showToast('Reschedule feature coming soon', 'info');
    }
}

/**
 * Cancel appointment
 * @param {string} appointmentId - Appointment ID
 */
function cancelAppointment(appointmentId) {
    if (confirm('Are you sure you want to cancel this appointment?')) {
        const user = JSON.parse(localStorage.getItem('user'));
        if (user) {
            let appointments = JSON.parse(localStorage.getItem(`appointments_${user.email}`)) || [];
            appointments = appointments.filter(a => a.id !== appointmentId);
            localStorage.setItem(`appointments_${user.email}`, JSON.stringify(appointments));
            
            loadAppointmentHistory();
            
            if (window.COPDApp) {
                window.COPDApp.showToast('Appointment cancelled', 'success');
            }
        }
    }
}

// ==================== EXPORT FUNCTIONS ====================

// Make functions globally available
window.initDoctorSearch = initDoctorSearch;
window.performSearch = performSearch;
window.resetSearch = resetSearch;
window.sortDoctors = sortDoctors;
window.selectDoctor = selectDoctor;
window.loadDoctorDetails = loadDoctorDetails;
window.bookAppointment = bookAppointment;
window.detectLocation = detectLocation;
window.expandRadius = expandRadius;
window.loadAppointmentHistory = loadAppointmentHistory;
window.rescheduleAppointment = rescheduleAppointment;
window.cancelAppointment = cancelAppointment;

// Initialize on page load based on current page
document.addEventListener('DOMContentLoaded', function() {
    const currentPage = window.location.pathname.split('/').pop();
    
    if (currentPage === 'doctor-search.html') {
        initDoctorSearch();
    } else if (currentPage === 'doctor-details.html') {
        loadDoctorDetails();
    } else if (currentPage === 'appointments.html') {
        loadAppointmentHistory();
    }
});

// ==================== EXPORT FOR MODULE USE ====================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        DOCTORS_DATABASE,
        DOCTOR_CONFIG,
        initDoctorSearch,
        performSearch,
        selectDoctor,
        bookAppointment,
        detectLocation
    };
}