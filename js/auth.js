/**
 * COPD Care AI - Authentication JavaScript File
 * Technovation 2026 Project
 * Handles all user authentication, registration, password management, and session handling
 */

// ==================== AUTHENTICATION CONFIGURATION ====================

const AUTH_CONFIG = {
    MIN_PASSWORD_LENGTH: 8,
    MAX_LOGIN_ATTEMPTS: 5,
    LOCKOUT_DURATION: 15 * 60 * 1000, // 15 minutes in milliseconds
    SESSION_DURATION: 7 * 24 * 60 * 60 * 1000, // 7 days in milliseconds
    TOKEN_KEY: 'auth_token',
    USER_KEY: 'user',
    REMEMBER_ME_KEY: 'remember_me',
    ATTEMPTS_KEY: 'login_attempts',
    LOCKOUT_KEY: 'lockout_until'
};

// ==================== USER DATABASE (Mock - In production, this would be on server) ====================

// This is a mock database for demo purposes
// In production, this would be replaced with actual API calls
const MOCK_USERS = [
    {
        id: 1,
        firstName: 'Bharat',
        lastName: 'Goswami',
        email: 'bharat@example.com',
        password: 'patient123', // In production, this would be hashed
        phone: '9876543210',
        dob: '1985-06-15',
        gender: 'male',
        city: 'Kota',
        state: 'Rajasthan',
        bloodGroup: 'B+',
        emergencyContact: '9876543211',
        role: 'patient',
        isActive: true,
        createdAt: '2026-01-01T00:00:00.000Z',
        lastLogin: null,
        profileComplete: true
    },
    {
        id: 2,
        firstName: 'Rahul',
        lastName: 'Sharma',
        email: 'rahul@example.com',
        password: 'patient123',
        phone: '9876543212',
        dob: '1990-03-20',
        gender: 'male',
        city: 'Jaipur',
        state: 'Rajasthan',
        bloodGroup: 'O+',
        emergencyContact: '9876543213',
        role: 'patient',
        isActive: true,
        createdAt: '2026-01-15T00:00:00.000Z',
        lastLogin: null,
        profileComplete: true
    },
    {
        id: 3,
        firstName: 'Priya',
        lastName: 'Verma',
        email: 'priya@example.com',
        password: 'patient123',
        phone: '9876543214',
        dob: '1988-11-10',
        gender: 'female',
        city: 'Delhi',
        state: 'Delhi',
        bloodGroup: 'A+',
        emergencyContact: '9876543215',
        role: 'patient',
        isActive: true,
        createdAt: '2026-02-01T00:00:00.000Z',
        lastLogin: null,
        profileComplete: false
    }
];

// ==================== HELPER FUNCTIONS ====================

/**
 * Generate a random token
 * @param {number} length - Token length
 * @returns {string} Random token
 */
function generateToken(length = 32) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let token = '';
    for (let i = 0; i < length; i++) {
        token += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return token;
}

/**
 * Hash password (mock - in production use bcrypt)
 * @param {string} password - Plain text password
 * @returns {string} Hashed password
 */
function hashPassword(password) {
    // This is a mock hash function - DO NOT USE IN PRODUCTION
    return btoa(password + '_salt');
}

/**
 * Verify password (mock - in production use bcrypt)
 * @param {string} plainPassword - Plain text password
 * @param {string} hashedPassword - Hashed password
 * @returns {boolean} Password match
 */
function verifyPassword(plainPassword, hashedPassword) {
    // This is a mock verification - DO NOT USE IN PRODUCTION
    return hashPassword(plainPassword) === hashedPassword;
}

/**
 * Validate email format
 * @param {string} email - Email to validate
 * @returns {boolean} Is valid email
 */
function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

/**
 * Validate phone number (Indian format)
 * @param {string} phone - Phone number to validate
 * @returns {boolean} Is valid phone number
 */
function isValidPhone(phone) {
    const re = /^[6-9]\d{9}$/;
    return re.test(phone.replace(/[^0-9]/g, ''));
}

/**
 * Validate password strength
 * @param {string} password - Password to validate
 * @returns {object} Validation result
 */
function validatePasswordStrength(password) {
    const result = {
        isValid: true,
        errors: []
    };
    
    if (!password || password.length < AUTH_CONFIG.MIN_PASSWORD_LENGTH) {
        result.isValid = false;
        result.errors.push(`Password must be at least ${AUTH_CONFIG.MIN_PASSWORD_LENGTH} characters long`);
    }
    
    if (!/[A-Z]/.test(password)) {
        result.isValid = false;
        result.errors.push('Password must contain at least one uppercase letter');
    }
    
    if (!/[a-z]/.test(password)) {
        result.isValid = false;
        result.errors.push('Password must contain at least one lowercase letter');
    }
    
    if (!/[0-9]/.test(password)) {
        result.isValid = false;
        result.errors.push('Password must contain at least one number');
    }
    
    if (!/[!@#$%^&*]/.test(password)) {
        result.isValid = false;
        result.errors.push('Password must contain at least one special character (!@#$%^&*)');
    }
    
    return result;
}

/**
 * Format phone number for display
 * @param {string} phone - Phone number
 * @returns {string} Formatted phone number
 */
function formatPhoneForDisplay(phone) {
    const cleaned = phone.replace(/[^0-9]/g, '');
    if (cleaned.length === 10) {
        return `+91 ${cleaned.slice(0, 5)} ${cleaned.slice(5)}`;
    }
    return phone;
}

/**
 * Calculate age from date of birth
 * @param {string} dob - Date of birth (YYYY-MM-DD)
 * @returns {number} Age in years
 */
function calculateAge(dob) {
    const birthDate = new Date(dob);
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
        age--;
    }
    
    return age;
}

// ==================== LOGIN ATTEMPT TRACKING ====================

/**
 * Check if user is locked out
 * @param {string} identifier - Email or phone
 * @returns {boolean} Is locked out
 */
function isLockedOut(identifier) {
    const lockoutUntil = localStorage.getItem(`${AUTH_CONFIG.LOCKOUT_KEY}_${identifier}`);
    if (!lockoutUntil) return false;
    
    const lockoutTime = parseInt(lockoutUntil);
    if (Date.now() > lockoutTime) {
        // Lockout expired
        localStorage.removeItem(`${AUTH_CONFIG.LOCKOUT_KEY}_${identifier}`);
        localStorage.removeItem(`${AUTH_CONFIG.ATTEMPTS_KEY}_${identifier}`);
        return false;
    }
    
    return true;
}

/**
 * Record failed login attempt
 * @param {string} identifier - Email or phone
 */
function recordFailedAttempt(identifier) {
    const attemptsKey = `${AUTH_CONFIG.ATTEMPTS_KEY}_${identifier}`;
    let attempts = parseInt(localStorage.getItem(attemptsKey)) || 0;
    attempts++;
    
    localStorage.setItem(attemptsKey, attempts.toString());
    
    if (attempts >= AUTH_CONFIG.MAX_LOGIN_ATTEMPTS) {
        // Lock out user
        const lockoutUntil = Date.now() + AUTH_CONFIG.LOCKOUT_DURATION;
        localStorage.setItem(`${AUTH_CONFIG.LOCKOUT_KEY}_${identifier}`, lockoutUntil.toString());
        
        // Clear attempts
        localStorage.removeItem(attemptsKey);
        
        return {
            locked: true,
            lockoutDuration: AUTH_CONFIG.LOCKOUT_DURATION / 60000 // minutes
        };
    }
    
    return {
        locked: false,
        remainingAttempts: AUTH_CONFIG.MAX_LOGIN_ATTEMPTS - attempts
    };
}

/**
 * Reset login attempts
 * @param {string} identifier - Email or phone
 */
function resetLoginAttempts(identifier) {
    localStorage.removeItem(`${AUTH_CONFIG.ATTEMPTS_KEY}_${identifier}`);
    localStorage.removeItem(`${AUTH_CONFIG.LOCKOUT_KEY}_${identifier}`);
}

// ==================== SESSION MANAGEMENT ====================

/**
 * Create user session
 * @param {object} user - User object
 * @param {boolean} rememberMe - Remember me flag
 */
function createSession(user, rememberMe = false) {
    const token = generateToken();
    const sessionData = {
        userId: user.id,
        email: user.email,
        token: token,
        createdAt: new Date().toISOString(),
        expiresAt: rememberMe ? 
            new Date(Date.now() + AUTH_CONFIG.SESSION_DURATION).toISOString() : 
            new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString() // 24 hours
    };
    
    // Remove sensitive data
    const { password, ...safeUser } = user;
    
    // Update last login
    safeUser.lastLogin = new Date().toISOString();
    
    // Store session
    localStorage.setItem(AUTH_CONFIG.TOKEN_KEY, JSON.stringify(sessionData));
    localStorage.setItem(AUTH_CONFIG.USER_KEY, JSON.stringify(safeUser));
    
    if (rememberMe) {
        localStorage.setItem(AUTH_CONFIG.REMEMBER_ME_KEY, 'true');
    }
    
    return sessionData;
}

/**
 * Get current session
 * @returns {object|null} Session data or null
 */
function getCurrentSession() {
    const sessionStr = localStorage.getItem(AUTH_CONFIG.TOKEN_KEY);
    if (!sessionStr) return null;
    
    try {
        const session = JSON.parse(sessionStr);
        
        // Check if session is expired
        if (new Date(session.expiresAt) < new Date()) {
            destroySession();
            return null;
        }
        
        return session;
    } catch (error) {
        console.error('Error parsing session:', error);
        return null;
    }
}

/**
 * Get current user
 * @returns {object|null} User object or null
 */
function getCurrentUser() {
    const session = getCurrentSession();
    if (!session) return null;
    
    const userStr = localStorage.getItem(AUTH_CONFIG.USER_KEY);
    if (!userStr) return null;
    
    try {
        return JSON.parse(userStr);
    } catch (error) {
        console.error('Error parsing user:', error);
        return null;
    }
}

/**
 * Check if user is logged in
 * @returns {boolean} Login status
 */
function isLoggedIn() {
    return getCurrentSession() !== null;
}

/**
 * Destroy current session (logout)
 */
function destroySession() {
    localStorage.removeItem(AUTH_CONFIG.TOKEN_KEY);
    localStorage.removeItem(AUTH_CONFIG.USER_KEY);
    // Keep remember me preference
}

/**
 * Require authentication - redirect to login if not authenticated
 * @param {string} redirectUrl - URL to redirect to after login
 * @returns {boolean} Is authenticated
 */
function requireAuth(redirectUrl = null) {
    if (!isLoggedIn()) {
        const currentPage = window.location.pathname;
        const loginUrl = redirectUrl || `/pages/login.html?redirect=${encodeURIComponent(currentPage)}`;
        window.location.href = loginUrl;
        return false;
    }
    return true;
}

/**
 * Require guest - redirect to dashboard if already logged in
 * @param {string} redirectUrl - URL to redirect to
 * @returns {boolean} Is guest
 */
function requireGuest(redirectUrl = '/pages/dashboard.html') {
    if (isLoggedIn()) {
        window.location.href = redirectUrl;
        return false;
    }
    return true;
}

// ==================== AUTHENTICATION FUNCTIONS ====================

/**
 * Login user
 * @param {string} identifier - Email or phone
 * @param {string} password - Password
 * @param {boolean} rememberMe - Remember me flag
 * @returns {Promise} Login result
 */
async function login(identifier, password, rememberMe = false) {
    // Check if locked out
    if (isLockedOut(identifier)) {
        return {
            success: false,
            error: 'account_locked',
            message: 'Too many failed attempts. Please try again after 15 minutes.'
        };
    }
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Find user by email or phone
    const user = MOCK_USERS.find(u => 
        u.email.toLowerCase() === identifier.toLowerCase() || 
        u.phone === identifier.replace(/[^0-9]/g, '')
    );
    
    // Check credentials
    if (!user || !verifyPassword(password, hashPassword(user.password))) {
        const attemptResult = recordFailedAttempt(identifier);
        
        if (attemptResult.locked) {
            return {
                success: false,
                error: 'account_locked',
                message: `Too many failed attempts. Please try again after ${attemptResult.lockoutDuration} minutes.`
            };
        }
        
        return {
            success: false,
            error: 'invalid_credentials',
            message: 'Invalid email/phone or password',
            remainingAttempts: attemptResult.remainingAttempts
        };
    }
    
    // Check if account is active
    if (!user.isActive) {
        return {
            success: false,
            error: 'account_inactive',
            message: 'Your account has been deactivated. Please contact support.'
        };
    }
    
    // Reset login attempts on successful login
    resetLoginAttempts(identifier);
    
    // Create session
    const session = createSession(user, rememberMe);
    
    // Track login event
    trackLogin(user.id, identifier);
    
    return {
        success: true,
        user: { ...user, password: undefined },
        session,
        message: 'Login successful!'
    };
}

/**
 * Register new user
 * @param {object} userData - User registration data
 * @returns {Promise} Registration result
 */
async function register(userData) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Validate required fields
    const requiredFields = ['firstName', 'lastName', 'email', 'password', 'phone'];
    for (const field of requiredFields) {
        if (!userData[field]) {
            return {
                success: false,
                error: 'missing_fields',
                message: `${field} is required`
            };
        }
    }
    
    // Validate email
    if (!isValidEmail(userData.email)) {
        return {
            success: false,
            error: 'invalid_email',
            message: 'Please enter a valid email address'
        };
    }
    
    // Validate phone
    if (!isValidPhone(userData.phone)) {
        return {
            success: false,
            error: 'invalid_phone',
            message: 'Please enter a valid 10-digit Indian phone number'
        };
    }
    
    // Validate password strength
    const passwordValidation = validatePasswordStrength(userData.password);
    if (!passwordValidation.isValid) {
        return {
            success: false,
            error: 'weak_password',
            message: passwordValidation.errors[0]
        };
    }
    
    // Check if passwords match
    if (userData.password !== userData.confirmPassword) {
        return {
            success: false,
            error: 'password_mismatch',
            message: 'Passwords do not match'
        };
    }
    
    // Check if user already exists
    const existingUser = MOCK_USERS.find(u => 
        u.email.toLowerCase() === userData.email.toLowerCase() || 
        u.phone === userData.phone.replace(/[^0-9]/g, '')
    );
    
    if (existingUser) {
        return {
            success: false,
            error: 'user_exists',
            message: 'User with this email or phone already exists'
        };
    }
    
    // Create new user (in production, this would be an API call)
    const newUser = {
        id: MOCK_USERS.length + 1,
        firstName: userData.firstName,
        lastName: userData.lastName,
        email: userData.email.toLowerCase(),
        password: hashPassword(userData.password), // Store hashed password
        phone: userData.phone.replace(/[^0-9]/g, ''),
        dob: userData.dob || null,
        gender: userData.gender || null,
        city: userData.city || '',
        state: userData.state || '',
        bloodGroup: userData.bloodGroup || null,
        emergencyContact: userData.emergencyContact || null,
        role: 'patient',
        isActive: true,
        createdAt: new Date().toISOString(),
        lastLogin: null,
        profileComplete: !!(userData.dob && userData.gender)
    };
    
    // Add to mock database (in production, this would be server-side)
    MOCK_USERS.push(newUser);
    
    // Create session and log user in
    const session = createSession(newUser, userData.rememberMe || false);
    
    // Track registration event
    trackRegistration(newUser.id, newUser.email);
    
    return {
        success: true,
        user: { ...newUser, password: undefined },
        session,
        message: 'Registration successful!'
    };
}

/**
 * Logout user
 */
function logout() {
    const user = getCurrentUser();
    if (user) {
        trackLogout(user.id);
    }
    destroySession();
    
    // Redirect to home page
    window.location.href = '/index.html';
}

// ==================== PASSWORD MANAGEMENT ====================

/**
 * Request password reset
 * @param {string} email - User email
 * @returns {Promise} Reset request result
 */
async function requestPasswordReset(email) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Validate email
    if (!isValidEmail(email)) {
        return {
            success: false,
            error: 'invalid_email',
            message: 'Please enter a valid email address'
        };
    }
    
    // Find user
    const user = MOCK_USERS.find(u => u.email.toLowerCase() === email.toLowerCase());
    
    // Always return success to prevent email enumeration
    if (!user) {
        return {
            success: true,
            message: 'If an account exists with this email, you will receive password reset instructions.'
        };
    }
    
    // Generate reset token
    const resetToken = generateToken(48);
    const resetExpires = new Date(Date.now() + 60 * 60 * 1000); // 1 hour
    
    // Store reset token (in production, this would be in database)
    localStorage.setItem(`reset_token_${user.id}`, JSON.stringify({
        token: resetToken,
        expires: resetExpires.toISOString()
    }));
    
    // In production, send email here
    console.log(`Password reset link: /reset-password.html?token=${resetToken}`);
    
    return {
        success: true,
        message: 'If an account exists with this email, you will receive password reset instructions.'
    };
}

/**
 * Reset password with token
 * @param {string} token - Reset token
 * @param {string} newPassword - New password
 * @returns {Promise} Reset result
 */
async function resetPassword(token, newPassword) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Validate password strength
    const passwordValidation = validatePasswordStrength(newPassword);
    if (!passwordValidation.isValid) {
        return {
            success: false,
            error: 'weak_password',
            message: passwordValidation.errors[0]
        };
    }
    
    // Find token in storage (in production, this would be in database)
    let foundUser = null;
    let tokenData = null;
    
    for (const user of MOCK_USERS) {
        const storedToken = localStorage.getItem(`reset_token_${user.id}`);
        if (storedToken) {
            const data = JSON.parse(storedToken);
            if (data.token === token && new Date(data.expires) > new Date()) {
                foundUser = user;
                tokenData = data;
                break;
            }
        }
    }
    
    if (!foundUser) {
        return {
            success: false,
            error: 'invalid_token',
            message: 'Invalid or expired reset token'
        };
    }
    
    // Update password (in production, hash it)
    foundUser.password = hashPassword(newPassword);
    
    // Clear reset token
    localStorage.removeItem(`reset_token_${foundUser.id}`);
    
    return {
        success: true,
        message: 'Password reset successful! You can now login with your new password.'
    };
}

/**
 * Change password (when logged in)
 * @param {string} currentPassword - Current password
 * @param {string} newPassword - New password
 * @returns {Promise} Change result
 */
async function changePassword(currentPassword, newPassword) {
    const user = getCurrentUser();
    if (!user) {
        return {
            success: false,
            error: 'not_authenticated',
            message: 'You must be logged in to change password'
        };
    }
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Find actual user in database
    const dbUser = MOCK_USERS.find(u => u.id === user.id);
    
    // Verify current password
    if (!verifyPassword(currentPassword, dbUser.password)) {
        return {
            success: false,
            error: 'invalid_password',
            message: 'Current password is incorrect'
        };
    }
    
    // Validate new password
    const passwordValidation = validatePasswordStrength(newPassword);
    if (!passwordValidation.isValid) {
        return {
            success: false,
            error: 'weak_password',
            message: passwordValidation.errors[0]
        };
    }
    
    // Update password
    dbUser.password = hashPassword(newPassword);
    
    return {
        success: true,
        message: 'Password changed successfully!'
    };
}

// ==================== PROFILE MANAGEMENT ====================

/**
 * Update user profile
 * @param {object} profileData - Profile data to update
 * @returns {Promise} Update result
 */
async function updateProfile(profileData) {
    const user = getCurrentUser();
    if (!user) {
        return {
            success: false,
            error: 'not_authenticated',
            message: 'You must be logged in to update profile'
        };
    }
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Find actual user in database
    const dbUser = MOCK_USERS.find(u => u.id === user.id);
    if (!dbUser) {
        return {
            success: false,
            error: 'user_not_found',
            message: 'User not found'
        };
    }
    
    // Update allowed fields
    const allowedFields = ['firstName', 'lastName', 'phone', 'dob', 'gender', 'city', 'state', 'bloodGroup', 'emergencyContact'];
    
    allowedFields.forEach(field => {
        if (profileData[field] !== undefined) {
            dbUser[field] = profileData[field];
        }
    });
    
    // Check if profile is now complete
    dbUser.profileComplete = !!(dbUser.dob && dbUser.gender && dbUser.bloodGroup && dbUser.emergencyContact);
    
    // Update session with new user data
    const { password, ...safeUser } = dbUser;
    localStorage.setItem(AUTH_CONFIG.USER_KEY, JSON.stringify(safeUser));
    
    return {
        success: true,
        user: safeUser,
        message: 'Profile updated successfully!'
    };
}

/**
 * Get user profile
 * @returns {object|null} User profile
 */
function getProfile() {
    return getCurrentUser();
}

/**
 * Check if profile is complete
 * @returns {boolean} Is profile complete
 */
function isProfileComplete() {
    const user = getCurrentUser();
    if (!user) return false;
    return user.profileComplete || false;
}

// ==================== SOCIAL AUTHENTICATION ====================

/**
 * Google login
 * @param {string} token - Google OAuth token
 * @returns {Promise} Login result
 */
async function googleLogin(token) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Mock Google authentication
    // In production, this would verify the token with Google
    const mockGoogleUser = {
        id: 'google_12345',
        email: 'user@gmail.com',
        firstName: 'Google',
        lastName: 'User',
        picture: 'https://via.placeholder.com/150'
    };
    
    // Check if user exists
    let user = MOCK_USERS.find(u => u.email === mockGoogleUser.email);
    
    if (!user) {
        // Create new user from Google data
        user = {
            id: MOCK_USERS.length + 1,
            firstName: mockGoogleUser.firstName,
            lastName: mockGoogleUser.lastName,
            email: mockGoogleUser.email,
            password: null, // No password for OAuth users
            phone: '',
            dob: null,
            gender: null,
            city: '',
            state: '',
            bloodGroup: null,
            emergencyContact: null,
            role: 'patient',
            isActive: true,
            createdAt: new Date().toISOString(),
            lastLogin: null,
            profileComplete: false,
            authProvider: 'google'
        };
        
        MOCK_USERS.push(user);
    }
    
    // Create session
    const session = createSession(user, true);
    
    return {
        success: true,
        user: { ...user, password: undefined },
        session,
        message: 'Google login successful!'
    };
}

/**
 * Facebook login
 * @param {string} token - Facebook OAuth token
 * @returns {Promise} Login result
 */
async function facebookLogin(token) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Mock Facebook authentication
    const mockFacebookUser = {
        id: 'fb_12345',
        email: 'fbuser@gmail.com',
        firstName: 'Facebook',
        lastName: 'User',
        picture: 'https://via.placeholder.com/150'
    };
    
    // Similar implementation as googleLogin
    let user = MOCK_USERS.find(u => u.email === mockFacebookUser.email);
    
    if (!user) {
        user = {
            id: MOCK_USERS.length + 1,
            firstName: mockFacebookUser.firstName,
            lastName: mockFacebookUser.lastName,
            email: mockFacebookUser.email,
            password: null,
            phone: '',
            dob: null,
            gender: null,
            city: '',
            state: '',
            bloodGroup: null,
            emergencyContact: null,
            role: 'patient',
            isActive: true,
            createdAt: new Date().toISOString(),
            lastLogin: null,
            profileComplete: false,
            authProvider: 'facebook'
        };
        
        MOCK_USERS.push(user);
    }
    
    const session = createSession(user, true);
    
    return {
        success: true,
        user: { ...user, password: undefined },
        session,
        message: 'Facebook login successful!'
    };
}

// ==================== ACCOUNT MANAGEMENT ====================

/**
 * Deactivate account
 * @returns {Promise} Deactivation result
 */
async function deactivateAccount() {
    const user = getCurrentUser();
    if (!user) {
        return {
            success: false,
            error: 'not_authenticated',
            message: 'You must be logged in to deactivate your account'
        };
    }
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Find and deactivate user
    const dbUser = MOCK_USERS.find(u => u.id === user.id);
    if (dbUser) {
        dbUser.isActive = false;
    }
    
    // Destroy session
    destroySession();
    
    return {
        success: true,
        message: 'Account deactivated successfully'
    };
}

/**
 * Delete account permanently
 * @returns {Promise} Deletion result
 */
async function deleteAccount() {
    const user = getCurrentUser();
    if (!user) {
        return {
            success: false,
            error: 'not_authenticated',
            message: 'You must be logged in to delete your account'
        };
    }
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // In production, this would trigger a permanent deletion process
    // For demo, we'll just remove from our mock array
    const index = MOCK_USERS.findIndex(u => u.id === user.id);
    if (index !== -1) {
        MOCK_USERS.splice(index, 1);
    }
    
    // Clear all user data from localStorage
    const userPrefix = `_${user.id}`;
    const keysToRemove = [];
    
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key.includes(userPrefix) || key === AUTH_CONFIG.USER_KEY || key === AUTH_CONFIG.TOKEN_KEY) {
            keysToRemove.push(key);
        }
    }
    
    keysToRemove.forEach(key => localStorage.removeItem(key));
    
    return {
        success: true,
        message: 'Account deleted permanently'
    };
}

// ==================== TRACKING & ANALYTICS ====================

/**
 * Track login event
 * @param {number} userId - User ID
 * @param {string} identifier - Login identifier
 */
function trackLogin(userId, identifier) {
    const loginEvents = JSON.parse(localStorage.getItem('login_events') || '[]');
    loginEvents.push({
        userId,
        identifier: identifier.replace(/[^@]/g, '***'), // Mask for privacy
        timestamp: new Date().toISOString(),
        success: true
    });
    
    // Keep only last 50 events
    if (loginEvents.length > 50) {
        loginEvents.shift();
    }
    
    localStorage.setItem('login_events', JSON.stringify(loginEvents));
}

/**
 * Track registration event
 * @param {number} userId - User ID
 * @param {string} email - User email
 */
function trackRegistration(userId, email) {
    const registrations = JSON.parse(localStorage.getItem('registrations') || '[]');
    registrations.push({
        userId,
        email: email.replace(/(.{2})(.*)(?=@)/, '$1***'), // Mask email
        timestamp: new Date().toISOString()
    });
    
    localStorage.setItem('registrations', JSON.stringify(registrations));
}

/**
 * Track logout event
 * @param {number} userId - User ID
 */
function trackLogout(userId) {
    const logouts = JSON.parse(localStorage.getItem('logouts') || '[]');
    logouts.push({
        userId,
        timestamp: new Date().toISOString()
    });
    
    localStorage.setItem('logouts', JSON.stringify(logouts));
}

// ==================== UI UPDATE FUNCTIONS ====================

/**
 * Update UI based on authentication state
 */
function updateAuthUI() {
    const isLoggedIn = !!getCurrentUser();
    const user = getCurrentUser();
    
    // Update navigation items
    document.querySelectorAll('.auth-required').forEach(el => {
        el.style.display = isLoggedIn ? 'block' : 'none';
    });
    
    document.querySelectorAll('.guest-only').forEach(el => {
        el.style.display = isLoggedIn ? 'none' : 'block';
    });
    
    // Update user name displays
    if (isLoggedIn && user) {
        document.querySelectorAll('.user-name-display').forEach(el => {
            el.textContent = `${user.firstName} ${user.lastName}`;
        });
        
        document.querySelectorAll('.user-initials').forEach(el => {
            el.textContent = `${user.firstName[0]}${user.lastName[0]}`;
        });
    }
}

// ==================== INITIALIZATION ====================

/**
 * Initialize authentication on page load
 */
function initAuth() {
    // Check for existing session
    const session = getCurrentSession();
    const user = getCurrentUser();
    
    if (session && user) {
        // Update last login (in production, this would be on the server)
        const dbUser = MOCK_USERS.find(u => u.id === user.id);
        if (dbUser) {
            dbUser.lastLogin = new Date().toISOString();
        }
    }
    
    // Update UI
    updateAuthUI();
    
    // Setup event listeners for login forms
    setupAuthForms();
}

/**
 * Setup authentication form event listeners
 */
function setupAuthForms() {
    // Login form
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const email = document.getElementById('email')?.value;
            const password = document.getElementById('password')?.value;
            const rememberMe = document.getElementById('rememberMe')?.checked || false;
            
            if (!email || !password) {
                showToast('Please fill in all fields', 'error');
                return;
            }
            
            // Show loading
            const submitBtn = loginForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Logging in...';
            submitBtn.disabled = true;
            
            const result = await login(email, password, rememberMe);
            
            // Restore button
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
            
            if (result.success) {
                showToast(result.message, 'success');
                
                // Check for redirect
                const urlParams = new URLSearchParams(window.location.search);
                const redirect = urlParams.get('redirect') || '/pages/dashboard.html';
                
                setTimeout(() => {
                    window.location.href = redirect;
                }, 1000);
            } else {
                let errorMessage = result.message;
                if (result.remainingAttempts) {
                    errorMessage += ` (${result.remainingAttempts} attempts remaining)`;
                }
                showToast(errorMessage, 'error');
            }
        });
    }
    
    // Signup form
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                firstName: document.getElementById('firstName')?.value,
                lastName: document.getElementById('lastName')?.value,
                email: document.getElementById('email')?.value,
                phone: document.getElementById('phone')?.value,
                password: document.getElementById('password')?.value,
                confirmPassword: document.getElementById('confirmPassword')?.value,
                dob: document.getElementById('dob')?.value,
                gender: document.getElementById('gender')?.value,
                city: document.getElementById('city')?.value,
                state: document.getElementById('state')?.value,
                bloodGroup: document.getElementById('bloodGroup')?.value,
                emergencyContact: document.getElementById('emergencyContact')?.value,
                rememberMe: document.getElementById('rememberMe')?.checked || false
            };
            
            // Validate required fields
            if (!formData.firstName || !formData.lastName || !formData.email || !formData.phone || !formData.password) {
                showToast('Please fill in all required fields', 'error');
                return;
            }
            
            // Show loading
            const submitBtn = signupForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Creating account...';
            submitBtn.disabled = true;
            
            const result = await register(formData);
            
            // Restore button
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
            
            if (result.success) {
                showToast(result.message, 'success');
                setTimeout(() => {
                    window.location.href = '/pages/dashboard.html';
                }, 1000);
            } else {
                showToast(result.message, 'error');
            }
        });
    }
    
    // Password reset request form
    const forgotForm = document.getElementById('forgotPasswordForm');
    if (forgotForm) {
        forgotForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const email = document.getElementById('email')?.value;
            
            if (!email) {
                showToast('Please enter your email', 'error');
                return;
            }
            
            const submitBtn = forgotForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Sending...';
            submitBtn.disabled = true;
            
            const result = await requestPasswordReset(email);
            
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
            
            showToast(result.message, result.success ? 'success' : 'error');
            
            if (result.success) {
                setTimeout(() => {
                    window.location.href = '/pages/login.html';
                }, 3000);
            }
        });
    }
}

// ==================== TOAST NOTIFICATION (Fallback) ====================

/**
 * Show toast notification (fallback if main.js not loaded)
 * @param {string} message - Message to display
 * @param {string} type - success/error/info
 */
function showToast(message, type = 'info') {
    // Check if main.js toast is available
    if (window.COPDApp && window.COPDApp.showToast) {
        window.COPDApp.showToast(message, type);
        return;
    }
    
    // Fallback toast implementation
    const toast = document.createElement('div');
    toast.className = `fixed bottom-4 right-4 px-6 py-4 rounded-lg shadow-xl text-white z-50 animate-slide-in ${
        type === 'success' ? 'bg-green-600' : 
        type === 'error' ? 'bg-red-600' : 
        'bg-blue-600'
    }`;
    toast.innerHTML = `<div class="flex items-center space-x-3"><i class="fas ${
        type === 'success' ? 'fa-check-circle' : 
        type === 'error' ? 'fa-exclamation-circle' : 
        'fa-info-circle'
    } text-xl"></i><span>${message}</span></div>`;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ==================== EXPORT FUNCTIONS ====================

// Make functions globally available
window.Auth = {
    // Core auth
    login,
    register,
    logout,
    
    // Session
    getCurrentUser,
    getCurrentSession,
    isLoggedIn,
    requireAuth,
    requireGuest,
    
    // Password
    requestPasswordReset,
    resetPassword,
    changePassword,
    
    // Profile
    updateProfile,
    getProfile,
    isProfileComplete,
    
    // Account
    deactivateAccount,
    deleteAccount,
    
    // Social
    googleLogin,
    facebookLogin,
    
    // UI
    updateAuthUI,
    
    // Config
    CONFIG: AUTH_CONFIG
};

// Make individual functions available
window.login = login;
window.register = register;
window.logout = logout;
window.isLoggedIn = isLoggedIn;
window.getCurrentUser = getCurrentUser;
window.requireAuth = requireAuth;
window.requireGuest = requireGuest;
window.requestPasswordReset = requestPasswordReset;
window.resetPassword = resetPassword;
window.changePassword = changePassword;
window.updateProfile = updateProfile;

// Initialize on page load
document.addEventListener('DOMContentLoaded', initAuth);

// ==================== EXPORT FOR MODULE USE ====================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        login,
        register,
        logout,
        getCurrentUser,
        isLoggedIn,
        requireAuth,
        requestPasswordReset,
        resetPassword,
        changePassword,
        updateProfile,
        AUTH_CONFIG
    };
}