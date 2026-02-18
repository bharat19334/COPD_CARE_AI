document.addEventListener('DOMContentLoaded', () => {
    const risk = parseInt(localStorage.getItem('copdRisk')) || 0;
    const dietContent = document.getElementById('dietContent');

    if (risk > 60) {
        dietContent.innerHTML = `
            <h3>High Risk Diet Plan</h3>
            <ul>
                <li>High Protein: Eggs, Chicken, Pulses</li>
                <li>Low Carbs: Avoid heavy sugary foods</li>
                <li>Small, frequent meals to reduce breathlessness</li>
            </ul>`;
    } else {
        dietContent.innerHTML = `
            <h3>Preventive/Low Risk Diet Plan</h3>
            <ul>
                <li>Antioxidant rich foods: Berries, Spinach</li>
                <li>Stay Hydrated: 3-4 liters of water</li>
                <li>Vitamin D and Omega-3 supplements</li>
            </ul>`;
    }
});