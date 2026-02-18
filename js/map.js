function initMap() {
    navigator.geolocation.getCurrentPosition((position) => {
        const { latitude, longitude } = position.coords;
        const map = L.map('map').setView([latitude, longitude], 13);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
        
        // User Marker
        L.marker([latitude, longitude]).addTo(map).bindPopup('You are here').openPopup();

        // Nearby Doctors fetching (Overpass API)
        fetchDoctors(latitude, longitude, map);
    });
}

async function fetchDoctors(lat, lon, map) {
    const query = `[out:json];node["amenity"="doctors"](around:5000,${lat},${lon});out;`;
    const response = await fetch(`https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`);
    const data = await response.json();

    data.elements.forEach(doc => {
        L.marker([doc.lat, doc.lon]).addTo(map).bindPopup(doc.tags.name || "Doctor");
    });
}