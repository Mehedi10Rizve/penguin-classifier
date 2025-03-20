import requests
import pickle
import pandas as pd

print("Fetching new penguin data from API...")

# Fetch new data from API
response = requests.get("http://130.225.39.127:8000/new_penguin/")

if response.status_code == 200:
    penguin_data = response.json()
    print("Penguin data received:", penguin_data)
else:
    print("Failed to fetch data! Status code:", response.status_code)
    exit()

# Convert to DataFrame
features = pd.DataFrame([penguin_data])

# Keep only the relevant features for prediction
features = features[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# Load trained model
print("Loading trained model...")
with open("models/classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
print("Loading scaler...")
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Standardize features before prediction
features_scaled = scaler.transform(features)

print("Making prediction...")
predicted_species = model.predict(features_scaled)[0]

# Save results to HTML file for GitHub Pages
html_content = f"""
<html>
<head><title>Penguin Prediction</title></head>
<body>
<h2>New Penguin Spotted!</h2>
<p>Bill Length: {penguin_data['bill_length_mm']} mm</p>
<p>Bill Depth: {penguin_data['bill_depth_mm']} mm</p>
<p>Flipper Length: {penguin_data['flipper_length_mm']} mm</p>
<p>Body Mass: {penguin_data['body_mass_g']} g</p>
<h3>Predicted Species: {predicted_species}</h3>
</body>
</html>
"""

with open("results/index.html", "w") as f:
    f.write(html_content)

print("Prediction saved successfully! Check results/index.html")