import sqlite3
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data from SQLite
conn = sqlite3.connect("data/penguins.db")
df = pd.read_sql("SELECT * FROM penguins", conn)
conn.close()

# Feature selection
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

# Standardize features (important for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save trained model
with open("models/classifier.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler for future use (needed for predictions)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Logistic Regression model trained and saved successfully!")
