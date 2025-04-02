import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sample dataset (Replace with real medical data)
data = {
    "glucose_level": [90, 200, 150, 130, 180],
    "chest_pain": [0, 1, 1, 0, 1],
    "cough": [1, 0, 1, 1, 0],
    "disease": ["Normal", "Diabetes", "Heart Disease", "Lung Disease", "Diabetes"]
}

df = pd.DataFrame(data)

# Encode labels
df["disease"] = df["disease"].astype("category").cat.codes

# Split dataset
X = df.drop("disease", axis=1)
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved!")
