from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load dataset (features and labels)
X = np.load('X_music.npy')  # Features
y = np.load('y_music.npy')  # Labels (genre names)

# Initialize LabelEncoder to convert genre labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Encoding the genre labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and label encoder for future use
joblib.dump(model_rf, 'models/genre_model.pkl')
joblib.dump(label_encoder, 'models/genre_label_encoder.pkl')

print("Model training, evaluation, and saving completed successfully!")
