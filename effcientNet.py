import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.utils import to_categorical

# Load the data
merged_data = gpd.read_file(r"D:\datascienEO\merged_data.geojson")

# Encode species labels to integers
unique_species = merged_data['l3_species'].unique()
species_to_index = {species: idx for idx, species in enumerate(unique_species)}
labels = merged_data['l3_species'].map(species_to_index).values
num_classes = len(unique_species)

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=num_classes)

# Extract band data as 4D array
band_data = merged_data.iloc[:, 1:-4]
import ast

band_data['B2_3'] = band_data['B2_3'].apply(ast.literal_eval)
band_data['B4_3'] = band_data['B4_3'].apply(ast.literal_eval)
band_data['B8_3'] = band_data['B8_3'].apply(ast.literal_eval)
band_data['NDVI_3'] = band_data['NDVI_3'].apply(ast.literal_eval)

rows = len(band_data)
X = np.zeros((rows, 11, 11, 4), dtype=np.float32)

for i in range(rows):
    X[i, :, :, 0] = np.array(band_data['B2_3'][i], dtype=np.float32)  # Band B2
    X[i, :, :, 1] = np.array(band_data['B4_3'][i], dtype=np.float32)  # Band B4
    X[i, :, :, 2] = np.array(band_data['B8_3'][i], dtype=np.float32)  # Band B8
    X[i, :, :, 3] = np.array(band_data['NDVI_3'][i], dtype=np.float32)  # NDVI

# Normalize data to [0, 1]
X = X / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=24, shuffle=True)

# Define EfficientNetB0 model with custom input for 4 channels
input_tensor = Input(shape=(11, 11, 4))
base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=input_tensor)

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
output = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test_classes, y_pred_classes)
plt.show()

# Print accuracy
acc = accuracy_score(y_test_classes, y_pred_classes)
print("Accuracy with EfficientNet: ", acc)
