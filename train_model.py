import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model

# Step 1: Load and Prepare the Dataset
# Use the correct separator for your CSV file
df = pd.read_csv('F:/AIforhealth/backend/data/disease_data.csv', sep=',')  # Changed sep to ','

# Fill any missing values with empty strings
df = df.fillna('')

# Print the DataFrame columns to check if they are loaded correctly
print("DataFrame Columns:", df.columns.tolist())

# Define the columns
symptoms_columns = [f'Symptom {i}' for i in range(1, 8)]
output_columns = ['Disease', 'Description', 'Drugs Used to Treat']

# Combine symptoms into one string for encoding
df['combined_symptoms'] = df[symptoms_columns].apply(lambda x: ' '.join(x), axis=1)

# Encode symptoms as input features
symptoms_encoder = LabelEncoder()
df['symptoms_encoded'] = symptoms_encoder.fit_transform(df['combined_symptoms'])

# Encode the output labels (multi-output classification)
disease_encoder = LabelEncoder()
df['disease_encoded'] = disease_encoder.fit_transform(df['Disease'])

description_encoder = LabelEncoder()
df['description_encoded'] = description_encoder.fit_transform(df['Description'])

drugs_encoder = LabelEncoder()
df['drugs_encoded'] = drugs_encoder.fit_transform(df['Drugs Used to Treat'])

# Features (inputs) and Labels (outputs)
X = df['symptoms_encoded']
y_disease = df['disease_encoded']
y_description = df['description_encoded']
y_drugs = df['drugs_encoded']

# One-hot encoding of input features
X = tf.keras.utils.to_categorical(X, num_classes=len(symptoms_encoder.classes_))

# Split the data into training and testing sets
X_train, X_test, y_disease_train, y_disease_test, y_description_train, y_description_test, y_drugs_train, y_drugs_test = train_test_split(
    X, y_disease, y_description, y_drugs, test_size=0.2, random_state=42
)

# Step 2: Build the Multi-Output Model
# Input layer
input_layer = Input(shape=(X_train.shape[1],))

# Hidden layers
hidden_layer_1 = Dense(128, activation='relu')(input_layer)
hidden_layer_2 = Dense(64, activation='relu')(hidden_layer_1)

# Output layers (Multi-output)
output_disease = Dense(len(disease_encoder.classes_), activation='softmax', name='disease_output')(hidden_layer_2)
output_description = Dense(len(description_encoder.classes_), activation='softmax', name='description_output')(hidden_layer_2)
output_drugs = Dense(len(drugs_encoder.classes_), activation='softmax', name='drugs_output')(hidden_layer_2)

# Define the model
model = Model(inputs=input_layer, outputs=[output_disease, output_description, output_drugs])

# Compile the model
model.compile(optimizer='adam', 
              loss={'disease_output': 'sparse_categorical_crossentropy',
                    'description_output': 'sparse_categorical_crossentropy',
                    'drugs_output': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'])

# Summary of the model
model.summary()

# Step 3: Train the Model
history = model.fit(X_train, 
                    {'disease_output': y_disease_train, 
                     'description_output': y_description_train, 
                     'drugs_output': y_drugs_train},
                    epochs=50,
                    validation_data=(X_test, 
                                     {'disease_output': y_disease_test, 
                                      'description_output': y_description_test, 
                                      'drugs_output': y_drugs_test}),
                    batch_size=16)

# Step 4: Evaluate the Model
loss, disease_loss, description_loss, drugs_loss, disease_accuracy, description_accuracy, drugs_accuracy = model.evaluate(X_test, 
                                                                                                  {'disease_output': y_disease_test,
                                                                                                   'description_output': y_description_test,
                                                                                                   'drugs_output': y_drugs_test})

print(f"Disease Accuracy: {disease_accuracy}")
print(f"Description Accuracy: {description_accuracy}")
print(f"Drugs Accuracy: {drugs_accuracy}")

# Step 5: Make Predictions
# Example prediction
sample_input = X_test[0:1]
predictions = model.predict(sample_input)

predicted_disease = disease_encoder.inverse_transform([np.argmax(predictions[0])])
predicted_description = description_encoder.inverse_transform([np.argmax(predictions[1])])
predicted_drugs = drugs_encoder.inverse_transform([np.argmax(predictions[2])])

print(f"Predicted Disease: {predicted_disease}")
print(f"Predicted Description: {predicted_description}")
print(f"Predicted Drugs: {predicted_drugs}")
print(df.columns)

# Save the model (optional)
model.save('multi_output_disease_model.h5')
