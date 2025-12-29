#!/usr/bin/env python
# coding: utf-8

# In[1]:


import speech_recognition as sr

# ==========================================
# SPEECH TO TEXT FUNCTION
# ==========================================
def listen_to_voice():
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Use the default microphone
    try:
        with sr.Microphone() as source:
            print("\n🎤 Adjusting for ambient noise... (Please wait)")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("🔴 Listening... (Say something like 'Move the sofa left')")
            
            # Listen for audio (stops automatically when you stop speaking)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("✅ Audio captured. Converting to text...")

            # Convert to text using Google's free API
            # (For offline, you would use Vosk or Whisper)
            text = recognizer.recognize_google(audio)
            
            return text.lower()

    except sr.WaitTimeoutError:
        print("❌ No speech detected. Try again.")
        return None
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"❌ API Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error accessing microphone: {e}")
        print("Tip: Ensure you have 'pyaudio' installed.")
        return None

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    print("--- Voice Command Tester ---")
    
    while True:
        command = listen_to_voice()
        
        if command:
            print(f"🗣️  You said: '{command}'")
            
            # Simple Keyword Check (Simulating what your AI will do later)
            if "move" in command:
                print("   -> Intent: MOVE_OBJECT")
            elif "rotate" in command:
                print("   -> Intent: ROTATE_OBJECT")
            elif "color" in command or "paint" in command:
                print("   -> Intent: CHANGE_COLOR")
            elif "exit" in command or "stop" in command:
                print("👋 Exiting...")
                break
        
        print("-" * 30)


# In[2]:


import json
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_FILE = 'voice_dataset.json'
VOCAB_SIZE = 1000      # Max number of words to keep
MAX_LENGTH = 20        # Max length of a sentence
EMBEDDING_DIM = 16     # Size of the word vectors
NUM_EPOCHS = 30        # How many times to train on the dataset

# ==========================================
# 2. LOAD & PREPROCESS DATA
# ==========================================
if not os.path.exists(DATASET_FILE):
    print(f"❌ Error: '{DATASET_FILE}' not found. Run generate_voice_dataset.py first!")
    exit()

print(f"Loading dataset from {DATASET_FILE}...")
with open(DATASET_FILE, 'r') as f:
    data = json.load(f)

sentences = [item['text'] for item in data]
labels = [item['intent'] for item in data]

print(f"Loaded {len(sentences)} examples.")

# --- Encode Labels (Text -> Numbers) ---
# e.g., "MOVE_OBJECT" -> 0, "ROTATE_OBJECT" -> 1
label_encoder = LabelEncoder()
training_labels_final = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

print(f"Classes found: {list(label_encoder.classes_)}")

# --- Tokenize Text (Sentences -> Numbers) ---
# The Tokenizer builds a dictionary of word -> index
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Convert sentences to sequences of numbers
sequences = tokenizer.texts_to_sequences(sentences)
# Pad sequences so they are all the same length (MAX_LENGTH)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

# ==========================================
# 3. BUILD THE MODEL
# ==========================================
print("\n--- Building Neural Network ---")
model = tf.keras.Sequential([
    # Embedding Layer: Converts word indices to dense vectors of fixed size
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    # GlobalAveragePooling1D: Averages vectors, good for simple text classification
    tf.keras.layers.GlobalAveragePooling1D(),
    # Dense Layer: Learns patterns
    tf.keras.layers.Dense(24, activation='relu'),
    # Output Layer: Probability for each class
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
print("\n--- Starting Training ---")
history = model.fit(
    np.array(padded_sequences), 
    np.array(training_labels_final), 
    epochs=NUM_EPOCHS, 
    verbose=2
)

# ==========================================
# 5. SAVE ARTIFACTS
# ==========================================
print("\n--- Saving Files ---")

# A. Save Tokenizer Dictionary (Crucial for the App)
# We save the word_index so the app knows "sofa" = 42
tokenizer_json = tokenizer.to_json()
with open('tokenizer_dict.json', 'w') as f:
    f.write(tokenizer_json)
print("✅ Saved 'tokenizer_dict.json'")

# B. Save Label Map
# The app needs to know 0 = "CHANGE_COLOR"
label_map = {int(index): label for index, label in enumerate(label_encoder.classes_)}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f, indent=2)
print("✅ Saved 'label_map.json'")

# C. Convert & Save TFLite Model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('voice_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ Saved 'voice_model.tflite'")

print("\n🎉 Training Complete! Move these 3 files to your backend or app assets folder.")


# In[ ]:




