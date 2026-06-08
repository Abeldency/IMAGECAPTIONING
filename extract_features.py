import os

# ==========================
# Store Keras cache on D:
# ==========================
os.makedirs(r"D:\AI_PROJECTS\keras_cache", exist_ok=True)
os.environ["KERAS_HOME"] = r"D:\AI_PROJECTS\keras_cache"

import pickle
from tqdm import tqdm
import numpy as np

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)
from tensorflow.keras.models import Model

# ==========================
# Paths
# ==========================
IMAGE_FOLDER = r"D:\AI_PROJECTS\ImageCaptioning\dataset\Images"
FEATURES_FOLDER = r"D:\AI_PROJECTS\ImageCaptioning\features"
FEATURES_FILE = r"D:\AI_PROJECTS\ImageCaptioning\features\features.pkl"

# ==========================
# Debug Info
# ==========================
print("Current Directory:")
print(os.getcwd())

# ==========================
# Load ResNet50
# ==========================
print("\nLoading ResNet50...")

model = ResNet50(weights="imagenet")

model = Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

print("ResNet50 loaded successfully.")

# ==========================
# Extract Features
# ==========================
features = {}

image_list = os.listdir(IMAGE_FOLDER)

print(f"\nTotal Images Found: {len(image_list)}")
print("Starting feature extraction...\n")

for image_name in tqdm(image_list):

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    try:

        image = load_img(
            image_path,
            target_size=(224, 224)
        )

        image = img_to_array(image)

        image = np.expand_dims(
            image,
            axis=0
        )

        image = preprocess_input(image)

        feature = model.predict(
            image,
            verbose=0
        )

        features[image_name] = feature

    except Exception as e:

        print(f"\nError processing: {image_name}")
        print(e)

# ==========================
# Save Features
# ==========================
print("\nSaving features...")
print("Total extracted:", len(features))

with open(FEATURES_FILE, "wb") as f:
    pickle.dump(features, f)


# Verify Save
file_size = os.path.getsize(FEATURES_FILE)

print("\nFeature extraction completed successfully.")
print("Total features:", len(features))
print(f"Saved File Size: {file_size / (1024 * 1024):.2f} MB")
print(f"Saved To: {FEATURES_FILE}")