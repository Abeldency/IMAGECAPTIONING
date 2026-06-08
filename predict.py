import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from tensorflow.keras.models import Model

# ====================================
# PATHS
# ====================================

MODEL_PATH = r"D:\AI_PROJECTS\ImageCaptioning\models\caption_model.keras"
TOKENIZER_PATH = r"D:\AI_PROJECTS\ImageCaptioning\models\tokenizer.pkl"

IMAGE_PATH = r"D:\AI_PROJECTS\ImageCaptioning\test1.jpg"

# ====================================
# LOAD MODEL
# ====================================

print("Loading model...")

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("Model loaded.")

# ====================================
# FEATURE EXTRACTOR
# ====================================

resnet = ResNet50(weights="imagenet")

resnet = Model(
    inputs=resnet.inputs,
    outputs=resnet.layers[-2].output
)

# ====================================
# IMAGE FEATURE
# ====================================

image = load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

image = img_to_array(image)

image = np.expand_dims(
    image,
    axis=0
)

image = preprocess_input(image)

feature = resnet.predict(
    image,
    verbose=0
)

# ====================================
# HELPERS
# ====================================

max_length = 38

index_word = {
    v: k
    for k, v in tokenizer.word_index.items()
}

# ====================================
# CAPTION GENERATION
# ====================================

caption = "startseq"

for _ in range(max_length):

    sequence = tokenizer.texts_to_sequences(
        [caption]
    )[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_length
    )

    yhat = model.predict(
        [feature, sequence],
        verbose=0
    )

    yhat = np.argmax(yhat)

    word = index_word.get(yhat)

    if word is None:
        break

    caption += " " + word

    if word == "endseq":
        break

# ====================================
# CLEAN OUTPUT
# ====================================

caption = caption.replace(
    "startseq",
    ""
)

caption = caption.replace(
    "endseq",
    ""
)

caption = caption.strip()

print("\nGenerated Caption:")
print(caption)