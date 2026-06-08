import pickle
import numpy as np
import gradio as gr

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
# LOAD MODEL
# ====================================

MODEL_PATH = r"D:\AI_PROJECTS\ImageCaptioning\models\caption_model.keras"
TOKENIZER_PATH = r"D:\AI_PROJECTS\ImageCaptioning\models\tokenizer.pkl"

print("Loading caption model...")
caption_model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("Loading ResNet50...")
resnet = ResNet50(weights="imagenet")

resnet = Model(
    inputs=resnet.inputs,
    outputs=resnet.layers[-2].output
)

max_length = 38

index_word = {
    v: k
    for k, v in tokenizer.word_index.items()
}

# ====================================
# CAPTION FUNCTION
# ====================================

def generate_caption(image):

    image = image.resize((224, 224))

    image = np.array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(image)

    feature = resnet.predict(
        image,
        verbose=0
    )

    caption = "startseq"

    for _ in range(max_length):

        sequence = tokenizer.texts_to_sequences(
            [caption]
        )[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length
        )

        yhat = caption_model.predict(
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

    caption = caption.replace(
        "startseq",
        ""
    )

    caption = caption.replace(
        "endseq",
        ""
    )

    return caption.strip()

# ====================================
# UI
# ====================================

demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="AI Generated Caption"),
    title="📷 AI Image Caption Generator",
    description="""
    Upload an image and the AI will generate a caption using
    ResNet50 feature extraction and an LSTM-based captioning model.
    """,
)

print("Loading caption model...")
caption_model = load_model(MODEL_PATH)
print("Caption model loaded.")

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("Tokenizer loaded.")

print("Loading ResNet50...")
resnet = ResNet50(weights="imagenet")
print("ResNet50 loaded.")

resnet = Model(
    inputs=resnet.inputs,
    outputs=resnet.layers[-2].output
)

print("Feature extractor ready.")

print("Launching Gradio...")
demo.launch(
    share=True,
    inbrowser=True
)