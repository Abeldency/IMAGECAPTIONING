import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os
import random

# Base paths relative to the directory of this script for shareability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "caption_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")

# Page config (Must be the first Streamlit command)
st.set_page_config(
    page_title="VISION AI - Image Captioning",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Theme, Glassmorphism, Responsive Grid, Animations)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 50%, #020617 100%);
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #f8fafc;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #080c14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #cbd5e1;
}

/* Glassmorphic Container Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 24px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    border: 1px solid rgba(99, 102, 241, 0.25);
    box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.08);
    transform: translateY(-2px);
}

/* Hero Section */
.hero-container {
    position: relative;
    text-align: center;
    padding: 50px 20px;
    border-radius: 24px;
    background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.12), transparent), rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 30px;
}
.hero-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.12);
    color: #818cf8;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(99, 102, 241, 0.25);
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.hero-title {
    font-size: 60px;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 40%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 20px;
    font-weight: 500;
    color: #94a3b8;
    margin-bottom: 12px;
}
.hero-desc {
    font-size: 15px;
    color: #cbd5e1;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Dashboard Metric Grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 20px 10px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    background: rgba(99, 102, 241, 0.04);
    border-color: rgba(99, 102, 241, 0.2);
}
.metric-val {
    font-size: 28px;
    font-weight: 700;
    color: #6366f1;
    margin-bottom: 4px;
}
.metric-lbl {
    font-size: 11px;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tech Stack Cards Grid */
.tech-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.tech-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px 16px;
    text-align: center;
    transition: all 0.3s ease;
}
.tech-card:hover {
    background: rgba(99, 102, 241, 0.04);
    border-color: rgba(99, 102, 241, 0.2);
    transform: translateY(-2px);
}
.tech-card h3 {
    margin: 0 0 8px 0;
    font-size: 16px;
    font-weight: 600;
    color: #e2e8f0;
}
.tech-card p {
    margin: 0;
    font-size: 14px;
    color: #64748b;
    font-weight: 500;
}

/* Custom Overrides for Streamlit Widgets */
[data-testid="stFileUploader"] {
    background-color: rgba(15, 23, 42, 0.3) !important;
    border: 2px dashed rgba(99, 102, 241, 0.2) !important;
    border-radius: 16px !important;
    padding: 10px !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35) !important;
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
}

/* AI Caption Output Display Box */
.caption-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(168, 85, 247, 0.1);
    margin-top: 20px;
    animation: pulseGlow 3s infinite alternate;
}
.caption-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    background: #818cf8;
    color: white;
    padding: 3px 8px;
    border-radius: 4px;
    margin-bottom: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.caption-text {
    font-size: 20px;
    font-weight: 600;
    color: #f8fafc;
    line-height: 1.4;
    font-style: italic;
    margin: 0;
}

/* Section Header Style */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 30px;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Footer Section */
.footer-container {
    text-align: center;
    padding: 50px 20px 20px 20px;
    color: #64748b;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 50px;
}
.footer-logo {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: #f8fafc;
    margin-bottom: 5px;
}
.footer-author {
    font-size: 15px;
    color: #94a3b8;
    margin-bottom: 10px;
}
.footer-copy {
    font-size: 12px;
}

/* Responsive Media Queries (Mobile Specific Styles) */
@media (max-width: 992px) {
    .metric-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .tech-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
@media (max-width: 768px) {
    .hero-title {
        font-size: 38px !important;
    }
    .hero-subtitle {
        font-size: 16px !important;
    }
    .hero-desc {
        font-size: 13px !important;
    }
    .hero-container {
        padding: 35px 15px !important;
    }
    .glass-card {
        padding: 16px !important;
    }
    .caption-text {
        font-size: 16px !important;
    }
}
@media (max-width: 480px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }
    .tech-grid {
        grid-template-columns: 1fr;
    }
}

@keyframes pulseGlow {
    0% { box-shadow: 0 4px 20px rgba(168, 85, 247, 0.1); }
    100% { box-shadow: 0 4px 30px rgba(99, 102, 241, 0.2); }
}
</style>
""", unsafe_allow_html=True)

# Cache Resource with Dynamic Loading, Library Checking, & Graceful Fallback
@st.cache_resource
def load_everything():
    # Attempt lazy loading to avoid startup crashes if libraries are missing in user env
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        from tensorflow.keras.models import Model
    except (ImportError, ModuleNotFoundError) as e:
        return None, None, None, None, None, f"Missing required deep learning libraries (TensorFlow/Keras). Error: {str(e)}"

    if not os.path.exists(MODEL_PATH):
        return None, None, None, None, None, f"Model file not found at: `{MODEL_PATH}`"
    if not os.path.exists(TOKENIZER_PATH):
        return None, None, None, None, None, f"Tokenizer file not found at: `{TOKENIZER_PATH}`"

    try:
        caption_model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        resnet = ResNet50(weights="imagenet")
        resnet = Model(inputs=resnet.inputs, outputs=resnet.layers[-2].output)

        return caption_model, tokenizer, resnet, pad_sequences, preprocess_input, None
    except Exception as e:
        return None, None, None, None, None, f"Error initializing deep learning model: {str(e)}"

# Attempt initialization
load_result = load_everything()
if load_result and load_result[-1] is None:
    caption_model, tokenizer, resnet, pad_sequences, preprocess_input, _ = load_result
    model_loaded = True
else:
    model_loaded = False
    error_message = load_result[-1] if load_result else "Unknown initialization error"

# Generate caption function
def generate_caption(image):
    if not model_loaded:
        # High-fidelity simulated demo predictions when model is not loaded
        import time
        time.sleep(2.0)  # Simulate AI modeling latency
        
        simulated_captions = [
            "a group of children playing soccer on a green field",
            "a close-up of a delicious pizza with melted cheese and fresh basil",
            "a modern office space with laptop computers on wooden tables",
            "a beautiful snow-capped mountain range reflecting in a quiet lake",
            "a cute fluffy kitten curled up asleep on a warm blanket",
            "a busy city intersection lit up with bright neon signs at night",
            "a young surfer riding a large wave in the blue ocean",
            "a tranquil forest path winding through tall redwood trees",
            "a classic cup of hot coffee with latte art on a wooden surface"
        ]
        return random.choice(simulated_captions)

    # Real model prediction pipeline
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = resnet.predict(image, verbose=0)

    caption = "startseq"
    max_length = 38
    index_word = {v: k for k, v in tokenizer.word_index.items()}

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = caption_model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = index_word.get(yhat)

        if word is None:
            break

        caption += " " + word

        if word == "endseq":
            break

    caption = caption.replace("startseq", "").replace("endseq", "")
    return caption.strip()

# Hero Section HTML
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ Next-Gen Vision Intelligence</div>
    <div class="hero-title">VISION AI</div>
    <div class="hero-subtitle">Intelligent Image Understanding Platform</div>
    <div class="hero-desc">
        Translate pixel-level image semantics into natural, human-readable sentences using state-of-the-art Computer Vision and Deep Learning.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Styling & Content
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 20px;">'
                '<span style="font-size: 40px;">📷</span>'
                '<h2 style="margin: 10px 0 5px 0; font-weight: 800; color: white;">VISION AI</h2>'
                '<span style="background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.25); padding: 3px 10px; border-radius: 99px; font-size: 11px; font-weight: 600;">v1.0.0</span>'
                '</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Abel Dency Johnson**")
    
    st.markdown("### 🏢 Company")
    st.markdown("ADJ Software Productions")
    
    st.markdown("### 🛠️ AI Stack")
    st.markdown("""
    *   **Feature Extractor**: ResNet50 CNN
    *   **Sequence Decoder**: LSTM Network
    *   **Core Engine**: TensorFlow / Keras
    *   **Data Pipeline**: NumPy & PIL
    """)
    
    st.markdown("### 📊 Dataset")
    st.markdown("Flickr8k")
    
    st.markdown("### ⚡ System Status")
    if model_loaded:
        st.success("Deep Learning Model Active")
    else:
        st.warning("Running in Demo Mode")
        st.info("💡 Real model files missing. Simulated predictions active.")

# AI Dashboard Metrics Grid
st.markdown('<div class="section-title">📊 AI Dashboard</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-val">8,091</div>
        <div class="metric-lbl">Dataset Images</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">8,831</div>
        <div class="metric-lbl">Vocabulary Size</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">10</div>
        <div class="metric-lbl">Training Epochs</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">{"Ready" if model_loaded else "Demo Mode"}</div>
        <div class="metric-lbl">Inference Engine</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Image Workspace Section
st.markdown('<div class="section-title">🖥️ Image Analysis Workspace</div>', unsafe_allow_html=True)

# Wrap input controls in glassmorphic card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

input_mode = st.radio(
    "Choose Input Stream",
    ["Upload Image File", "Capture Live Camera"],
    horizontal=True
)

if input_mode == "Upload Image File":
    uploaded_file = st.file_uploader(
        "Upload image (PNG, JPG, or JPEG)",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.camera_input(
        "Capture Live Stream"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Show input error/demo banner if not loaded
if not model_loaded and uploaded_file:
    st.info(f"💡 **Demo Mode Note**: The system is processing your uploaded file. Since the model weights are not loaded, a high-fidelity mock prediction will be generated. (Reason: {error_message})")

# Image analysis workspace layout
if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Responsive grid layout using Streamlit columns
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-top: 0; color: #f1f5f9;">Uploaded Image</h3>', unsafe_allow_html=True)
        st.image(image, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with right_col:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-top: 0; color: #f1f5f9;">AI Generated Caption</h3>', unsafe_allow_html=True)
        st.write("Click the button below to feed the image through our visual intelligence network.")
        
        # Use full width button styled beautifully via global CSS
        if st.button("Generate Description", width='stretch'):
            with st.spinner("AI Engine is translating visual semantics..."):
                caption = generate_caption(image)
                
                # Render caption output in beautiful custom glow card
                st.markdown(f"""
                <div class="caption-card">
                    <div class="caption-badge">✨ AI Translation</div>
                    <p class="caption-text">"{caption}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Include a text-friendly version for easy mobile copying
                st.markdown("<br>", unsafe_allow_html=True)
                st.text_input("Copy Caption:", value=caption, disabled=True)
                
        st.markdown('</div>', unsafe_allow_html=True)

# Technology Stack Cards Grid
st.markdown('<div class="section-title">🛠️ Core Technology Stack</div>', unsafe_allow_html=True)
st.markdown("""
<div class="tech-grid">
    <div class="tech-card">
        <h3>Computer Vision</h3>
        <p>ResNet50 Feature Extractor</p>
    </div>
    <div class="tech-card">
        <h3>Sequence Modeling</h3>
        <p>LSTM Natural Language Decoder</p>
    </div>
    <div class="tech-card">
        <h3>AI Framework</h3>
        <p>TensorFlow & Keras API</p>
    </div>
    <div class="tech-card">
        <h3>Training Ground</h3>
        <p>Flickr8k Dataset</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Technical Description in Glass Card
st.markdown('<div class="section-title">📖 Deep Learning Architecture</div>', unsafe_allow_html=True)
st.markdown("""
<div class="glass-card">
    <p style="margin: 0; line-height: 1.6; color: #cbd5e1; font-size: 15px;">
        <strong>Vision AI</strong> is an intelligent image understanding platform that marries computer vision and natural language processing to describe visual scenes. 
        <br><br>
        The pipeline operates in two major phases: 
        1. <strong>Feature Extraction:</strong> Images are processed through a ResNet50 neural network pre-trained on ImageNet. Instead of performing classification, we extract the dense feature bottleneck vector representing global visual semantics.
        2. <strong>Language Generation:</strong> The visual feature vector is injected into an LSTM (Long Short-Term Memory) sequence decoder. Working in tandem with a word tokenizer trained on Flickr8k, the network recursively predicts the next word in the sequence until a complete, grammatically sound caption is assembled.
    </p>
</div>
""", unsafe_allow_html=True)

# Footer Section
st.markdown("""
<div class="footer-container">
    <div class="footer-logo">VISION AI</div>
    <div class="footer-author">Developed by <strong>Abel Dency Johnson</strong> &bull; ADJ Software Productions</div>
    <div style="font-size: 13px; color: #475569; margin-bottom: 15px;">
        E-Mail: <a href="mailto:abeldency@gmail.com" style="color: #6366f1; text-decoration: none;">abeldency@gmail.com</a>
    </div>
    <div class="footer-copy">&copy; 2026 ADJ Software Productions. All Rights Reserved.</div>
</div>
""", unsafe_allow_html=True)
