#!/usr/bin/env python3
"""
Streamlit demo: Multi-language code segmentation visualizer
Models:
  1. BoC (Logistic Regression)
  2. BiLSTM (Uncentered)
  3. BiLSTM (Centered)
  4. CNN-BiLSTM (Uncentered Hybrid)
  5. Transformer (Fine-tuned DistilRoBERTa from Hugging Face Hub)

Supports: Python, Java, JavaScript, and Combined ("all")
"""

import streamlit as st
import torch, joblib, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_REPO = "nurbek/semantic-segmentation-transformer"  # <── replace with your HF repo name
st.set_page_config(page_title="Multi-language Code Segmentation", layout="wide")
st.title("🔍 Multi-language Code Segmentation Comparison")
st.caption("Compare segmentation predictions across Python, Java, JavaScript, and Combined datasets using five models.")

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
def highlight_segments(code, probs, threshold=0.5):
    """Highlight predicted segmentation boundaries."""
    highlighted = ""
    for i, ch in enumerate(code):
        if i < len(probs) and probs[i] > threshold:
            highlighted += f"<span style='background-color:#ffd54f'>{ch}</span>"
        else:
            highlighted += ch
    return highlighted.replace("\n", "<br>")

def char_tensor(code):
    """Convert string to tensor of ASCII/byte values."""
    return torch.tensor([ord(c) if ord(c) < 256 else 0 for c in code],
                        dtype=torch.long).unsqueeze(0).to(DEVICE)

# ----------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------
@st.cache_resource
def load_models(lang):
    models = {}
    suffix = lang.lower()

    # 1️⃣ BoC
    try:
        models["boc"] = joblib.load(f"runs/lr_boc/lr_{suffix}.joblib")
    except:
        models["boc"] = None

    # 2️⃣ BiLSTM (Centered)
    try:
        from train_lstm_centered import BiLSTMCentered
        m_centered = BiLSTMCentered().to(DEVICE)
        m_centered.load_state_dict(torch.load(f"runs/lstm_centered/lstm_centered_{suffix}.pt", map_location=DEVICE))
        m_centered.eval()
        models["bilstm_centered"] = m_centered
    except:
        models["bilstm_centered"] = None

    # 3️⃣ BiLSTM (Uncentered)
    try:
        from train_lstm_uncentered import BiLSTM
        m_unc = BiLSTM().to(DEVICE)
        m_unc.load_state_dict(torch.load(f"runs/lstm_uncentered/lstm_uncentered_{suffix}.pt", map_location=DEVICE))
        m_unc.eval()
        models["bilstm_uncentered"] = m_unc
    except:
        models["bilstm_uncentered"] = None

    # 4️⃣ CNN-BiLSTM (Hybrid)
    try:
        from train_cnn_bilstm_uncentered_v3 import CNNBiLSTM
        m_cnn_bilstm = CNNBiLSTM().to(DEVICE)
        m_cnn_bilstm.load_state_dict(torch.load(f"runs/cnn_bilstm_uncentered_v3/cnn_bilstm_uncentered_{suffix}.pt", map_location=DEVICE))
        m_cnn_bilstm.eval()
        models["cnn_bilstm"] = m_cnn_bilstm
    except:
        models["cnn_bilstm"] = None

    # 5️⃣ Transformer (Fine-tuned from Hugging Face)
    try:
        st.info(f"[INFO] Loading fine-tuned Transformer from Hugging Face Hub → {HF_MODEL_REPO}")
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        transformer = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO).to(DEVICE)
        transformer.eval()
        models["transformer"] = (transformer, tokenizer)
    except Exception as e:
        st.warning(f"[WARN] Could not load Hugging Face model: {e}")
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        transformer = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2).to(DEVICE)
        transformer.eval()
        models["transformer"] = (transformer, tokenizer)

    return models

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
lang = st.selectbox("🌐 Select Language", ["python", "java", "javascript", "all"])

code_sample = {
    "python": """def compute_sum(a, b):
    total = a + b
    print('Result:', total)
    return total
""",
    "java": """public class Example {
    public static void main(String[] args) {
        int total = add(5, 7);
        System.out.println(total);
    }
}""",
    "javascript": """function add(a, b) {
  let total = a + b;
  console.log(total);
  return total;
}""",
    "all": """def mix_code(a, b): return a + b // works for python or java style syntax"""
}

code_input = st.text_area("✍️ Paste or edit your code snippet:", code_sample[lang], height=200)
models = load_models(lang)

# ----------------------------------------------------------
# RUN INFERENCE
# ----------------------------------------------------------
if st.button("🔎 Segment Code"):
    st.write(f"### 🧠 Model Predictions for *{lang.upper()}*")
    x = char_tensor(code_input)

    col1, col2 = st.columns(2)

    # BiLSTM Centered
    if models["bilstm_centered"]:
        with torch.no_grad():
            probs_centered = torch.sigmoid(models["bilstm_centered"](x).squeeze().cpu()).numpy()
        col1.markdown("**BiLSTM (Centered)**")
        col1.markdown(highlight_segments(code_input, probs_centered), unsafe_allow_html=True)

    # BiLSTM Uncentered
    if models["bilstm_uncentered"]:
        with torch.no_grad():
            probs_unc = torch.sigmoid(models["bilstm_uncentered"](x).squeeze().cpu()).numpy()
        col2.markdown("**BiLSTM (Uncentered)**")
        col2.markdown(highlight_segments(code_input, probs_unc), unsafe_allow_html=True)

    # CNN-BiLSTM
    if models["cnn_bilstm"]:
        with torch.no_grad():
            probs_cnn_bilstm = torch.sigmoid(models["cnn_bilstm"](x).squeeze().cpu()).numpy()
        st.markdown("**CNN-BiLSTM (Hybrid)**")
        st.markdown(highlight_segments(code_input, probs_cnn_bilstm), unsafe_allow_html=True)

    # Transformer
    transformer, tokenizer = models["transformer"]
    inputs = tokenizer(code_input, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = transformer(**inputs).logits
        pred = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).cpu().item()
        confidence = torch.softmax(logits, dim=-1).max().cpu().item()
    st.markdown("**Transformer (Fine-tuned DistilRoBERTa)**")
    st.write(f"Predicted class → `{pred}` Confidence → `{confidence:.3f}`")

    st.caption("🟨 Highlighted regions indicate where each model predicts likely code segment boundaries.")

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.header("⚙️ Model Info")
st.sidebar.write("""
**Included Models**
- BoC (Bag of Characters)
- BiLSTM (Centered / Uncentered)
- CNN-BiLSTM (Hybrid)
- Transformer (Fine-tuned DistilRoBERTa)
""")
st.sidebar.write(f"Device: **{DEVICE}**")
st.sidebar.write(f"Language selected: **{lang.upper()}**")
