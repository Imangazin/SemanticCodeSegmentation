#!/usr/bin/env python3
"""
Streamlit demo: Multi-language code segmentation visualizer

Models:
  1. BoC (Logistic Regression)
  2. BiLSTM (Uncentered)
  3. BiLSTM (Centered)
  4. CNN-BiLSTM (Hybrid)
  5. Transformer (Fine-tuned DistilRoBERTa from Hugging Face)

Repo: nurbekimangazin/semanticcodesegmentation
"""

import os, torch, joblib, numpy as np, streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_REPO = "nurbekimangazin/semanticcodesegmentation"

st.set_page_config(page_title="Multi-language Code Segmentation", layout="wide")
st.title("🔍 Multi-language Code Segmentation Comparison")
st.caption("Compare segmentation predictions across Python, Java, JavaScript, and Combined datasets using five models.")

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
def highlight_segments(code, probs, threshold=0.5):
    html = ""
    for i, ch in enumerate(code):
        if i < len(probs) and probs[i] > threshold:
            html += f"<span style='background-color:#ffd54f'>{ch}</span>"
        else:
            html += ch
    return html.replace("\n", "<br>")

def char_tensor(code):
    return torch.tensor([ord(c) if ord(c) < 256 else 0 for c in code],
                        dtype=torch.long).unsqueeze(0).to(DEVICE)

# ----------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------
@st.cache_resource
def load_models(lang):
    models = {}
    suffix = lang.lower()

    # BoC
    try:
        models["boc"] = joblib.load(f"runs/lr_boc/lr_{suffix}.joblib")
    except:
        models["boc"] = None

    # BiLSTM (Centered)
    try:
        from train_lstm_centered import BiLSTMCentered
        m = BiLSTMCentered().to(DEVICE)
        m.load_state_dict(torch.load(f"runs/lstm_centered/lstm_centered_{suffix}.pt", map_location=DEVICE))
        m.eval()
        models["bilstm_centered"] = m
    except:
        models["bilstm_centered"] = None

    # BiLSTM (Uncentered)
    try:
        from train_lstm_uncentered import BiLSTM
        m = BiLSTM().to(DEVICE)
        m.load_state_dict(torch.load(f"runs/lstm_uncentered/lstm_uncentered_{suffix}.pt", map_location=DEVICE))
        m.eval()
        models["bilstm_uncentered"] = m
    except:
        models["bilstm_uncentered"] = None

    # CNN-BiLSTM
    try:
        from train_cnn_bilstm_uncentered_v3 import CNNBiLSTM
        m = CNNBiLSTM().to(DEVICE)
        m.load_state_dict(torch.load(f"runs/cnn_bilstm_uncentered_v3/cnn_bilstm_uncentered_{suffix}.pt", map_location=DEVICE))
        m.eval()
        models["cnn_bilstm"] = m
    except:
        models["cnn_bilstm"] = None

    # Transformer from Hugging Face
    try:
        st.info(f"Loading fine-tuned Transformer from Hugging Face → {HF_MODEL_REPO}")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
        transformer = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO).to(DEVICE)
        transformer.eval()
        models["transformer"] = (transformer, tokenizer)
    except Exception as e:
        st.warning(f"Could not load HF model: {e}")
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        transformer = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2).to(DEVICE)
        transformer.eval()
        models["transformer"] = (transformer, tokenizer)

    return models

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
lang = st.selectbox("🌐 Select Language", ["python", "java", "javascript", "all"])

samples = {
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
    "all": """def mix_code(a, b): return a + b // generic sample"""
}

code_input = st.text_area("✍️ Paste or edit code:", samples[lang], height=200)
models = load_models(lang)

# ----------------------------------------------------------
# RUN INFERENCE
# ----------------------------------------------------------
if st.button("🔎 Segment Code"):
    st.write(f"### 🧠 Predictions for *{lang.upper()}*")
    x = char_tensor(code_input)

    col1, col2 = st.columns(2)

    if models["bilstm_centered"]:
        with torch.no_grad():
            probs = torch.sigmoid(models["bilstm_centered"](x).squeeze().cpu()).numpy()
        col1.markdown("**BiLSTM (Centered)**")
        col1.markdown(highlight_segments(code_input, probs), unsafe_allow_html=True)

    if models["bilstm_uncentered"]:
        with torch.no_grad():
            probs = torch.sigmoid(models["bilstm_uncentered"](x).squeeze().cpu()).numpy()
        col2.markdown("**BiLSTM (Uncentered)**")
        col2.markdown(highlight_segments(code_input, probs), unsafe_allow_html=True)

    if models["cnn_bilstm"]:
        with torch.no_grad():
            probs = torch.sigmoid(models["cnn_bilstm"](x).squeeze().cpu()).numpy()
        st.markdown("**CNN-BiLSTM (Hybrid)**")
        st.markdown(highlight_segments(code_input, probs), unsafe_allow_html=True)

    # Transformer prediction
    transformer, tokenizer = models["transformer"]
    inputs = tokenizer(code_input, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = transformer(**inputs).logits
        pred = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).cpu().item()
        conf = torch.softmax(logits, dim=-1).max().cpu().item()
    st.markdown("**Transformer (Fine-tuned DistilRoBERTa)**")
    st.write(f"Predicted class → `{pred}` Confidence → `{conf:.3f}`")

    st.caption("🟨 Highlighted regions show likely code segment boundaries.")

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
st.sidebar.write(f"Language: **{lang.upper()}**")
