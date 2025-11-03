#!/usr/bin/env python3
"""
Streamlit demo: Multi-language code segmentation visualizer

Models included:
  1. BoC (Logistic Regression)
  2. BiLSTM (Uncentered)
  3. BiLSTM (Centered)
  4. CNN-BiLSTM (Hybrid)

Supports: Python, Java, JavaScript, and Combined ("all")
"""

import os, torch, joblib, numpy as np, streamlit as st

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Multi-language Code Segmentation", layout="wide")
st.title("🔍 Multi-language Code Segmentation Comparison")
st.caption("Compare segmentation predictions across Python, Java, JavaScript, and Combined datasets using four neural and baseline models.")

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
def highlight_segments(code, probs, threshold=0.5):
    """Highlight predicted segmentation boundaries."""
    html = ""
    for i, ch in enumerate(code):
        if i < len(probs) and probs[i] > threshold:
            html += f"<span style='background-color:#ffd54f'>{ch}</span>"
        else:
            html += ch
    return html.replace("\n", "<br>")

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

    return models

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
lang = st.selectbox("🌐 Select Language", ["python", "java", "javascript", "all"])

code_samples = {
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
    "all": """def mix_code(a, b): return a + b // generic syntax for all"""
}

code_input = st.text_area("✍️ Paste or edit your code snippet:", code_samples[lang], height=200)
models = load_models(lang)

# ----------------------------------------------------------
# RUN INFERENCE
# ----------------------------------------------------------
if st.button("🔎 Segment Code"):
    st.write(f"### 🧠 Predictions for *{lang.upper()}*")
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

    st.caption("🟨 Highlighted regions indicate likely code segment boundaries predicted by each model.")

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.header("⚙️ Model Info")
st.sidebar.write("""
**Included Models**
- BoC (Bag of Characters)
- BiLSTM (Centered)
- BiLSTM (Uncentered)
- CNN-BiLSTM (Hybrid)
""")
st.sidebar.write(f"Device: **{DEVICE}**")
st.sidebar.write(f"Language: **{lang.upper()}**")
