#!/usr/bin/env python3
"""
Interactive Streamlit app for multi-language code segmentation

Includes:
- BoC (Logistic Regression)
- BiLSTM (Centered / Uncentered)
- CNN-BiLSTM (Hybrid)
"""

import os, torch, joblib, numpy as np, streamlit as st

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Code Segmentation Demo", layout="wide")
st.title("🔍 Multi-language Code Segmentation Comparison")

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def highlight_segments(code, probs, threshold=0.5):
    """Return HTML string with highlighted predicted boundaries."""
    html = ""
    for i, ch in enumerate(code):
        if i < len(probs) and probs[i] > threshold:
            html += f"<span style='background-color:#ffd54f'>{ch}</span>"
        else:
            html += ch
    return html.replace("\n", "<br>")

def char_tensor(code):
    """Convert text to tensor."""
    return torch.tensor([ord(c) if ord(c) < 256 else 0 for c in code],
                        dtype=torch.long).unsqueeze(0).to(DEVICE)

@st.cache_resource
def load_model(lang, model_name):
    """Load a specific model for a given language."""
    suffix = lang.lower()
    model = None

    try:
        if model_name == "BoC":
            path = f"runs/lr_boc/lr_{suffix}.joblib"
            model = joblib.load(path) if os.path.exists(path) else None

        elif model_name == "BiLSTM (Centered)":
            from train_lstm_centered import BiLSTMCentered
            m = BiLSTMCentered().to(DEVICE)
            m.load_state_dict(torch.load(f"runs/lstm_centered/lstm_centered_{suffix}.pt", map_location=DEVICE))
            m.eval()
            model = m

        elif model_name == "BiLSTM (Uncentered)":
            from train_lstm_uncentered import BiLSTM
            m = BiLSTM().to(DEVICE)
            m.load_state_dict(torch.load(f"runs/lstm_uncentered/lstm_uncentered_{suffix}.pt", map_location=DEVICE))
            m.eval()
            model = m

        elif model_name == "CNN-BiLSTM (Hybrid)":
            from train_cnn_bilstm_uncentered_v3 import CNNBiLSTM
            m = CNNBiLSTM().to(DEVICE)
            m.load_state_dict(torch.load(f"runs/cnn_bilstm_uncentered_v3/cnn_bilstm_uncentered_{suffix}.pt", map_location=DEVICE))
            m.eval()
            model = m

    except Exception as e:
        st.warning(f"⚠️ Could not load {model_name} for {lang}: {e}")
        model = None

    return model


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.sidebar.header("⚙️ Settings")

lang = st.sidebar.selectbox("🌐 Language", ["python", "java", "javascript", "all"])
model_choice = st.sidebar.selectbox("🧠 Model", [
    "BoC",
    "BiLSTM (Centered)",
    "BiLSTM (Uncentered)",
    "CNN-BiLSTM (Hybrid)"
])

st.sidebar.write(f"Device: **{DEVICE}**")

# ----------------------------------------------------------
# Sample inputs
# ----------------------------------------------------------
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

code_input = st.text_area("✍️ Paste or edit code snippet:", samples[lang], height=200)

# ----------------------------------------------------------
# Load and run
# ----------------------------------------------------------
model = load_model(lang, model_choice)

if st.button("🔎 Segment Code"):
    if not model:
        st.error(f"Model `{model_choice}` not available for {lang.upper()}.")
    else:
        st.subheader(f"🧩 Predictions — {model_choice} ({lang.upper()})")

        x = char_tensor(code_input)
        with torch.no_grad():
            probs = torch.sigmoid(model(x).squeeze().cpu()).numpy()

        st.markdown(highlight_segments(code_input, probs), unsafe_allow_html=True)
        st.caption("🟨 Highlighted areas = likely code segment boundaries.")


# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------
st.markdown("---")
st.caption("© 2025 Semantic Code Segmentation — Demo app for academic presentation.")
