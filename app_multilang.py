#!/usr/bin/env python3
"""
Streamlit App: Multi-Language Code Segmentation
Supports: BoC, BiLSTM, CNN, CNN-BiLSTM
Author: Nurbek Imangazin
"""

import streamlit as st
import torch, joblib, numpy as np, os

# -------------------------------------------------------
# 🧩 CONFIG
# -------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANGS = ["python", "java", "javascript", "multilanguage_all"]
MODELS = {
    "BoC": "runs/lr_boc",
    "BiLSTM (Uncentered)": "runs/lstm_uncentered",
    "BiLSTM (Centered)": "runs/lstm_centered",
    "CNN (Uncentered)": "runs/cnn_uncentered",
    "CNN-BiLSTM (Uncentered)": "runs/cnn_bilstm_uncentered_v3",
}

st.set_page_config(page_title="Semantic Code Segmentation", layout="wide")
st.title("🧠 Semantic Code Segmentation")
st.markdown("Select a language and model to segment your code automatically.")

# -------------------------------------------------------
# 📦 MODEL LOADER
# -------------------------------------------------------
@st.cache_resource
def load_model(model_type, lang):
    """Load model safely depending on type."""
    try:
        if model_type == "BoC":
            model_path = os.path.join(MODELS["BoC"], f"lr_{lang}.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                return model, "BoC"
            else:
                st.warning(f"⚠️ Missing BoC model for {lang}.")
                return None, "BoC"

        elif "lstm_uncentered" in MODELS["BiLSTM (Uncentered)"]:
            from train_lstm_uncentered import BiLSTM
            model_path = os.path.join(MODELS["BiLSTM (Uncentered)"], f"lstm_uncentered_{lang}.pt")
            if os.path.exists(model_path):
                model = BiLSTM().to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                return model, "BiLSTM (Uncentered)"
            else:
                st.warning(f"⚠️ Missing BiLSTM (Uncentered) for {lang}.")
                return None, "BiLSTM (Uncentered)"

        elif "lstm_centered" in MODELS["BiLSTM (Centered)"]:
            from train_lstm_centered import BiLSTMCentered
            model_path = os.path.join(MODELS["BiLSTM (Centered)"], f"lstm_centered_{lang}.pt")
            if os.path.exists(model_path):
                model = BiLSTMCentered().to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                return model, "BiLSTM (Centered)"
            else:
                st.warning(f"⚠️ Missing BiLSTM (Centered) for {lang}.")
                return None, "BiLSTM (Centered)"

        elif "cnn_uncentered" in MODELS["CNN (Uncentered)"]:
            from train_cnn_uncentered import CharCNN
            model_path = os.path.join(MODELS["CNN (Uncentered)"], f"cnn_uncentered_{lang}.pt")
            if os.path.exists(model_path):
                model = CharCNN().to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                return model, "CNN (Uncentered)"
            else:
                st.warning(f"⚠️ Missing CNN (Uncentered) for {lang}.")
                return None, "CNN (Uncentered)"

        elif "cnn_bilstm" in MODELS["CNN-BiLSTM (Uncentered)"]:
            from train_cnn_bilstm_uncentered_v3 import CNNBiLSTM
            model_path = os.path.join(MODELS["CNN-BiLSTM (Uncentered)"], f"cnn_bilstm_uncentered_{lang}.pt")
            if os.path.exists(model_path):
                model = CNNBiLSTM().to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                return model, "CNN-BiLSTM (Uncentered)"
            else:
                st.warning(f"⚠️ Missing CNN-BiLSTM (Uncentered) for {lang}.")
                return None, "CNN-BiLSTM (Uncentered)"

    except Exception as e:
        st.error(f"⚠️ Could not load {model_type} for {lang}: {e}")
        return None, model_type


# -------------------------------------------------------
# 🔧 PREDICTION FUNCTION (FIXED)
# -------------------------------------------------------
def get_predictions(model, model_type, code_snippet, device="cpu"):
    """Return per-character probabilities for segmentation."""
    if model is None:
        raise ValueError("Model not loaded.")

    # --- BoC model (scikit-learn) ---
    if model_type.lower().startswith("boc"):
        X = np.array([[ord(c) / 255.0 for c in code_snippet]])
        try:
            probs = model.predict_proba(X)[0][1]
        except AttributeError:
            probs = model.predict(X)[0]
        # Repeat same probability per character
        return [float(probs)] * len(code_snippet)

    # --- PyTorch models (LSTM/CNN/CNN-BiLSTM) ---
    elif any(k in model_type.lower() for k in ["lstm", "cnn"]):
        seq = [ord(c) if ord(c) < 256 else 0 for c in code_snippet]
        x = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x).squeeze().cpu()
            probs = torch.sigmoid(logits).numpy()

        # Ensure it’s per-character
        if np.isscalar(probs):
            probs = [float(probs)] * len(code_snippet)
        elif len(probs) != len(code_snippet):
            probs = np.resize(probs, len(code_snippet)).tolist()
        else:
            probs = probs.tolist()

        # Optional smoothing for stability
        window = 3
        smoothed = np.convolve(probs, np.ones(window) / window, mode="same")
        return smoothed.tolist()

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# -------------------------------------------------------
# 🖥️ UI
# -------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    selected_lang = st.selectbox("Select Programming Language", LANGS)
with col2:
    model_type = st.selectbox("Select Model", list(MODELS.keys()))

code_input = st.text_area(
    "Paste your code here:",
    height=250,
    placeholder="def example():\n    print('Hello World')"
)

if st.button("🔎 Segment Code"):
    threshold = st.slider("Segmentation threshold", 0.0, 1.0, 0.5, 0.05)

    with st.spinner(f"Loading {model_type} model for {selected_lang}..."):
        model, mtype = load_model(model_type, selected_lang)

    if model is None:
        st.error(f"❌ {model_type} model not found for {selected_lang}.")
    else:
        with st.spinner(f"Running {mtype} segmentation..."):
            try:
                probs = get_predictions(model, mtype, code_input, DEVICE)
                if not isinstance(probs, list):
                    probs = [float(probs)] * len(code_input)

                # Create visual segmentation (HTML)
                html_output = ""
                for c, p in zip(code_input, probs):
                    color = "rgba(255,255,0,0.3)" if p > threshold else "transparent"
                    html_output += f"<span style='background-color:{color}'>{c}</span>"

                st.success("✅ Segmentation complete!")
                st.markdown(
                    f"<pre style='background:#f8f9fa;padding:10px;border-radius:8px;'>{html_output}</pre>",
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"⚠️ Error while segmenting code: {e}")
# -------------------------------------------------------
# 🔍 Notes
# -------------------------------------------------------
st.markdown("---")
st.caption("🧠 Models: BoC (Logistic Regression), BiLSTM, CNN, CNN-BiLSTM")
st.caption("📊 Device used: **" + DEVICE + "**")
