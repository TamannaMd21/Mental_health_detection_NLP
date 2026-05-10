
import gradio as gr
import numpy as np
import joblib
import json
import re
import string
import os
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ── Load artifacts ──────────────────────────────────────
ARTIFACTS = "mental_health_artifacts"

# Load tokenizer from JSON (how _8_ notebook saves it)
with open(f"{ARTIFACTS}/keras_tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

le     = joblib.load(f"{ARTIFACTS}/label_encoder.pkl")
config = joblib.load(f"{ARTIFACTS}/config.pkl")

max_len        = config["max_len"]
CONTRACTIONS   = config["contractions"]
NEGATION_WORDS = set(config["negation_words"])

lstm_model = load_model(f"{ARTIFACTS}/model.keras",     safe_mode=False)
gru_model  = load_model(f"{ARTIFACTS}/gru_model.keras", safe_mode=False)
print("✅ All artifacts loaded")

# ── Preprocessing (same as training) ────────────────────
stop_words = set(stopwords.words("english")) - NEGATION_WORDS
lemmatizer = WordNetLemmatizer()

def expand_contractions(text):
    for c, e in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(c) + r"\b", e, text)
    return text

def preprocess(text):
    text = str(text).lower()
    text = text.encode("ascii","ignore").decode()
    text = expand_contractions(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = text.translate(str.maketrans("","",string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w)>2 and w.isalpha()]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# ── Status metadata ──────────────────────────────────────
STATUS_META = {
    "Normal"              : ("🟢", "No mental health concern detected. You seem to be doing well!"),
    "Depression"          : ("🔵", "Signs of depression detected. Please consider speaking to a professional."),
    "Anxiety"             : ("🟡", "Signs of anxiety detected. Try breathing exercises and seek support."),
    "Suicidal"            : ("🔴", "Urgent: Please contact a mental health professional immediately."),
    "Stress"              : ("🟠", "High stress levels detected. Take breaks and practice self-care."),
    "Bipolar"             : ("🟣", "Bipolar patterns detected. Consistent professional support is recommended."),
    "Personality disorder": ("🔵", "Personality disorder signs detected. Professional therapy can help greatly."),
}

# ── Prediction function ──────────────────────────────────
def gradio_predict(text, model_choice):
    if not text or not text.strip():
        return "Please enter some text.", "", {}
    clean  = preprocess(text)
    seq    = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=max_len)
    model  = lstm_model if model_choice == "LSTM" else gru_model
    probs  = model.predict(padded, verbose=0)[0]
    idx    = int(np.argmax(probs))
    label  = le.classes_[idx]
    conf   = float(probs[idx])
    emoji, msg = STATUS_META.get(label, ("❓", ""))
    result     = f"{emoji}  {label}  ({conf:.2%} confidence)"
    prob_dict  = {le.classes_[i]: float(probs[i]) for i in range(len(probs))}
    return result, msg, prob_dict

# ── Gradio UI ────────────────────────────────────────────
examples = [
    ["I feel completely hopeless and do not want to get out of bed.",       "LSTM"],
    ["I am doing great today, feeling very happy and grateful!",             "GRU"],
    ["I have constant panic attacks and cannot stop worrying.",              "LSTM"],
    ["I do not see any point in continuing, everything feels meaningless.",  "LSTM"],
    ["My mood switches so fast, one moment energetic, next devastated.",     "GRU"],
    ["I feel like I do not know who I am, my identity keeps changing.",      "GRU"],
    ["Too many deadlines, I feel completely burned out at work.",            "LSTM"],
]

with gr.Blocks(title="Mental Health Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Mental Health Detection
    **Semantic Analysis using NLP — LSTM & GRU Models with Attention**
    Detects: `Normal` | `Depression` | `Anxiety` | `Suicidal` | `Stress` | `Bipolar` | `Personality Disorder`
    ---
    """)
    with gr.Row():
        with gr.Column(scale=2):
            text_input  = gr.Textbox(label="Enter your statement",
                                     placeholder="Type how you are feeling today...",
                                     lines=4)
            model_radio = gr.Radio(choices=["LSTM", "GRU"], value="LSTM", label="Choose Model")
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                clear_btn   = gr.Button("🗑️ Clear",   variant="secondary")
        with gr.Column(scale=2):
            result_out = gr.Textbox(label="Prediction",          interactive=False)
            advice_out = gr.Textbox(label="Advice",              interactive=False, lines=2)
            prob_chart = gr.Label( label="Confidence per Class", num_top_classes=7)

    gr.Examples(examples=examples, inputs=[text_input, model_radio], label="Try Example Inputs")
    gr.Markdown("---\n⚠️ *For educational purposes only. Always consult a mental health professional.*")

    analyze_btn.click(fn=gradio_predict,
                      inputs=[text_input, model_radio],
                      outputs=[result_out, advice_out, prob_chart])
    clear_btn.click(fn=lambda: ("","LSTM","","",{}),
                    inputs=[],
                    outputs=[text_input, model_radio, result_out, advice_out, prob_chart])

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true")
parser.add_argument("--port",  type=int, default=7860)
args, _ = parser.parse_known_args()

demo.launch(share=args.share, server_port=args.port, server_name="0.0.0.0", debug=False)
