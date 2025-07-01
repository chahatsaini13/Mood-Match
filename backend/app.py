from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import torch
from transformers import AutoTokenizer

# === Flask setup ===
app = Flask(__name__)
CORS(app)

# === Load design dataset ===
with open("emotion_design_dataset.json", "r", encoding="utf-8") as f:
    design_data = json.load(f)

# === Emotion mapping ===
    emotion_mapping = {
    "joy": "JOY/HAPPINESS",
    "happiness": "JOY/HAPPINESS",
    "amusement": "JOY/HAPPINESS",
    "excitement": "JOY/HAPPINESS",
    "love": "LOVE/BLUSH",
    "gratitude": "LOVE/BLUSH",
    "approval": "LOVE/BLUSH",
    "surprise": "SURPRISE/WONDER",
    "wonder": "SURPRISE/WONDER",
    "neutral": "NEUTRAL/CLARITY",
    "peace": "PEACE/SERENITY",
    "pride": "PEACE/SERENITY",
    "trust": "TRUST/SECURITY",
    "optimism": "TRUST/SECURITY",
    "fear": "FEAR/ANXIETY",
    "nervousness": "FEAR/ANXIETY",
    "anger": "POWER/ANGER",
    "annoyance": "POWER/ANGER",
    "disapproval": "POWER/ANGER",
    "sorrow": "SORROW/DUSK",
    "sadness": "SORROW/DUSK",
    "disappointment": "SORROW/DUSK",
    "grief": "SORROW/DUSK",
    "remorse": "SORROW/DUSK",
    "confusion": "SORROW/DUSK",
    "curiosity": "SURPRISE/WONDER",
    "realization": "SURPRISE/WONDER",
    "relief": "PEACE/SERENITY",
    "embarrassment": "SORROW/DUSK",
    "disgust": "SORROW/DUSK",
    "caring": "LOVE/BLUSH",
    "desire": "LOVE/BLUSH"
}



# === Load ML model & tokenizer ===
from ML.model import CustomRobertaModel  # ✅ Correct: keep model.py inside ML/

model_checkpoint = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

num_labels = 28  # ✅ Update if needed
pos_weights = torch.ones(num_labels)

# ✅ Correct path if weights are in ML folder:
model = CustomRobertaModel(model_checkpoint, num_labels, pos_weights)
model.load_state_dict(torch.load("ML/pytorch_model.bin", map_location=torch.device("cpu")))
model.eval()

label_names = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral"
]

# === Prediction logic ===
def get_emotion_from_model(text, threshold=0.5):
    device = torch.device("cpu")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)

    probs = probs.cpu().numpy()[0]
    predicted_labels = [label_names[i] for i, prob in enumerate(probs) if prob >= threshold]

    if not predicted_labels:
        return "neutral"

    return predicted_labels[0]

# === API route ===
@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    data = request.get_json()
    print("✅ Received data:", data)

    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    user_text = data.get("text", "")
    print("📩 User text:", user_text)

    predicted_emotion = get_emotion_from_model(user_text)
    print("🔮 Predicted emotion:", predicted_emotion)

    mapped_emotion = emotion_mapping.get(predicted_emotion, predicted_emotion)
    design = design_data.get(mapped_emotion, {})
    print("🎨 Design data:", design)

    return jsonify({
        "emotion": predicted_emotion,
        "design": design
    })

if __name__ == '__main__':
    app.run(debug=True)
