# /backend/ML/inference.py

import torch
from transformers import AutoTokenizer
from .model import CustomRobertaModel

# --------------------------
# ✅ CONFIG:
model_checkpoint = "roberta-large"
num_labels = 28  # Total emotion labels
pos_weights = [1.0] * num_labels  # Same as your training

# --------------------------
# ✅ Load tokenizer & model:
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = CustomRobertaModel(model_checkpoint, num_labels, pos_weights)
model.load_state_dict(torch.load("backend/ML/pytorch_model.bin", map_location=torch.device("cpu")))
model.eval()

# --------------------------
# ✅ Predict function:
def predict_emotion(text, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    predicted_labels = []
    for i, prob in enumerate(probs):
        if prob >= threshold:
            predicted_labels.append(i)  # Or store label names if you want

    return predicted_labels
