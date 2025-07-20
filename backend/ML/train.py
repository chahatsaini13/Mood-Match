!pip install -q pandas==2.2.2 scikit-learn==1.6.1 fsspec==2025.3.2 transformers datasets accelerate --upgrade --no-deps
import json
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from google.colab import files
uploaded = files.upload()

with open('prompts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

records = [
    {"prompt": p.strip(), "emotion": e.strip()}
    for e, prompts in data.items()
    for p in prompts
]
df = pd.DataFrame(records).dropna(subset=['prompt', 'emotion'])

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"[‘’]", "'", text)
    return text.strip()

df['prompt'] = df['prompt'].apply(clean_text)

min_n = df['emotion'].value_counts().min()
balanced_df = pd.concat([
    df[df['emotion'] == cls].sample(n=min_n, random_state=42)
    for cls in df['emotion'].unique()
], ignore_index=True)

le = LabelEncoder()
balanced_df['label'] = le.fit_transform(balanced_df['emotion'])
label2id = {str(e): int(i) for e, i in zip(le.classes_, le.transform(le.classes_))}
id2label = {int(i): str(e) for i, e in zip(le.transform(le.classes_), le.classes_)}

train_df, test_df = train_test_split(
    balanced_df[['prompt', 'label']],
    stratify=balanced_df['label'],
    test_size=0.2,
    random_state=42,
)

hf_datasets = {
    'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
    'test':  Dataset.from_pandas(test_df.reset_index(drop=True))
}


for split in hf_datasets:
    hf_datasets[split] = hf_datasets[split].rename_column('label', 'labels')

model_ckpt = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_fn(example):
    tokens = tokenizer(
        example['prompt'],
        truncation=True,
        padding='max_length',
        max_length=128
    )
    tokens['labels'] = example['labels']
    return tokens

tokenized_ds = {
    split: hf_datasets[split].map(tokenize_fn, batched=True)
    for split in hf_datasets
}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall":    recall_score(labels, preds, average="weighted"),
    }


model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=len(le.classes_),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=int(len(tokenized_ds['train']) * 0.1),
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['test'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model('./best_emotion_model')

tokenizer.save_pretrained('./best_emotion_model')

with open('./best_emotion_model/label2id.json', 'w') as f:
    json.dump(label2id, f)
with open('./best_emotion_model/id2label.json', 'w') as f:
    json.dump(id2label, f)

from sklearn.metrics import confusion_matrix, classification_report

preds_output = trainer.predict(tokenized_ds['test'])
y_true = tokenized_ds['test']['labels']
y_pred = preds_output.predictions.argmax(axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_emotion(text: str) -> str:
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(dim=-1).item()
    return id2label[pred_id]


if __name__ == "__main__":
    while True:
        txt = input("Enter a prompt (or 'quit' to exit): ")
        if txt.lower() == 'quit':
            break
        print("Predicted emotion:", predict_emotion(txt))
