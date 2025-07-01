import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import f1_score
from model import CustomRobertaModel

# Load dataset & tokenizer
model_checkpoint = "roberta-large"
dataset = load_dataset("go_emotions", "simplified")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

label_names = dataset["train"].features["labels"].feature.names
num_labels = len(label_names)

# Tokenize & preprocess
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

def prepare_labels_binary(examples):
    batch_labels = []
    for label_list in examples["labels"]:
        one_hot = [0.0] * num_labels
        for idx in label_list:
            one_hot[idx] = 1.0
        batch_labels.append(one_hot)
    examples["labels"] = batch_labels
    return examples

dataset = dataset.map(tokenize_function, batched=True).map(prepare_labels_binary, batched=True)
dataset = dataset.remove_columns(["text", "id"])

def enforce_float_transform(example):
    example["labels"] = torch.tensor(example["labels"], dtype=torch.float32)
    return example

dataset = dataset.with_transform(enforce_float_transform)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Class weights
label_matrix = np.zeros((len(train_dataset), num_labels))
for i, labels in enumerate(dataset["train"]["labels"]):
    for label in labels:
        label_matrix[i][label] = 1
label_counts = label_matrix.sum(axis=0)
pos_weights = (len(train_dataset) - label_counts) / label_counts

# Model
model = CustomRobertaModel(model_checkpoint, num_labels, pos_weights)

# Data collator
base_collator = DataCollatorWithPadding(tokenizer=tokenizer)
def custom_collator(features):
    labels = [f["labels"] for f in features]
    for f in features:
        del f["labels"]
    batch = base_collator(features)
    batch["labels"] = torch.stack(labels)
    return batch

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    best_thresh, best_ema = 0.5, 0
    for t in np.arange(0.3, 0.6, 0.05):
        preds = (probs >= t).astype(int)
        ema = np.mean(np.all(preds == labels, axis=1))
        if ema > best_ema:
            best_ema, best_thresh = ema, t
    final_preds = (probs >= best_thresh).astype(int)
    macro_f1 = f1_score(labels, final_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, final_preds, average='micro', zero_division=0)
    exact_match = np.mean(np.all(final_preds == labels, axis=1))
    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "exact_match_accuracy": exact_match}

# Training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="exact_match_accuracy",
    greater_is_better=True,
    report_to="none",
    gradient_accumulation_steps=2,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=custom_collator,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate(test_dataset)
print(results)

# Save
model_save_dir = "./my_emotion_model"
torch.save(model.state_dict(), f"{model_save_dir}/pytorch_model.bin")
tokenizer.save_pretrained(model_save_dir)
