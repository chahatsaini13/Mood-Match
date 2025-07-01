import torch
import torch.nn as nn
from transformers import AutoModel

class CustomRobertaModel(nn.Module):
    def __init__(self, checkpoint, num_labels, pos_weights):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights, dtype=torch.float))

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
