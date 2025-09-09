# fake_news_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file


# --- Dual Encoder Model ---
class DualEncoderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_encoder = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")
        self.en_encoder = AutoModel.from_pretrained("roberta-base")
        hidden = self.bn_encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, labels=None, lang_id=None):
        outs = []
        for i in range(input_ids.size(0)):
            if lang_id[i] == 0:  # Bangla
                o = self.bn_encoder(input_ids=input_ids[i].unsqueeze(0),
                                    attention_mask=attention_mask[i].unsqueeze(0))
            else:  # English
                o = self.en_encoder(input_ids=input_ids[i].unsqueeze(0),
                                    attention_mask=attention_mask[i].unsqueeze(0))
            cls = o.last_hidden_state[:, 0, :]
            outs.append(cls)
        logits = self.classifier(torch.cat(outs, dim=0))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# --- Load model + tokenizers ---
def load_model(model_path: str):
    model = DualEncoderClassifier()
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    bn_tok = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
    en_tok = AutoTokenizer.from_pretrained("roberta-base")

    return model, bn_tok, en_tok


# --- Helper: Bangla text detector ---
def is_bangla(s: str) -> bool:
    return any('\u0980' <= ch <= '\u09FF' for ch in s)


# --- Prediction function ---
@torch.inference_mode()
def predict(text: str, model, bn_tok, en_tok, max_len: int = 128):
    lang_id = 0 if is_bangla(text) else 1
    tok = bn_tok if lang_id == 0 else en_tok

    enc = tok(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    out = model(input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                lang_id=torch.tensor([lang_id]))
    probs = torch.softmax(out["logits"], dim=-1).squeeze(0)
    pred = int(torch.argmax(probs).item())  # 0=fake, 1=real
    return pred, probs.tolist()


if __name__ == "__main__":
    model, bn_tok, en_tok = load_model("../../model/bangla_english_fake_news_classifier/model.safetensors")

    samples = [
        "ডেঙ্গু প্রতিরোধে নতুন প্রতিষেধক আসছে",  # Bangla
        "New vaccine developed to prevent dengue fever."  # English
    ]

    for s in samples:
        pred, probs = predict(s, model, bn_tok, en_tok)
        label = "Real" if pred == 1 else "Fake"
        print(f"Title: {s}\nPrediction: {label} (Probs: {probs})\n")
