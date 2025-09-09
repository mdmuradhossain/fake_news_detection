import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModel,
    Trainer, TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------
# Load Datasets
# --------------------------
logger.info(" Loading datasets...")

try:
    if not os.path.exists('../../dataset/bengali_fake_news_dataset/cleanbn_fakenews.csv'):
        raise FileNotFoundError('Bangla dataset not found')
    bangla_df = pd.read_csv('../../dataset/bengali_fake_news_dataset/cleanbn_fakenews.csv')  # text, label

    if not os.path.exists(
            '../../dataset/english_news_articles_of_bd/bangladeshi_all_english_newspapers_daily_news_combined_dataset'
            '.csv'):
        raise FileNotFoundError('English dataset not found')
    english_df = pd.read_csv(
        '../../dataset/english_news_articles_of_bd/bangladeshi_all_english_newspapers_daily_news_combined_dataset.csv')  # text, label

    if not os.path.exists(
            '../../dataset/Fact_Checked_Facebook_News_Corpus_from_Bangladesh/FactWatch_manually_labeled_data.xlsx'):
        raise FileNotFoundError('Facebook dataset not found')
    facebook_bangla_df = pd.read_excel(
        '../../dataset/Fact_Checked_Facebook_News_Corpus_from_Bangladesh/FactWatch_manually_labeled_data.xlsx')
except FileNotFoundError as e:
    logger.error(f"Dataset file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading datasets: {e}")
    raise

# Map labels if needed
rating_map = {'True': 1, 'Partly True': 1, 'False': 0, 'Partly False': 0, 'Altered': 0}
if facebook_bangla_df['label'].dtype == object:
    facebook_bangla_df['label'] = facebook_bangla_df['label'].map(rating_map)
facebook_bangla_df = facebook_bangla_df.dropna(subset=['label'])
facebook_bangla_df = facebook_bangla_df[['text', 'label']]

# --------------------------
# Add language column
# --------------------------
bangla_df['lang'] = 'bn'
english_df['lang'] = 'en'
facebook_bangla_df['lang'] = 'bn'
# Will add Facebook English later: facebook_english_df['lang'] = 'en'

# Combining all datasets
all_data = pd.concat([bangla_df, english_df, facebook_bangla_df], ignore_index=True)

# --------------------------
# Train/Val/Test Split
# --------------------------
train_df, temp_df = train_test_split(all_data, test_size=0.2, stratify=all_data['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# --------------------------
# Tokenizers
# --------------------------
try:
    bangla_tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
    english_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
except Exception as e:
    logger.error(f"Error loading tokenizers: {e}")
    raise
MAX_LEN = 128


class ClaimDataset(Dataset):
    def __init__(self, df, bangla_tokenizer, english_tokenizer, max_len=128):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.langs = df['lang'].tolist()
        self.bn_tokenizer = bangla_tokenizer
        self.en_tokenizer = english_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        lang = self.langs[idx]

        if lang == "bn":
            encoding = self.bn_tokenizer(
                text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
            )
            lang_id = 0
        else:
            encoding = self.en_tokenizer(
                text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
            )
            lang_id = 1

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_id": torch.tensor(lang_id, dtype=torch.long),
        }


# --------------------------
# Dual Encoder Model
# --------------------------
class DualEncoderClassifier(nn.Module):
    def __init__(self):
        super(DualEncoderClassifier, self).__init__()
        try:
            self.bn_encoder = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")
            self.en_encoder = AutoModel.from_pretrained("roberta-base")
        except Exception as e:
            logger.error(f"Error loading pretrained models: {e}")
            raise
        hidden_size = self.bn_encoder.config.hidden_size  # 768
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None, lang_id=None):
        # Separate indices for each language
        bn_mask = (lang_id == 0)
        en_mask = (lang_id == 1)

        outputs = torch.zeros(input_ids.size(0), self.bn_encoder.config.hidden_size, device=input_ids.device)

        # Process Bangla texts in batch
        if bn_mask.any():
            bn_indices = bn_mask.nonzero(as_tuple=True)[0]
            bn_input_ids = input_ids[bn_indices]
            bn_attention_mask = attention_mask[bn_indices]
            bn_out = self.bn_encoder(bn_input_ids, attention_mask=bn_attention_mask)
            outputs[bn_indices] = bn_out.last_hidden_state[:, 0, :]  # CLS token

        # Process English texts in batch
        if en_mask.any():
            en_indices = en_mask.nonzero(as_tuple=True)[0]
            en_input_ids = input_ids[en_indices]
            en_attention_mask = attention_mask[en_indices]
            en_out = self.en_encoder(en_input_ids, attention_mask=en_attention_mask)
            outputs[en_indices] = en_out.last_hidden_state[:, 0, :]  # CLS token

        logits = self.classifier(outputs)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# --------------------------
# Metrics
# --------------------------
def compute_metrics(predicts):
    labels = predicts.label_ids
    predicts = predicts.predictions.argmax(-1)
    return {
        "f1": f1_score(labels, predicts),
        "precision": precision_score(labels, predicts),
        "recall": recall_score(labels, predicts),
    }


# --------------------------
# Prepare datasets
# --------------------------
train_dataset = ClaimDataset(train_df, bangla_tokenizer, english_tokenizer, MAX_LEN)
val_dataset = ClaimDataset(val_df, bangla_tokenizer, english_tokenizer, MAX_LEN)
test_dataset = ClaimDataset(test_df, bangla_tokenizer, english_tokenizer, MAX_LEN)

# --------------------------
# Training
# --------------------------
training_args = TrainingArguments(
    output_dir="../../model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

logger.info(" Starting training training...")
model = DualEncoderClassifier()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
logger.info(" Model training completed.")
logger.info(" Evaluating on test set...")
results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Results: ", results)

# Save evaluation results
with open('../../model/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save training with custom name
model_save_path = "../../model/bangla_english_fake_news_classifier"
trainer.save_model(model_save_path)

torch.save(model.state_dict(), "bangla_english_fake_news_model.pth")

logger.info(f"Model saved to: {model_save_path}")
logger.info("Model training and evaluation completed.")
