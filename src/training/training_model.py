# Dual-Model Claim Detection: BanglaBERT + RoBERTa for Bangla and English News Articles and Facebook Posts

import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------
# Load Bangla and English datasets
# --------------------------------
# Bangla news dataset, columns: text, label
logger.info(" Loading Bangla fake news datasets...")
bangla_df = pd.read_csv('../../dataset/bengali_fake_news_dataset/cleanbn_fakenews.csv')
# English news dataset, columns: text, label
logger.info(" Loading English fake news datasets...")
english_df = pd.read_csv('../../dataset/english_news_articles_of_bd'
                         '/bangladeshi_all_english_newspapers_daily_news_combined_dataset.csv')
# columns: text, label

# Load Facebook posts datasets
logger.info(" Loading Facebook posts datasets...")
facebook_posts_bangla_df = pd.read_excel('../../dataset/Fact_Checked_Facebook_News_Corpus_from_Bangladesh'
                                         '/FactWatch_manually_labeled_data.xlsx')  # columns: text, label

# Debug: Print unique label values before mapping
print("Unique values in facebook_posts_bangla_df['label'] before mapping:", facebook_posts_bangla_df['label'].unique())

# If labels are already 0/1, no mapping needed
if set(facebook_posts_bangla_df['label'].unique()) <= {0, 1}:
    facebook_posts_bangla_df = facebook_posts_bangla_df[['text', 'label']]
else:
    rating_map = {
        'True': 1,
        'Partly True': 1,
        'False': 0,
        'Partly False': 0,
        'Altered': 0
    }
    facebook_posts_bangla_df['label'] = facebook_posts_bangla_df['label'].map(rating_map)
    facebook_posts_bangla_df = facebook_posts_bangla_df[['text', 'label']]
    facebook_posts_bangla_df = facebook_posts_bangla_df.dropna(subset=['label'])

# Check if DataFrame is empty after filteringc
if facebook_posts_bangla_df.empty:
    raise ValueError(
        "facebook_posts_bangla_df is empty after mapping and filtering. "
        "Check the unique values in the 'label' column and update the rating_map accordingly."
    )

# Ensure both classes have at least 2 samples for stratification
label_counts = facebook_posts_bangla_df['label'].value_counts()
facebook_posts_bangla_df = facebook_posts_bangla_df[
    facebook_posts_bangla_df['label'].isin(label_counts[label_counts >= 2].index)
]


# facebook_posts_bangla_df = pd.read_csv('../../dataset/facebook_posts_bangla.csv')  # columns: text, label
# facebook_posts_english_df = pd.read_csv('../../dataset/facebook_posts_english.csv')  # columns: text, label


# ---------------------------
# Train/Validation/Test Split
# ---------------------------
def split_dataset(df):
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    return train_df, val_df, test_df


bangla_train, bangla_val, bangla_test = split_dataset(bangla_df)
english_train, english_val, english_test = split_dataset(english_df)

# Split Facebook posts datasets
facebook_bangla_train, facebook_bangla_val, facebook_bangla_test = split_dataset(facebook_posts_bangla_df)
# facebook_english_train, facebook_english_val, facebook_english_test = split_dataset(facebook_posts_english_df)

# --------------
# Dataset Class
# --------------
MAX_LEN = 128


class ClaimDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    predicts = pred.predictions.argmax(-1)
    f1 = f1_score(labels, predicts)
    precision = precision_score(labels, predicts)
    recall = recall_score(labels, predicts)
    return {'f1': f1, 'precision': precision, 'recall': recall}


# -----------------------------
# Function to train the training
# -----------------------------
def train_model(train_df, val_df, model_name, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = ClaimDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, MAX_LEN)
    val_dataset = ClaimDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, MAX_LEN)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer


# -----------------------------
# Train BanglaBERT (Bangla dataset)
# -----------------------------
bangla_trainer = train_model(bangla_train, bangla_val, 'sagorsarker/bangla-bert-base', './bangla_model')

# -----------------------------
# Train RoBERTa (English dataset)
# -----------------------------
english_trainer = train_model(english_train, english_val, 'roberta-base', './english_model')

# -----------------------------
# Train BanglaBERT (Facebook Bangla posts)
# -----------------------------
facebook_bangla_trainer = train_model(
    facebook_bangla_train, facebook_bangla_val, 'sagorsarker/bangla-bert-base', './facebook_bangla_model'
)

# -----------------------------
# Train RoBERTa (Facebook English posts)
# -----------------------------
# facebook_english_trainer = train_model(
#     facebook_english_train, facebook_english_val, 'roberta-base', './facebook_english_model'
# )

# -----------------------------
# Evaluate Models on Test Sets
# -----------------------------
bangla_test_dataset = ClaimDataset(bangla_test['text'].tolist(), bangla_test['label'].tolist(),
                                   AutoTokenizer.from_pretrained('sagorsarker/bangla-bert-base'), MAX_LEN)
english_test_dataset = ClaimDataset(english_test['text'].tolist(), english_test['label'].tolist(),
                                    AutoTokenizer.from_pretrained('roberta-base'), MAX_LEN)

facebook_bangla_test_dataset = ClaimDataset(
    facebook_bangla_test['text'].tolist(), facebook_bangla_test['label'].tolist(),
    AutoTokenizer.from_pretrained('sagorsarker/bangla-bert-base'), MAX_LEN
)
# facebook_english_test_dataset = ClaimDataset(
#     facebook_english_test['text'].tolist(), facebook_english_test['label'].tolist(),
#     AutoTokenizer.from_pretrained('roberta-base'), MAX_LEN
# )

bangla_results = bangla_trainer.predict(bangla_test_dataset)
english_results = english_trainer.predict(english_test_dataset)
facebook_bangla_results = facebook_bangla_trainer.predict(facebook_bangla_test_dataset)
# # facebook_english_results = facebook_english_trainer.predict(facebook_english_test_dataset)
#
logger.info(" Evaluation Test Metrics Results: ")
logger.info(" Bangla Test Metrics......")
print("Bangla Test Metrics:", bangla_results.metrics)
logger.info(" English Test Metrics......")
print("English Test Metrics:", english_results.metrics)
logger.info(" Facebook Bangla Posts Test Metrics......")
print("Facebook Bangla Posts Test Metrics:", facebook_bangla_results.metrics)
# print("Facebook English Test Metrics:", facebook_english_results.metrics)

# -----------------------------
# Evaluate using trainer.evaluate()
# -----------------------------
bangla_eval = bangla_trainer.evaluate(bangla_test_dataset)
english_eval = english_trainer.evaluate(english_test_dataset)
facebook_bangla_eval = facebook_bangla_trainer.evaluate(facebook_bangla_test_dataset)
# facebook_english_eval = facebook_english_trainer.evaluate(facebook_english_test_dataset)

logger.info(" Evaluation using trainer.evaluate(): ")
logger.info(" Bangla Test Evaluation......")
print("Bangla Test Evaluation (trainer.evaluate()):", bangla_eval)
logger.info(" English Test Evaluation......")
print("English Test Evaluation (trainer.evaluate()):", english_eval)
logger.info(" Facebook Bangla Posts Test Evaluation......")
print("Facebook Bangla Posts Test Evaluation (trainer.evaluate()):", facebook_bangla_eval)
# print("Facebook English Test Evaluation (trainer.evaluate()):", facebook_english_eval)
