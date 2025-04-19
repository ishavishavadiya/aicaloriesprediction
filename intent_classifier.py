from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import pandas as pd

# Sample training data
data = {
    "text": [
        "Suggest a high protein meal after workout",
        "I want a light dinner under 400 kcal",
        "What’s a good vegan breakfast?",
        "Give me something spicy and low calorie",
        "What can I eat after gym?",
        "Thanks!",
        "Hi there!",
        "Suggest gluten-free Indian food"
    ],
    "label": [
        "post_workout_food",
        "light_meal",
        "meal_suggestion",
        "light_meal",
        "post_workout_food",
        "thanks",
        "greeting",
        "diet_restricted"
    ]
}

df = pd.DataFrame(data)
labels = list(df["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df['label_id'] = df['label'].map(label2id)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label_id'], test_size=0.2)

train_dataset = IntentDataset(X_train.tolist(), y_train.tolist())
val_dataset = IntentDataset(X_val.tolist(), y_val.tolist())

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels)
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir='./logs',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=1,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
model.save_pretrained("intent_model")
tokenizer.save_pretrained("intent_model")
