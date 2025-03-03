import os
import wandb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = "0.0"
wandb.init(mode="disabled")

import torch
import numpy as np
import pickle

import evaluate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)



if torch.cuda.is_available():
    device_type = "cuda"
    device_index = 0
    print("CUDA GPU is available. Using GPU acceleration.")
elif torch.backends.mps.is_available():
    device_type = "mps"
    device_index = 0
    print("MPS is available. Using Apple Metal acceleration.")
else:
    device_type = "cpu"
    device_index = -1
    print("No MPS or CUDA found. Using CPU.")


imdb = load_dataset("stanfordnlp/imdb")
small_train_dataset = imdb["train"]
test_dataset = imdb["test"]


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


accuracy_metric = evaluate.load("accuracy", trust_remote_code=True)
f1_metric = evaluate.load("f1", trust_remote_code=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


use_cpu_flag = (device_type == "cpu")

training_args = TrainingArguments(
    output_dir = 'IMDB-sentimental-analysis',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=False,
    use_cpu=use_cpu_flag
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()


eval_value = trainer.evaluate()
print("Evaluation:", eval_value)


model_dict = {
    "config": model.config,
    "state_dict": model.state_dict()
}

os.makedirs("models/deep", exist_ok=True)
pickle_path = "models/deep/distilbert.pkl"

with open(pickle_path, "wb") as f:
    pickle.dump(model_dict, f)

print(f"Model saved to: {pickle_path}")
