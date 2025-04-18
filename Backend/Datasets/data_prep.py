
#Tokenization and data preparation for the GoEmotions and Empathetic Dialogues datasets
from datasets import load_dataset
from transformers import BertTokenizer

# Load datasets
raw_goemotions = load_dataset("go_emotions")
raw_empathetic = load_dataset("empathetic_dialogues")


# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("Models/bert-base-uncased")

# Tokenization functions
def tokenize_goemotions(example):
    tokens = tokenizer(example["text"], padding="max_length", truncation=True)
    tokens["labels"] = example["labels"]  
    return tokens


def tokenize_empathetic(batch):
    tokens = tokenizer(batch["utterance"], padding="max_length", truncation=True)
    tokens["tags"] = batch["tags"]  
    return tokens


# Tokenize each split
goemotions = {}
for split in raw_goemotions:
    goemotions[split] = raw_goemotions[split].map(tokenize_goemotions, batched=True)

empathetic = {}
for split in raw_empathetic:
    empathetic[split] = raw_empathetic[split].map(tokenize_empathetic, batched=True)

# Label functions
def add_goemotions_labels(example):
    example["labels"] = example["labels"][0] if isinstance(example["labels"], list) else example["labels"]
    return example

def add_empathetic_labels(example):
    example["labels"] = int(example["tags"])
    return example

# Apply labels to each split
for split in goemotions:
    goemotions[split] = goemotions[split].map(add_goemotions_labels)

for split in empathetic:
    empathetic[split] = empathetic[split].map(add_empathetic_labels)
