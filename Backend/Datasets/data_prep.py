from datasets import load_dataset
from transformers import BertTokenizer

# Load datasets
goemotions = load_dataset("go_emotions")
empathetic = load_dataset("empathetic_dialogues")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("Models/bert-base-uncased")

# Tokenization functions
def tokenize_goemotions(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

def tokenize_empathetic(example):
    return tokenizer(example["utterance"], padding="max_length", truncation=True)

# Apply tokenization
goemotions = goemotions.map(tokenize_goemotions, batched=True)
empathetic = empathetic.map(tokenize_empathetic, batched=True)
