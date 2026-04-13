from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path("model")
MAX_LENGTH = 128


if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Could not find model folder: {MODEL_DIR.resolve()}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


id2label = model.config.id2label
if not id2label:
    id2label = {i: f"LABEL_{i}" for i in range(model.config.num_labels)}
else:
    id2label = {int(k): v for k, v in id2label.items()}


def classify(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()
    predicted_id = int(torch.argmax(probabilities).item())

    return {
        "text": text,
        "prediction": id2label[predicted_id],
        "probabilities": {
            id2label[i]: round(float(probabilities[i].item()), 4)
            for i in range(len(id2label))
        },
    }


sample_messages = [
    "Eh bro, want to go makan later at the hawker centre?",
    "Congratulations! You have been selected for a $5000 reward. Click here to claim now.",
    "Officer here. Your bank account is under investigation and you must respond immediately.",
    "Hey darling, I miss you so much. Can you send me money for flight ticket to come see you?",
]


print(f"Loaded model from: {MODEL_DIR.resolve()}")
print(f"Device: {device}")
print(f"Labels: {id2label}")
print()

for message in sample_messages:
    result = classify(message)
    print(f"Text:       {result['text']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Probs:      {result['probabilities']}")
    print()
