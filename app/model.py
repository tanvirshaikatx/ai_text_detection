import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd

# Load the model
model_path = "model/model.pkl"
xgb_model = joblib.load(model_path)

# Load pre-trained BERT model for embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_text(text, max_length=100):
    """Tokenize and extract BERT embeddings for input text."""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(ids))
    
    input_tensor = torch.tensor([ids])
    
    with torch.no_grad():
        outputs = bert_model(input_tensor)
        embedding = outputs[0][:, 0, :].numpy().flatten()
    
    return pd.DataFrame([embedding])

def predict(text):
    """Predict if the text is AI-generated or human-written."""
    processed_text = preprocess_text(text)
    prediction = xgb_model.predict(processed_text)[0]
    return "AI-Generated" if prediction == 1 else "Human-Written"
