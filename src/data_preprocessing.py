import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df.fillna('', inplace=True)
    df['combined_text'] = df['title'] + ' ' + df['description'] + ' ' + df['url_content']
    return df

def split_data(df, label_column, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        df['combined_text'], df[label_column], test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def load_parsbert_model():
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    model = AutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    return tokenizer, model

def extract_features_with_parsbert(X, tokenizer, model):
    features = []
    for text in X:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        mean_last_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        features.append(mean_last_hidden_state)
    return np.array(features)

def main(filepath, label_column):
    df = load_data(filepath)
    df_preprocessed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_preprocessed, label_column)
    tokenizer, model = load_parsbert_model()
    X_train_features = extract_features_with_parsbert(X_train, tokenizer, model)
    X_test_features = extract_features_with_parsbert(X_test, tokenizer, model)
    print(f"X_train_features shape: {X_train_features.shape}")
    print(f"X_test_features shape: {X_test_features.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

if __name__ == "__main__":
    filepath = 'path/to/your/csv/file.csv'  # Change this to your actual file path
    label_column = 'your_label_column_name'  # Change this to your actual label column name
    main(filepath, label_column)
