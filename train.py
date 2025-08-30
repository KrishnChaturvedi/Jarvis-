# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import CommandClassifier

df = pd.read_csv("commands_dataset.csv")
commands = df["command"].values
labels = df["label"].values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(commands).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

input_dim = X_train_tensor.shape[1]
num_classes = len(set(y_train))
model = CommandClassifier(input_size=input_dim, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = accuracy_score(y_train_tensor, torch.argmax(model(X_train_tensor), 1))
        test_acc = accuracy_score(y_test_tensor, torch.argmax(model(X_test_tensor), 1))

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f}")

torch.save(model.state_dict(), "command_model.pth")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model, vectorizer, and label encoder saved.")

