import torch
import pickle
from model import CommandClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load vectorizer & label encoder
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load model
input_size = len(vectorizer.get_feature_names_out())
num_classes = len(label_encoder.classes_)
model = CommandClassifier(input_size=input_size, num_classes=num_classes)
model.load_state_dict(torch.load("command_model.pth"))
model.eval()

# Predict
def predict_command(command):
    vec = vectorizer.transform([command]).toarray()
    tensor = torch.tensor(vec, dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        label = label_encoder.inverse_transform(predicted.numpy())[0]
        return label

print("🎙️ Jarvis is ready. Type a command or 'exit' to quit.")
while True:
    text = input("🗣️ You: ")
    if text.lower() == "exit":
        break
    result = predict_command(text)
    print(f"🤖 Jarvis: Detected Intent → {result}")
