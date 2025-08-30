
import os
import torch
import pickle
import speech_recognition as sr
import pyttsx3
from model import CommandClassifier

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


input_size = len(vectorizer.get_feature_names_out())
num_classes = len(label_encoder.classes_)
model = CommandClassifier(input_size=input_size, num_classes=num_classes)
model.load_state_dict(torch.load("command_model.pth"))
model.eval()

recognizer = sr.Recognizer()
mic = sr.Microphone()

engine = pyttsx3.init()
def speak(text):
    print(f"🗣️ Jarvis: {text}")
    engine.say(text)
    engine.runAndWait()

def predict_command(command):
    vec = vectorizer.transform([command]).toarray()
    tensor = torch.tensor(vec, dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        label = label_encoder.inverse_transform(predicted.numpy())[0]
        return label

def perform_action(command, category):
    command = command.lower()

    if category == "open_app":
        if "chrome" in command:
            
            os.system("start chrome")
        elif "spotify" in command:
            os.system("start chrome https://www.spotify.com")
        elif "youtube" in command:
            os.system("start chrome https://www.youtube.com")
        elif "calculator" in command:
            os.system("start calc")
        elif "notepad" in command:
            os.system("start notepad")
        elif "vs code" in command:
            os.system("code")
        else:
            speak("I don't know that app, sorry!")

    elif category == "play_music":
        speak("Playing music...")
        os.system("start wmplayer")  # Windows Media Player

    elif category == "get_time":
        import datetime
        now = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The time is {now}")

    elif category == "search_web":
        import webbrowser
        query = command.replace("search for", "").replace("look up", "").strip()
        webbrowser.open(f"https://www.google.com/search?q={query}")
        speak(f"Searching the web for {query}")

    elif category == "tell_joke":
        speak("Why did the computer go to therapy? Because it had too many bytes of emotional baggage.")

    elif category == "set_alarm":
        speak("Alarms are not yet supported on this system. I'm still learning that trick!")

    elif category == "shutdown":
        speak("Shutting down system. Goodbye!")
        os.system("shutdown /s /t 1")

    elif category == "weather":
        speak("You’ll need an API for that! Or just look outside. ☔")

    elif category == "reminder":
        speak("I’ll remind you — just kidding. Reminders will be added soon.")

    elif category == "send_message":
        speak("Texting feature coming soon!")

    else:
        speak("I couldn't understand what to do.")

speak("Hello, I'm Jarvis. How can I help you? Say 'exit' to quit.")

while True:
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("🎤 Listening...")
            audio = recognizer.listen(source, timeout=5)

        text = recognizer.recognize_google(audio)
        print(f"🧠 You said: {text}")

       

        if text.lower() in ["exit", "quit", "stop"]:
            speak("Goodbye Raghav.")
            break

        category = predict_command(text)
        speak(f"Understood: {category.replace('_', ' ')}")
        perform_action(text, category)

    except sr.UnknownValueError:
        speak("Sorry, I didn’t catch that. Please try again.")
    except sr.WaitTimeoutError:
        speak("No voice detected. Try again.")
    except Exception as e:
        print("❌ Error:", str(e))
        speak("Something went wrong.")
