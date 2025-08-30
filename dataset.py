# dataset.py

import random
import pandas as pd

def generate_dataset(save_path="commands_dataset.csv"):
    categories = {
        "open_app": ["Open {}", "Launch {}", "Start the {} app", "Run {}", "Fire up {}", "{} Kholiyo", "{} Kholde"],
        "play_music": ["Play music", "Start some songs", "Put on some tunes", "I want to hear music", "Begin my playlist"],
        "get_time": ["What time is it", "Tell me the current time", "Give me the time", "Show time", "Current time please"],
        "search_web": ["Search for {}", "Look up {}", "Google {}", "Find {}", "Browse {}", "{} ke baare mein kya pta hai", "What is {}?", "{} kya hota hai?", "{} dhundiyo web pr"],
        "tell_joke": ["Tell me a joke", "Make me laugh", "Say something funny", "Give me a joke", "I want to hear a joke"],
        "set_alarm": ["Set alarm for {}", "Wake me up at {}", "Alarm for {}", "Schedule an alarm at {}", "Make an alarm at {}"],
        "shutdown": ["Shutdown the system", "Turn off the laptop","Bund karde Laptop", "So jao Jarvis" "Power off", "Shut this thing down", "Kill the power"],
        "reminder": ["Remind me to {}", "Set a reminder to {}", "Don't let me forget to {}", "I need to remember to {}", "Create a reminder for {}"],
        "send_message": ["Send message to {}", "Text {}", "WhatsApp {}", "Shoot a message to {}", "Ping {} with a message"]
    }

    apps = ["Chrome", "Spotify", "YouTube", "Calculator", "VS Code", "Notepad"]
    search_queries = ["latest movies", "machine learning", "AI trends", "Python tutorials", "news headlines", "Python"]
    alarm_times = ["5 AM", "6:30 AM", "8 PM", "10 PM", "12:45"]
    reminder_tasks = ["take medicine", "attend meeting", "call mom", "submit assignment", "drink water"]
    contacts = ["Rahul", "Mom", "Boss", "Riya", "Amit", "Didi", "Keshav", "Nunu Bhaiya", "Papa", "Sahil"]

    data = []

    for category, templates in categories.items():
        for _ in range(100):
            template = random.choice(templates)
            if "{}" in template:
                if category == "open_app":
                    filled = template.format(random.choice(apps))
                elif category == "search_web":
                    filled = template.format(random.choice(search_queries))
                elif category == "set_alarm":
                    filled = template.format(random.choice(alarm_times))
                elif category == "reminder":
                    filled = template.format(random.choice(reminder_tasks))
                elif category == "send_message":
                    filled = template.format(random.choice(contacts))
                else:
                    filled = template.format("unknown")
            else:
                filled = template
            data.append([filled, category])

    df = pd.DataFrame(data, columns=["command", "label"])
    df.to_csv(save_path, index=False)
    print(f"✅ Dataset saved to {save_path} with {len(df)} samples.")

if __name__ == "__main__":
    generate_dataset()
