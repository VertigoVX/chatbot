import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext, END

# Load data
with open('chatbot_data.json', 'r') as file:
    data = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess user input
def preprocess(sentence):
    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence)
    sentence = [lemmatizer.lemmatize(word) for word in sentence if word not in string.punctuation]
    return ' '.join(sentence)

# Prepare data for TF-IDF
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(preprocess(pattern))
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Generate Responses
def generate_response(user_input):
    user_input = preprocess(user_input)
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = np.argmax(similarity)
    if similarity[0][index] < 0.3:  # Threshold for similarity
        return "I'm sorry, I don't understand that."
    tag = tags[index]
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Chatbot GUI
def send():
    user_input = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", END)

    if user_input:
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "You: " + user_input + '\n\n')
        response = generate_response(user_input)
        chat_window.insert(tk.END, "Chatbot: " + response + '\n\n')
        chat_window.config(state=tk.DISABLED)
        chat_window.yview(tk.END)

# Create GUI window
base = tk.Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=tk.FALSE, height=tk.FALSE)

# Create chat window
chat_window = scrolledtext.ScrolledText(base, bd=1, bg="white", width=50, height=8, font="Arial")
chat_window.config(state=tk.DISABLED)

# Bind scrollbar to chat window
scrollbar = tk.Scrollbar(base, command=chat_window.yview)
chat_window['yscrollcommand'] = scrollbar.set

# Create button to send message
send_button = tk.Button(base, font=("Verdana", 12, 'bold'), text="Send", width=12, height=5, bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', command=send)

# Create entry box to type message
entry_box = tk.Text(base, bd=0, bg="white", width=29, height=5, font="Arial")

# Place components on the screen
chat_window.place(x=6, y=6, height=386, width=370)
entry_box.place(x=6, y=401, height=90, width=265)
send_button.place(x=280, y=401, height=90)
scrollbar.place(x=376, y=6, height=386)

base.mainloop()
