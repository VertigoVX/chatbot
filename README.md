# Chatbot Application

This is a simple chatbot application built using Python. The chatbot uses Natural Language Processing (NLP) techniques to preprocess user input and generate responses. It includes a graphical user interface (GUI) created with `tkinter`.

## Features

- Natural Language Processing using `nltk` and `scikit-learn`
- Intent matching with TF-IDF and cosine similarity
- Expanded intents and patterns for versatile responses
- Graphical User Interface (GUI) using `tkinter`

## Requirements

- Python 3.x
- `nltk`
- `numpy`
- `scikit-learn`
- `tkinter`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/VertigoVX/chatbot.git
    cd chatbot
    ```

2. Install the required libraries:
    ```bash
    pip install nltk numpy scikit-learn
    ```

3. Download the necessary NLTK data files:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    ```
