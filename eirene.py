import json
import pickle
import random

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


ERROR_THRESHOLD = 0.25

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("eirene.h5")

word_index = {w: i for i, w in enumerate(words)}


def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


def bag_of_words(sentence):
    sentence_words = clean_up_sentences(sentence)
    bagOfWords = [0] * len(words)

    for w in sentence_words:
        if w in word_index:
            bagOfWords[word_index[w]] = 1

    return np.array(bagOfWords)


def predict_class(sentence):
    bagOfWords = bag_of_words(sentence)
    result = model.predict(np.array([bagOfWords]), verbose=0)[0]

    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [
        {"intent": classes[r[0]], "probability": str(r[1])}
        for r in results
    ]


def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understood that. Can you rephrase?"

    tag = intents_list[0]["intent"]

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            if "responses" in intent:
                return random.choice(intent["responses"])
            if "response" in intent:
                return random.choice(intent["response"])

    return "I'm not sure I understood that. Can you rephrase?"


print("Eirene online.")

while True:
    message = input("").strip()
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    print(response)
