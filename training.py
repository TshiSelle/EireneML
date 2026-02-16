import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


INTENTS_PATH = "intents.json"
WORDS_PATH = "words.pkl"
CLASSES_PATH = "classes.pkl"
MODEL_PATH = "eirene.h5"

IGNORE_TOKENS = {"?", "!", ".", ","}


lemmatizer = WordNetLemmatizer()


def load_intents(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def normalize_tokens(tokens: list[str]) -> list[str]:
    return [lemmatizer.lemmatize(t.lower()) for t in tokens if t not in IGNORE_TOKENS]


def build_dataset(intents_json: dict) -> tuple[list[str], list[str], list[tuple[list[str], str]]]:
    words: list[str] = []
    classes: list[str] = []
    documents: list[tuple[list[str], str]] = []

    for intent in intents_json.get("intents", []):
        tag = intent.get("tag")
        patterns = intent.get("patterns", [])

        if not tag or not patterns:
            continue

        if tag not in classes:
            classes.append(tag)

        for p in patterns:
            tokens = nltk.word_tokenize(p)
            tokens = normalize_tokens(tokens)
            if not tokens:
                continue
            words.extend(tokens)
            documents.append((tokens, tag))

    words = sorted(set(words))
    classes = sorted(set(classes))
    return words, classes, documents


def vectorize(
    documents: list[tuple[list[str], str]],
    words: list[str],
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    word_index = {w: i for i, w in enumerate(words)}
    class_index = {c: i for i, c in enumerate(classes)}

    x = np.zeros((len(documents), len(words)), dtype=np.float32)
    y = np.zeros((len(documents), len(classes)), dtype=np.float32)

    for row, (tokens, tag) in enumerate(documents):
        for t in set(tokens):
            idx = word_index.get(t)
            if idx is not None:
                x[row, idx] = 1.0

        y[row, class_index[tag]] = 1.0

    return x, y


def build_model(input_dim: int, output_dim: int) -> Sequential:
    model = Sequential()
    model.add(Dense(128, input_shape=(input_dim,), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation="softmax"))

    try:
        opt = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    except TypeError:
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )
    return model


intents = load_intents(INTENTS_PATH)

words, classes, documents = build_dataset(intents)
dump_pickle(words, WORDS_PATH)
dump_pickle(classes, CLASSES_PATH)

x, y = vectorize(documents, words, classes)

idx = list(range(len(x)))
random.shuffle(idx)
x = x[idx]
y = y[idx]

model = build_model(input_dim=x.shape[1], output_dim=y.shape[1])
model.fit(x, y, epochs=200, batch_size=5, verbose=1)
model.save(MODEL_PATH)

print("Done")
