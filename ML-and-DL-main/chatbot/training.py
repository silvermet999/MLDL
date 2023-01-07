import random
import json
import pickle
import numpy as np


import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import sequential
from tensorflow.keras.layer import Dense, Activation, Droupout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ['?', ".", "!", ",", "comment", "quel", "quelle", "je", "veux", "savoir", "vous", "tu", "j'ai", "quels", "quelles", "son", "est", "vos", "votre", "tes", "toi", "ton", "ta", "questions", "question", "des", "de", "le", "la", "les", "sur", "à propos"]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        word.append(word_list)
        documents.append((word_list), intent["tag"])
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

print(documents)