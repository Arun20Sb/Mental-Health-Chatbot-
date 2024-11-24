import nltk
nltk.data.path.append('./nltk_data')  # Add the local nltk_data path
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import streamlit as st

# Load intents and training data
intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('data/texts.pkl', 'rb'))
classes = pickle.load(open('data/labels.pkl', 'rb'))

# Preprocessing functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "Sorry, I didn't understand that."

# Main chatbot function
def chatbot_response(msg):
    res = getResponse(predict_class(msg, model), intents)
    return res

# Load the trained model
model = load_model('model/model.h5')

# Streamlit UI Setup
st.title("Mental Health Chatbot")
user_input = st.text_input("Ask me something about mental health:")

if user_input:
    st.write(f"User: {user_input}")
    chatbot_response_text = chatbot_response(user_input)
    st.write(f"Bot: {chatbot_response_text}")


