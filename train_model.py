# Import necessary libraries
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

import random

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize variables
words = []
classes = []
documents = []
ignore_words = ['?', '!']

#-------------------------------------------------------------
# Load intents file
data_file = open('data/intents.json').read()
intents = json.loads(data_file)

# Display the contents of the intents file
# print(json.dumps(intents, indent=4))


#-------------------------------------------------------------
nltk.download('punkt', force=True)
nltk.download('punkt_tab')

# Tokenize patterns, prepare documents, and collect classes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)  # Add tokenized words to the words list
        documents.append((w, intent['tag']))  # Add tokenized words with their tag
        if intent['tag'] not in classes:  # Add unique tags to the classes list
            classes.append(intent['tag'])

print("Tokenization complete!")
# print(f"Documents: {len(documents)}")
# print(f"Classes: {len(classes)}, {classes}")
# print(f"Words collected: {len(words)}")

#-------------------------------------------------------------
nltk.download('wordnet')

# Lemmatize and remove stop words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# Remove duplicates and sort words
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Print information about the corpus
# print(f"{len(documents)} documents")
# print(f"{len(classes)} classes: {classes}")
# print(f"{len(words)} unique lemmatized words: {words}")

#-------------------------------------------------------------
# Save the words and classes to pickle files
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

print("Words and classes saved!")

#-------------------------------------------------------------
# Create training data
training = []
# Create an empty array for the output
output_empty = [0] * len(classes)

# Loop through each document to create the training data
for doc in documents:
    # Initialize the bag of words
    bag = []
    # Tokenized words from the pattern
    pattern_words = doc[0]
    # Lemmatize each word in the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create the bag of words array (1 if the word is in the pattern)
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Output: '1' for the current tag, '0' for others
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    # Append the bag and output to the training data
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Split the data into features (X) and labels (Y)
train_x = np.array([item[0] for item in training])  # Features (bag of words)
train_y = np.array([item[1] for item in training])  # Labels (output row)

print("Training data created!")

#-------------------------------------------------------------
# Create the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
# Compile the model with updated SGD optimizer
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("Model created and compiled!")

#-------------------------------------------------------------
# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('model.h5')
print("Model created and saved!")

#-------------------------------------------------------------
# Evaluate the model accuracy on the training data
# loss, accuracy = model.evaluate(np.array(train_x), np.array(train_y), verbose=0)
# print(f"Training Accuracy: {accuracy * 100:.2f}%")
