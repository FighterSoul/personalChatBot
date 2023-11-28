import random 
import json
import pickle
import numpy as np

import nltk 
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load training data

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

word = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        word.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates

words = [lemmatizer.lemmatize(w.lower()) for w in word if w not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Now you can split your data
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3)

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.6))  # Increase dropout
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.6))  # Increase dropout
model.add(Dense(len(train_y[0]), activation='softmax'))

# Define your optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)

hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1, validation_data=(val_x, val_y), callbacks=[early_stopping, model_checkpoint])

# Load the best model
model = tf.keras.models.load_model('best_model.h5')

model.save('chatbot_model.h5')
print('Done')