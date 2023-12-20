import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the dataset
data = pd.read_csv('dataset/csv_files/movies_train.csv')

# Preprocess the text
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Hàm tiền xử lý văn bản
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokens = nltk.word_tokenize((str(text).lower()))
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if
                       token.lower() not in stop_words and token.isalnum()]
    return filtered_tokens


# Áp dụng hàm preprocess_text cho cột 'Title'
data['processed_Title'] = data['Title'].apply(preprocess_text)

# Preprocess Genres
data['Genre'] = data['Genre'].apply(lambda x: str(x).split(','))

# Encode Genres
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(data['Genre'])

for i, genre in enumerate(mlb.classes_):
    data[genre] = genre_encoded[:, i]

# Load the pre-trained GloVe embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Tokenize and pad the text sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['processed_Title'])
sequences = tokenizer.texts_to_sequences(data['processed_Title'])
word_index = tokenizer.word_index
padded_sequences = pad_sequences(sequences, maxlen=300)

# Create the embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Define the LSTM model
model = Sequential()
model.add(Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=300, trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(mlb.classes_), activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, genre_encoded, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='micro')
recall = recall_score(y_test, y_pred_classes, average='micro')
f1 = f1_score(y_test, y_pred_classes, average='micro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
