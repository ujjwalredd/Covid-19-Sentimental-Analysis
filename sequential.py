import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, ELU, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def sequential(df):
    dff = df.drop(['-----------------------'],axis=1) # drop unnessary column 
    train_df, test_df = train_test_split(dff, test_size=0.2)
    tokenizer = Tokenizer(num_words=5000) #tokenzing the text
    tokenizer.fit_on_texts(train_df['-------------']) # Headline column
    X_train = tokenizer.texts_to_sequences(train_df['-----------']) # Headline column
    X_test = tokenizer.texts_to_sequences(test_df['---------------'])      # Headline column
    max_len = 100
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    y_train = pd.get_dummies(train_df['-----------']).values # Sentiment column
    y_test = pd.get_dummies(test_df['---------']).values # Sentiment column
    model = Sequential()        # model
    model.add(Embedding(5000, 32, input_length=max_len))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(ELU(alpha=1.0))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(ELU(alpha=1.0))
    model.add(Dense(2, activation='softmax'))


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # complie 
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64) # train
    
    # Predict labels for test data
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy