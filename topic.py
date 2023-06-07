import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim import models, corpora
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')


def topic(df):
    
    def preprocess_text(text):
        tokens = word_tokenize(text.lower()) #Tokenize
        
        # Remove punctuation and stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        
        return tokens
    
    def perform_sentiment_analysis(text):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = analyzer.polarity_scores(text)
        
        # Classify sentiment based on compound score
        if sentiment_score['compound'] >= 0.05:
            return 'positive'
        elif sentiment_score['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'
        
    def perform_topic_modeling(documents, num_topics):
        # Create a dictionary and corpus
        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        
        # Train LDA model
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
        
        return lda_model

    def generate_word_clouds(lda_model):
        topics = lda_model.show_topics(num_topics=-1, formatted=False)
        for topic_id, words in topics:
            word_cloud = WordCloud(background_color='white').generate_from_frequencies(dict(words))
            
            # Plot word cloud
            print("\nWordCloud of Topic MOdelling:")
            plt.figure()
            plt.imshow(word_cloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Topic ' + str(topic_id))
            plt.show()
            
    dataset = df['-----------'] #Headline column change
    
    preprocessed_docs = [preprocess_text(doc) for doc in dataset]

    # Perform sentiment analysis
    sentiments = [perform_sentiment_analysis(doc) for doc in dataset]

    # Perform topic modeling
    num_topics = 10
    lda_model = perform_topic_modeling(preprocessed_docs, num_topics)

    # Generate word clouds
    generate_word_clouds(lda_model)
    
    # Preprocess dataset
    def preprocess_text(text):
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        
        return ' '.join(tokens)
    
    def train_sentiment_classifier(texts, labels):
        # Preprocess texts
        preprocessed_texts = [preprocess_text(text) for text in texts]
        
        # Vectorize texts
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(preprocessed_texts)
        
        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        
        # Train logistic regression classifier
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        
        return classifier, vectorizer, X_test, y_test
    
    def perform_sentiment_analysis(texts, classifier, vectorizer):
        preprocessed_texts = [preprocess_text(text) for text in texts]
        X = vectorizer.transform(preprocessed_texts)
        predictions = classifier.predict(X)
        
        return predictions

    # Calculate accuracy
    def calculate_accuracy(predictions, ground_truth):
        return accuracy_score(ground_truth, predictions)
    
    classifier, vectorizer, X_test, y_test = train_sentiment_classifier(texts, labels)

    # Perform sentiment analysis on test set
    predictions = perform_sentiment_analysis(texts, classifier, vectorizer)

    # Calculate accuracy
    accuracy = calculate_accuracy(predictions, labels)
    
    return accuracy