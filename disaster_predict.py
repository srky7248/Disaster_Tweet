import string
import pickle
from keras.models import load_model
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
def receive_text():
    entered_text = input("Please enter a string (Press Enter to finish): ")
    if not entered_text:
        return
    return entered_text
entered_text = receive_text()
def data_cleaning(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'', text)
    
    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)
    
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
                               
    table=str.maketrans('','',string.punctuation)
    text = text.translate(table)
    
    text = text.lower()
    return text

entered_text = data_cleaning(entered_text)
def create_corpus(text):
    corpus = []
    words = [word.lower() for word in word_tokenize(text) if((word.isalpha()==1) & (word not in stop))]
    corpus.append(words)
    return corpus

entered_text = create_corpus(entered_text)

with open('./model/tokenize_model/tokenizer_obj.pkl', 'rb') as file:
    tokenizer_obj = pickle.load(file)
MAX_LEN = 50
sequences=tokenizer_obj.texts_to_sequences(entered_text)
tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
model = load_model('./model/studyed_model/desaster.h5')
predict = model.predict(tweet_pad)
if predict >= 0.5:
    print("disaster")
else:
    print("not disaster")
