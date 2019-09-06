import pandas as pd
from pymongo import MongoClient 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split

# Getting stopwords set from english language
STOPWORDS = set(stopwords.words('english'))

flairs_list = ["Political", "Non-political", "Reddiquette", "AskIndia", "Science/Technology", "Policy/Economy", "Finance/Business", "Sports", "Photogrpahy", "AMA"]

def ConvertToString(value):
    return str(value)

def Lemmatization(text):
	token_words = word_tokenize(text)
	ls = WordNetLemmatizer()
	list_lemma = [ls.lemmatize(word) for word in token_words if word.isalnum()]
	text = (" ".join(list_lemma))
	return text

def Stemming(text):
	token_words = word_tokenize(text)
	ps = PorterStemmer()
	list_stem = [ps.stem(word) for word in token_words if word.isalnum()]
	text = (" ".join(list_stem))
	return text

def RemoveStopwords(text):
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def PreProcessing(feature):
	df[feature] = df[feature].apply(ConvertToString)
	df[feature] = df[feature].str.lower()
	df[feature] = df[feature].apply(Stemming)
	df[feature] = df[feature].apply(Lemmatization)
	df[feature] = df[feature].apply(RemoveStopwords)

# Setting up MongoDB client 
client = MongoClient() 
client = MongoClient("mongodb://localhost:27017/") 

# Connecting to the reddit data collection in the database
mydatabase = client.redditFlair
mycollection = mydatabase['myTable']
Data = mydatabase.mycollection
df = pd.DataFrame(list(Data.find()))
df = df.fillna("")

selected_features = ['title', 'body', 'comments']

# Pre-processing the text contained in the selected features
for feature in selected_features:
	PreProcessing(feature)

# Getting combination of features to train models
combination_of_features = df["title"] + df["comments"] + df["body"] + df["url"]
df = df.assign(combination_of_features = combination_of_features)

x = df.combination_of_features
y = df.flair
# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=10)


