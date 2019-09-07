import pandas as pd
import pickle
from pymongo import MongoClient 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Getting stopwords set from english language
STOPWORDS = set(stopwords.words('english'))

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
	# df[feature] = df[feature].apply(Stemming)
	# df[feature] = df[feature].apply(Lemmatization)
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

def MultinomialNaiveBayes(x_train,y_train,x_test,y_test):
	nb = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
	nb.fit(x_train,y_train)
	y_pred = nb.predict(x_test)
	print("Naive Bayes accuracy_score: "+str(accuracy_score(y_pred,y_test)))

def LinearSVM(x_train,y_train,x_test,y_test):
	sgd = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None))])
	sgd.fit(x_train,y_train)
	y_pred = sgd.predict(x_test)
	print("Linear SVM accuracy_score: "+str(accuracy_score(y_pred,y_test)))
	model = 'finalized_model.sav'
	pickle.dump(sgd, open(model, 'wb'))

def LogisticRegressionC(x_train, y_train, x_test, y_test):
	logistic_reg = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LogisticRegression(n_jobs = 1, C = 1e5))])
	logistic_reg.fit(x_train,y_train)
	y_pred = logistic_reg.predict(x_test)
	print("Logistic Regression accuracy_score: "+str(accuracy_score(y_pred,y_test)))

# Training different models
MultinomialNaiveBayes(x_train,y_train,x_test,y_test)
LinearSVM(x_train,y_train,x_test,y_test)
LogisticRegressionC(x_train,y_train,x_test,y_test)



