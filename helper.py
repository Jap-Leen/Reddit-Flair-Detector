import joblib
import nltk
from nltk.corpus import stopwords
import praw as pr
import pandas as pd
import sklearn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

stops = set(stopwords.words("english"))
model = joblib.load("./finalized_model.sav")

# Creating an instance of Reddit
reddit = pr.Reddit(client_id='Pr1H4ZD88nm0ag',
                   client_secret='6wjk3Y6PnD-P1FunblDuP0KCibs',
                   password='reddit@123',
                   user_agent='Reddit Flair Detector',
                   username='JapLeen')

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

def helper(url):
	submission_info = {"id":[], "title":[], "body":[], "comments":[]}
	submission = reddit.submission(url = url)
	submission_info["id"] = submission.id
	submission_info["title"] = submission.title
	submission_info["body"] = submission.selftext
	comment = ''
	submission.comments.replace_more(limit=0)
	for comment_c in submission.comments:
		comment+= ' ' + comment_c.body
	submission_info["comments"].append(comment)
	data = pd.DataFrame(submission_info)
	data.fillna("")
	selected_features = ['title', 'body', 'comments']
	# Pre-processing the text contained in the selected features
	for feature in selected_features:
		PreProcessing(feature)
	# Getting combination of features
	combination_of_features = data["title"] + data["comments"] + data["body"]
	data = data.assign(combination_of_features = combination_of_features)
	return(model.predict(data['combination_of_features'])[0])

