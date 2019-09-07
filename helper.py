import joblib
import nltk
from nltk.corpus import stopwords
import praw as pr
import pandas as pd
import sklearn
from training_models import PreProcessing

stops = set(stopwords.words("english"))
model = joblib.load("./finalized_model.sav")

# Creating an instance of Reddit
reddit = pr.Reddit(client_id='Pr1H4ZD88nm0ag',
                   client_secret='6wjk3Y6PnD-P1FunblDuP0KCibs',
                   password='reddit@123',
                   user_agent='Reddit Flair Detector',
                   username='JapLeen')

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

