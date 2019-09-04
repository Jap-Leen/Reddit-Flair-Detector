import praw as pr
import pandas as pd
import datetime as dt

# Extracting relevant information about each flair.
def ExtractFlairInfo(flair):
  get_subreddits = subreddit.search(flair, limit=30)
  for submission in get_subreddits:
    submission_info["id"].append(submission.id)
    submission_info["title"].append(submission.title)
    submission_info["url"].append(submission.url)
    submission_info["flair"].append(flair)
    submission_info["body"].append(submission.selftext)
    submission_info["number_of_comments"].append(submission.num_comments)
    comment = ''
    submission.comments.replace_more(limit=0)
    for comment_c in submission.comments:
    	comment+=' ' + comment_c.body
    submission_info["comments"].append(comment)
    submission_info["score"].append(submission.score)
    submission_info["upvote_ratio"].append(submission.upvote_ratio)
    submission_info["creation_date"].append(submission.created)

# Creating an instance of Reddit
reddit = pr.Reddit(client_id='Pr1H4ZD88nm0ag',
                   client_secret='6wjk3Y6PnD-P1FunblDuP0KCibs',
                   password='reddit@123',
                   user_agent='Reddit Flair Detector',
                   username='JapLeen')

# Collecting posts from r/india subreddit
subreddit = reddit.subreddit('india')

# List of flairs on the subreddit mentioned
flairs_list = ["Political", "Non-political", "Reddiquette", "AskIndia", "Science/Technology", "Policy/Economy", "Finance/Business", "Sports", "Photogrpahy", "AMA"]

# Storing relevant information about each flair on the subreddit
submission_info = {"id":[], "title":[], "url": [], "flair": [], "body":[], "number_of_comments":[], "comments":[], "score":[], "upvote_ratio":[], "creation_date":[]}

# Iteratnig through all the flairs and extracting relevant information
for flair in flairs_list:
	ExtractFlairInfo(flair)

# Storing results in tabular form
redditData = pd.DataFrame(submission_info)

# Storing results in CSV format
redditData.to_csv(r'./redditData.csv')