# Reddit-Flair-Detector - flaiReddit

flaiReddit is a Reddit Flair Detector for subreddit [r/india](https://www.reddit.com/r/india/), that takes any post's URL as input and predicts the flair for the post using Machine Learning models. The web application for the same is hosted on Heroku at [flaiReddit(https://flaireddit.herokuapp.com/)]. The web-application also contans some useful data plots obtained after analysis of collected data.
  
## Codebase

The code has been developed using Python programming language, utilizing it's text processing and machine learning modules. The web application has been developed using Flask, HTML, CSS and hosted on Heroku web server.

## Dependencies

The dependencies can be found in [requirements.txt](https://github.com/Jap-Leen/Reddit-Flair-Detector/blob/master/requirements.txt). 

## Directory and File Structure

- [app.py](https://github.com/Jap-Leen/Reddit-Flair-Detector/blob/master/app.py): Used to start up the Flask app.
- [scrapeData.py](https://github.com/Jap-Leen/Reddit-Flair-Detector/blob/master/scrapeData.py): Used to scrape r/India posts from Reddit.
- [training_models.py](https://github.com/Jap-Leen/Reddit-Flair-Detector/blob/master/training_models.py): Used to pre-process text and train various models. It was also used to analyse data by plotting trends.
- [helper.py](https://github.com/Jap-Leen/Reddit-Flair-Detector/blob/master/helper.py): Used to get predicted flair for given URL test.
- [requirements.txt](https://github.com/Jap-Leen/Reddit-Flair-Detector/blob/master/requirements.txt): Contains all dependencies for the project
- [nltk.txt](https://github.com/Jap-Leen/Reddit-Flair-Detector/blob/master/nltk.txt): Contains NLTK library dependencies for deployment on Heroku.
- [data](https://github.com/Jap-Leen/Reddit-Flair-Detector/tree/master/data): Contains CSV and JSON files of collected posts.
- [templates](https://github.com/Jap-Leen/Reddit-Flair-Detector/tree/master/templates): Contains HTML script for the web application
- [static](https://github.com/Jap-Leen/Reddit-Flair-Detector/tree/master/static): Contains [images](https://github.com/Jap-Leen/Reddit-Flair-Detector/tree/master/static/images) folder having the plots displayed on the web-application, obtained after data analysis.


## How to execute?

  1. Open the `Terminal`.
  2. Clone the repository by entering `git clone https://github.com/Jap-Leen/Reddit-Flair-Detector.git`.
  3. Ensure that `Python3` and `pip` is installed on the system.
  4. Create a `virtualenv` by executing the following command: `virtualenv venv`.
  5. Activate the `venv` virtual environment by executing the follwing command: `source venv/bin/activate`.
  6. Enter the cloned repository directory and execute `pip install -r requirements.txt`.
  7. Run `python app.py` from Terminal. 

## Approach 
### Data Scraping
The python library PRAW has been used to scrape data from the subreddit [r/india](https://www.reddit.com/r/india/). 300 posts belonging to each of thee flairs were collected and analysed.

### Data pre - preprocessing
The following procedures have been executed on the title, body and comments to clean the data:
1. Lowercasing
2. Tokenizing and stemming
3. Lemmatization
4. Removing stopwords

### Storing Data
Data so collected is stored as a MongoDB collection. Its JSON file can be found [here](https://github.com/Jap-Leen/Reddit-Flair-Detector/blob/master/data/reddit_flair_initial_data.json).

### Data spliting
The collected data is split as follows:
0.25% as Test Data and 0.75% as Training Data

### Training 
Features of the posts like Title, Comment, Body and URL are used in various possible combinations and trained on three algorithms: Multinomial Naive Bayes, Linear SVM and Logistic Regression.

### Flair Prediction
The model with highest accuracy score is saved and loaded for predicting the flair and the returned result is displayed on the web application.
    
## Results

[Result Analysis Sheet](https://docs.google.com/spreadsheets/d/1HLhxVlx-4OxuncdlFTC4xGs_cMbevKM7W9vSA9F5UBA/edit?usp=sharing)

The resulting scores for different stages of pre-processing, features and models can be found above.

The best accuracy score obtained was of 0.793248945147679. The features selected were the combination of Title, Body, Comments and URL. The model trained was Linear SVM. (Includes pre-processing, without stemming and lemmatization)

## References

1. http://www.storybench.org/how-to-scrape-reddit-with-python/
2. https://praw.readthedocs.io/en/latest/code_overview/praw_models.html
3. https://devcenter.heroku.com/articles/getting-started-with-python
