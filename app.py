from flask import Flask, render_template,request
from helper import helper
import os

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/flairDetect', methods=['POST'])
def flairDetect():
	redditURL = request.form['redditpost']
	print(redditURL)
	flair = str(helper(str(redditURL)))
	print(flair)
	return render_template('index.html',flair=flair)

port = int(os.getenv('VCAP_APP_PORT', 5000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)