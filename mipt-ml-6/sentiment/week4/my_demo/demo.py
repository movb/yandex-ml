# -*- coding: utf-8 -*-

from sentiment_classifier import SentimentClassifier
from codecs import open
import time
from flask import Flask, render_template, request
app = Flask(__name__)

print "Preparing classifier"
start_time = time.time()
classifier = SentimentClassifier()
print "Classifier is ready"
print time.time() - start_time, "seconds"

@app.route("/sentiment-demo", methods=["POST", "GET"])
def index_page(text="", prediction_message="", prob=0.0):
    if request.method == "POST":
        text = request.form["text"]
        logfile = open("ydf_demo_logs.txt", "a", "utf-8")
	print text
	print >> logfile, "<response>"
	print >> logfile, text
        prediction_message, prob = classifier.get_prediction_message_and_score(text)
        print prediction_message
	print >> logfile, prediction_message
	print >> logfile, prob
	print >> logfile, "</response>"
	logfile.close()
	
    return render_template('hello.html', text=text, prediction_message=prediction_message, prob=prob)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5550, debug=False)
