# -*- coding: utf-8 -*-

from sentiment_classifier import SentimentClassifier

clf = SentimentClassifier()

pred, score = clf.get_prediction_message_and_score("It was bad bank")

print pred, score
