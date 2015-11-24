#!/usr/bin/python
from classifier import NaiveBayesClassifier
from spamclassifier import FakeReviewClassifier

a = FakeReviewClassifier("training_data/hotelT-train.txt","training_data/hotelF-train.txt")
