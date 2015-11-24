#!/usr/bin/python
from math import log
from filereader import FileReader
import random
import re
from collections import Counter

class NaiveBayesClassifier(object):
    def __init__(self, positive_file, negative_file):
        # Parse positive reviews
        self.pos_filereader = FileReader(positive_file, "True")
        self.positive_words = self.pos_filereader.ParseFile()
        # self.positive_word_count = len(self.positive_words)

        self.positive_reviews = self.pos_filereader.GetReviewList()
        # print self.positive_reviews

        self.neg_filereader = FileReader(negative_file, "Fake")
        self.negative_words = self.neg_filereader.ParseFile()
        self.negative_reviews = self.neg_filereader.GetReviewList()

        # Merge both the lists
        self.combined_reviews = []
        self.combined_reviews.extend(self.positive_reviews)
        self.combined_reviews.extend(self.negative_reviews)
        # Split 80-20
        data_lenth = len(self.combined_reviews)
        random.shuffle(self.combined_reviews)
        train_set_length = int(0.8 * data_lenth)
        self.training_set = self.combined_reviews[:train_set_length]
        self.test_set = self.combined_reviews[train_set_length:]

        # Do the counting for test_set
        str = ""
        self.positive_word_count = 0
        self.negative_word_count = 0
        self.positive_words = {}
        self.negative_words = {}
        self.pos_words = []
        self.neg_words = []
        for review in self.training_set:
            # review = review_item.iteritems()
            # ipdb.set_trace()
            str += review["review"]
            words = re.findall(r'\w+', review["review"])
            wc = len(words)
            if review["sentiment"] == "True":
                self.positive_word_count += wc
                self.pos_words.extend(words)
            else:
                self.negative_word_count += wc
                self.neg_words.extend(words)

        words = re.findall(r'\w+', str)
        self.bag_of_words = Counter(words)
        self.total_word_type_count = len(self.bag_of_words)
        self.positive_words = Counter(self.pos_words)
        self.negative_words = Counter(self.neg_words)
        print self.positive_word_count
        print self.negative_word_count
        # print len(self.positive_words)
        # print len(self.negative_words)
        # Classify
        self.Classify_Test()

        # print self.bag_of_words

        # self.negative_word_count = len(self.negative_words)
        # self.bag_of_words = self.positive_words + self.negative_words
        # self.total_word_type_count = len(self.bag_of_words)

        # print self.total_word_type_count

    def Classify_Test(self):
        total = 0
        correct = 0
        for review in self.test_set:
            cur_review = review["review"]
            cur_review = cur_review.replace(",", " ")
            cur_review = cur_review.replace(".", " ")
            cur_review = cur_review.replace(";", " ")
            cur_review = cur_review.replace(";", " ")
            cur_review = cur_review.replace("/", " ")
            cur_review = cur_review.replace("-", " ")
            cur_review = cur_review.replace("!", " ")
            cur_review = cur_review.replace("\"", "")
            cur_review = cur_review.lower()
            review_words = cur_review.split()
            positive_probability = 0
            negative_probability = 0
            for word in review_words:
                # if word is in the positive wordlist
                if word in self.positive_words:
                    # print self.bag_of_words[word]
                    positive_probability += log(self.positive_words[word] + 1) - (log(self.total_word_type_count + self.positive_word_count))
                else:
                    # just use the count as 1
                    positive_probability += 0 - log(self.total_word_type_count + self.positive_word_count)
                if word in self.negative_words:
                    negative_probability += log(self.negative_words[word] + 1) - (log(self.total_word_type_count + self.negative_word_count))
                else:
                    negative_probability += 0 - log(self.total_word_type_count + self.negative_word_count)
            # print positive_probability, negative_probability
            total += 1
            if positive_probability > negative_probability:
                print review["id"], "\t" ,"True"
                if review["sentiment"] == "True":
                    correct += 1
            else:
                print review["id"], "\t" , "Fake"
                if review["sentiment"] == "Fake":
                    correct += 1
        print "Accuracy: ", float(correct * 100 / total)
    def Classify(self, test_file):
        # create new filereader
        # don't pass sentiment to the function. It will be set to None
        test_file_reader = FileReader(test_file)
        # Parse the file to read all the reviews
        test_file_reader.ParseFile()
        reviews = test_file_reader.GetReviews()
        # print reviews
        # Iterate over the reviews in the test set
        for id, review in reviews.iteritems():
            # print review
            # split the review into words
            cur_review = review["review"]
            review_words = cur_review.split()
            # print review_words
            # For each word in the review calculate a probability
            # use add-1 smoothing
            positive_probability = 0
            negative_probability = 0
            for word in review_words:
                # if word is in the positive wordlist
                if word in self.positive_words:
                    # print self.bag_of_words[word]
                    positive_probability += log(self.positive_words[word] + 1) - (log(self.total_word_type_count + self.positive_word_count))
                else:
                    # just use the count as 1
                    positive_probability += 0 - log(self.total_word_type_count + self.positive_word_count)
                if word in self.negative_words:
                    negative_probability += log(self.negative_words[word] + 1) - (log(self.total_word_type_count + self.negative_word_count))
                else:
                    negative_probability += 0 - log(self.total_word_type_count + self.negative_word_count)
            # print positive_probability, negative_probability
            if positive_probability > negative_probability:
                print review["id"], "\t" ,"POS"
            else:
                print review["id"], "\t" , "NEG"
