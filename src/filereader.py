#! /usr/bin/python
# This module parses the file and creates a dict of all reviews
from collections import Counter
import re
import codecs

class FileReader(object):
    def __init__(self, filename, sentiment=None):
        # set the initial filename
        self.filename = filename
        # Set the sentiment
        self.sentiment = sentiment

        # Initialize empty dicts
        self.reviews = {}

        # # Read the stop words
        # self.stop_words = []
        # with open("stop-word-list.txt", "rb") as stopwordfile:
        #     for line in stopwordfile:
        #         self.stop_words.append(line.strip())

    def SetFile(self, filename):
        # To set the filename
        self.filename = filename

    def SetSentiment(self, sentiment):
        # Set the sentiment
        self.sentiment = sentiment

    def GetWordCount(self):
        return self.word_count
    def GetReviews(self):
        return self.reviews

    def GetReviewList(self):
        reviewList = []
        # ipdb.set_trace()
        for id, review in self.reviews.iteritems():
            # ipdb.set_trace()
            reviewList.append(review)
        return reviewList

    def RemovePunctuation(self, str):
        # Remove commas, full stops, semicolons, /, hyphens
        str = str.replace(",", " ")
        # str = str.replace(".", " ")
        str = str.replace(";", " ")
        str = str.replace(";", " ")
        str = str.replace("/", " ")
        str = str.replace("-", " ")
        str = str.replace("!", " ")
        # str = str.lower()
        return str

    def ParseFile(self):
        self.reviews = {}
        self.word_count = 0
        # Parse the file and create the dict
        if self.filename != None:
            with codecs.open(self.filename, "rb", encoding='utf8') as training_file:
                str = ""
                for line in training_file:
                    # Get ID
                    split_array = line.split()
                    # ipdb.set_trace()
                    id = split_array[0]
                    # print id
                    review = " ".join(split_array[1:])
                    # print review
                    review = self.RemovePunctuation(review)
                    self.reviews[id] = {
                    "review": review,
                    "id": id,
                    "sentiment": self.sentiment
                    }
                    str += review
                words = re.findall(r'\w+', str)
                self.word_count += len(words)
                cnt = Counter(words)
            return cnt
