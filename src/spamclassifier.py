from filereader import FileReader
from collections import Counter
import random
import nltk
import numpy as np
import codecs
from sklearn.linear_model import LogisticRegression

class FakeReviewClassifier(object):
    def __init__(self, genuine_file, fake_file):
        # Get True Reviews
        self.true_filereader = FileReader(genuine_file, "True")
        self.true_filereader.ParseFile()
        self.genuine_reviews = self.true_filereader.GetReviewList()
        # print self.genuine_reviews

        # Get Fake Reviews
        self.fake_file_reader = FileReader(fake_file, "Fake")
        self.fake_file_reader.ParseFile()
        self.fake_reviews = self.fake_file_reader.GetReviewList()

        # Merge both the Reviews
        self.combined_reviews = []
        self.combined_reviews.extend(self.genuine_reviews)
        self.combined_reviews.extend(self.fake_reviews)
        # Split 80-20
        data_length = len(self.combined_reviews)
        random.shuffle(self.combined_reviews)
        train_set_length = int(0.9 * data_length)
        self.training_set = self.combined_reviews[:train_set_length]
        self.test_set = self.combined_reviews[train_set_length:]
        # print self.test_set
        # print self.test_set
        self.MaxEntClassifierTrain()

    def MaxEntClassifierTrain(self):
        features = []
        correct_labels = []
        print len(self.training_set)
        for review_object in self.training_set:
            # Get review text
            feat = []
            review = review_object["review"]
            # pronoun_count = self.GetNumberofPronounsAndVerbs(review)
            feat.append(self.GetPercentageOfSentimentWords(review))
            pronoun_count, verb_count = self.GetNumberofPronounsAndVerbs(review)
            feat.append(pronoun_count)
            feat.append(verb_count)
            features.append(feat)
            # features.append(pronoun_count)
            if(review_object["sentiment"] == "True"):
                correct_labels.append(1)
            else:
                correct_labels.append(0)
            # correct_labels.append(review_object["id"])
        # print features
        X = np.matrix(features)
        # X = X.reshape(-1, 1)
        print X
        y = np.array(correct_labels)
        # print X.shape
        # print y.shape
        # print y
        log_reg_classifier = LogisticRegression()
        log_reg_classifier.fit(X, y)
        # TODO pickle the classifier object and just read from memory
        # Classify the first review

        test_features = []
        test_labels = []
        for rev in self.test_set:
            feat = []
            test_review = rev["review"]
            # test_features = np.matrix(self.GetNumberofPronounsAndVerbs(test_review))
            # test_features.reshape(1, -1)
            # test_features.append(self.GetPercentageOfSentimentWords(test_review))
            feat.append(self.GetPercentageOfSentimentWords(test_review))
            pronoun_count, verb_count = self.GetNumberofPronounsAndVerbs(test_review)
            feat.append(pronoun_count)
            feat.append(verb_count)
            test_features.append(feat)
            if rev["sentiment"] == "True":
                test_labels.append(1)
            else:
                test_labels.append(0)
            # print test_features.shape
            # print rev["id"], ":", log_reg_classifier.predict(test_features)
        X1 = np.matrix(test_features)
        # X1 = X1.reshape(-1, 1)
        y1 = np.array(test_labels)
        print log_reg_classifier.score(X1, y1)


    def GetNumberofPronounsAndVerbs(self, review):
        # tokenize the review
        text = nltk.word_tokenize(review)
        # print nltk.pos_tag(text)
        pos_count = Counter(elem[1] for elem in nltk.pos_tag(text))
        # print pos_count
        # Get number of pronouns
        if "NNP" in pos_count:
            pronoun_count = pos_count["NNP"]
        else:
            pronoun_count = 0
        if "VBP" in pos_count:
            verb_count = pos_count["VBP"]
        else:
            verb_count = 0
        # print pronoun_count
        return (pronoun_count, verb_count)

    def GetPercentageOfSentimentWords(self, review):
        text = nltk.word_tokenize(review)
        word_count = len(text)
        sentiment_word_count = 0
        for word in text:
            with codecs.open('positive-words.txt', 'rb', encoding='utf8') as pos_words:
                for line in pos_words:
                    if word == line.strip():
                        sentiment_word_count += 1
                        break
            with codecs.open('negative-words.txt', 'rb', encoding='utf8') as neg_words:
                for line in neg_words:
                    if word == line.strip():
                        sentiment_word_count += 1
                        break
        # sentiment_percentage = float(sentiment_word_count / word_count)
        # print sentiment_word_count
        return sentiment_word_count
