from filereader import FileReader
from collections import Counter
import random
import nltk
import numpy as np
import codecs
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import neighbors
import cPickle as pickle

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
        self.MaxEntClassification()
        # print self.test_set
        # print self.test_set
        # self.GetNumberofBigramTypes(self.test_set[0]["review"])
        # self.MaxEntClassifierTrain()

    def MaxEntClassification(self):
        features = []
        correct_labels = []
        self.review_pos = {}
        self.review_sentiment_counts = {}
        # print len(self.combined_reviews)
        for review_object in self.combined_reviews:
            # Get review text
            feat = []
            review = review_object["review"]
            id = review_object["id"]
            # feat.append(self.GetPercentageOfSentimentWords(review, id))
            # feat.append(self.GetLengthofReview(review))
            feat.append(self.GetSentimentWordCount(id))
            # pronoun_count, verb_count = self.GetNumberofPronounsAndVerbs(review, id)
            pronoun_count, verb_count = self.GetPosCounts(id)
            feat.append(pronoun_count)
            feat.append(verb_count)
            # feat.append(noun_count)
            feat.append(self.GetNumberofRelationshipWords(review))
            features.append(feat)
            # features.append(pronoun_count)
            if(review_object["sentiment"] == "True"):
                correct_labels.append(1)
            else:
                correct_labels.append(0)
            # correct_labels.append(review_object["id"])
        # print features
		# with open('review_pos', 'wb') as dump:
		# 	pickle.dump(self.review_pos, dump, pickle.HIGHEST_PROTOCOL)
		# with open('review_pos_sentiment_counts', 'wb') as dump:
		# 	pickle.dump(self.review_sentiment_counts, dump, pickle.HIGHEST_PROTOCOL)
        X = np.matrix(features)
        # X = X.reshape(-1, 1)
        y = np.array(correct_labels)
        log_reg_classifier = LogisticRegression()
        # log_reg_classifier = neighbors.KNeighborsClassifier(100)
        # log_reg_classifier = svm.SVC()
        results_array = []
        # loo = cross_validation.LeaveOneOut(len(self.combined_reviews))
        # print "Predicting..."
        # for training_index, test_index in loo:
        #     X_train, X_test = X[training_index], X[test_index]
        #     y_train, y_test = y[training_index], y[test_index]
        #     log_reg_classifier.fit(X_train, y_train)
        #     prediction_res = log_reg_classifier.predict(X_test)
        #     results_array.append(accuracy_score(y_test, prediction_res))
        # print results_array.count(1) / float(len(results_array))
        log_reg_classifier.fit(X, y)

        # Read test data
        # with codecs.open("hotelDeceptionTest.txt", "rb", encoding='utf8') as test_file:
        test_filereader = FileReader("hotelDeceptionTest.txt")
        test_filereader.ParseFile()
        test_reviews = test_filereader.GetReviewList()

        features = []
        for review_object in test_reviews:
            # Get review text
            feat = []
            review = review_object["review"]
            id = review_object["id"]
            feat.append(self.GetPercentageOfSentimentWords(review, id))
            # feat.append(self.GetLengthofReview(review))
            # feat.append(self.GetPercentageOfSentimentWords(id))
            pronoun_count, verb_count = self.GetNumberofPronounsAndVerbs(review, id)
            # pronoun_count, verb_count = self.GetPosCounts(id)
            feat.append(pronoun_count)
            feat.append(verb_count)
            # feat.append(noun_count)
            feat.append(self.GetNumberofRelationshipWords(review))
            features.append(feat)

        X_test = np.matrix(features)
        results_array = log_reg_classifier.predict(X_test)

        # Print the results
        for i in range(0, len(results_array)):
            if results_array[i] == 1:
                res = "T"
            else:
                res="F"
            print test_reviews[i]["id"],"\t", res



    def MaxEntClassifierTrain(self):
        features = []
        correct_labels = []
        # print len(self.training_set)
        for review_object in self.training_set:
            # Get review text
            feat = []
            review = review_object["review"]
            feat.append(self.GetPercentageOfSentimentWords(review))
            pronoun_count, verb_count = self.GetNumberofPronounsAndVerbs(review)
            feat.append(pronoun_count)
            feat.append(verb_count)
            feat.append(self.GetNumberofBigramTypes(review))
            # feat.append(self.GetNumberofRelationshipWords(review))
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
            # feat.append(self.GetNumberofBigramTypes(review))
            # feat.append(self.GetNumberofRelationshipWords(test_review))
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

    def GetSentimentWordCount(self, id):
        with open('review_pos_sentiment_counts', 'rb') as dump_file:
            pickled_data = pickle.load(dump_file);
        return pickled_data[id]

    def GetPosCounts(self, id):
        with open('review_pos', 'rb') as dump_file:
            pickled_data = pickle.load(dump_file);
        element = pickled_data[id]
        return (element["NN"], element["VB"])

    def GetLengthofReview(self, review):
        text = nltk.word_tokenize(review)
        # print len(text)
        return len(text)

    def GetNumberofBigramTypes(self, review):
        # print review
        bigrams = nltk.bigrams(nltk.word_tokenize(review))
        # number_of_bigram_types = len(Counter(bigrams))
        final_list = [s for s in bigrams if s[0] == s[1]]
        return len(final_list)
        # print final_list
        # return number_of_bigram_types
        # print number_of_bigram_types

    def GetNumberofRelationshipWords(self, review):
        text = nltk.word_tokenize(review)
        relationship_word_count = 0
        rel_words = ['wife', 'husband', 'children', 'child', 'son', 'daughter', 'aunt', 'uncle', 'nephew', 'niece', 'sister', 'brother', 'grandfather', 'father', 'mother']
        # with codecs.open('relation-lexicon.txt', 'rb', encoding='utf8') as rel_words:
        for word in text:
            if word.lower() in rel_words:
                relationship_word_count += 1
        # print relationship_word_count
        return relationship_word_count


    def GetNumberofPronounsAndVerbs(self, review, id):
        # tokenize the review
        text = nltk.word_tokenize(review)
        # print nltk.pos_tag(text)
        pos_count = Counter(elem[1] for elem in nltk.pos_tag(text))
        self.review_pos[id] = pos_count
        # print pos_count
        # Get number of pronouns
        if "NN" in pos_count:
            pronoun_count = pos_count["NN"]
        else:
            pronoun_count = 0
        if "VB" in pos_count:
            verb_count = pos_count["VB"]
        else:
            verb_count = 0
        # print pronoun_count
        return (pronoun_count, verb_count)

    def GetPercentageOfSentimentWords(self, review, id):
        text = nltk.word_tokenize(review)
        sentiment_word_count = 0
        for word in text:
            with codecs.open('positive-words.txt', 'rb', encoding='utf8') as pos_words:
                for line in pos_words:
                    if word == line.strip():
                        sentiment_word_count += 1
                        break
            # with codecs.open('negative-words.txt', 'rb', encoding='utf8') as neg_words:
            #     for line in neg_words:
            #         if word == line.strip():
            #             sentiment_word_count += 1
            #             break
        # sentiment_percentage = float(sentiment_word_count / word_count)
        # print sentiment_word_count
        self.review_sentiment_counts[id] = sentiment_word_count
        return sentiment_word_count
