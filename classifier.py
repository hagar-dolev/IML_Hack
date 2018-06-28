"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2018

            **  Code Classifier  **

Auther(s):

===================================================
"""

class Classifier(object):

    def classify(self,X):
        """
        Recieves a list of m unclassified pieces of code, and predicts for each
        one the Github project it belongs to.
        :param X: A list of length m containing the code segments (strings)
        :return: y_hat - a list where each entry is a number between 0 and 8
        0 - Sonar
        1 - Dragonfly
        2 - tensorflow
        3 - devilution
        4 - flutter
        5 - react
        6 - spritejs
        """
    raise NotImplementedError("TODO: Implement this method by 12pm Friday!")
