from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import line_profiler
import joblib


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=LogisticRegression, **classifier_kwargs):
        """
        Initialization of the hierarchical classifier.
        This classifier works as follows:
        P(Cat1, Cat2, Cat3) = P(Cat3 | Cat1, Cat2) * P(Cat2 | Cat1) * P(Cat1),
        where each of the probability distributions is modeled by one classifier model base_classifier.
        TfidfVectorizer is used for text vectorization.
        :param base_classifier: Base Sklearn classifier
        :param classifier_kwargs: parameters for the base classifier (each has its own)
        """
        self.base_classifier = base_classifier
        self.classifier_kwargs = classifier_kwargs
        # To predict Cat1 we only need one model
        self.model_cat1 = None
        # For Cat2 predictions, multiple models are used to predict Cat2 given Cat1
        self.models_cat2 = {}
        # For Cat3 predictions, multiple models are used to predict Cat3 given (Cat1, Cat2)
        self.models_cat3 = {}

    @line_profiler.profile
    def fit(self, X, y):
        """
        Training a hierarchical classifier.
        :param X: Texts
        :param y: np.array of hierarchical class labels (Cat1, Cat2, Cat3).
        """
        # Training a classifier for Cat1
        self.model_cat1 = Pipeline([
            ('tf-idf', TfidfVectorizer(max_features=10000)),
            ('clf', self.base_classifier(**self.classifier_kwargs))
        ])
        self.model_cat1.fit(X, y[:, 0])

        # Training classifiers for Cat2 - going through all unique values of Cat1
        for cat1 in np.unique(y[:, 0]):
            # select objects that have a given Cat1 to predict Cat2 for them
            cat_indices = (y[:, 0] == cat1)
            # train the corresponding model
            self.models_cat2[cat1] = Pipeline([
                ('tf-idf', TfidfVectorizer(max_features=10000)),
                ('clf', self.base_classifier(**self.classifier_kwargs))
            ])
            self.models_cat2[cat1].fit(X[cat_indices], y[cat_indices, 1])

        # Training classifiers for Cat3 - going through all sorts of pairs (Cat1, Cat2)
        for cat1 in np.unique(y[:, 0]):
            for cat2 in np.unique(y[y[:, 0] == cat1, 1]):
                # select objects that have data (Cat1, Cat2) to predict Cat3 for them
                cat_indices = (y[:, 0] == cat1) & (y[:, 1] == cat2)
                unique_cat3 = np.unique(y[cat_indices, 2])
                # The case when for a given pair (Cat1, Cat2) there is only one possible Cat3 - in this case
                # using DummyClassifier we simply predict this Cat3
                if len(unique_cat3) < 2:
                    self.models_cat3[(cat1, cat2)] = Pipeline([
                        ('tf-idf', TfidfVectorizer(max_features=10000)),
                        ('clf', DummyClassifier(strategy='most_frequent'))
                    ])
                    self.models_cat3[(cat1, cat2)].fit(X[cat_indices], y[cat_indices, 2])
                else:
                    self.models_cat3[(cat1, cat2)] = Pipeline([
                        ('tf-idf', TfidfVectorizer(max_features=10000)),
                        ('clf', self.base_classifier(**self.classifier_kwargs))
                    ])
                    self.models_cat3[(cat1, cat2)].fit(X[cat_indices], y[cat_indices, 2])

    def predict(self, X):
        """
        Prediction using a hierarchical classifier.
        :param X: Texts
        :return: Predicted class labels (Cat1, Cat2, Cat3).
        """
        # First, Cat1 is predicted for all objects from X
        y_pred_cat1 = self.model_cat1.predict(X)
        # Arrays for predictions Cat2, Cat3 for all objects from X
        y_pred_cat2 = []
        y_pred_cat3 = []

        # Predict Cat2 and Cat3 based on predicted Cat1
        for i, cat1 in enumerate(y_pred_cat1):
            # Based on predicted Cat1 we predict Cat2
            cat2_pred = self.models_cat2[cat1].predict([X[i]])[0]
            y_pred_cat2.append(cat2_pred)

            # На основе предсказанных Cat1 и Cat2 предсказываем Cat3
            cat3_pred = self.models_cat3[(cat1, cat2_pred)].predict([X[i]])[0]
            y_pred_cat3.append(cat3_pred)

        y_pred_cat2 = np.array(y_pred_cat2)
        y_pred_cat3 = np.array(y_pred_cat3)

        return np.column_stack((y_pred_cat1, y_pred_cat2, y_pred_cat3))

    def save_model(self, filename):
        """
        Saving the model to a file.
        :param filename: the name of the file in which the model will be saved
        """
        joblib.dump(self, filename)
        print(f"The model was saved in {filename}!")

    @staticmethod
    def load_model(filename):
        """
        Loading a model from a file
        :param filename: the name of the file from which the model will be loaded
        :return: loaded model
        """
        model = joblib.load(filename)
        print(f"Model loaded from {filename}!")
        return model
