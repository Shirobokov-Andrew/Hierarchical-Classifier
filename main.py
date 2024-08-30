import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import line_profiler
from hierarchical_classifier import HierarchicalClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import time


@line_profiler.profile
def test_flat_classifier(train_df, test_df, classifier):
    _ = classifier.fit(train_df['Text'], train_df['Cat1Cat2Cat3'])
    y_predicted_flat_logreg = classifier.predict(test_df['Text'])


def main():
    data_path_train = './amazon/train_40k.csv'
    data_path_test = './amazon/val_10k.csv'

    train_df = pd.read_csv(data_path_train)
    train_df = train_df[['Text', 'Cat1', 'Cat2', 'Cat3']].drop_duplicates(ignore_index=True)
    test_df = pd.read_csv(data_path_test).drop_duplicates()
    test_df = test_df[['Text', 'Cat1', 'Cat2', 'Cat3']].drop_duplicates(ignore_index=True)

    train_df['Cat1Cat2Cat3'] = train_df['Cat1'] + '/' + train_df['Cat2'] + '/' + train_df['Cat3']
    test_df['Cat1Cat2Cat3'] = test_df['Cat1'] + '/' + test_df['Cat2'] + '/' + test_df['Cat3']

    logreg_cls = HierarchicalClassifier(LogisticRegression, max_iter=500)
    cur_time = time.time()
    _ = logreg_cls.fit(train_df['Text'], train_df[['Cat1', 'Cat2', 'Cat3']].to_numpy())
    end_time = time.time()
    print(f'Time to fit the hierarchical classifier: {end_time-cur_time:.3f}')

    cur_time = time.time()
    y_predicted_logreg = logreg_cls.predict(test_df['Text'])
    end_time = time.time()
    print(f'Time to predict with hierarchical classifier: {end_time-cur_time:.3f}')

    flat_logreg_cls = Pipeline([
            ('tf-idf', TfidfVectorizer(max_features=10000)),
            ('clf', LogisticRegression(max_iter=500))
        ])
    cur_time = time.time()
    test_flat_classifier(train_df, test_df, flat_logreg_cls)
    end_time = time.time()
    print(f'Time to fit and predict with flat classifier: {end_time-cur_time:.3f}')


if __name__ == '__main__':
    main()
