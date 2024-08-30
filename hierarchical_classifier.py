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
        Инициализация иерархического классификатора.
        Данный классификатор работает следующим образом:
        P(Cat1, Cat2, Cat3) = P(Cat3 | Cat1, Cat2) * P(Cat2 | Cat1) * P(Cat1),
        где каждое из вероятностных распределений моделируется одной моделью-классификатором base_classifier.
        Для векторизации текстов используется TfidfVectorizer.
        :param base_classifier: Базовый классификатор sklearn
        :param classifier_kwargs: параметры для базового классификатора (для каждого свои)
        """
        self.base_classifier = base_classifier
        self.classifier_kwargs = classifier_kwargs
        # Для предсказания Cat1 нам достаточно одной модели
        self.model_cat1 = None
        # Для предсказаний Cat2 используется множество моделей, чтобы предсказать Cat2 при данном Cat1
        self.models_cat2 = {}
        # Для предсказаний Cat3 используется множество моделей, чтобы предсказать Cat3 при данном (Cat1, Cat2)
        self.models_cat3 = {}

    @line_profiler.profile
    def fit(self, X, y):
        """
        Обучение иерархического классификатора.
        :param X: Тексты
        :param y: np.array иерархических меток классов (Cat1, Cat2, Cat3).
        """
        # Обучаем классификатор для Cat1
        self.model_cat1 = Pipeline([
            ('tf-idf', TfidfVectorizer(max_features=10000)),
            ('clf', self.base_classifier(**self.classifier_kwargs))
        ])
        self.model_cat1.fit(X, y[:, 0])

        # Обучаем классификаторы для Cat2 - проходимся по всем уникальным значениям Cat1
        for cat1 in np.unique(y[:, 0]):
            # выбираем объекты, которые имеют данную Cat1, чтобы предсказать им Cat2
            cat_indices = (y[:, 0] == cat1)
            # обучаем соответствующую модель
            self.models_cat2[cat1] = Pipeline([
                ('tf-idf', TfidfVectorizer(max_features=10000)),
                ('clf', self.base_classifier(**self.classifier_kwargs))
            ])
            self.models_cat2[cat1].fit(X[cat_indices], y[cat_indices, 1])

        # Обучаем классификаторы для Cat3 - проходимся по всевозможным парам (Cat1, Cat2)
        for cat1 in np.unique(y[:, 0]):
            for cat2 in np.unique(y[y[:, 0] == cat1, 1]):
                # выбираем объекты, которые имеют данные (Cat1, Cat2), чтобы предсказать им Cat3
                cat_indices = (y[:, 0] == cat1) & (y[:, 1] == cat2)
                unique_cat3 = np.unique(y[cat_indices, 2])
                # Случай, когда для данной пары (Cat1, Cat2) оказывается всего один возможный Cat3 - в этом случае
                # с помощью DummyClassifier просто предсказываем этот самый Cat3
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
        Предсказание с помощью иерархического классификатора.
        :param X: Тексты.
        :return: Предсказанные метки классов (Cat1, Cat2, Cat3).
        """
        # Сначала предсказывается Cat1 для всех объектов X
        y_pred_cat1 = self.model_cat1.predict(X)
        # Массивы для предсказаний Cat2, Cat3 для всех объектов X
        y_pred_cat2 = []
        y_pred_cat3 = []

        # Предсказание Cat2 и Cat3 на основе предсказанных Cat1
        for i, cat1 in enumerate(y_pred_cat1):
            # На основе предсказанной Cat1 предсказываем Cat2
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
        Сохранение модели в файл.
        :param filename: имя файла, в котором будет сохранена модель
        """
        joblib.dump(self, filename)
        print(f"Модель сохранена в {filename}!")

    @staticmethod
    def load_model(filename):
        """
        Загрузка модели из файла
        :param filename: имя файла, из которого модель будет загружена
        :return: загруженная модель
        """
        model = joblib.load(filename)
        print(f"Модель загружена из {filename}!")
        return model
