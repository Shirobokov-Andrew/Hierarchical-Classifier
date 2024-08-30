from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from hierarchical_classifier import HierarchicalClassifier
import numpy as np

# Создаем объект FastAPI
app = FastAPI()

# Загружаем предобученную модель
model = joblib.load('hierarchical_logreg_cls_model.pkl')


# Модель для входных данных
class Review(BaseModel):
    text: str


# Главная страница для проверки, что сервис работает
@app.get("/")
def read_root():
    return {"message": "Hierarchical Classification API"}


# Предсказанеи категории по тексту
@app.post("/predict/")
def predict_category(review: Review):
    try:
        # Получаем текст из запроса
        text = review.text

        # Предсказываем с помощью загруженной модели
        prediction = model.predict([text])
        print('Предсказания успешно выполнены!')

        return {
            "Cat1": prediction[0][0],
            "Cat2": prediction[0][1],
            "Cat3": prediction[0][2]
        }
    except Exception as e:
        # В случае ошибки возвращаем HTTP 500 и сообщение об ошибке
        raise HTTPException(status_code=500, detail=str(e))

