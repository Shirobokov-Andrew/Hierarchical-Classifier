from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from hierarchical_classifier import HierarchicalClassifier
import numpy as np

# Create a FastAPI object
app = FastAPI()

# Loading the pre-trained model
model = joblib.load('hierarchical_logreg_cls_model.pkl')


# Model for input data
class Review(BaseModel):
    text: str


# Home page to check if the service is working
@app.get("/")
def read_root():
    return {"message": "Hierarchical Classification API"}


# Predicting category from text
@app.post("/predict/")
def predict_category(review: Review):
    try:
        # Getting text from request
        text = review.text

        # Predict using the loaded model
        prediction = model.predict([text])
        print('Предсказания успешно выполнены!')

        return {
            "Cat1": prediction[0][0],
            "Cat2": prediction[0][1],
            "Cat3": prediction[0][2]
        }
    except Exception as e:
        # In case of an error, we return HTTP 500 and an error message
        raise HTTPException(status_code=500, detail=str(e))

