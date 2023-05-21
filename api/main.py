#!pip install FastAPI
#!pip install pydantic
import pickle
import sklearn
import pandas as pd
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Wine Quality Classification",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model = pickle.load(
    open('model.pkl', 'rb')
)

@app.get("/")
def read_root(text: str = ""):
    if not text:
        return f"Try to appendd ?text=something in the URL!"
    else:
        return text


class Wine(BaseModel):
    fixed_acidity: int
    volatile_acidity: int
    cytric_acid: int
    residual_sugar: int
    chlorides: int
    free_sulfur_dioxide: int
    total_sulfur_dioxide: int
    density: int
    pH: int
    sulphates: int
    alcohol: int

@app.post("/predict/")
def predict(wines: List[Wine]) -> List[str]:
    X = pd.DataFrame([dict(wine) for wine in wines])
    y_pred = model.predict(X)
    return list(y_pred)
