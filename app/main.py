from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

preprocessor = load('../models/preprocessor.joblib')
model = torch.load("../models/pytorch_beer_classifier.pt")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer classificatio is ready to go!'

def format_features(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int, beer_abv: int):
    return {
        'brewery_name': [brewery_name],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'beer_abv': [beer_abv]
    }

@app.get("/beer/classification")
def predict(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int, beer_abv: int):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)
    data = preprocessor.transform(obs)
    data_tensor = torch.Tensor(np.array(data))
    pred = model(data_tensor).argmax(1)
    return JSONResponse(pred.tolist())