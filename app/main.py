from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mod import PytorchMultiClass

app = FastAPI()

preprocessor = load('../models/preprocessor.joblib')

# class PytorchMultiClass(nn.Module):
#     def __init__(self, num_features):
#         super(PytorchMultiClass, self).__init__()
        
#         self.layer_1 = nn.Linear(num_features, 32)
#         self.layer_out = nn.Linear(32, 10)

#     def forward(self, x):
#         x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
#         return self.layer_out(x)

#device = torch.device("cpu")

mod = PytorchMultiClass(14)
mod.load_state_dict(torch.load("../models/pytorch_beer_classifier.pt"))  #, map_location=device))
mod.eval()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer classification is ready to go!'

def format_features(brewery_name: str,	review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    return {
        'brewery_name': [brewery_name],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'beer_abv': [beer_abv]
    }

@app.get("/beer/type")
def predict(brewery_name: str,	review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)
    data = preprocessor.transform(obs)
    data_tensor = torch.Tensor(np.array(data))
    pred = mod(data_tensor).argmax(1)
    return JSONResponse(pred.tolist())
    # return JSONResponse(data.tolist())
    # return data

@app.get("/beers/type")
def predict(brewery_name: str,	review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float, brewery_name_2: str,	review_aroma_2: float, review_appearance_2: float, review_palate_2: float, review_taste_2: float, beer_abv_2: float):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    features2 = format_features(brewery_name_2, review_aroma_2, review_appearance_2, review_palate_2, review_taste_2, beer_abv_2)
    obs_1 = pd.DataFrame(features)
    obs_2 = pd.DataFrame(features2)
    obs = obs_1.append(obs_2)
    data = preprocessor.transform(obs)
    data_tensor = torch.Tensor(np.array(data))
    pred = mod(data_tensor).argmax(1)
    return JSONResponse(pred.tolist())

@app.get("/model/architecture")
def architecture():
    return JSONResponse(str(mod))