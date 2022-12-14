from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import PytorchMultiClass

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

model = PytorchMultiClass(14)
model.load_state_dict(torch.load("../models/pytorch_beer_classifier.pt"))  #, map_location=device))
# model.eval()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer classification is ready to go!'

def format_features(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int, beer_abv: int):
    return {
        'brewery_name': [brewery_name],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'beer_abv': [beer_abv]
    }

@app.get("/beer/type")
def predict(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int, beer_abv: int):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)
    data = preprocessor.transform(obs)
    # data_tensor = torch.Tensor(np.array(data))
    # pred = model(data_tensor).argmax(1)
    # return JSONResponse(pred.tolist())
    return JSONResponse(data.tolist())
    # return data