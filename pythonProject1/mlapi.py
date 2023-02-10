import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd


app = FastAPI()

#send json file to api
class ScoringItem(BaseModel):
    #structure def
    YearsAtCompany: float
    EmployeeSatisfaction: float
    Position: str
    Salary: int

# rb as binary
with open('rfmodel.pkl', 'rb') as f:
    model = pickle.load(f)


    # pass through endpnt
@app.post("/")
    # whats called when we go to route
async def scoring_endpoint(item:ScoringItem):
    # specify columns for lightweight dataframe
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)

    return {"prediction":int(yhat)}

