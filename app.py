import io
from typing import Optional

import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from inference import load_predictor, Predictor

predictor: Optional[Predictor] = None
templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = load_predictor()
    yield


app = FastAPI(title="Fraud Detector", lifespan=lifespan)


class Transaction(BaseModel):
    step:           int   = Field(..., ge=1)
    type:           str   = Field(..., pattern="^(CASH_IN|CASH_OUT|DEBIT|PAYMENT|TRANSFER)$")
    amount:         float = Field(..., ge=0)
    oldbalanceOrg:  float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(tx: Transaction):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predictor.predict(tx.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail="Inference error")


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not (file.filename or "").endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    required = set(Transaction.model_fields.keys())
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV file.")
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=400,
                            detail=f"Missing columns: {sorted(missing)}")
    valid_types = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    invalid_types = set(df["type"].dropna().unique()) - valid_types
    if invalid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid transaction types: {sorted(invalid_types)}"
        )
    try:
        df_input = df[list(required)]
        preds    = predictor.predict_batch(df_input)
        records  = df_input[sorted(required)].to_dict(orient="records")
        for rec, pred in zip(records, preds):
            rec.update(pred)
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail="Batch inference error")
