from fastapi import FastAPI
import fasttext
import pythainlp
from pythainlp import word_tokenize

app = FastAPI()

model = fasttext.load_model("sentiment_quan.model")

app = FastAPI()

@app.get("/")
async def read_root():
  return {"message": "Test"}

@app.get("/predict/{text}")
async def predict(text : str):
    text_to = ' '.join(word_tokenize(text))
    print('text_token >', text_to)
    result = model.predict(text_to)
    return result

