from fastapi import FastAPI
import fasttext
import pythainlp
from pythainlp import word_tokenize

app = FastAPI()

model = fasttext.load_model("sentiment_quan.model")

db = {}

@app.get("/")
async def read_root():
  return {"message": "Test"}

@app.get("/predict/{text}/{userid}")
async def predict(text : str,userid : str):
  collect ={
    userid : text
  }
  db.update(collect)
  text_to = ' '.join(word_tokenize(db[userid]))
  print('text_token >', text_to)
  result = model.predict(text_to)
  if result[0][0] == "__label__neg":
    predict = "negative"
  elif result[0][0] == "__label__pos":
    predict = "positive"
  probability = int(result[1][0]*100)
  sum = "it's "+predict+" "+str(probability)+"%"
  return sum
