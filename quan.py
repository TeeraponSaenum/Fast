import fasttext
import pythainlp
from pythainlp import word_tokenize

model = fasttext.load_model("NIDA-DATA/sentiment.model")

model.quantize(input=None,
                  qout=False,
                  cutoff=0,
                  retrain=False,
                  epoch=None,
                  lr=None,
                  thread=None,
                  verbose=None,
                  dsub=2,
                  qnorm=False,
                 )
model.save_model('sentiment_quan.model')
quan = fasttext.load_model("sentiment_quan.model")

text = 'ชอบทำงานสายคอมพิวเตอร์'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
quan.predict(text_to)
