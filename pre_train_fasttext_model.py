import fasttext
import pythainlp
from pythainlp import word_tokenize

listall = []
with open('/content/NIDA-DATA/news20k.txt', 'r', encoding = 'utf8') as f:
    [listall.append(line.strip()) for line in f]

tokenized_list = [' '.join(word_tokenize(line.strip())) for line in listall]
f = open('news20k_tokenized.txt','w')
for t in tokenized_list:
  f.write(t+'\n')
f.close()

#train word vector in 50 dimentions

w2v = fasttext.train_unsupervised(
    'news20k_tokenized.txt', 
    dim=50, min_count=5, word_ngrams=3
    )
words = w2v.get_words()
len(words)

#save model to file

words = w2v.get_words()

with open('50d.vec', 'w') as f:
    f.write(f'{len(words)} {w2v.get_dimension()}\n')
    for w in words:
        vec    = w2v.get_word_vector(w)
        vecstr = ' '.join([ f'{v:4f}' for v in vec ])
        try:
            f.write(f'{w} {vecstr}\n')
        except:
            pass

w2v.get_nearest_neighbors('ไข้หวัด')

w2v.get_nearest_neighbors('ป่วย')

import pandas as pd

"""### Train fastText for Topic Segmentation model

datapath = '/content/topic.xlsx'
pretrain_path = 'NIDA-DATA/fasttext50d.vec'
target = 'data' #change from column 'data' to 'preprocess_data'
label = 'topic'

df = pd.read_excel(datapath) #DataFrame

df['topic'].value_counts().plot.bar()

#preprocess data 
text = df[target].tolist()
tokens = [word_tokenize(row) for row in text] #ตัดคำโดยใช้ newmm 
tokens = [' '.join(row) for row in tokens] # เปลี่ยนให้อยู่ใน format ของ fasttext

def name_label(t):
  return '__label__' + str(t)

df['preprocess_data'] = tokens
df['label'] = df.topic.apply(name_label)
df

#write data to trainset

with open("NIDA-DATA/topic.train", "w") as f:
    for lb, item in zip(df['label'], df['preprocess_data'].tolist()):
        f.write(f'{lb} {item}\n')
f.close()

model = fasttext.train_supervised('NIDA-DATA/topic.train', epoch=100, dim=50, 
                                  wordNgrams=2,ws=5,lr=0.01, 
                                  pretrainedVectors=pretrain_path)

text = 'อยากช่วยชีวิตคนโดยใช้การแพทย์'

text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'ชอบสร้างเทคโนโลยีเจ๋งๆ'

text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'ชอบสร้างเทคโนโลยีเจ๋งๆทางการแพทย์ เพื่อนำไปช่วยชีวิตคน'

text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

model.save_model('NIDA-DATA/topic.model')
"""
"""---

### Train fastText for Sentiment Analysis model
"""

datapath = '/content/sentiment.xlsx'
pretrain_path = '/content/NIDA-DATA/fasttext50d.vec'
target = 'data' #change from column 'data' to 'preprocess_data'
label = 'sentiment'

df = pd.read_excel(datapath) #DataFrame

print(df['sentiment'].value_counts())
df['sentiment'].value_counts().plot.bar()

#preprocess data 
text = df[target].tolist()
tokens = [word_tokenize(row) for row in text] #ตัดคำโดยใช้ newmm 
tokens = [' '.join(row) for row in tokens] # เปลี่ยนให้อยู่ใน format ของ fasttext

def name_label(t):
  return '__label__' + str(t)

df['preprocess_data'] = tokens
df['label'] = df.sentiment.apply(name_label)

#write data to trainset

with open("NIDA-DATA/sentiment.train", "w") as f:
    for lb, item in zip(df['label'], df['preprocess_data'].tolist()):
        f.write(f'{lb} {item}\n')
f.close()

model = fasttext.train_supervised('NIDA-DATA/sentiment.train', epoch=100, dim=50, 
                                  wordNgrams=2,ws=3,lr=0.01, 
                                  pretrainedVectors=pretrain_path)

text = 'ชอบการสร้างเทคโนโลยี และออกแบบงานต่างๆ'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'การออกแบบและสร้างเทคโนโลยีช่างน่าสนุกจริงๆ แถมเวลาเขียนโค๊ตก็รู้สึกสดใสมาก'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'ไม่ชอบเวลาทำงานอื่นนอกจากคอมพิวเตอร์'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'ไม่อยากเข้าใกล้การเขียนโค๊ตเลย'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'การสร้างเทคโนโลยีคือหายนะชัดๆ'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'สามารถนั่งโปรแกรมมิ่งยาวๆทั้งวันได้โดยไม่เบื่อ'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'ฉันเชื่อว่าเทคโนโลยีจะเป็นสิ่งที่สร้างความเปลี่ยนแปลงในโลก'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'ฉันสนใจที่จะเรียนรู้เกี่ยวกับเทคโนโลยี และจะนำมาใช้ในการแก้ไขปัญหาในสังคม'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'การแก้ error คือฝันร้ายในยามกลางวันชัดๆ'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'คอมพิวเตอร์มันง่ายเกินจนไม่ชอบเลย'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

"""7/10"""

text = 'เกลียดโปรแกรมมิ่ง'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'งานสายเทคโนโลยีไปต่อยาก'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'งานสายเทคโนโลยีสามารถนำไปต่อยอดและสารารถสิ่งต่างๆได้'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'งานคอมพิวเตอร์นั้นดูไม่ใช่แนวฉันเลย'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'สายคอมพิวเตอร์สิ่งที่ฉันใฝ่ฝันมานาน'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

"""12/15"""

text = 'ถ้าได้มาเรียนสายคอมพิวเตอร์ ชีวิตนี้จะดีมาก'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'เหนื่อยกับการเขียนโปรแกรมเหลือเกิน'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'เขียนโค๊ตคือนรกมาก'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'รักการสร้างเทคโนโลยี'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

text = 'ชอบทำงานสายคอมพิวเตอร์'
text_to = ' '.join(word_tokenize(text))
print('text_token >', text_to)
model.predict(text_to)

"""16/20"""

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

model.save_model('NIDA-DATA/sentiment_quan.model')

"""send assignment to data.analytics.aiml+nida@gmail.com"""