import pandas as pd
from datasets import load_dataset
import re
from pyarabic import araby
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import arabicstopwords.arabicstopwords as ast


def load_data():
  ds = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")
  return ds

def reshape_data (ds):
  models =["allam", "jais", "llama", "openai"]
  rows =[]
  for sample in ds["from_title_and_content"]: #to scan every row
    if sample.get("original_abstract"): # if it is original abstarct -> human label = 0
       rows.append({
          "text": sample["original_abstract"],
          "label":0,
          "source": "human"
      })
    for model in models: #to go through the 4 models
       col = f"{model}_generated_abstract"
       if sample.get(col): # if the col = any of 4 models -> ai-generated label = 1
          rows.append({
            "text": sample[col],
            "label":1,
            "source":model
        })
return pd.DataFrame(rows)
  
def check_quality(df):
  print("missing data: ",df.isnull().sum().sum())
  print("duplicates: ", df["text"].duplicated().sum())
  print("blanks: ", df["text"].str.strip().eq("").sum())
  print(pd.crosstab(df["source"], df["label"]))
  return df
  
#phase 2: preprocessing
def remove_duplicate(df):
  mask = df.duplicated(subset=["text"], keep=False)
  return df[~mask].reset_index(drop=True)
  
def normalize(t):
  t = araby.strip_tatweel(t)
  t = re.sub(r"[اآإأ]", "ا" ,t) #الالف
  t = re.sub(r"ى", "ي",t) #الف مقصورة
  t = re.sub(r"ؤ", "و" ,t) # واو همزة
  t = re.sub(r"ئ", "ي" ,t) # ياء همزة
  t = re.sub(r"[^\u0621-\u064A\s]", " " ,t) # non arabic
  t = re.sub(r"\s+", " " ,t).strip() #repeated spaces to one space
  return t

def bulid_stopwords(t):
  nltk_stop = set(normalize(w) for w in stopwords.words("arabic"))
  ast_stop = set(normalize(w) for w in ast.stopwords_list())
  return nltk_stop | ast_stop

def remove_stopwords(t, stop_set):
  return " ".join([w for w in t.split() if w not in stop_set])
  
stemmer = ISRIStemmer()

def stem_text(t):
  return " ".join([stemmer.stem(w) for w in tsplit()])

def preprocess(df):
  stop_set = bulid_stopwords(t)
  df["text_no_tashkeel"] = df["text"].apply(araby.strip_tashkeel)
  df["text_normalized"] = df["text_no_tashkeel"].apply(normalize)
  df["text_no_stopwords"] = df["text_normalized"].apply(lambda t: remove_stopwords(t, stop_set))
  df["text_stemmed"] = df["text_no_stopwords"].apply(stem_text)
  return df
# statistics

def count_words(t):
  return len(t.split())

def average_word_length(t):
  words = t.split()
  if len(words) == 0:
    return 0
  total = 0
  for w in words:
    total = total + len(w)
  return total /len(words)

def average_sentence_length(t):
  sentences = t.replace("!",".").replace("؟",".").split(".")
  sentences = [s for s in sentences if s.strip()]
  if len(sentences) == 0:
    return 0
  total = 0
  for s in sentences:
    total = total +len(s.split())
  return total / len(sentences)

def type_token_ratio(t):
  words = t.split()
  if  len(words) == 0:
    return 0
  unique_words = set(words)
  return len(unique_words) / len(words)

def add_statistics(df):
  df["count_words"] = df["text"].apply(count_words) # number of words
  df["average_word_length"] = df["text"].apply(average_word_length) #average number of letters: is the word long or short
  df["average_sentence_length"] = df["text"].apply(average_sentence_length)# average number of words: is the sentnce long or short
  df["type_token_ratio"] = df["text"].apply(type_token_ratio)
  return df

if __name__ == "__main__":
  ds = load_data()
  df = reshape_data(ds)
  print("raw: ", df.shape)
  check_quality(df)
  df = remove_duplicate(df)
  print("after removing duplicates: ", df.shape)
  df = preprocess(df)
  add_statistics(df)
  print(df.groupby("label")[["count_words" , "average_word_length", "average_sentence_length" , "type_token_ratio"]].mean().round(3))
  





 
