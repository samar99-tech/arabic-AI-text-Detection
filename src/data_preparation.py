import pandas as pd
from datasets import load_dataset
import re
from pyarabic import araby
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import arabicstopwords.arabicstopwords as ast
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

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
  
def get_summary(df):
  summary = df.groupby("label")[["count_words" , "average_word_length", "average_sentence_length" , "type_token_ratio"]].mean()
  summary.index = ["Human (0)", "AI-generated (1)"]
  return summary
  
def top_ngrams(texts , n=2, top=10):
  counts = Counter()
  for t in texts:
    words = t.split()
    for i in range(len(words) - n +1):
      counts[" ".join(words[i:i+n])] +=1
  return counts.most_common(top)   

# f13:
def hapax_ratio(t):
    words = str(t).split()
    if len(words) == 0:
        return 0
    counts = Counter(words)
    hapax = 0
    for w in counts:
        if counts[w] == 1:
            hapax = hapax + 1
    return hapax / len(words)

# f35:
def count_paragraphs(t):
    lines = str(t).split("\n")
    paragraphs = [p for p in lines if p.strip() != ""]
    return len(paragraphs)

# f57: 
def load_language_model(name="aubmindlab/aragpt2-base"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return tok, model, device

def top50_count(t, tok, model, device):
    import torch
    ids = tok(str(t), return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    if ids.shape[1] < 2:
        return 0
    with torch.no_grad():
        scores = model(ids).logits[0, :-1]
    real = ids[0, 1:]
    real_scores = scores.gather(1, real.unsqueeze(1))
    ranks = (scores > real_scores).sum(1)
    return int((ranks < 50).sum())

# f79:
def split_sentences(t):
    t = str(t).replace("!", ".").replace("؟", ".").replace("?", ".")
    return [s for s in t.split(".") if s.strip() != ""]

def burstiness(t):
    lengths = [len(s.split()) for s in split_sentences(t)]
    if len(lengths) < 2:
        return 0
    return np.std(lengths) / np.mean(lengths)

# 
def diacritic_ratio(t):
    t = str(t)
    if len(t) == 0:
        return 0
    diacritized = len(re.findall(r"[\u0621-\u064A][\u064B-\u0652]", t))
    return diacritized / len(t)

def add_features(df):
    df["f13_hapax"] = df["text_normalized"].apply(hapax_ratio)
    df["f35_paragraphs"] = df["text"].apply(count_paragraphs)
    tok, model, device = load_language_model()
    df["f57_count"] = df["text"].apply(lambda t: top50_count(t, tok, model, device))
    df["f79_burstiness"] = df["text"].apply(burstiness)
    df["f101_diacritics"] = df["text"].apply(diacritic_ratio)
    return df


def split_data(df):
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=42)
    return train, val, test
                                                  
if __name__ == "__main__":
  ds = load_data()
  df = reshape_data(ds)
  print("raw: ", df.shape)
  check_quality(df)
  df = remove_duplicate(df)
  print("after removing duplicates: ", df.shape)
  df = preprocess(df)
  df = add_statistics(df)
  print(get_summary(df).round(3))
  print(top_ngrams(df[df["label"] == 1]["text_no_stopwords"],2,10))
  df = add_features(df)
  train, val, test = split_data(df)
  print("train:", train.shape, "val:", val.shape, "test:", test.shape)
  





 
