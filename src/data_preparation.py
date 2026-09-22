import pandas as pd
from datasets import load_dataset

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

ds = load_data()
df = reshape_data(ds)
print ("raw: ", df.shape)
check_quality(df)
 

 
