import pandas as pd
from datasets import load_dataset

def load_data():
  ds = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")
  return ds

def reshape_data (ds):
  subsets = ["by_polishing", "from_title", "from_title_and_content"]
  models =["allam", "jais", "llama", "openai"]
  rows =[]
  for split in subsets: #to go through the 3 subsets
    for sample in ds[split]: #to scan every row
      if sample.get("original_abstract"): # if it is original abstarct -> human label = 0
        rows.append({
            "text": sample["original_abstract"],
            "label":0,
            "source": "human",
            "split": split
      })
      for model in models: #to go through the 4 models
        col = f"{model}_generated_abstract"
        if sample.get(col): # if the col = any of 4 models -> ai-generated label = 1
          rows.append({        
            "text": sample[col],
            "label":1,
            "source":model,
            "split": split
        })
  return pd.DataFrame(rows)

ds = load_data()
df = reshape_data(ds)
print ("raw: ", df.shape)
 

 
