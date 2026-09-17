# Detection of AI-Generated Arabic Text: A Data Mining Approach

## objective
build a binary classifier to differentiate between human-written abstracts and AI-generated

## dataset
**KFUPM-JRCAI/arabic-generated-abstracts (Hugging Face)**
contains three subsets:
- by_polishing: (2851 samples) models polish existing human abstracts
- from_title: (2963 samples) free-form generation from titles only
- from_title_and_content: (2574 samples) content-aware generation using both title and paper content 

each sample contains the  original human abstracts and generated versions from 4 models :allam, jais, llama, and openai

## project structure:
'data/raw/' original untouched dataset(raw_data.csv)
'data/processed/' cleaned data from duplicates(clean_data.csv)
'notebooks/' dataloading.ipynb (Google Colab)
'doc/' documentation
'requirements.txt' python packages 

## setup
pip install -r requirements.txt

## status
phase 1: project setup and acquisition [x]
phase 2: data preprocessing and EDA [ ]
phase 3: feature engineering [ ]
phase 4: modele building, training and evaluation [ ]
phase 5: analysis, interpretation and reporting

