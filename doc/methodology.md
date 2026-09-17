# methodology 
## data acquisition and initial exploration
- download the KFUPM-JRCAI/arabic-generated-abstracts dataset using the Hugging Face 'datasets' library in Google Colab
- inspected structure: 3 subsets (by_polishing: 2851 / from_title: 2963 / from_title_and_content: 2574 rows), each with five columns
- combined all subsets into one wide-format dataframe (8388 rows)
- reshaped to long format: one text per row with binary (human = 0, AI = 1) and a source column.
- exported as data/processed/clean_data.csv
