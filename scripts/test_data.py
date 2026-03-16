# Testing the data features pipeline
# Load -. Preprocess -> Feature Engineering
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath("src"))


from src.load.load import load_data
from src.load.preprocess import preprocess
from src.feature.build_feat import build_feature




path = "./data/lung_cancer_dataset.csv"

def main(path):
    
    # Loading of dataset
    print("\n Loading Dataset...")
    df = load_data(path)
    print("Succeccfully Loaded Dataset✅✅")
    print(df.head())
    
    
    # Display information about dataset
    print("\n Columns In Dataset")
    print(df.columns)
    print("\n Basic Information About Dataset")
    print(df.info())
    print("\n Description Of Dataset")
    print(df.describe(include='all'))
    print("\n Datatypes of each columns")
    print(df.dtypes)
    
    # preprocessing Of Dataset
    
    print("\n Preprocessing Dataset ...")
    
    df_clean = preprocess(df,target_col="lung_cancer")
    
    print("Dataset Successfully Preprocessed ✅✅")
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df.head())
    
    # Feature Engineering
    print("\n Feature Engineering")
    
    df_feat = build_feature(df_clean,target_cols="lung_cancer")
    print(f"Data after feature engineering. Shape: {df_feat.shape}")
    print(df_feat.head(3))
    
    print("\n✅ Phase 1 pipeline completed successfully!")
    return df_feat
    

if __name__ == "__main__":
    main(path)