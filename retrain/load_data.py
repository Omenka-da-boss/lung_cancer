import pandas as pd
import os

ref_df = pd.read_csv("data\\lung_cancer_dataset.csv")
current_df = pd.read_csv("monitoring\\new_data.csv")


def load_data():
    if os.path.exists("monitoring\\new_data.csv"):
        df = pd.concat([ref_df,current_df],axis=0,ignore_index=True)
        return df
    
    else:
        return ref_df