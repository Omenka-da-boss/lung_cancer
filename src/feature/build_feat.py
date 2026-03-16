import pandas as pd
from sklearn.preprocessing import LabelEncoder

encode = LabelEncoder()


def _maping_func__(s: pd.Series) -> pd.Series:
    
    vals = list(pd.Series(s.dropna().unique().astype(str)))
    valset = set(vals)
    
    if valset == {"Yes","No"}:
        return s.map({"Yes":1,"No":0})
    
    if valset == {"Male","Female"}:
        return s.map({"Male":1,"Female":0})
    
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]:1,sorted_vals[1]:1}
        return s.astype(str).map(mapping).astype(int)
    
    if len(vals) ==  2:
        sort = sorted(vals)
        mapp = {sort[0]:0,sort[1]:1,sort[2]:2}
        return s.astype(str).map(mapp).astype(int)  
    
    return s

def build_feature(df: pd.DataFrame,target_cols:str = "lung_cancer") -> pd.DataFrame:
    
    obj_cols = [c for c in df.select_dtypes(include='object').columns if c != target_cols]
    
    bin_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    
    print(f"   🔢 Binary features: {len(bin_cols)} | Multi-category features: {len(multi_cols)}")
    if bin_cols:
        print(f"      Binary: {bin_cols}")
    if multi_cols:
        print(f"      Multi-category: {multi_cols}")
        
    for c in bin_cols:
        original = df[c].dtype
        df[c] = df[c].map(_maping_func__(df[c].astype(str)))
        
        print(f"      ✅ {c}: {original} → binary (0/1)")
    
    bool_cols = [c for c in df.select_dtypes(include='bool').columns]
    
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   🔄 Converted {len(bool_cols)} boolean columns to int: {bool_cols}")
    
    for c in multi_cols:
        print("Before Encoding ")
        print(df[c].value_counts())
        df[c] = encode.fit_transform(df[c])
        print("After Encoding ")
        print(df[c].value_counts())
    
    return df