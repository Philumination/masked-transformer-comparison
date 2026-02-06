from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from analysis_src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()


import pandas as pd
import numpy as np

def preprocess_data(adata):
    # preprocess age column
    
    age_numeric = []
    
    for age_val in adata.obs['age']:
        if pd.isna(age_val):
            age_numeric.append(np.nan)
            continue
        
        age_str = str(age_val).strip().lower()
        
        # Skip invalid entries
        if any(x in age_str for x in ['not', 'unspecified', 'restricted', 'unknown']):
            age_numeric.append(np.nan)
            continue
        
        # Handle negative
        if age_str.startswith('-'):
            age_numeric.append(np.nan)
            continue
        
        # Handle '>=100'
        if age_str.startswith('>='):
            age_numeric.append(100)
            continue


        # Handle ranges like '85-89' -> 87
        if '-' in age_str and age_str[0].isdigit():
            parts = age_str.split('-')
            try:
                age_numeric.append((float(parts[0]) + float(parts[1])) / 2)
                continue
            except:
                age_numeric.append(np.nan)
                continue
        
        # Handle weeks like '52 weeks' -> 1 year
        if 'week' in age_str:
            try:
                weeks = float(age_str.split()[0])
                age_numeric.append(weeks / 52)
                continue
            except:
                age_numeric.append(np.nan)
                continue
        
        # Handle days like '29 days' -> 0.08 years
        if 'day' in age_str:
            try:
                days = float(age_str.split()[0])
                age_numeric.append(days / 365)
                continue
            except:
                age_numeric.append(np.nan)
                continue
        
        # Handle months
        if 'month' in age_str:
            try:
                months = float(age_str.split()[0])
                age_numeric.append(months / 12)
                continue
            except:
                age_numeric.append(np.nan)
                continue
        
        # Handle 'year' suffix
        if 'year' in age_str:
            try:
                age_numeric.append(float(age_str.split()[0]))
                continue
            except:
                age_numeric.append(np.nan)
                continue
        
        # Direct conversion for numbers
        try:
            age = float(age_str)
            if age < 18 or age > 120: 
                age_numeric.append(np.nan)
            else:
                age_numeric.append(age)
        except:
            age_numeric.append(np.nan)
    
    # Save to adata
    adata.obs['age_numeric'] = age_numeric
    
    # Create categories
    age_category = []
    for age in adata.obs['age_numeric']:
        if pd.isna(age):
            age_category.append('Unknown')
        elif age < 1:  
            age_category.append('0-1') #  should no longer exist
        elif age < 18:
            age_category.append('1-18') # should also no longer exist
        elif age < 30:
            age_category.append('18-30')
        elif age < 50:
            age_category.append('30-50')
        elif age < 70:
            age_category.append('50-70')
        else:
            age_category.append('70+')
    
    adata.obs['age_category'] = age_category

    # preprocess sex column
    sex_cleaned = []
    
    for sval in adata.obs["sex"]:
        if pd.isna(sval):
            sex_cleaned.append(np.nan)
            continue
        
        if sval == "female" or sval == "famale":
            sex_cleaned.append("female")
        elif sval == "male":
            sex_cleaned.append("male")
        else:
            sex_cleaned.append(np.nan)  # only female/male for 2 classes classification
    
    adata.obs['sex_cleaned'] = sex_cleaned

    # preprocess bmi category
    bmi_cat_cleaned = []
    bmi_keep = {"underweight", "normal", "overweight", "obese"}
    bmi_unknown = {"", "unspecified", "labcontrol test", "not collected"}

    for bmi_val in adata.obs["bmi_cat"]:
        if pd.isna(bmi_val) or bmi_val in bmi_unknown:
            bmi_cat_cleaned.append("unknown")
        elif bmi_val in bmi_keep:
            bmi_cat_cleaned.append(bmi_val)
        else:
            bmi_cat_cleaned.append("unknown")

    adata.obs["bmi_cat_cleaned"] = bmi_cat_cleaned
    
    
    # preprocess IBD column (exact values observed)
    ibd_cleaned = []
    ibd_yes = {
        "colitis",
        "crohns",
        "diagnosed by a medical professional (doctor, physician assistant)",
        "self-diagnosed",
        "diagnosed by an alternative medicine practitioner",
        "yes.ibs"
    }
    ibd_no = {
        "i do not have this condition",
        "no",
    }
    ibd_unknown = {"", "unspecified", "not provided", "not collected"}

    for ibd_val in adata.obs["IBD"]:
        if pd.isna(ibd_val) or ibd_val in ibd_unknown:
            ibd_cleaned.append("unknown")
        elif ibd_val in ibd_no:
            ibd_cleaned.append("no")
        elif ibd_val in ibd_yes:
            ibd_cleaned.append("yes")
        else:
            ibd_cleaned.append("unknown")

    adata.obs["ibd_cleaned"] = ibd_cleaned
    
    
    # preprocess diabetes
    diabetes_cleaned = []
    diab_yes = {
        "diabetic",
        "yes.type.i",
        "diagnosed by a medical professional (doctor, physician assistant)",
        "self-diagnosed",
        "diagnosed by an alternative medicine practitioner",
        "prediabetic",  # maybe wrong here
    }
    diab_no = {
        "i do not have this condition",
        "no",
        "normoglycemic", # means normal blood sugar
    }
    diab_unknown = {
        "", "unspecified", "not provided", "not collected",
        "1.0", "2.0", "3.0", "4.0", "5.0", "6.0",  # codes for diabets type ??
    }

    for d_val in adata.obs["diabetes"]:
        if pd.isna(d_val) or d_val in diab_unknown:
            diabetes_cleaned.append("unknown")
        elif d_val in diab_no:
            diabetes_cleaned.append("no")
        elif d_val in diab_yes:
            diabetes_cleaned.append("yes")
        else:
            diabetes_cleaned.append("unknown")

    adata.obs["diabetes_cleaned"] = diabetes_cleaned

    from skbio.stats.composition import clr
    # Clr transformed Counts
    adata.layers["Clrs"] = clr(adata.X.toarray() + 1)  # Add pseudocount for zeros

    

    return adata