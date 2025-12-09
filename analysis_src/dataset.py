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

def preprocess_age(adata):
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
            if age < 0 or age > 120:
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
            age_category.append('0-1')
        elif age < 18:
            age_category.append('1-18')
        elif age < 30:
            age_category.append('18-30')
        elif age < 50:
            age_category.append('30-50')
        elif age < 70:
            age_category.append('50-70')
        else:
            age_category.append('70+')
    
    adata.obs['age_category'] = age_category
    
    return adata