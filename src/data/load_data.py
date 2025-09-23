import os
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")
OUT_SAMPLE = Path("data/processed/sample.csv")

def main():
    """
    Load and process telco customer churn data.
    
    Reads the raw telco customer churn data, displays basic information,
    and saves a sample to the processed directory.
    """
    try:
        # Check if raw data file exists
        if not RAW_PATH.exists():
            print(f"❌ ERROR: Raw data file not found at {RAW_PATH}")
            print(f"   Please ensure 'Telco-Customer-Churn.csv' is placed in the data/raw/ directory")
            return
        
        print(f"📂 Loading data from {RAW_PATH}")
        
        # Load the data
        df = pd.read_csv(RAW_PATH)
        
        # Display dataset information
        print(f"\n📊 Dataset Information:")
        print(f"   Shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")
        
        print(f"\n📋 Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        print(f"\n🔍 Missing Values Count:")
        missing_counts = df.isnull().sum()
        if missing_counts.sum() == 0:
            print("   No missing values found! ✅")
        else:
            for col, count in missing_counts.items():
                if count > 0:
                    percentage = (count / len(df)) * 100
                    print(f"   {col}: {count} ({percentage:.2f}%)")
        
        # Create output directory if it doesn't exist
        OUT_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        
        # Save first 100 rows as sample
        sample_df = df.head(100)
        sample_df.to_csv(OUT_SAMPLE, index=False)
        print(f"\n💾 Saved first 100 rows to {OUT_SAMPLE}")
        print(f"   Sample shape: {sample_df.shape}")
        
    except FileNotFoundError:
        print(f"❌ ERROR: File not found - {RAW_PATH}")
        print(f"   Please check if the file exists and the path is correct")
    except pd.errors.EmptyDataError:
        print(f"❌ ERROR: The file {RAW_PATH} is empty")
    except pd.errors.ParserError as e:
        print(f"❌ ERROR: Failed to parse CSV file - {e}")
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred - {e}")

if __name__ == "__main__":
    main()
