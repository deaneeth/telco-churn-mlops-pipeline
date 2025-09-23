import pandas as pd
import numpy as np
import json
from pathlib import Path

RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")
COLUMNS_OUTPUT = Path("data/processed/columns.json")

def main():
    """
    Perform exploratory data analysis on telco customer churn data.
    
    Loads the raw data, performs basic data cleaning, analyzes churn distribution,
    detects column types, and saves metadata for future use.
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
        print(f"   Dataset shape: {df.shape}")
        
        # Convert TotalCharges to numeric (some values might be strings)
        print(f"\n🔧 Data Cleaning:")
        original_type = df['TotalCharges'].dtype
        print(f"   TotalCharges original type: {original_type}")
        
        # Convert TotalCharges to numeric, coercing errors to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Check for any values that couldn't be converted
        nan_count = df['TotalCharges'].isna().sum()
        if nan_count > 0:
            print(f"   ⚠️  Found {nan_count} non-numeric values in TotalCharges (converted to NaN)")
        else:
            print(f"   ✅ All TotalCharges values converted successfully")
        
        print(f"   TotalCharges new type: {df['TotalCharges'].dtype}")
        
        # Analyze churn distribution
        print(f"\n📊 Churn Distribution:")
        churn_counts = df['Churn'].value_counts()
        churn_percentages = df['Churn'].value_counts(normalize=True) * 100
        
        for category in churn_counts.index:
            count = churn_counts[category]
            percentage = churn_percentages[category]
            print(f"   {category}: {count:,} ({percentage:.2f}%)")
        
        # Calculate churn rate
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        print(f"   Churn Rate: {churn_rate:.2f}%")
        
        # Detect numeric and categorical columns
        print(f"\n🔍 Column Type Detection:")
        
        # Exclude customerID and target variable from features
        feature_columns = [col for col in df.columns if col not in ['customerID', 'Churn']]
        
        numeric_columns = []
        categorical_columns = []
        
        for col in feature_columns:
            # Check if column is numeric (int or float)
            if df[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
        
        print(f"   Numeric columns ({len(numeric_columns)}):")
        for col in numeric_columns:
            print(f"      • {col} ({df[col].dtype})")
        
        print(f"   Categorical columns ({len(categorical_columns)}):")
        for col in categorical_columns:
            unique_count = df[col].nunique()
            print(f"      • {col} ({unique_count} unique values)")
        
        # Create metadata dictionary
        metadata = {
            "dataset_info": {
                "total_rows": int(df.shape[0]),
                "total_columns": int(df.shape[1]),
                "features_count": len(feature_columns),
                "target_column": "Churn"
            },
            "churn_distribution": {
                "Yes": int(churn_counts.get('Yes', 0)),
                "No": int(churn_counts.get('No', 0)),
                "churn_rate_percent": round(churn_rate, 2)
            },
            "columns": {
                "numeric": numeric_columns,
                "categorical": categorical_columns,
                "target": "Churn",
                "identifier": "customerID"
            },
            "data_quality": {
                "total_charges_na_count": int(nan_count),
                "total_charges_converted": True
            }
        }
        
        # Create output directory if it doesn't exist
        COLUMNS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata to JSON file
        with open(COLUMNS_OUTPUT, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Saved column metadata to {COLUMNS_OUTPUT}")
        print(f"   Summary: {len(numeric_columns)} numeric, {len(categorical_columns)} categorical columns")
        
    except FileNotFoundError:
        print(f"❌ ERROR: File not found - {RAW_PATH}")
    except pd.errors.EmptyDataError:
        print(f"❌ ERROR: The file {RAW_PATH} is empty")
    except pd.errors.ParserError as e:
        print(f"❌ ERROR: Failed to parse CSV file - {e}")
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred - {e}")

if __name__ == "__main__":
    main()