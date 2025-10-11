import pandas as pd

def count_nulls(data):
 
    if isinstance(data, pd.DataFrame):
        null_counts = data.isnull().sum()
        print("Null values per column:")
        for col, count in null_counts.items():
            print(f"{col}: {count}")
    elif isinstance(data, pd.Series):
        count = data.isnull().sum()
        print(f"Null values in Series: {count}")
    else:
        print("Input must be a pandas DataFrame or Series.")

def load_dataframe(file_path):
   
 
    if file_path.lower().endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")
    

def xlsx_to_csv(xlsx_path, csv_path):
    """
    Converts an XLSX file to a CSV file.

    Parameters:
        xlsx_path (str): Path to the input XLSX file.
        csv_path (str): Path to the output CSV file.

    Returns:
        None
    """
    df = pd.read_excel(xlsx_path)
    df.to_csv(csv_path, index=False)
    print(f"Converted {xlsx_path} to {csv_path}")