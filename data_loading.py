import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=";", encoding='latin1')
        return df
    except UnicodeDecodeError as e:
        msg = f"Fail to import CSV, err: {str(msg)}"
        print(msg)
        return None
