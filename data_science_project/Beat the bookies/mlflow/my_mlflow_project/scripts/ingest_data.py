import pandas as pd

def ingest_data():
    # Ingest raw data
    data = pd.read_csv('/Users/christianhellum/Cand. Merc./Data-Science-Project/data_science_project/Beat the bookies/data/df_preprocessed.csv')
    data.to_csv('/Users/christianhellum/Cand. Merc./Data-Science-Project/data_science_project/Beat the bookies/data/df_preprocessed.csv', index=False)

if __name__ == "__main__":
    ingest_data()
# Placeholder for ingest_data.py
