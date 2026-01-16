import pandas as pd
from pathlib import Path


def process_data(raw_dir=None, output_dir=None):
    """Clean and combine all CSV files from raw/eod directory."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    if raw_dir is None:
        raw_dir = BASE_DIR / 'data' / 'raw' / 'eod'
    if output_dir is None:
        output_dir = BASE_DIR / 'data' / 'processed'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(Path(raw_dir).glob('*.csv'))
    if not csv_files:
        return None
    
    all_dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['ticker'] = file.stem
        all_dataframes.append(df)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df = combined_df.drop_duplicates().dropna(how='all')
    
    output_file = output_dir / "data_combined.csv"
    combined_df.to_csv(output_file, index=False)
    
    return combined_df

if __name__ == "__main__":
    process_data()