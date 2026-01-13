"""Coleta dados EOD (OHLCV) diários dos últimos 3 meses."""

from pathlib import Path
import yaml
from scripts.brapi_client import fetch_daily_history, BrapiError

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yml"

# Config
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

assets = config["market"]["assets"]
output_dir = Path(config["data"]["path_raw"]) / "eod"
output_dir.mkdir(parents=True, exist_ok=True)

# Coleta EOD
for asset in assets:
    try:
        df = fetch_daily_history(asset)
        df.to_csv(output_dir / f"{asset}.csv")
        print(f"{asset}: {len(df)} registros")
    except BrapiError as e:
        print(f"{asset}: {e}")
