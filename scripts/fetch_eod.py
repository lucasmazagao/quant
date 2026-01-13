"""Coleta dados EOD (OHLCV) diários dos últimos 3 meses."""

from pathlib import Path
import yaml
from brapi_client import fetch_daily_history, BrapiError

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yml"

# Config
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

assets = config["market"]["assets"]
output_dir = Path(config["data"]["path_raw"]) / "eod"
output_dir.mkdir(parents=True, exist_ok=True)

# Coleta EOD (1d, 3mo - fixo)
print(f"Coletando EOD para {len(assets)} ativos (1d, 3mo)...")
for asset in assets:
    try:
        df = fetch_daily_history(asset)
        df.to_parquet(output_dir / f"{asset}.parquet")
        print(f"✓ {asset}: {len(df)} registros")
    except BrapiError as e:
        print(f"✗ {asset}: {e}")
