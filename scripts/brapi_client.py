"""Cliente BRAPI - Dados de Mercado da B3
Coleta dados diários (1d) dos últimos 3 meses (3mo) apenas.
"""

import requests
import pandas as pd
import time
from typing import Dict, List
from pathlib import Path

# Configurações
TOKEN = "8tV76NTpdFUEE3zqLwa1Ne"
BASE_URL = "https://brapi.dev/api"
USER_AGENT = "quant-pipeline/1.0"
TIMEOUT = 15
RETRIES = 3
BACKOFF = 0.5

# Exceções
class BrapiError(Exception): pass
class BrapiAuthError(BrapiError): pass
class BrapiDataError(BrapiError): pass

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def _request(path: str, params: Dict, retries: int = RETRIES, 
             backoff: float = BACKOFF) -> Dict:
    """Faz requisição HTTP com retry automático."""
    headers = {"User-Agent": USER_AGENT}
    params = {**params, "token": TOKEN}
    url = f"{BASE_URL}{path}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            
            if response.status_code == 401:
                raise BrapiAuthError(f"Token inválido: {response.text[:100]}")
            elif response.status_code == 429:
                raise BrapiError("Rate limit excedido")
            elif response.status_code >= 400:
                raise BrapiError(f"HTTP {response.status_code}: {response.text[:100]}")
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise BrapiError(f"Falha após {retries} tentativas: {e}")
            time.sleep(backoff * (attempt + 1))

    raise BrapiError("Erro na requisição")


# =============================================================================
# FUNÇÕES PRINCIPAIS DE COLETA
# =============================================================================

def fetch_daily_history(ticker: str) -> pd.DataFrame:
    """Busca histórico de preços diários (1d, 3mo) de um ativo."""
    try:
        data = _request(f"/quote/{ticker}", {"range": "3mo", "interval": "1d"})
        
        results = data.get("results", [])
        if not results:
            raise BrapiDataError(f"Sem resultados para {ticker}")
        
        historical = results[0].get("historicalDataPrice", [])
        if not historical:
            raise BrapiDataError(f"Sem dados históricos para {ticker}")
        
        df = pd.DataFrame(historical)
        df = _normalize_df(df)
        
        if df.empty:
            raise BrapiDataError(f"DataFrame vazio para {ticker}")
        
        return df
        
    except (BrapiError, BrapiAuthError):
        raise
    except Exception as e:
        raise BrapiDataError(f"Erro ao processar {ticker}: {e}")

def fetch_multiple_assets(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Busca histórico diário (1d, 3mo) de múltiplos ativos."""
    results = {}
    
    for ticker in tickers:
        try:
            results[ticker] = fetch_daily_history(ticker)
            time.sleep(0.2)  # Delay entre requisições
        except Exception as e:
            print(f"Erro ao coletar {ticker}: {e}")
            
    return results

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza DataFrame: processa datas, renomeia colunas, mantém OHLCV."""
    if df.empty:
        return df
    
    # Processar datas
    if "date" in df.columns:
        df["date"] = (pd.to_datetime(df["date"], unit="s", utc=True)
                      .dt.tz_convert("America/Sao_Paulo")
                      .dt.tz_localize(None))
        df = df.sort_values("date").set_index("date")
    
    # Manter apenas OHLCV
    ohlcv = ["open", "high", "low", "close", "volume"]
    available = [col for col in ohlcv if col in df.columns]
    
    if not available:
        raise BrapiDataError("Nenhuma coluna OHLCV encontrada")
    
    return df[available]

def save_to_parquet(dataframes: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Salva DataFrames em arquivos Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ticker, df in dataframes.items():
        try:
            df.to_parquet(output_dir / f"{ticker}.parquet")
        except Exception as e:
            print(f"Erro ao salvar {ticker}: {e}")

def load_from_parquet(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """Carrega DataFrames de arquivos Parquet."""
    dataframes = {}
    
    for parquet_file in input_dir.glob("*.parquet"):
        try:
            dataframes[parquet_file.stem] = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"Erro ao carregar {parquet_file.stem}: {e}")
    
    return dataframes

