# Volatility Features
import pandas as pd
import numpy as np

# --- Teorização e Modelagem de Variáveis de Volatilidade ---

def realized_volatility(df: pd.DataFrame, window: int = 21, price_col: str = 'close') -> pd.Series:
    """
    Volatilidade realizada (desvio padrão dos retornos logarítmicos).
    window: janela em dias úteis (ex: 21 para 1 mês)
    """
    log_ret = np.log(df[price_col]).diff()
    return log_ret.rolling(window).std() * np.sqrt(252)  # anualizada

def parkinson_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Volatilidade de Parkinson (usa high/low intra-dia, mais eficiente que close-close).
    """
    hl = np.log(df['high'] / df['low'])
    parkinson = hl.rolling(window).apply(lambda x: (x**2).sum() / (4 * window * np.log(2)), raw=True)
    return np.sqrt(parkinson) * np.sqrt(252)

def garman_klass_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Volatilidade de Garman-Klass (usa open, high, low, close).
    """
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    gk = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    gk_vol = gk.rolling(window).mean()
    return np.sqrt(gk_vol) * np.sqrt(252)

def close_to_close_volatility(df: pd.DataFrame, window: int = 21, price_col: str = 'close') -> pd.Series:
    """
    Volatilidade tradicional: desvio padrão dos retornos de fechamento.
    """
    returns = df[price_col].pct_change()
    return returns.rolling(window).std() * np.sqrt(252)

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR) - volatilidade absoluta, não percentual.
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

# Exemplo de uso (supondo DataFrame OHLCV):
# df['vol_realizada'] = realized_volatility(df)
# df['vol_parkinson'] = parkinson_volatility(df)
# df['vol_gk'] = garman_klass_volatility(df)
# df['vol_close2close'] = close_to_close_volatility(df)
# df['atr'] = atr(df)

# Outras ideias:
# - Volatilidade histórica de retornos de volume
# - Volatilidade de retornos intraday (se disponível)
# - Volatilidade rolling de spreads (high-low, open-close)
# - Volatilidade de múltiplos ativos (cross-sectional)
# - Volatilidade implícita (se disponível via opções)