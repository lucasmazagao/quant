"""Funções de dimensionamento de posição."""
from __future__ import annotations
import pandas as pd


def volatility_target(returns: pd.Series, target_vol: float = 0.15, lookback: int = 20) -> pd.Series:
    realized = returns.rolling(lookback).std() * (252 ** 0.5)
    weight = target_vol / realized
    return weight.clip(upper=5)  # limita alavancagem
