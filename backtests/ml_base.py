"""
Ensemble ML Predictor - Adaptado para Ações Brasileiras (B3)
Baseado na estratégia Jane Street com LightGBM + XGBoost + CatBoost

Diferenças do código original:
1. Features técnicas ao invés de features_00-78
2. Prediz retornos futuros ao invés de responder_6
3. Dados diários (EOD) ao invés de intraday
4. Ações brasileiras ao invés de dados Jane Street
"""

import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt

from pathlib import Path
from typing import List, Dict, Tuple


class EnsembleStockPredictor:
    """
    Sistema de predição usando ensemble de 3 algoritmos de gradient boosting.
    
    Arquitetura:
    - 5 folds de cross-validation temporal
    - 3 modelos por fold (LGB, XGB, CatBoost) = 15 modelos totais
    - Previsão final = média das 15 previsões
    """
    
    def __init__(
        self, 
        n_folds: int = 5,
        use_gpu: bool = True,
        model_dir: str = './models/ensemble'
    ):
        self.n_folds = n_folds
        self.use_gpu = use_gpu
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = []
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features técnicas para cada ação.
        
        Features criadas:
        - Retornos: 1d, 5d, 21d (mensal)
        - Volatilidade: rolling std
        - Médias móveis: SMA 5, 21, 50
        - RSI (Relative Strength Index)
        - Volume relativo
        """
        df = df.copy()
        
        # Features de preço
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_21d'] = df['close'].pct_change(21)
        
        # Volatilidade
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_21d'] = df['return_1d'].rolling(21).std()
        
        # Médias móveis
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_21'] = df['close'].rolling(21).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Distância das médias (normalizadas)
        df['distance_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
        df['distance_sma21'] = (df['close'] - df['sma_21']) / df['sma_21']
        
        # RSI (14 períodos)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['volume_sma_21'] = df['volume'].rolling(21).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_21']
        
        # High-Low range (volatilidade intraday)
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Target: retorno futuro em 5 dias
        df['target'] = df['close'].pct_change(5).shift(-5)
        
        return df
    
    def reduce_mem_usage(self, df: pd.DataFrame, float16_as32: bool = True) -> pd.DataFrame:
        """
        Otimiza uso de memória (idêntico ao código Jane Street).
        """
        start_mem = df.memory_usage().sum() / 1024**2
        print(f'💾 Memória inicial: {start_mem:.2f} MB')
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object and str(col_type) != 'category':
                c_min, c_max = df[col].min(), df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32 if float16_as32 else np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        
        end_mem = df.memory_usage().sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f'✅ Memória otimizada: {end_mem:.2f} MB (redução de {reduction:.1f}%)')
        
        return df
    
    def prepare_data(
        self, 
        data_path: str = './data/processed',
        skip_days: int = 100,
        valid_days: int = 60
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Carrega e prepara dados para treinamento.
        
        Args:
            data_path: Caminho para dados processados
            skip_days: Dias iniciais a pular (warm-up para features)
            valid_days: Dias para validação
        
        Returns:
            df: DataFrame completo com features
            train_dates: Lista de datas de treino
            valid_dates: Lista de datas de validação
        """
        # Você precisará criar este arquivo agregado
        # Veja função load_all_stocks() abaixo
        df = pd.read_parquet(f'{data_path}/all_stocks_features.parquet')
        
        print(f"📊 Dataset: {len(df):,} linhas, {len(df.columns)} colunas")
        
        # Remove NaN das features (warm-up period)
        df = df.dropna()
        df = self.reduce_mem_usage(df)
        
        # Divide por data
        unique_dates = sorted(df['date'].unique())
        
        if len(unique_dates) < skip_days + valid_days:
            raise ValueError(f"Dados insuficientes! Precisa de pelo menos {skip_days + valid_days} dias.")
        
        valid_dates = unique_dates[-(valid_days):]
        train_dates = unique_dates[skip_days:-valid_days]
        
        print(f"📅 Período de treino: {train_dates[0]} a {train_dates[-1]} ({len(train_dates)} dias)")
        print(f"📅 Período de validação: {valid_dates[0]} a {valid_dates[-1]} ({len(valid_dates)} dias)")
        
        return df, train_dates, valid_dates
    
    def get_model_dict(self) -> Dict:
        """
        Define modelos com hiperparâmetros (similar ao Jane Street).
        
        Adaptações:
        - Menos estimators (dados diários vs intraday)
        - Learning rates ajustados para menor volume de dados
        """
        device_lgb = 'gpu' if self.use_gpu else 'cpu'
        device_xgb = 'cuda' if self.use_gpu else 'cpu'
        task_cbt = 'GPU' if self.use_gpu else 'CPU'
        
        return {
            'lgb': lgb.LGBMRegressor(
                n_estimators=300,  # Menos que Jane Street (500)
                learning_rate=0.05,
                max_depth=5,
                device=device_lgb,
                objective='l2',
                random_state=42
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=1000,  # Menos que Jane Street (2000)
                learning_rate=0.05,
                max_depth=5,
                tree_method='hist',
                device=device_xgb,
                objective='reg:squarederror',
                random_state=42
            ),
            'cbt': cbt.CatBoostRegressor(
                iterations=500,  # Menos que Jane Street (1000)
                learning_rate=0.03,
                depth=5,
                task_type=task_cbt,
                loss_function='RMSE',
                random_state=42,
                verbose=False
            )
        }
    
    def r2_weighted(self, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
        """
        R² ponderado (métrica Jane Street adaptada).
        """
        numerator = np.average((y_pred - y_true) ** 2, weights=weights)
        denominator = np.average((y_true) ** 2, weights=weights) + 1e-38
        return 1 - numerator / denominator
    
    def train_fold(
        self,
        df: pd.DataFrame,
        train_dates: List[str],
        valid_dates: List[str],
        fold_idx: int,
        model_name: str,
        model_dict: Dict
    ):
        """
        Treina um modelo para um fold específico.
        """
        # Seleciona datas do fold (pula 1 a cada N_fold)
        selected_dates = [date for i, date in enumerate(train_dates) if i % self.n_folds != fold_idx]
        
        # Prepara dados de treino
        train_mask = df['date'].isin(selected_dates)
        X_train = df.loc[train_mask, self.feature_names]
        y_train = df.loc[train_mask, 'target']
        
        # Pesos uniformes (pode adicionar lógica customizada)
        w_train = np.ones(len(y_train))
        
        # Prepara dados de validação
        valid_mask = df['date'].isin(valid_dates)
        X_valid = df.loc[valid_mask, self.feature_names]
        y_valid = df.loc[valid_mask, 'target']
        w_valid = np.ones(len(y_valid))
        
        # Obtém modelo
        model = model_dict[model_name]
        
        print(f"\n🔄 Treinando {model_name.upper()} - Fold {fold_idx}")
        print(f"   Treino: {len(X_train):,} amostras | Validação: {len(X_valid):,} amostras")
        
        # Treina baseado no tipo
        if model_name == 'lgb':
            model.fit(
                X_train, y_train, sample_weight=w_train,
                eval_set=[(X_valid, y_valid)],
                eval_sample_weight=[w_valid],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(20)
                ]
            )
        elif model_name == 'cbt':
            evalset = cbt.Pool(X_valid, y_valid, weight=w_valid)
            model.fit(
                X_train, y_train, sample_weight=w_train,
                eval_set=[evalset],
                early_stopping_rounds=50,
                verbose=20
            )
        else:  # xgb
            model.fit(
                X_train, y_train, sample_weight=w_train,
                eval_set=[(X_valid, y_valid)],
                sample_weight_eval_set=[w_valid],
                early_stopping_rounds=50,
                verbose=20
            )
        
        # Avalia
        y_pred = model.predict(X_valid)
        r2 = self.r2_weighted(y_valid.values, y_pred, w_valid)
        print(f"   ✅ R² = {r2:.4f}")
        
        # Salva
        model_path = self.model_dir / f'{model_name}_fold{fold_idx}.pkl'
        joblib.dump(model, model_path)
        print(f"   💾 Salvo em {model_path}")
        
        self.models.append(model)
        
        return model
    
    def train_all(self, df: pd.DataFrame, train_dates: List[str], valid_dates: List[str]):
        """
        Treina todos os modelos (15 no total: 5 folds × 3 algoritmos).
        """
        # Define features (exclui colunas não-feature)
        exclude_cols = ['date', 'ticker', 'target', 'open', 'high', 'low', 'close', 'volume']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\n🎯 Features selecionadas ({len(self.feature_names)}): {self.feature_names[:5]}...")
        
        model_dict = self.get_model_dict()
        
        print(f"\n🚀 Iniciando treinamento de {self.n_folds * 3} modelos...\n")
        
        for fold_idx in range(self.n_folds):
            print(f"\n{'='*60}")
            print(f"📁 FOLD {fold_idx + 1}/{self.n_folds}")
            print(f"{'='*60}")
            
            for model_name in ['lgb', 'xgb', 'cbt']:
                self.train_fold(df, train_dates, valid_dates, fold_idx, model_name, model_dict)
        
        print(f"\n✅ Treinamento completo! {len(self.models)} modelos prontos.")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz previsão usando ensemble (média dos 15 modelos).
        """
        if not self.models:
            raise ValueError("Nenhum modelo treinado! Execute train_all() primeiro.")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X[self.feature_names])
            predictions.append(pred)
        
        # Média das previsões
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def load_models(self):
        """
        Carrega modelos salvos.
        """
        self.models = []
        
        for fold_idx in range(self.n_folds):
            for model_name in ['lgb', 'xgb', 'cbt']:
                model_path = self.model_dir / f'{model_name}_fold{fold_idx}.pkl'
                if model_path.exists():
                    model = joblib.load(model_path)
                    self.models.append(model)
                else:
                    raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        print(f"✅ {len(self.models)} modelos carregados de {self.model_dir}")


def load_all_stocks(data_dir: str = './data/raw/eod') -> pd.DataFrame:
    """
    Carrega todos os CSVs de ações e cria um DataFrame unificado.
    
    Assume formato CSV com colunas: Date, Open, High, Low, Close, Volume
    """
    data_path = Path(data_dir)
    all_files = list(data_path.glob('*.csv'))
    
    print(f"📁 Encontrados {len(all_files)} arquivos CSV em {data_dir}")
    
    dfs = []
    
    for file in all_files:
        ticker = file.stem  # Nome do arquivo sem extensão
        
        try:
            df = pd.read_csv(file, parse_dates=['Date'])
            df.columns = df.columns.str.lower()
            df['ticker'] = ticker
            df = df.rename(columns={'date': 'date'})
            
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Erro ao carregar {file}: {e}")
    
    # Concatena tudo
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"✅ Dataset consolidado: {len(df_all):,} linhas, {df_all['ticker'].nunique()} tickers")
    
    return df_all


# Exemplo de uso
if __name__ == "__main__":
    print("🎯 Ensemble Stock Predictor - Baseado em Jane Street\n")
    
    # 1. Carrega dados
    print("ETAPA 1: Carregando dados...")
    df_raw = load_all_stocks('./data/raw/eod')
    
    # 2. Cria features
    print("\nETAPA 2: Criando features técnicas...")
    predictor = EnsembleStockPredictor(n_folds=5, use_gpu=False)
    
    # Aplica features por ticker
    df_list = []
    for ticker in df_raw['ticker'].unique():
        df_ticker = df_raw[df_raw['ticker'] == ticker].copy()
        df_ticker = predictor.create_features(df_ticker)
        df_list.append(df_ticker)
    
    df_features = pd.concat(df_list, ignore_index=True)
    
    # Salva para não precisar recalcular
    os.makedirs('./data/processed', exist_ok=True)
    df_features.to_parquet('./data/processed/all_stocks_features.parquet')
    print("💾 Features salvas em ./data/processed/all_stocks_features.parquet")
    
    # 3. Prepara dados
    print("\nETAPA 3: Preparando dados de treino/validação...")
    df, train_dates, valid_dates = predictor.prepare_data()
    
    # 4. Treina
    print("\nETAPA 4: Treinamento do ensemble...")
    predictor.train_all(df, train_dates, valid_dates)
    
    print("\n🎉 Processo completo!")
