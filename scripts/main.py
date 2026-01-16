"""Orquestra a execução completa do pipeline:
1. Coleta de dados (EOD)
2. Padronização e limpeza
3. Feature engineering
4. Labeling
5. Model training
6. Signal generation
7. Risk management
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Criar diretório de logs
Path('logs').mkdir(exist_ok=True)

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"

    print(log_msg)
    with open('logs/pipeline.log', 'a') as f:
        f.write(log_msg + '\n')

def run_script(script: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, f"scripts/{script}"],
            check=True,
            capture_output=True,
            text=True
        )
        log(f"{script} concluído")
        return True
    except subprocess.CalledProcessError as e:
        log(f"{script} falhou: {e.stderr[:200]}")
        return False

def main():
    steps = [
        ("Coleta EOD", "fetch_eod.py"),
        ("Padronização e Limpeza", "clean_data.py"),
        # ("Feature Engineering", "features.py"),
        # ("Labeling", "labeling.py"),
        # ("Training", "train_model.py"),
    ]

    for i, (name, script) in enumerate(steps, 1):
        log(f"\n[Etapa {i}/{len(steps)}] {name}")

        log(f"Executando {script}")
        
        if not run_script(script):
            log(f"Pipeline interrompido em: {name}")
            sys.exit(1)
        else:
            log(f"Etapa concluída: {name}")

if __name__ == "__main__":
    main()