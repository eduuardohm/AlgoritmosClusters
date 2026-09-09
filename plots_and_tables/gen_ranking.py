import pandas as pd
import re
import os
from scipy.stats import rankdata

# Lista de datasets e métodos
datasets = ['AR10P', 'COIL20', 'Heart-Statlog', 'Ionosphere', 'Lymphography', 'Madelon', 'PIE10P', 'Scene', 'Sonar', 'TOX171', 'WDBC', 'Wine', 'Zoo']
caminho_base = 'results/'
metodos = ["BASELINE", "LS", "MCFS", "UDFS", "DUFS", "VCSDFS", "FMIUFS", "SRCFS", "MF_M", "MF_V"]
metricas = ["sillhouette", "db"]

# Dicionário para armazenar resultados por métrica
tabelas_resultado = {métrica: {} for métrica in metricas}

for dataset in datasets:
    colored_dataset_name = colored_rank = f"\033[92m{dataset}\033[0m"
    print(f'\nDataset: {colored_dataset_name}')
    dados_metodos = {}

    for metodo in metodos:
        arquivo = os.path.join(caminho_base, dataset, f"{metodo}.csv")
        df = pd.read_csv(arquivo)
        for metrica in metricas:
            # Armazena os valores da métrica como uma lista por linha
            if metrica not in dados_metodos:
                dados_metodos[metrica] = {}
            dados_metodos[metrica][metodo] = df[metrica].values

    for metrica in metricas:
        print('\n' + metrica.upper())
        
        n_linhas = 9
        all_ranks = []

        for i in range(n_linhas):
            valores = [dados_metodos[metrica][metodo][i] for metodo in metodos]
            # Se for 'db', rank em ordem crescente (minimizar). Senão, decrescente (maximizar)
            if metrica == "db":
                ranks = rankdata(pd.Series(valores), method='average')  # menor = melhor
            else:
                ranks = rankdata(-pd.Series(valores), method='average')  # maior = melhor

            all_ranks.append(ranks)

        for j, ranks in enumerate(all_ranks):
            print(f'0.{j + 1}:', end=' ')
            for i, rank in enumerate(ranks):
                rank = int(rank) if rank.is_integer() else rank
                if rank == 1:
                    colored_rank = f"\033[94m{rank}\033[0m"
                elif rank == 10:
                    colored_rank = f"\033[91m{rank}\033[0m"
                else:    
                    colored_rank = f"\033[93m{rank}\033[0m"
                
                if rank == 10:
                    print(f"{colored_rank} ({metodos[i]})", end=' ')
                else:
                    print(f"{colored_rank}  ({metodos[i]})", end=' ')
            print('')