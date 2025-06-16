import pandas as pd
import os
from scipy.stats import rankdata

# Lista de datasets e métodos
datasets = ['COIL20', 'AR10P', 'PIE10P', 'TOX171', 'Sonar', 'Wine', 'Ionosphere', 'WDBC']
caminho_base = 'results/'
metodos = ["MF_M", "MF_V", "VCSDFS", "FMIUFS", "SRCFS", "LSCAE"]
metricas = ["ari", "nmi", "sillhouette", "db"]

# Dicionário para armazenar resultados por métrica
tabelas_resultado = {métrica: {} for métrica in metricas}

for dataset in datasets:
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
        linhas_metricas = []
        n_linhas = 9

        melhor_linha = None
        melhor_rank_total = float('inf')

        for i in range(n_linhas):
            valores = [dados_metodos[metrica][metodo][i] for metodo in metodos]
            # Se for 'db', rank em ordem crescente (minimizar). Senão, decrescente (maximizar)
            if metrica == "db":
                ranks = rankdata(pd.Series(valores), method='min')  # menor = melhor
            else:
                ranks = rankdata(-pd.Series(valores), method='min')  # maior = melhor

            rank_MF_M = ranks[metodos.index("MF_M")]
            rank_MF_V = ranks[metodos.index("MF_V")]
            soma_rank = rank_MF_M + rank_MF_V

            if soma_rank < melhor_rank_total:
                melhor_rank_total = soma_rank
                melhor_linha = (valores, ranks)

        # Formatar a linha com valores e rankings
        valores, ranks = melhor_linha
        linha_formatada = {
            metodo: f"{valores[i]:.4f} ({int(ranks[i])}º)"
            for i, metodo in enumerate(metodos)
        }
        tabelas_resultado[metrica][dataset] = linha_formatada

import re

# Criar DataFrames finais para cada métrica
for metrica, dados in tabelas_resultado.items():
    df_resultado = pd.DataFrame.from_dict(dados, orient='index')[metodos]

    # Extrair apenas os rankings e calcular média
    rankings_numericos = df_resultado.map(lambda x: int(re.search(r'\((\d+)º\)', x).group(1)))
    medias = rankings_numericos.mean().round(2)

    # Criar linha com as médias dos rankings
    linha_media = {metodo: f"Média: {medias[metodo]:.2f}" for metodo in metodos}
    df_resultado.loc["Média Ranking"] = linha_media

    print(f"\n=== Resultado para {metrica.upper()} ===")
    print(df_resultado)
