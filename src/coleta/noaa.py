"""
Modulo para coleta de dados NOAA ONI (Oceanic Nino Index)
Fonte: Climate Prediction Center (CPC/NOAA)
Dados sao originalmente TRIMESTRAIS (DJF, MAM, JJA, SON)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


def coletar_noaa_oni(
    force_atualizacao: bool = True,
    raw_dir: Path = None,
    processed_dir: Path = None,
    anos_filtro: List[int] = None,
) -> pd.DataFrame:
    """
    Coleta dados do indice ONI (Oceanic Nino Index) da NOAA
    Mantem os dados no formato TRIMESTRAL original
    
    Args:
        force_atualizacao: Se True, recarrega do arquivo original
        raw_dir: Diretorio com dados brutos
        processed_dir: Diretorio para salvar dados processados
        anos_filtro: Lista de anos para filtrar
        
    Returns:
        DataFrame com dados ONI trimestrais
    """
    # Configurar diretorios
    if raw_dir is None:
        raw_dir = Path("data/coletados/noaa")
    if processed_dir is None:
        processed_dir = Path("data/processed/noaa")
    
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    # Definir periodo
    if anos_filtro:
        ano_min = min(anos_filtro)
        ano_max = max(anos_filtro)
    else:
        ano_min = 1950
        ano_max = datetime.now().year
    
    # Nome do arquivo de cache
    cache_file = processed_dir / f"noaa_oni_trimestral_{ano_min}_{ano_max}.csv"
    
    # Verificar cache
    if not force_atualizacao and cache_file.exists():
        logger.info(f"Carregando do cache: {cache_file}")
        df = pd.read_csv(cache_file)
        if anos_filtro:
            df = df[df['ano'].isin(anos_filtro)]
        return df
    
    logger.info("Carregando dados NOAA ONI do arquivo local...")
    
    # Arquivo original TXT
    arquivo_txt = raw_dir / "oni.ascii.txt"
    
    if not arquivo_txt.exists():
        logger.error(f"Arquivo nao encontrado: {arquivo_txt}")
        return pd.DataFrame()
    
    dados = []
    
    with open(arquivo_txt, 'r') as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith('SEAS'):
                continue
            
            # Formato: "  DJF 1950  24.72  -1.53"
            partes = linha.split()
            if len(partes) >= 4:
                try:
                    trimestre = partes[0]  # DJF, JFM, etc.
                    ano = int(partes[1])
                    # Coluna ANOM e a quarta (indice 3)
                    oni = float(partes[3])
                    
                    dados.append({
                        'ano': ano,
                        'trimestre': trimestre,
                        'oni_index': oni
                    })
                except Exception as e:
                    continue
    
    if not dados:
        logger.error("Nao foi possivel processar os dados do NOAA")
        return pd.DataFrame()
    
    df = pd.DataFrame(dados)
    
    # Classificar fases ENOS
    def classificar_fase(oni):
        if pd.isna(oni):
            return 'Neutro'
        elif oni >= 0.5:
            return 'El Nino'
        elif oni <= -0.5:
            return 'La Nina'
        else:
            return 'Neutro'
    
    df['fase_enos'] = df['oni_index'].apply(classificar_fase)
    
    # Ordenar
    df = df.sort_values(['ano', 'trimestre']).reset_index(drop=True)
    
    # Filtrar por periodo
    if anos_filtro:
        df = df[df['ano'].isin(anos_filtro)]
    
    # Estatisticas
    logger.info(f"Dados coletados (TRIMESTRAIS):")
    logger.info(f"   Total de registros: {len(df)}")
    logger.info(f"   Periodo: {df['ano'].min()} a {df['ano'].max()}")
    logger.info(f"   Trimestres: {df['trimestre'].unique()}")
    logger.info(f"   ONI - Media: {df['oni_index'].mean():.2f}, Min: {df['oni_index'].min():.2f}, Max: {df['oni_index'].max():.2f}")
    logger.info(f"   Fases ENOS: {df['fase_enos'].value_counts().to_dict()}")
    
    # Salvar cache
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(cache_file, index=False, encoding='utf-8')
    logger.info(f"Cache salvo em: {cache_file}")
    
    return df


if __name__ == "__main__":
    # Teste
    df = coletar_noaa_oni(anos_filtro=[2020, 2021, 2022])
    print(f"\nResultado: {len(df)} registros")
    print(df.head(10))