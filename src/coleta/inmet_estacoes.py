"""
Módulo para coleta e processamento de dados das estações do INMET.
"""
import pandas as pd
import numpy as np
import os
import glob
from typing import Tuple, Optional, Dict
import logging
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ler_arquivo_inmet(caminho_arquivo: str) -> Optional[pd.DataFrame]:
    """
    Le arquivo CSV do INMET com formato brasileiro corretamente.
    """
    logger.info(f"Lendo: {os.path.basename(caminho_arquivo)}")
    
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            linhas = f.readlines()
        
        cabecalho_idx = None
        for i, linha in enumerate(linhas):
            if 'Data Medicao' in linha:
                cabecalho_idx = i
                break
        
        if cabecalho_idx is None:
            return None
        
        df = pd.read_csv(
            caminho_arquivo,
            sep=';',
            decimal=',',
            encoding='utf-8',
            skiprows=cabecalho_idx,
            na_values=['null', 'NULL', '', ' ', 'NaN', 'None'],
            on_bad_lines='skip'
        )
        
        df.columns = df.columns.str.strip()
        
        if 'Data Medicao' in df.columns:
            df['Data Medicao'] = pd.to_datetime(df['Data Medicao'], errors='coerce')
            df = df.dropna(subset=['Data Medicao'])
        
        if len(df) == 0:
            return None
        
        # Extrai metadados
        metadados = {}
        for linha in linhas[:cabecalho_idx]:
            if ':' in linha:
                key, value = linha.split(':', 1)
                metadados[key.strip()] = value.strip()
        
        df.attrs['codigo'] = metadados.get('Codigo Estacao', 'N/A')
        df.attrs['nome'] = metadados.get('Nome', 'N/A')
        df.attrs['latitude'] = metadados.get('Latitude', 'N/A')
        df.attrs['longitude'] = metadados.get('Longitude', 'N/A')
        df.attrs['altitude'] = metadados.get('Altitude', 'N/A')
        
        return df
        
    except Exception as e:
        logger.error(f"Erro: {e}")
        return None


def processar_estacao(caminho_arquivo: str) -> Optional[pd.DataFrame]:
    """Processa os dados de uma estacao."""
    df = ler_arquivo_inmet(caminho_arquivo)
    
    if df is None:
        return None
    
    mapeamento = {
        'Data Medicao': 'data',
        'PRECIPITACAO TOTAL, MENSAL(mm)': 'precipitacao_mm',
        'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)': 'precipitacao_mm',
        'TEMPERATURA MAXIMA MEDIA, MENSAL(°C)': 'temp_max_c',
        'TEMPERATURA MEDIA COMPENSADA, MENSAL(°C)': 'temp_media_c',
        'TEMPERATURA MEDIA, MENSAL (AUT)(°C)': 'temp_media_c',
        'TEMPERATURA MINIMA MEDIA, MENSAL(°C)': 'temp_min_c',
        'UMIDADE RELATIVA DO AR, MEDIA MENSAL(%)': 'umidade_percent',
        'PRESSAO ATMOSFERICA, MEDIA MENSAL(mB)': 'pressao_mb',
        'VENTO, VELOCIDADE MEDIA MENSAL(m/s)': 'vento_medio_ms'
    }
    
    colunas_renomear = {k: v for k, v in mapeamento.items() if k in df.columns}
    df = df.rename(columns=colunas_renomear)
    
    df['codigo_estacao'] = df.attrs.get('codigo', 'N/A')
    df['nome_estacao'] = df.attrs.get('nome', 'N/A')
    
    try:
        df['latitude'] = float(df.attrs.get('latitude', '0')) if df.attrs.get('latitude', '0') != 'N/A' else np.nan
        df['longitude'] = float(df.attrs.get('longitude', '0')) if df.attrs.get('longitude', '0') != 'N/A' else np.nan
        df['altitude'] = float(df.attrs.get('altitude', '0')) if df.attrs.get('altitude', '0') != 'N/A' else np.nan
    except:
        df['latitude'] = df['longitude'] = df['altitude'] = np.nan
    
    df['ano'] = df['data'].dt.year
    df['mes'] = df['data'].dt.month
    
    return df


def processar_todas_estacoes(
    dados_brutos_dir: str = "data/coletados/inmet/rs/estacoes",
    dados_processados_dir: str = "data/processed",
    anos_filtro: Optional[Tuple[int, int]] = None,
    salvar: bool = True
) -> pd.DataFrame:
    """
    Processa todas as estacoes e retorna um DataFrame combinado.
    
    Args:
        dados_brutos_dir: Diretorio com os arquivos CSV
        dados_processados_dir: Diretorio para salvar os dados processados
        anos_filtro: Tupla com (ano_inicio, ano_fim) para filtrar
        salvar: Se True, salva os dados processados
        
    Returns:
        DataFrame combinado com dados de todas as estacoes
    """
    print(f"\n{'='*60}")
    print(f"PROCESSANDO ESTACOES DO INMET")
    print(f"{'='*60}")
    print(f"Diretorio: {dados_brutos_dir}")
    
    if anos_filtro:
        ano_inicio, ano_fim = anos_filtro
        print(f"Periodo filtrado: {ano_inicio} a {ano_fim}")
    else:
        ano_inicio, ano_fim = None, None
        print(f"Periodo: todos os anos")
    
    if not os.path.exists(dados_brutos_dir):
        print(f"AVISO: Diretorio nao encontrado")
        return pd.DataFrame()
    
    arquivos = glob.glob(os.path.join(dados_brutos_dir, "dados_*.csv"))
    arquivos = [f for f in arquivos if 'Catalogo' not in f]
    
    if not arquivos:
        print(f"AVISO: Nenhum arquivo encontrado")
        return pd.DataFrame()
    
    print(f"Encontrados {len(arquivos)} arquivos")
    
    dados_combinados = []
    
    for arquivo in arquivos:
        df = processar_estacao(arquivo)
        
        if df is not None and len(df) > 0:
            if anos_filtro:
                df = df[(df['ano'] >= ano_inicio) & (df['ano'] <= ano_fim)]
            
            if len(df) > 0:
                dados_combinados.append(df)
                codigo = df['codigo_estacao'].iloc[0]
                nome = df['nome_estacao'].iloc[0]
                print(f"   OK: {codigo} - {nome}: {len(df)} registros ({df['ano'].min()}-{df['ano'].max()})")
    
    if not dados_combinados:
        print("AVISO: Nenhum dado coletado")
        return pd.DataFrame()
    
    df_combinado = pd.concat(dados_combinados, ignore_index=True)
    
    # Renomeia para compatibilidade
    df_combinado = df_combinado.rename(columns={
        'codigo_estacao': 'estacao_codigo',
        'nome_estacao': 'estacao_nome'
    })
    
    if salvar:
        os.makedirs(dados_processados_dir, exist_ok=True)
        caminho = os.path.join(dados_processados_dir, "inmet_estacoes_combinado.csv")
        df_combinado.to_csv(caminho, index=False, encoding='utf-8-sig')
        print(f"\nDados salvos: {caminho}")
    
    print(f"\nTotal: {df_combinado['estacao_codigo'].nunique()} estacoes, {len(df_combinado)} registros")
    print(f"Periodo: {df_combinado['ano'].min()} a {df_combinado['ano'].max()}")
    
    return df_combinado


if __name__ == "__main__":
    df = processar_todas_estacoes()
    print(f"\nDataFrame resultante: {df.shape}")