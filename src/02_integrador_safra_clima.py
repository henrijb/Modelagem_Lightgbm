"""
Integrador de dados de safra e clima - VERSAO TRIMESTRAL
Combina dados do SIDRA, NOAA ONI (trimestral) e INMET/ERA5-Land (mensal -> trimestral)
Autor: TCC
Data: 2025
MODIFICACOES:
- Suporte a coordenadas dos municipios (IBGE) quando --fonte-clima era5
- INMET so e usado quando --fonte-clima inmet
- Coordenadas dos municipios substituem coordenadas das estacoes para ERA5
"""
import pandas as pd
import numpy as np
import logging
import argparse
import re
import unicodedata
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Limiar de precipitacao mensal (mm) abaixo do qual o mes conta como seco.
LIMIAR_SECA_MM = 30

# Temperatura base (C) para o calculo de graus-dia de crescimento (GDD) da soja.
GDD_BASE_C = 10.0

def normalizar_nome(nome):
    texto = unicodedata.normalize('NFKD', str(nome)).upper()
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"['`]", '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def chave_nome(nome):
    return re.sub(r'[^A-Z0-9]+', '', normalizar_nome(nome))

def carregar_coordenadas_municipios(caminho_csv):
    """Carrega coordenadas dos municipios a partir do CSV gerado pelo 01_coleta.py."""
    df = pd.read_csv(caminho_csv)
    df['municipio'] = df['municipio'].apply(normalizar_nome)
    return df.set_index('municipio')

def carregar_era5_land(caminho, ano_inicial, ano_final):
    import xarray as xr
    dados = xr.open_dataset(caminho)
    nomes = set(dados.variables)
    nome_tempo = 'time' if 'time' in nomes else 'valid_time'
    nome_lat = 'latitude' if 'latitude' in nomes else 'lat'
    nome_lon = 'longitude' if 'longitude' in nomes else 'lon'
    
    if not {'t2m', 'tp'}.issubset(nomes):
        raise ValueError("NetCDF ERA5-Land nao possui as variaveis t2m e tp.")
    
    return dados, nome_tempo, nome_lat, nome_lon

def obter_clima_era5(dados, nome_tempo, nome_lat, nome_lon, latitude, longitude, ano, trimestre):
    meses = {
        'SON': [9, 10, 11],
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
    }
    
    # CORRECAO CRITICA: DJF da safra 2022 usa Dez/2021, Jan/2022, Fev/2022
    ano_clima = ano - 1 if trimestre == 'SON' else ano
    
    selecao = dados.sel({
        nome_lat: latitude,
        nome_lon: longitude,
    }, method='nearest')
    
    periodo = selecao.where(
        (selecao[nome_tempo].dt.year == ano_clima)
        & selecao[nome_tempo].dt.month.isin(meses[trimestre]),
        drop=True,
    )
    
    if periodo.sizes.get(nome_tempo, 0) == 0:
        return np.nan, np.nan, ano_clima, 0, np.nan, np.nan, 0
    
    # No produto mensal do ERA5-Land, tp representa a acumulacao media
    # diaria do mes em metros. Convertemos para mm mensais antes da soma.
    precipitacao_mensal_mm = periodo['tp'] * periodo[nome_tempo].dt.days_in_month * 1000.0
    precipitacao = float(precipitacao_mensal_mm.sum().values)
    
    temperatura_mensal_c = periodo['t2m'].values - 273.15
    temperatura = float(temperatura_mensal_c.mean())
    
    # Estatisticas extras calculadas apenas com os meses ja baixados
    temp_desvio = float(np.std(temperatura_mensal_c)) if temperatura_mensal_c.size > 0 else np.nan
    temp_amplitude = (
        float(np.max(temperatura_mensal_c) - np.min(temperatura_mensal_c))
        if temperatura_mensal_c.size > 0 else np.nan
    )
    meses_seca = int((precipitacao_mensal_mm.values < LIMIAR_SECA_MM).sum())
    
    return (
        precipitacao, temperatura, ano_clima, periodo.sizes.get(nome_tempo, 0),
        temp_desvio, temp_amplitude, meses_seca
    )

def nome_municipio_estacao(nome):
    nome_base = re.split(r'\s*-\s*|\s*\(', str(nome), maxsplit=1)[0]
    return chave_nome(nome_base)

def carregar_era5_land_diario(diretorio, ano_inicial, ano_final):
    """Abre os arquivos diarios consolidados do ERA5-Land (temp min/max)"""
    import xarray as xr
    diretorio = Path(diretorio)
    sufixo = f"{ano_inicial - 1}_{ano_final}"
    
    arquivos = {
        't2m_min': diretorio / f"era5_land_diario_temp_min_{sufixo}.nc",
        't2m_max': diretorio / f"era5_land_diario_temp_max_{sufixo}.nc",
    }
    
    faltando = [str(caminho) for caminho in arquivos.values() if not caminho.exists()]
    if faltando:
        raise FileNotFoundError(
            "Arquivos do ERA5-Land diario nao encontrados: " + ', '.join(faltando)
        )
    
    variaveis_origem = {'t2m_min': 't2m', 't2m_max': 't2m'}
    conjuntos = {}
    nome_tempo = nome_lat = nome_lon = None
    
    for chave, caminho in arquivos.items():
        dados = xr.open_dataset(caminho)
        nomes = set(dados.variables)
        nome_tempo = nome_tempo or ('time' if 'time' in nomes else 'valid_time')
        nome_lat = nome_lat or ('latitude' if 'latitude' in nomes else 'lat')
        nome_lon = nome_lon or ('longitude' if 'longitude' in nomes else 'lon')
        
        variavel_origem = variaveis_origem[chave]
        if variavel_origem not in nomes:
            raise ValueError(f"Arquivo {caminho} nao possui a variavel {variavel_origem}.")
        
        conjuntos[chave] = dados[variavel_origem].rename(chave)
    
    combinado = xr.merge(conjuntos.values())
    return combinado, nome_tempo, nome_lat, nome_lon

def obter_estatisticas_diarias_era5(dados, nome_tempo, nome_lat, nome_lon, latitude, longitude, ano, trimestre):
    """Calcula GDD (graus-dia de crescimento) do trimestre a partir das
    series diarias de temperatura minima e maxima."""
    meses = {
        'SON': [9, 10, 11],
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
    }
    
    ano_clima = ano - 1 if trimestre == 'SON' else ano
    
    selecao = dados.sel({nome_lat: latitude, nome_lon: longitude}, method='nearest')
    periodo = selecao.where(
        (selecao[nome_tempo].dt.year == ano_clima)
        & selecao[nome_tempo].dt.month.isin(meses[trimestre]),
        drop=True,
    )
    
    dias_validos = int(periodo.sizes.get(nome_tempo, 0))
    if dias_validos == 0:
        return np.nan, 0
    
    t_min_c = periodo['t2m_min'].values - 273.15
    t_max_c = periodo['t2m_max'].values - 273.15
    
    # Calculo do GDD (base 10C)
    gdd_diario = np.clip((t_max_c + t_min_c) / 2.0 - GDD_BASE_C, a_min=0, a_max=None)
    gdd = float(np.nansum(gdd_diario))
    
    return gdd, dias_validos

def mapear_municipios_estacoes(df_sidra, df_inmet):
    cobertura = df_inmet.groupby('estacao_codigo').size().to_dict()
    candidatos = {}
    
    for _, estacao in df_inmet[['estacao_codigo', 'estacao_nome']].drop_duplicates().iterrows():
        municipio_estacao = nome_municipio_estacao(estacao['estacao_nome'])
        if municipio_estacao:
            candidatos.setdefault(municipio_estacao, []).append(estacao['estacao_codigo'])
    
    aliases = {
        'SANTANADOLIVRAMENTO': 'SANTANADOLIVRAMENTO',
        'SANTANADOLIVRAMENTO RS': 'SANTANADOLIVRAMENTO',
        'SANTAVITORIADOPALMAR': 'SANTAVITORIADOPALMAR',
        'SANTAVITORIADOPALMAR RS': 'SANTAVITORIADOPALMAR',
    }
    
    mapeamento = {}
    for municipio in df_sidra['municipio'].dropna().unique():
        municipio_normalizado = chave_nome(municipio)
        municipio_normalizado = aliases.get(municipio_normalizado, municipio_normalizado)
        opcoes = candidatos.get(municipio_normalizado, [])
        if opcoes:
            mapeamento[municipio_normalizado] = max(
                opcoes,
                key=lambda codigo: (cobertura.get(codigo, 0), str(codigo).isdigit())
            )
    
    df_sidra = df_sidra.copy()
    df_sidra['municipio'] = df_sidra['municipio'].map(normalizar_nome)
    df_sidra['municipio_norm'] = df_sidra['municipio'].map(chave_nome)
    df_sidra['estacao_codigo'] = df_sidra['municipio_norm'].map(mapeamento)
    
    municipios_sem_estacao = sorted(
        df_sidra.loc[df_sidra['estacao_codigo'].isna(), 'municipio'].unique()
    )
    if municipios_sem_estacao:
        logger.warning(
            "Municipios sem estacao INMET correspondente (%d): %s",
            len(municipios_sem_estacao),
            ', '.join(municipios_sem_estacao)
        )
    
    return df_sidra, mapeamento

def main(
    ano_inicial=1976,
    ano_final=2024,
    data_dir=Path("data/processed"),
    municipios=None,
    era5_file=None,
    fonte_clima="era5",
    min_anos_municipio=20,
    usar_era5_diario=False,
    split="aleatorio",
    target="desvio_pct",
    excluir_jja=True,
    excluir_ano=True,
):
    print("\n" + "=" * 60)
    print("INTEGRADOR SAFRA x CLIMA - VERSAO TRIMESTRAL (CORRIGIDO)")
    print("=" * 60)
    print(f"Split: {split}")
    print(f"Target: {target}")
    print(f"Excluir JJA: {excluir_jja}")
    print(f"Excluir 'ano': {excluir_ano}")
    
    data_dir = Path(data_dir)
    output_dir = data_dir / "integrado_trimestral"
    output_dir.mkdir(exist_ok=True)
    
    if fonte_clima == "era5":
        era5_file = Path(era5_file) if era5_file else (
            Path("data/coletados/era5_land") /
            f"era5_land_rs_{ano_inicial - 1}_{ano_final}.nc"
        )
        if not era5_file.exists():
            raise FileNotFoundError(f"Arquivo ERA5-Land nao encontrado: {era5_file}")
    
    usar_era5_diario = usar_era5_diario and fonte_clima == "era5"
    
    print(f"Fonte climatica selecionada: {fonte_clima.upper()}")
    if usar_era5_diario:
        print("GDD: calculado com ERA5-Land diario (seca via metrica mensal)")
    
    # ========================================================================
    # 1. CARREGAR DADOS SIDRA (Soja RS)
    # ========================================================================
    print("\n1. CARREGANDO DADOS SIDRA (Soja RS)")
    print("-" * 40)
    df_sidra = pd.read_csv(data_dir / "sidra" / "tabela1612_soja.csv")
    print(f"Registros SIDRA: {len(df_sidra)}")
    print(f"Municipios: {df_sidra['Municipio'].nunique()}")
    
    # Limpar nome do municipio
    df_sidra['municipio'] = df_sidra['Municipio'].str.replace(' (RS)', '', regex=False).str.strip()
    df_sidra['municipio'] = df_sidra['municipio'].map(normalizar_nome)
    
    if municipios:
        municipios_normalizados = {
            normalizar_nome(municipio) for municipio in municipios
        }
        df_sidra = df_sidra[
            df_sidra['municipio'].isin(municipios_normalizados)
        ].copy()
        print(f"Municipios solicitados: {sorted(municipios_normalizados)}")
    else:
        print("Municipios solicitados: todos os disponiveis no SIDRA")
    
    df_sidra['ano'] = df_sidra['Ano']
    df_sidra = df_sidra[df_sidra['ano'].between(ano_inicial, ano_final)].copy()
    
    produtividade = pd.to_numeric(df_sidra['Rendimento_Medio_kg_ha'], errors='coerce')
    invalidos = produtividade.isna() | (produtividade <= 0)
    if invalidos.any():
        logger.warning("SIDRA: removidos %d rendimentos ausentes ou invalidos.", invalidos.sum())
        df_sidra = df_sidra.loc[~invalidos].copy()
    
    duplicados = df_sidra.duplicated(['municipio', 'ano'], keep=False)
    if duplicados.any():
        duplicatas = df_sidra.loc[duplicados, ['municipio', 'ano']].drop_duplicates()
        raise ValueError(
            "SIDRA possui duplicidades para municipio/ano: "
            + duplicatas.to_dict('records').__repr__()
        )
    
    if df_sidra.empty:
        raise ValueError(
            f"SIDRA nao possui dados para os municipios selecionados "
            f"no periodo {ano_inicial}-{ano_final}."
        )
    
    # ========================================================================
    # 2. CARREGAR DADOS NOAA ONI (TRIMESTRAL)
    # ========================================================================
    print("\n2. CARREGANDO DADOS NOAA ONI (TRIMESTRAL)")
    print("-" * 40)
    noaa_file = data_dir / "noaa" / f"noaa_oni_trimestral_{ano_inicial - 1}_{ano_final}.csv"
    if not noaa_file.exists():
        logger.error(f"Arquivo NOAA trimestral nao encontrado: {noaa_file}")
        return
    df_noaa = pd.read_csv(noaa_file)
    df_noaa = df_noaa[df_noaa['ano'].between(ano_inicial - 1, ano_final)].copy()
    print(f"Registros NOAA: {len(df_noaa)}")
    print(f"Colunas: {df_noaa.columns.tolist()}")
    print(f"Trimestres: {df_noaa['trimestre'].unique()}")
    print(f"Periodo: {df_noaa['ano'].min()} a {df_noaa['ano'].max()}")
    
    trimestre_para_safra = {
        'SON': 0, 'OND': 1, 'NDJ': 2, 'DJF': 3, 'JFM': 4, 'FMA': 5,
        'MAM': 6, 'AMJ': 7, 'MJJ': 8, 'JJA': 9, 'JAS': 10, 'ASO': 11
    }
    df_noaa['ordem_safra'] = df_noaa['trimestre'].map(trimestre_para_safra)
    df_noaa = df_noaa.sort_values(['ano', 'ordem_safra']).reset_index(drop=True)
    
    # ========================================================================
    # 3. PREPARANDO DADOS CLIMÁTICOS (ERA5 ou INMET)
    # ========================================================================
    print("\n3. PREPARANDO DADOS CLIMÁTICOS")
    print("-" * 40)
    
    coordenadas_municipios = None
    coordenadas_estacoes = None
    df_inmet_trim = None
    
    if fonte_clima == "era5":
        # Carregar coordenadas dos municípios do CSV gerado
        caminho_coord = data_dir / "municipios_rs_coordenadas.csv"
        if not caminho_coord.exists():
            raise FileNotFoundError(
                f"CSV de coordenadas dos municípios não encontrado: {caminho_coord}\n"
                "Execute o 01_coleta.py primeiro ou verifique o caminho do shapefile."
            )
        coordenadas_municipios = carregar_coordenadas_municipios(caminho_coord)
        print(f"Coordenadas dos municípios carregadas: {len(coordenadas_municipios)} municípios")
    else:
        # Carregar e agregar INMET
        df_inmet = pd.read_csv(data_dir / "inmet_estacoes_combinado.csv")
        df_inmet['data'] = pd.to_datetime(df_inmet['data'])
        df_inmet['estacao_nome'] = df_inmet['estacao_nome'].map(normalizar_nome)
        df_inmet['ano'] = df_inmet['data'].dt.year
        df_inmet['mes'] = df_inmet['data'].dt.month
        
        def get_trimestre(mes):
            if mes in [9, 10, 11]: return 'SON'
            elif mes in [12, 1, 2]: return 'DJF'
            elif mes in [3, 4, 5]: return 'MAM'
            else: return 'JJA'
        
        df_inmet['trimestre'] = df_inmet['mes'].apply(get_trimestre)
        
        def ajustar_ano(row):
            if row['trimestre'] == 'DJF' and row['mes'] == 12:
                return row['ano'] + 1
            return row['ano']
        
        df_inmet['ano_trimestre'] = df_inmet.apply(ajustar_ano, axis=1)
        df_inmet['estacao_codigo'] = (
            df_inmet['estacao_codigo'].astype(str).str.replace(r'\.0$', '', regex=True)
        )
        
        coordenadas_estacoes = (
            df_inmet[['estacao_codigo', 'latitude', 'longitude']]
            .dropna()
            .drop_duplicates('estacao_codigo')
            .set_index('estacao_codigo')
        )
        
        df_inmet_trim = df_inmet.groupby(
            ['estacao_codigo', 'estacao_nome', 'ano_trimestre', 'trimestre']
        ).agg(
            precipitacao_mm=('precipitacao_mm', lambda valores: valores.sum(min_count=1)),
            temp_media_c=('temp_media_c', 'mean'),
            meses_validos=('mes', 'nunique'),
            meses_com_precipitacao=('precipitacao_mm', lambda valores: valores.notna().sum()),
            meses_com_temperatura=('temp_media_c', lambda valores: valores.notna().sum()),
            temp_desvio_c=('temp_media_c', 'std'),
            temp_min_mensal=('temp_media_c', 'min'),
            temp_max_mensal=('temp_media_c', 'max'),
            meses_seca=('precipitacao_mm', lambda valores: (valores < LIMIAR_SECA_MM).sum()),
        ).reset_index()
        
        df_inmet_trim['temp_amplitude_c'] = df_inmet_trim['temp_max_mensal'] - df_inmet_trim['temp_min_mensal']
        df_inmet_trim = df_inmet_trim.rename(columns={'ano_trimestre': 'ano'})
        df_inmet_trim = df_inmet_trim[
            df_inmet_trim['ano'].between(ano_inicial - 1, ano_final)
        ].copy()
        
        incompletos = df_inmet_trim[
            (df_inmet_trim['meses_com_precipitacao'] < 3)
            | (df_inmet_trim['meses_com_temperatura'] < 3)
        ]
        logger.warning(
            "INMET: %d trimestres possuem menos de tres meses climaticos validos.",
            len(incompletos)
        )
        print(f"Registros INMET trimestrais: {len(df_inmet_trim)}")
    
    # Carregar ERA5 (mensal e diario) se for o caso
    dados_era5 = nome_tempo = nome_lat = nome_lon = None
    if fonte_clima == "era5":
        dados_era5, nome_tempo, nome_lat, nome_lon = carregar_era5_land(
            era5_file, ano_inicial, ano_final
        )
    
    dados_era5_diario = nome_tempo_diario = nome_lat_diario = nome_lon_diario = None
    if usar_era5_diario:
        dados_era5_diario, nome_tempo_diario, nome_lat_diario, nome_lon_diario = carregar_era5_land_diario(
            Path("data/coletados/era5_land") / "diario", ano_inicial, ano_final
        )
    
    # ========================================================================
    # 4. PREPARANDO MUNICÍPIOS
    # ========================================================================
    print("\n4. PREPARANDO MUNICÍPIOS")
    print("-" * 40)
    
    if fonte_clima == "inmet":
        df_sidra, mapeamento = mapear_municipios_estacoes(df_sidra, df_inmet)
        df_sidra = df_sidra.dropna(subset=['estacao_codigo'])
        if df_sidra.empty:
            raise ValueError(
                "Nenhum dos municipios selecionados possui estacao INMET "
                "correspondente na rodada atual."
            )
        print(f"Registros SIDRA apos mapeamento: {len(df_sidra)}")
        print(f"Municipios mapeados para estacoes: {len(mapeamento)}")
    else:
        # Para ERA5, apenas garantir que os nomes estão normalizados
        df_sidra['municipio_norm'] = df_sidra['municipio'].map(chave_nome)
        print(f"Registros SIDRA: {len(df_sidra)}")
        print(f"Municipios: {df_sidra['municipio'].nunique()}")
    
    # ========================================================================
    # 5. INTEGRAR DADOS (EXPANDINDO SIDRA PARA TRIMESTRES)
    # ========================================================================
    print("\n5. INTEGRANDO DADOS TRIMESTRAIS")
    print("-" * 40)
    
    if excluir_jja:
        trimestres_safra = ['SON', 'DJF', 'MAM']
        print("Excluindo JJA (pos-colheita) dos dados integrados")
    else:
        trimestres_safra = ['SON', 'DJF', 'MAM', 'JJA']
        print("Mantendo JJA nos dados integrados")
    
    registros_integrados = []
    
    for _, row in df_sidra.iterrows():
        municipio = row['municipio']
        ano = row['ano']
        produtividade = row['Rendimento_Medio_kg_ha']
        estacao_codigo = row.get('estacao_codigo', 'ERA5')
        
        for i, trimestre in enumerate(trimestres_safra):
            ano_clima = ano - 1 if trimestre == 'SON' else ano
            
            if fonte_clima == "era5":
                # Usar coordenadas do município diretamente
                if municipio not in coordenadas_municipios.index:
                    logger.warning(f"Município {municipio} não encontrado nas coordenadas. Pulando.")
                    continue
                coordenada = coordenadas_municipios.loc[municipio]
                (
                    precip, temp, ano_clima, meses_validos,
                    temp_desvio_c, temp_amplitude_c, meses_seca
                ) = obter_clima_era5(
                    dados_era5, nome_tempo, nome_lat, nome_lon,
                    coordenada['latitude'], coordenada['longitude'], ano, trimestre
                )
                estacao_registro = 'ERA5'
            else:
                # Usar dados do INMET
                clima = df_inmet_trim[
                    (df_inmet_trim['estacao_codigo'] == estacao_codigo) &
                    (df_inmet_trim['ano'] == ano_clima) &
                    (df_inmet_trim['trimestre'] == trimestre)
                ]
                precip = clima['precipitacao_mm'].values[0] if len(clima) > 0 else np.nan
                temp = clima['temp_media_c'].values[0] if len(clima) > 0 else np.nan
                meses_validos = clima['meses_validos'].values[0] if len(clima) > 0 else 0
                temp_desvio_c = clima['temp_desvio_c'].values[0] if len(clima) > 0 else np.nan
                temp_amplitude_c = clima['temp_amplitude_c'].values[0] if len(clima) > 0 else np.nan
                meses_seca = clima['meses_seca'].values[0] if len(clima) > 0 else 0
                estacao_registro = estacao_codigo
            
            if usar_era5_diario:
                if fonte_clima == "era5":
                    coordenada = coordenadas_municipios.loc[municipio]
                else:
                    coordenada = coordenadas_estacoes.loc[estacao_registro]
                gdd, _ = obter_estatisticas_diarias_era5(
                    dados_era5_diario, nome_tempo_diario, nome_lat_diario, nome_lon_diario,
                    coordenada['latitude'], coordenada['longitude'], ano, trimestre
                )
                dias_secos = np.nan
                dias_secos_consecutivos = np.nan
            else:
                gdd, dias_secos, dias_secos_consecutivos = np.nan, np.nan, np.nan
            
            oni = df_noaa[
                (df_noaa['ano'] == ano_clima) &
                (df_noaa['trimestre'] == trimestre)
            ]
            oni_val = oni['oni_index'].values[0] if len(oni) > 0 else np.nan
            fase = oni['fase_enos'].values[0] if len(oni) > 0 else 'Neutro'
            
            registro = {
                'municipio': municipio,
                'ano': ano,
                'trimestre': trimestre,
                'ordem_trimestre': i,
                'produtividade_kg_ha': produtividade,
                'precipitacao_mm': precip,
                'temp_media_c': temp,
                'oni_index': oni_val,
                'fase_enos': fase,
                'estacao_codigo': estacao_registro,
                'ano_clima': ano_clima,
                'meses_validos': meses_validos,
                'temp_desvio_c': temp_desvio_c,
                'temp_amplitude_c': temp_amplitude_c,
                'meses_seca': meses_seca,
                'gdd': gdd,
                'dias_secos': dias_secos,
                'dias_secos_consecutivos': dias_secos_consecutivos
            }
            registros_integrados.append(registro)
    
    colunas_integradas = [
        'municipio', 'ano', 'trimestre', 'ordem_trimestre',
        'produtividade_kg_ha', 'precipitacao_mm', 'temp_media_c',
        'oni_index', 'fase_enos', 'estacao_codigo', 'ano_clima',
        'meses_validos', 'temp_desvio_c', 'temp_amplitude_c', 'meses_seca',
        'gdd', 'dias_secos', 'dias_secos_consecutivos'
    ]
    df_integrado = pd.DataFrame(registros_integrados, columns=colunas_integradas)
    
    if df_integrado.empty:
        raise ValueError(
            "Nenhum registro foi criado na integracao. "
            "Verifique os dados SIDRA e o mapeamento das estacoes."
        )
    
    if df_integrado[['precipitacao_mm', 'temp_media_c']].notna().sum().sum() == 0:
        raise ValueError("Nenhuma variavel climatica valida foi encontrada na integracao.")
    
    print(f"Registros integrados (trimestrais): {len(df_integrado)}")
    print(f"Municipios: {df_integrado['municipio'].nunique()}")
    print(f"Trimestres: {df_integrado['trimestre'].nunique()}")
    
    chaves_safra = ['municipio', 'ano', 'estacao_codigo']
    
    # CORRECAO CRITICA: Filtrar apenas trimestres da safra ANTES do pivot
    df_integrado = df_integrado[df_integrado['trimestre'].isin(trimestres_safra)].copy()
    
    clima_safra = df_integrado.pivot_table(
        index=chaves_safra,
        columns='trimestre',
        values=[
            'precipitacao_mm', 'temp_media_c', 'oni_index', 'meses_validos',
            'temp_desvio_c', 'temp_amplitude_c', 'meses_seca',
            'gdd', 'dias_secos', 'dias_secos_consecutivos'
        ],
        aggfunc='first'
    )
    clima_safra.columns = [f'{variavel}_{trimestre}' for variavel, trimestre in clima_safra.columns]
    clima_safra = clima_safra.reset_index()
    
    alvo_safra = (
        df_integrado[['municipio', 'ano', 'produtividade_kg_ha']]
        .drop_duplicates(['municipio', 'ano'])
    )
    
    df_safras = alvo_safra.merge(
        clima_safra,
        on=['municipio', 'ano'],
        how='inner'
    )
    
    contagem_anos = df_safras.groupby('municipio')['ano'].nunique()
    municipios_poucos_anos = sorted(contagem_anos[contagem_anos < min_anos_municipio].index)
    if municipios_poucos_anos:
        logger.warning(
            "Removidos %d municipios com menos de %d anos de historico: %s",
            len(municipios_poucos_anos), min_anos_municipio, ', '.join(municipios_poucos_anos)
        )
        df_safras = df_safras[~df_safras['municipio'].isin(municipios_poucos_anos)].copy()
    
    if df_safras.empty:
        raise ValueError(
            f"Nenhum municipio possui pelo menos {min_anos_municipio} anos de historico."
        )
    
    for trimestre in trimestres_safra:
        coluna_cobertura = f'meses_validos_{trimestre}'
        if coluna_cobertura not in df_safras:
            df_safras[coluna_cobertura] = 0
    
    cobertura_cols = [f'meses_validos_{trimestre}' for trimestre in trimestres_safra]
    df_safras['trimestres_completos'] = (df_safras[cobertura_cols] >= 3).sum(axis=1)
    df_safras['cobertura_climatica'] = df_safras['trimestres_completos'] / len(trimestres_safra)
    
    for trimestre in trimestres_safra:
        for prefixo in ('precipitacao_mm_', 'temp_media_c_', 'oni_index_', 'temp_desvio_c_', 'temp_amplitude_c_', 'gdd_'):
            coluna = f'{prefixo}{trimestre}'
            if coluna in df_safras:
                mediana = df_safras.groupby('estacao_codigo')[coluna].transform('median')
                df_safras[coluna] = df_safras[coluna].fillna(mediana)
                df_safras[coluna] = df_safras[coluna].fillna(df_safras[coluna].median())
        
        for prefixo_contagem in ('meses_seca_', 'dias_secos_', 'dias_secos_consecutivos_'):
            coluna_contagem = f'{prefixo_contagem}{trimestre}'
            if coluna_contagem in df_safras:
                df_safras[coluna_contagem] = df_safras[coluna_contagem].fillna(0)
    
    df_safras = df_safras.sort_values(['municipio', 'ano']).reset_index(drop=True)
    
    lag_produtividade = df_safras[['municipio', 'ano', 'produtividade_kg_ha']].copy()
    lag_produtividade['ano'] = lag_produtividade['ano'] + 1
    lag_produtividade = lag_produtividade.rename(
        columns={'produtividade_kg_ha': 'produtividade_kg_ha_lag1'}
    )
    df_safras = df_safras.merge(lag_produtividade, on=['municipio', 'ano'], how='left')
    
    # ========================================================================
    # CORRECAO CRITICA: DETRENDING COM SPLIT 80/20
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"CALCULANDO TARGET (split={split}, target={target})")
    print("=" * 60)
    
    if target == "desvio_pct":
        df_completo = df_safras.copy()
        if split == "aleatorio":
            df_treino, df_teste = train_test_split(
                df_completo,
                test_size=0.2,
                random_state=42,
                stratify=df_completo['municipio']
            )
            print(f"Split aleatorio estratificado por municipio")
        else:  # cronologico
            anos_ordenados = sorted(df_completo['ano'].unique())
            corte = int(len(anos_ordenados) * 0.8)
            anos_treino = anos_ordenados[:corte]
            anos_teste = anos_ordenados[corte:]
            df_treino = df_completo[df_completo['ano'].isin(anos_treino)].copy()
            df_teste = df_completo[df_completo['ano'].isin(anos_teste)].copy()
            print(f"Split cronologico: treino={anos_treino[0]}-{anos_treino[-1]}, teste={anos_teste[0]}-{anos_teste[-1]}")
        
        print(f"Treino: {len(df_treino)} amostras | Teste: {len(df_teste)} amostras")
        
        media_hist = df_treino.groupby('municipio')['produtividade_kg_ha'].mean().reset_index()
        media_hist.rename(columns={'produtividade_kg_ha': 'media_hist_kg_ha'}, inplace=True)
        
        df_treino = df_treino.merge(media_hist, on='municipio', how='left')
        df_teste = df_teste.merge(media_hist, on='municipio', how='left')
        
        media_geral = df_treino['produtividade_kg_ha'].mean()
        df_treino['media_hist_kg_ha'] = df_treino['media_hist_kg_ha'].fillna(media_geral)
        df_teste['media_hist_kg_ha'] = df_teste['media_hist_kg_ha'].fillna(media_geral)
        
        df_treino['target_desvio_pct'] = (
            (df_treino['produtividade_kg_ha'] - df_treino['media_hist_kg_ha']) /
            df_treino['media_hist_kg_ha']
        ) * 100
        df_teste['target_desvio_pct'] = (
            (df_teste['produtividade_kg_ha'] - df_teste['media_hist_kg_ha']) /
            df_teste['media_hist_kg_ha']
        ) * 100
        
        print(f"Desvio percentual calculado.")
        print(f"  Treino - Media: {df_treino['target_desvio_pct'].mean():.2f}%, DP: {df_treino['target_desvio_pct'].std():.2f}%")
        print(f"  Teste  - Media: {df_teste['target_desvio_pct'].mean():.2f}%, DP: {df_teste['target_desvio_pct'].std():.2f}%")
        
        df_treino.to_csv(output_dir / "treino_80pct.csv", index=False, encoding='utf-8-sig')
        df_teste.to_csv(output_dir / "teste_20pct.csv", index=False, encoding='utf-8-sig')
        print(f"Datasets salvos: treino_80pct.csv e teste_20pct.csv")
        
        df_safras = pd.concat([df_treino, df_teste], ignore_index=True)
    else:
        df_safras.to_csv(output_dir / "treino_80pct.csv", index=False, encoding='utf-8-sig')
        print(f"Target absoluto: dataset salvo sem desvio percentual")
    
    # ========================================================================
    # 6. ESTATISTICAS E CORRELACOES
    # ========================================================================
    print("\n" + "=" * 60)
    print("6. ESTATISTICAS DOS DADOS TRIMESTRAIS")
    print("=" * 60)
    
    print(f"\nProdutividade media: {df_integrado['produtividade_kg_ha'].mean():.0f} kg/ha")
    print(f"\nPrecipitacao media por trimestre:")
    for trim in trimestres_safra:
        precip = df_integrado[df_integrado['trimestre'] == trim]['precipitacao_mm'].mean()
        print(f"  {trim}: {precip:.0f} mm")
    
    print(f"\nTemperatura media por trimestre:")
    for trim in trimestres_safra:
        temp = df_integrado[df_integrado['trimestre'] == trim]['temp_media_c'].mean()
        print(f"  {trim}: {temp:.1f} C")
    
    print(f"\nCorrelacoes com Produtividade:")
    corr_precip = df_integrado['precipitacao_mm'].corr(df_integrado['produtividade_kg_ha'])
    corr_temp = df_integrado['temp_media_c'].corr(df_integrado['produtividade_kg_ha'])
    corr_oni = df_integrado['oni_index'].corr(df_integrado['produtividade_kg_ha'])
    print(f"  Precipitacao: {corr_precip:.3f}")
    print(f"  Temperatura: {corr_temp:.3f}")
    print(f"  ONI: {corr_oni:.3f}")
    
    print(f"\nCorrelacoes com Desvio Percentual de Produtividade:")
    if 'target_desvio_pct' in df_safras.columns:
        df_integrado_completo = df_integrado.merge(
            df_safras[['municipio', 'ano', 'target_desvio_pct']].drop_duplicates(),
            on=['municipio', 'ano'], how='left'
        )
        corr_precip_desvio = df_integrado_completo['precipitacao_mm'].corr(df_integrado_completo['target_desvio_pct'])
        corr_temp_desvio = df_integrado_completo['temp_media_c'].corr(df_integrado_completo['target_desvio_pct'])
        print(f"  Precipitacao: {corr_precip_desvio:.3f}")
        print(f"  Temperatura: {corr_temp_desvio:.3f}")
    
    print(f"\nCorrelacoes por Trimestre:")
    for trim in trimestres_safra:
        df_trim = df_integrado[df_integrado['trimestre'] == trim]
        if len(df_trim) > 0:
            corr = df_trim['temp_media_c'].corr(df_trim['produtividade_kg_ha'])
            print(f"  {trim} (Temperatura): {corr:.3f} (n={len(df_trim)})")
    
    # ========================================================================
    # 7. SALVAR RESULTADOS
    # ========================================================================
    print("\n" + "=" * 60)
    print("7. SALVANDO RESULTADOS")
    print("=" * 60)
    
    anomalias_longas = []
    for trimestre in trimestres_safra:
        colunas_trim = {
            f'anom_precipitacao_mm_{trimestre}': 'anom_precipitacao_mm',
            f'anom_temp_media_c_{trimestre}': 'anom_temp_media_c',
        }
        colunas_disponiveis = {
            origem: destino for origem, destino in colunas_trim.items()
            if origem in df_safras.columns
        }
        if not colunas_disponiveis:
            continue
        bloco = df_safras[['municipio', 'ano'] + list(colunas_disponiveis)].rename(columns=colunas_disponiveis)
        bloco['trimestre'] = trimestre
        anomalias_longas.append(bloco)
    
    if anomalias_longas:
        anomalias_longas = pd.concat(anomalias_longas, ignore_index=True)
        df_integrado = df_integrado.merge(
            anomalias_longas, on=['municipio', 'ano', 'trimestre'], how='left'
        )
    
    colunas_por_safra = [
        coluna for coluna in ('produtividade_kg_ha_lag1', 'cobertura_climatica', 'trimestres_completos', 'target_desvio_pct')
        if coluna in df_safras.columns
    ]
    if colunas_por_safra:
        df_integrado = df_integrado.merge(
            df_safras[['municipio', 'ano'] + colunas_por_safra],
            on=['municipio', 'ano'], how='left'
        )
    
    caminho = output_dir / "dados_integrados_trimestrais.csv"
    df_integrado.to_csv(caminho, index=False, encoding='utf-8-sig')
    print(f"Dataset salvo: {caminho} ({len(df_integrado)} registros)")
    
    caminho_safras = output_dir / "dados_integrados_safras.csv"
    df_safras.to_csv(caminho_safras, index=False, encoding='utf-8-sig')
    print(f"Dataset por safra salvo: {caminho_safras} ({len(df_safras)} registros)")
    
    df_agregado = df_integrado.groupby(['ano', 'trimestre', 'ordem_trimestre']).agg({
        'produtividade_kg_ha': 'mean',
        'precipitacao_mm': 'mean',
        'temp_media_c': 'mean',
        'oni_index': 'mean'
    }).reset_index()
    caminho_agregado = output_dir / "dados_agregados_trimestrais.csv"
    df_agregado.to_csv(caminho_agregado, index=False, encoding='utf-8-sig')
    print(f"Dataset agregado salvo: {caminho_agregado}")
    
    resumo = df_integrado.groupby('trimestre').agg({
        'produtividade_kg_ha': 'mean',
        'precipitacao_mm': 'mean',
        'temp_media_c': 'mean',
        'oni_index': 'mean'
    }).round(2)
    print("\n" + "=" * 60)
    print("RESUMO POR TRIMESTRE")
    print("=" * 60)
    print(resumo)
    
    cobertura = df_safras[['municipio', 'ano', 'cobertura_climatica', 'trimestres_completos'] + cobertura_cols]
    cobertura.to_csv(output_dir / 'relatorio_cobertura_municipio_ano.csv', index=False, encoding='utf-8-sig')
    cobertura_resumo = cobertura.groupby('municipio').agg(
        anos=('ano', 'count'),
        cobertura_media=('cobertura_climatica', 'mean'),
        safras_completas=('trimestres_completos', lambda valores: (valores == len(trimestres_safra)).sum()),
    ).reset_index()
    cobertura_resumo.to_markdown(
        output_dir / 'relatorio_cobertura_municipio_ano.md',
        index=False,
        floatfmt='.2f'
    )
    
    print("\n" + "=" * 60)
    print("INTEGRACAO TRIMESTRAL CONCLUIDA!")
    print(f"Arquivos salvos em: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integra os dados de safra e clima")
    parser.add_argument(
        "--ano-inicial",
        type=int,
        default=1976,
        help="Primeiro ano do periodo de integracao (padrao: 1976)",
    )
    parser.add_argument(
        "--ano-final",
        type=int,
        default=2024,
        help="Ultimo ano do periodo de integracao (padrao: 2024)",
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--era5-file", type=Path, default=None)
    parser.add_argument(
        "--fonte-clima",
        choices=["era5", "inmet"],
        default="era5",
        help="Fonte climatica a utilizar: era5 (ERA5-Land) ou inmet (estacoes)",
    )
    parser.add_argument(
        "--municipios",
        nargs="+",
        default=None,
        help="Municipios a utilizar; se omitido, usa todos os municipios disponiveis",
    )
    parser.add_argument(
        "--min-anos-municipio",
        type=int,
        default=20,
        help="Numero minimo de anos de historico para manter um municipio (padrao: 20)",
    )
    parser.add_argument(
        "--usar-era5-diario",
        choices=["sim", "nao"],
        default="nao",
        help="sim: calcula GDD a partir do ERA5-Land diario; nao (padrao)",
    )
    parser.add_argument(
        "--split",
        choices=["cronologico", "aleatorio"],
        default="aleatorio",
        help="cronologico: 80% anos antigos vs 20% recentes; "
             "aleatorio: 80/20 aleatorio estratificado por municipio (padrao)",
    )
    parser.add_argument(
        "--target",
        choices=["absoluto", "desvio_pct"],
        default="desvio_pct",
        help="absoluto: produtividade em kg/ha; "
             "desvio_pct: desvio percentual da media historica (padrao)",
    )
    parser.add_argument(
        "--excluir-jja",
        choices=["sim", "nao"],
        default="sim",
        help="sim: remove features de JJA (pos-colheita) (padrao); nao: mantem todas",
    )
    parser.add_argument(
        "--excluir-ano",
        choices=["sim", "nao"],
        default="sim",
        help="sim: remove feature 'ano' (vazamento temporal) (padrao); nao: mantem",
    )
    args = parser.parse_args()
    
    ano_minimo = 1976 if args.fonte_clima == "inmet" else 1951
    if args.ano_inicial < ano_minimo:
        parser.error(f"--ano-inicial deve ser maior ou igual a {ano_minimo} para --fonte-clima {args.fonte_clima}")
    if args.ano_inicial > args.ano_final:
        parser.error("--ano-inicial deve ser menor ou igual a --ano-final")
    
    main(
        args.ano_inicial,
        args.ano_final,
        args.processed_dir,
        args.municipios,
        args.era5_file,
        args.fonte_clima,
        args.min_anos_municipio,
        args.usar_era5_diario == "sim",
        split=args.split,
        target=args.target,
        excluir_jja=args.excluir_jja == "sim",
        excluir_ano=args.excluir_ano == "sim",
    )