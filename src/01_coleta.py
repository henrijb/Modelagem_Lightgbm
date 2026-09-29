"""
Script de coleta de dados para o TCC
Coleta dados de NOAA ONI, SIDRA (Soja RS), INMET e gera coordenadas dos municipios
"""
import sys
import logging
import argparse
import re
import unicodedata
from pathlib import Path
from datetime import datetime

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

# Configurar logging PRIMEIRO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dados brutos do ERA5-Land ficam em data/coletados, como as demais fontes
ERA5_RAW_DIR = Path("data/coletados/era5_land")

def normalizar_nome(nome):
    """Normaliza o nome do municipio para facilitar o merge."""
    texto = unicodedata.normalize('NFKD', str(nome)).upper()
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"['`]", '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def gerar_coordenadas_municipios(shapefile_path, output_path):
    """Extrai centroides dos municipios do RS a partir de um shapefile do IBGE."""
    try:
        import geopandas as gpd
    except ImportError:
        raise RuntimeError("Instale 'geopandas' para gerar coordenadas dos municipios (pip install geopandas).")
    
    shapefile_path = Path(shapefile_path)
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile nao encontrado: {shapefile_path}\nBaixe o arquivo RS_Municipios_2025.zip do IBGE e extraia-o.")
    
    logger.info("Extraindo coordenadas dos municipios do RS a partir do shapefile...")
    municipios_rs = gpd.read_file(shapefile_path)
    
    # Extrair centroides
    municipios_rs['latitude'] = municipios_rs.geometry.centroid.y
    municipios_rs['longitude'] = municipios_rs.geometry.centroid.x
    
    # O shapefile do IBGE geralmente tem a coluna 'NM_MUN' ou 'NOME'
    coluna_nome = 'NM_MUN' if 'NM_MUN' in municipios_rs.columns else 'NOME'
    coordenadas = municipios_rs[[coluna_nome, 'latitude', 'longitude']].copy()
    coordenadas.columns = ['municipio', 'latitude', 'longitude']
    
    # Normalizar nomes
    coordenadas['municipio'] = coordenadas['municipio'].apply(normalizar_nome)
    
    # Salvar CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coordenadas.to_csv(output_path, index=False)
    
    logger.info(f"Coordenadas extraidas: {len(coordenadas)} municipios")
    logger.info(f"CSV salvo em: {output_path}")
    return coordenadas

def obter_configuracao():
    parser = argparse.ArgumentParser(description="Coleta os dados do TCC")
    parser.add_argument(
        "--ano-inicial",
        type=int,
        default=1976,
        help="Primeiro ano da coleta (padrao: 1976)",
    )
    parser.add_argument(
        "--ano-final",
        type=int,
        default=2024,
        help="Ultimo ano da coleta (padrao: 2024)",
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--fonte-clima",
        choices=["era5", "inmet"],
        default="era5",
        help="Fonte climatica a utilizar: era5 (ERA5-Land) ou inmet (estacoes)",
    )
    parser.add_argument(
        "--era5-modo",
        choices=["baixar", "local"],
        default="baixar",
        help="baixar: obtem o arquivo pela API do CDS; local: usa um arquivo ja existente",
    )
    parser.add_argument(
        "--era5-arquivo",
        type=Path,
        default=None,
        help="Arquivo NetCDF do ERA5-Land; obrigatorio quando --era5-modo local",
    )
    parser.add_argument(
        "--era5-diario",
        choices=["sim", "nao"],
        default="nao",
        help="sim: baixa tambem o ERA5-Land diario (para GDD); nao (padrao): so o mensal",
    )
    parser.add_argument(
        "--shapefile-municipios",
        type=Path,
        default=Path("data\coletados\ibge\RS_Municipios_2025.shp"),
        help="Caminho para o shapefile dos municipios do RS (IBGE). Padrão: data\coletados\ibge\RS_Municipios_2025.shp",
    )
    
    args = parser.parse_args()
    
    # INMET so possui estacoes confiaveis a partir de 1976; o ERA5-Land
    # cobre desde 1950, entao o limite so vale para a fonte INMET.
    ano_minimo = 1976 if args.fonte_clima == "inmet" else 1951
    if args.ano_inicial < ano_minimo:
        parser.error(f"--ano-inicial deve ser maior ou igual a {ano_minimo} para --fonte-clima {args.fonte_clima}")
    if args.ano_inicial > args.ano_final:
        parser.error("--ano-inicial deve ser menor ou igual a --ano-final")
    if args.fonte_clima == "era5" and args.era5_modo == "local" and not args.era5_arquivo:
        parser.error("--era5-arquivo e obrigatorio quando --era5-modo local")
    
    return args

def baixar_era5_land(anos_clima, destino, force_atualizacao=True):
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("Instale cdsapi para baixar os dados do ERA5-Land.") from exc
    
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and not force_atualizacao:
        logger.info("ERA5-Land: usando arquivo existente %s", destino)
        return destino
    
    arquivo_config = Path(__file__).resolve().parents[1] / ".cdsapirc"
    opcoes_cliente = {}
    if arquivo_config.exists():
        configuracao = arquivo_config.read_text(encoding="utf-8")
        for linha in configuracao.splitlines():
            nome, separador, valor = linha.partition(":")
            if separador and nome.strip() in {"url", "key"}:
                opcoes_cliente[nome.strip()] = valor.strip().strip('"\'')
        if "key" not in opcoes_cliente:
            raise RuntimeError("O arquivo .cdsapirc nao possui a chave 'key'.")
    
    cliente = cdsapi.Client(**opcoes_cliente)
    logger.info("ERA5-Land: iniciando download pela API (%s-%s)", anos_clima[0], anos_clima[-1])
    cliente.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": ["2m_temperature", "total_precipitation"],
            "year": [str(ano) for ano in anos_clima],
            "month": [f"{mes:02d}" for mes in range(1, 13)],
            "time": "00:00",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [-27.0, -58.0, -34.0, -49.5],
        },
        str(destino),
    )
    if not destino.exists():
        raise RuntimeError(f"A API nao criou o arquivo esperado: {destino}")
    logger.info("ERA5-Land: arquivo salvo em %s", destino)
    return destino

def _cliente_cds():
    import cdsapi
    arquivo_config = Path(__file__).resolve().parents[1] / ".cdsapirc"
    opcoes_cliente = {}
    if arquivo_config.exists():
        configuracao = arquivo_config.read_text(encoding="utf-8")
        for linha in configuracao.splitlines():
            nome, separador, valor = linha.partition(":")
            if separador and nome.strip() in {"url", "key"}:
                opcoes_cliente[nome.strip()] = valor.strip().strip('"\'')
        if "key" not in opcoes_cliente:
            raise RuntimeError("O arquivo .cdsapirc nao possui a chave 'key'.")
    return cdsapi.Client(**opcoes_cliente)

def baixar_era5_land_diario(anos_clima, destino_dir, force_atualizacao=True):
    """
    Baixa estatisticas diarias do ERA5-Land (temperatura min/media/max) ANO A ANO
    para evitar o erro "cost limits exceeded" da API do Copernicus.
    """
    try:
        import cdsapi
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("Instale 'cdsapi' e 'xarray' para baixar o ERA5-Land diario.") from exc
    
    destino_dir = Path(destino_dir)
    destino_dir.mkdir(parents=True, exist_ok=True)
    area_rs = [-27.0, -58.0, -34.0, -49.5]
    meses = [f"{mes:02d}" for mes in range(1, 13)]
    dias = [f"{dia:02d}" for dia in range(1, 32)]
    
    # Apenas temperaturas (precipitacao diaria nao e suportada pela API para daily_sum)
    estatisticas = ["daily_mean", "daily_minimum", "daily_maximum"]
    cliente = _cliente_cds()
    
    for estatistica in estatisticas:
        sufixo_periodo = f"{anos_clima[0]}_{anos_clima[-1]}"
        nome_variavel = estatistica.replace("daily_", "temp_")
        arquivo_final = destino_dir / f"era5_land_diario_{nome_variavel}_{sufixo_periodo}.nc"
        
        if arquivo_final.exists() and not force_atualizacao:
            logger.info(f"ERA5-Land diario: arquivo consolidado {arquivo_final.name} ja existe.")
            continue
        
        logger.info(f"ERA5-Land diario: processando {estatistica} ano a ano...")
        arquivos_anuais = []
        for ano in anos_clima:
            nome_arquivo_ano = f"era5_land_diario_{nome_variavel}_{ano}.nc"
            destino_ano = destino_dir / nome_arquivo_ano
            arquivos_anuais.append(destino_ano)
            
            if destino_ano.exists() and not force_atualizacao:
                continue
            
            logger.info(f"  [Baixando] ano {ano}...")
            try:
                cliente.retrieve(
                    "derived-era5-land-daily-statistics",
                    {
                        "variable": ["2m_temperature"],
                        "year": [str(ano)],
                        "month": meses,
                        "day": dias,
                        "daily_statistic": estatistica,
                        "time_zone": "utc+00:00",
                        "frequency": "1_hourly",
                        "area": area_rs,
                    },
                    str(destino_ano),
                )
            except Exception as e:
                logger.error(f"  [ERRO] Falha ao baixar ano {ano}: {e}")
        
        # Consolidar arquivos anuais em um unico arquivo
        if len(arquivos_anuais) > 0 and all(p.exists() for p in arquivos_anuais):
            try:
                logger.info(f"  [Consolidando] {len(arquivos_anuais)} arquivos anuais...")
                ds_combinado = xr.open_mfdataset(arquivos_anuais, combine='by_coords')
                ds_combinado.to_netcdf(arquivo_final)
                ds_combinado.close()
                # Limpar arquivos anuais para economizar espaco
                for f in arquivos_anuais:
                    f.unlink()
                logger.info(f"  [OK] Consolidacao concluida: {arquivo_final.name}")
            except Exception as e:
                logger.error(f"  [ERRO] Falha ao consolidar: {e}")

# IMPORTAR DEPOIS do logging
try:
    from coleta.noaa import coletar_noaa_oni
    logger.info("Importado: coletar_noaa_oni")
except Exception as e:
    logger.error(f"Erro importando noaa: {e}")
    coletar_noaa_oni = None

try:
    from coleta.sidra_1612 import carregar_tabela_1612
    logger.info("Importado: carregar_tabela_1612")
except Exception as e:
    logger.error(f"Erro importando sidra_1612: {e}")
    carregar_tabela_1612 = None

try:
    from coleta.inmet_estacoes import processar_todas_estacoes
    logger.info("Importado: processar_todas_estacoes")
except Exception as e:
    logger.error(f"Erro importando inmet_estacoes: {e}")
    processar_todas_estacoes = None

print("=" * 60)
print("INICIANDO COLETA")
print("=" * 60)

# Configuração do período
args = obter_configuracao()
anos = list(range(args.ano_inicial, args.ano_final + 1))
anos_clima = list(range(args.ano_inicial - 1, args.ano_final + 1))
processed_dir = args.processed_dir
force = True

logger.info(f"Periodo configurado: {anos[0]} a {anos[-1]} ({len(anos)} anos)")

# ========================================================================
# 0. COLETA ERA5-LAND (API COPERNICUS)
# ========================================================================
print("\n" + "-" * 40)
print("0. COLETA ERA5-LAND (API COPERNICUS)")
print("-" * 40)

if args.fonte_clima == "era5":
    era5_file = args.era5_arquivo or (
        ERA5_RAW_DIR / f"era5_land_rs_{anos_clima[0]}_{anos_clima[-1]}.nc"
    )
    if args.era5_modo == "local":
        if not era5_file.exists():
            logger.error("ERA5-Land: arquivo local nao encontrado: %s", era5_file)
            sys.exit(1)
        logger.info("ERA5-Land: usando arquivo local %s", era5_file)
    else:
        try:
            baixar_era5_land(anos_clima, era5_file, force_atualizacao=force)
        except Exception as e:
            logger.exception("Erro no ERA5-Land: %s", e)
            sys.exit(1)
    
    if args.era5_diario == "sim":
        print("\n" + "-" * 40)
        print("0b. COLETA ERA5-LAND DIARIO (GDD)")
        print("-" * 40)
        try:
            baixar_era5_land_diario(
                anos_clima, ERA5_RAW_DIR / "diario",
                force_atualizacao=(args.era5_modo == "baixar"),
            )
        except Exception as e:
            logger.exception("Erro no ERA5-Land diario: %s", e)
            sys.exit(1)
else:
    logger.info("Fonte climatica selecionada: INMET (download do ERA5-Land ignorado)")

# ========================================================================
# 0c. GERAÇÃO DE COORDENADAS DOS MUNICÍPIOS (IBGE)
# ========================================================================
print("\n" + "-" * 40)
print("0c. GERACAO DE COORDENADAS DOS MUNICIPIOS (IBGE)")
print("-" * 40)

try:
    gerar_coordenadas_municipios(
        shapefile_path=args.shapefile_municipios,
        output_path=processed_dir / "municipios_rs_coordenadas.csv"
    )
except Exception as e:
    logger.error(f"Erro ao gerar coordenadas dos municipios: {e}")
    # Nao pare o pipeline, pois o INMET ainda pode ser usado

# ========================================================================
# 1. COLETA NOAA ONI
# ========================================================================
print("\n" + "-" * 40)
print("1. COLETA NOAA ONI")
print("-" * 40)

df_noaa = None
if coletar_noaa_oni is not None:
    try:
        df_noaa = coletar_noaa_oni(
            force_atualizacao=force,
            anos_filtro=anos_clima,
            processed_dir=processed_dir / "noaa"
        )
        if df_noaa is not None and not df_noaa.empty:
            logger.info(f"NOAA ONI: {len(df_noaa)} registros")
        else:
            logger.warning("NOAA ONI: sem dados")
    except Exception as e:
        logger.error(f"Erro no NOAA: {e}")
else:
    logger.error("Módulo NOAA não disponível")

# ========================================================================
# 2. COLETA SIDRA
# ========================================================================
print("\n" + "-" * 40)
print("2. COLETA SIDRA (Soja RS)")
print("-" * 40)

df_soja = None
if carregar_tabela_1612 is not None:
    try:
        df_raw = carregar_tabela_1612(
            force_atualizacao=force,
            anos_filtro=anos,
            salvar=True,
            processed_dir=str(processed_dir / "sidra")
        )
        if df_raw is not None and not df_raw.empty:
            # Filtrar soja
            if 'Produto' in df_raw.columns:
                df_soja = df_raw[df_raw['Produto'] == 'Soja (em grao)'].copy()
            else:
                df_soja = df_raw.copy()
            logger.info(f"SIDRA Soja: {len(df_soja)} registros")
            if 'ano' in df_soja.columns:
                logger.info(f"   Anos: {df_soja['ano'].min()} a {df_soja['ano'].max()}")
            if 'municipio' in df_soja.columns:
                logger.info(f"   Municipios: {df_soja['municipio'].nunique()}")
        else:
            logger.warning("SIDRA: sem dados")
    except Exception as e:
        logger.error(f"Erro no SIDRA: {e}")
        import traceback
        traceback.print_exc()
else:
    logger.error("Módulo SIDRA não disponível")

# ========================================================================
# 3. COLETA INMET (Apenas se --fonte-clima inmet)
# ========================================================================
print("\n" + "-" * 40)
print("3. COLETA INMET (Estacoes do RS)")
print("-" * 40)

df_inmet = None
if args.fonte_clima == "inmet":
    if processar_todas_estacoes is not None:
        try:
            df_inmet = processar_todas_estacoes(
                dados_brutos_dir="data/coletados/inmet/rs/estacoes",
                dados_processados_dir=str(processed_dir),
                anos_filtro=(anos_clima[0], anos_clima[-1]),
                salvar=True
            )
            if df_inmet is not None and not df_inmet.empty:
                logger.info(f"INMET: {len(df_inmet)} registros")
                if 'ano' in df_inmet.columns:
                    logger.info(f"   Periodo: {df_inmet['ano'].min()} a {df_inmet['ano'].max()}")
                if 'estacao_codigo' in df_inmet.columns:
                    logger.info(f"   Estacoes: {df_inmet['estacao_codigo'].nunique()}")
            else:
                logger.warning("INMET: sem dados")
        except Exception as e:
            logger.error(f"Erro no INMET: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error("Módulo INMET não disponível")
else:
    logger.info("Fonte climatica selecionada: ERA5. Coleta do INMET ignorada.")

# ========================================================================
# RESUMO FINAL
# ========================================================================
print("\n" + "=" * 60)
print("PIPELINE DE COLETA CONCLUIDO")
print("=" * 60)
print("\nRESUMO DOS DADOS COLETADOS:")
print("-" * 50)

if df_noaa is not None and not df_noaa.empty:
    print(f"\nNOAA ONI: {len(df_noaa)} registros")
else:
    print("\nNOAA ONI: FALHA")

if df_soja is not None and not df_soja.empty:
    print(f"\nSIDRA Soja: {len(df_soja)} registros")
else:
    print("\nSIDRA Soja: FALHA")

if args.fonte_clima == "inmet":
    if df_inmet is not None and not df_inmet.empty:
        print(f"\nINMET: {len(df_inmet)} registros")
    else:
        print("\nINMET: FALHA")
else:
    print("\nINMET: NAO COLETADO (fonte-clima=era5)")

print("\n" + "=" * 60)
print("FIM")
print("=" * 60)