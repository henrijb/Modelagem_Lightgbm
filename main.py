"""
main.py - Pipeline completo do TCC (Versao Unificada)
Autor: Seu Nome
TCC: Otimizacao da Janela de Plantio de Soja sob Influencia do Fenomeno ENOS
ATUALIZADO: Versao TRIMESTRAL - Orquestra todos os scripts do pipeline
NOVO: Suporte a split 80/20 aleatorio, target desvio percentual,
      exclusao de JJA e feature 'ano', e modelagem Random Forest (07)
"""
import sys
from pathlib import Path
import subprocess
import logging
import argparse
import csv as _csv
from datetime import datetime

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def obter_configuracao():
    parser = argparse.ArgumentParser(description="Executa a pipeline do TCC")
    parser.add_argument(
        "--ano-inicial",
        type=int,
        default=1976,
        help="Primeiro ano do periodo de analise (padrao: 1976)",
    )
    parser.add_argument(
        "--ano-final",
        type=int,
        default=2024,
        help="Ultimo ano do periodo de analise (padrao: 2024)",
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
        "--usar-anomalia",
        choices=["sim", "nao"],
        default="sim",
        help="sim (padrao): anomalia climatica por municipio; nao: precipitacao_mm_/temp_media_c_ brutos",
    )
    parser.add_argument(
        "--usar-lag1",
        choices=["sim", "nao"],
        default="sim",
        help="sim (padrao): inclui produtividade_kg_ha_lag1; nao: nao usa autoregressivo",
    )
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
        help="baixar: obtem o ERA5-Land pela API do CDS; local: usa um arquivo ja existente",
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
        help="sim: baixa tambem o ERA5-Land diario (para GDD e dias secos); nao (padrao): so o mensal",
    )
    parser.add_argument(
        "--usar-era5-diario",
        choices=["sim", "nao"],
        default="nao",
        help="sim: calcula GDD e dias secos no integrador; requer --era5-diario sim na coleta",
    )
    
    # NOVAS FLAGS PARA CORREÇÕES METODOLÓGICAS
    parser.add_argument(
        "--split",
        choices=["cronologico", "aleatorio"],
        default="aleatorio",
        help="cronologico: 80% anos antigos vs 20% recentes (padrao antigo); "
             "aleatorio: 80/20 aleatorio estratificado por municipio (NOVO, recomendado)",
    )
    parser.add_argument(
        "--target",
        choices=["absoluto", "desvio_pct"],
        default="desvio_pct",
        help="absoluto: produtividade em kg/ha (padrao antigo); "
             "desvio_pct: desvio percentual da media historica (NOVO, recomendado)",
    )
    parser.add_argument(
        "--excluir-jja",
        choices=["sim", "nao"],
        default="sim",
        help="sim: remove features de JJA (pos-colheita); nao: mantem todas (padrao antigo)",
    )
    parser.add_argument(
        "--excluir-ano",
        choices=["sim", "nao"],
        default="sim",
        help="sim: remove feature 'ano' (vazamento temporal); nao: mantem (padrao antigo)",
    )
    parser.add_argument(
        "--rodar-random-forest",
        choices=["sim", "nao"],
        default="sim",
        help="sim: executa 07_modelagem_random_forest.py; nao: pula (padrao: sim)",
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
    # Validacao cruzada: se quer usar diario, precisa ter baixado
    if args.usar_era5_diario == "sim" and args.fonte_clima != "era5":
        parser.error("--usar-era5-diario so pode ser usado com --fonte-clima era5")
    
    return args


args = obter_configuracao()

BASE_DIR = Path(__file__).resolve().parent
RODADA = datetime.now().strftime("%Y%m%d_%H%M%S")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / RODADA
OUTPUT_DIR = BASE_DIR / "output" / RODADA
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PIPELINE COMPLETO - TCC SOJA RS x ENOS (VERSAO TRIMESTRAL)")
print("=" * 80)

# ============================================================================
# CONFIGURACAO
# ============================================================================
ANOS = list(range(args.ano_inicial, args.ano_final + 1))
FORCE = True
EXECUTAR_SCRIPTS = True  # Se False, apenas mostra os comandos

print(f"\n[INFO] Periodo de analise: {ANOS[0]} a {ANOS[-1]} ({len(ANOS)} anos)")
print(f"[INFO] Pasta da rodada: {RODADA}")
print(f"[INFO] Modo: {'ATUALIZAR (forcar nova coleta)' if FORCE else 'USAR CACHE'}")
print(f"[INFO] Executar scripts: {'SIM' if EXECUTAR_SCRIPTS else 'NAO (apenas mostrar comandos)'}")
print(f"[INFO] Fonte climatica: {args.fonte_clima.upper()}")
print(f"[INFO] Usar anomalia: {args.usar_anomalia}")
print(f"[INFO] Usar lag1: {args.usar_lag1}")
print(f"[INFO] ERA5 diario (coleta): {args.era5_diario}")
print(f"[INFO] ERA5 diario (integrador): {args.usar_era5_diario}")
print(f"[INFO] Split: {args.split}")
print(f"[INFO] Target: {args.target}")
print(f"[INFO] Excluir JJA: {args.excluir_jja}")
print(f"[INFO] Excluir 'ano': {args.excluir_ano}")
print(f"[INFO] Random Forest: {args.rodar_random_forest}")

# ============================================================================
# 1. EXECUTAR COLETA
# ============================================================================
print("\n" + "=" * 80)
print("[ETAPA 1] COLETA DE DADOS")
print("=" * 80)

if EXECUTAR_SCRIPTS:
    print("\nExecutando 01_coleta.py...")
    try:
        comando_coleta = [
            sys.executable,
            "src/01_coleta.py",
            "--ano-inicial", str(args.ano_inicial),
            "--ano-final", str(args.ano_final),
            "--processed-dir", str(PROCESSED_DIR),
            "--fonte-clima", args.fonte_clima,
            "--era5-modo", args.era5_modo,
            "--era5-diario", args.era5_diario,
        ]
        if args.era5_arquivo:
            comando_coleta.extend(["--era5-arquivo", str(args.era5_arquivo)])
        
        result = subprocess.run(
            comando_coleta,
            capture_output=False,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("[OK] Coleta concluida com sucesso!")
        else:
            print(f"[ERRO] Coleta falhou com codigo {result.returncode}")
    except Exception as e:
        print(f"[ERRO] Ao executar coleta: {e}")
else:
    print(
        "\n[COMANDO] python src/01_coleta.py "
        f"--ano-inicial {args.ano_inicial} --ano-final {args.ano_final} "
        f"--fonte-clima {args.fonte_clima} --era5-modo {args.era5_modo} "
        f"--era5-diario {args.era5_diario}"
    )

# ============================================================================
# 2. EXECUTAR INTEGRADOR TRIMESTRAL
# ============================================================================
print("\n" + "=" * 80)
print("[ETAPA 2] INTEGRACAO DOS DADOS (VERSAO TRIMESTRAL)")
print("=" * 80)

if EXECUTAR_SCRIPTS:
    print("\nExecutando 02_integrador_safra_clima.py...")
    try:
        comando_integrador = [
            sys.executable,
            "src/02_integrador_safra_clima.py",
            "--ano-inicial", str(args.ano_inicial),
            "--ano-final", str(args.ano_final),
            "--processed-dir", str(PROCESSED_DIR),
            "--fonte-clima", args.fonte_clima,
            "--usar-era5-diario", args.usar_era5_diario,
            "--split", args.split,
            "--target", args.target,
            "--excluir-jja", args.excluir_jja,
            "--excluir-ano", args.excluir_ano,
        ]
        if args.fonte_clima == "era5" and args.era5_arquivo:
            comando_integrador.extend(["--era5-file", str(args.era5_arquivo)])
        if args.municipios:
            comando_integrador.extend(["--municipios", *args.municipios])
        comando_integrador.extend(["--min-anos-municipio", str(args.min_anos_municipio)])
        
        result = subprocess.run(
            comando_integrador,
            capture_output=False,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("[OK] Integracao concluida com sucesso!")
        else:
            print(f"[ERRO] Integracao falhou com codigo {result.returncode}")
    except Exception as e:
        print(f"[ERRO] Ao executar integrador: {e}")
else:
    comando_municipios = ""
    if args.municipios:
        comando_municipios = " --municipios " + " ".join(args.municipios)
    print(
        "\n[COMANDO] python src/02_integrador_safra_clima.py "
        f"--ano-inicial {args.ano_inicial} --ano-final {args.ano_final} "
        f"--fonte-clima {args.fonte_clima} --usar-era5-diario {args.usar_era5_diario} "
        f"--split {args.split} --target {args.target} "
        f"--excluir-jja {args.excluir_jja} --excluir-ano {args.excluir_ano} "
        f"{comando_municipios}"
    )

# ============================================================================
# 3. EXECUTAR ANALISE EXPLORATORIA TRIMESTRAL
# ============================================================================
print("\n" + "=" * 80)
print("[ETAPA 3] ANALISE EXPLORATORIA (VERSAO TRIMESTRAL)")
print("=" * 80)

if EXECUTAR_SCRIPTS:
    print("\nExecutando 03_analise_exploratoria.py...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "src/03_analise_exploratoria.py",
                "--data-dir", str(PROCESSED_DIR / "integrado_trimestral"),
                "--output-dir", str(OUTPUT_DIR / "analise_exploratoria_trimestral"),
            ],
            capture_output=False,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("[OK] Analise exploratoria concluida com sucesso!")
        else:
            print(f"[ERRO] Analise exploratoria falhou com codigo {result.returncode}")
    except Exception as e:
        print(f"[ERRO] Ao executar analise: {e}")
else:
    print(
        "\n[COMANDO] python src/03_analise_exploratoria.py "
        f"--data-dir {PROCESSED_DIR / 'integrado_trimestral'} "
        f"--output-dir {OUTPUT_DIR / 'analise_exploratoria_trimestral'}"
    )

# ============================================================================
# 4. EXECUTAR MODELAGEM TRIMESTRAL (LightGBM)
# ============================================================================
print("\n" + "=" * 80)
print("[ETAPA 4] MODELAGEM PREDITIVA LIGHTGBM")
print("=" * 80)

if EXECUTAR_SCRIPTS:
    print("\nExecutando 04_modelagem_lightgbm.py...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "src/04_modelagem_lightgbm.py",
                "--data-dir", str(PROCESSED_DIR / "integrado_trimestral"),
                "--output-dir", str(OUTPUT_DIR / "modelagem_trimestral"),
                "--usar-anomalia", args.usar_anomalia,
                "--usar-lag1", args.usar_lag1,
            ],
            capture_output=False,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("[OK] Modelagem LightGBM concluida com sucesso!")
        else:
            print(f"[ERRO] Modelagem LightGBM falhou com codigo {result.returncode}")
    except Exception as e:
        print(f"[ERRO] Ao executar modelagem LightGBM: {e}")
else:
    print(
        "\n[COMANDO] python src/04_modelagem_lightgbm.py "
        f"--data-dir {PROCESSED_DIR / 'integrado_trimestral'} "
        f"--output-dir {OUTPUT_DIR / 'modelagem_trimestral'} "
        f"--usar-anomalia {args.usar_anomalia} --usar-lag1 {args.usar_lag1}"
    )

# ============================================================================
# 5. MODELAGEM GRADIENT BOOSTING
# ============================================================================
print("\n" + "=" * 80)
print("[ETAPA 5] MODELAGEM COMPARATIVA GRADIENT BOOSTING")
print("=" * 80)

if EXECUTAR_SCRIPTS:
    print("\nExecutando 05_modelagem_Gradient_Boosting.py...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "src/05_modelagem_Gradient_Boosting.py",
                "--data-dir", str(PROCESSED_DIR / "integrado_trimestral"),
                "--output-dir", str(OUTPUT_DIR / "modelagem_gradient_boosting"),
                "--comparison-dir", str(OUTPUT_DIR),
                "--usar-anomalia", args.usar_anomalia,
                "--usar-lag1", args.usar_lag1,
            ],
            capture_output=False,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("[OK] Gradient Boosting concluido com sucesso!")
        else:
            print(f"[ERRO] Gradient Boosting falhou com codigo {result.returncode}")
    except Exception as e:
        print(f"[ERRO] Ao executar Gradient Boosting: {e}")
else:
    print(
        "\n[COMANDO] python src/05_modelagem_Gradient_Boosting.py "
        f"--data-dir {PROCESSED_DIR / 'integrado_trimestral'} "
        f"--output-dir {OUTPUT_DIR / 'modelagem_gradient_boosting'} "
        f"--comparison-dir {OUTPUT_DIR} "
        f"--usar-anomalia {args.usar_anomalia} --usar-lag1 {args.usar_lag1}"
    )

# ============================================================================
# 6. MODELAGEM RIDGE
# ============================================================================
print("\n" + "=" * 80)
print("[ETAPA 6] MODELAGEM COMPARATIVA RIDGE")
print("=" * 80)

if EXECUTAR_SCRIPTS:
    print("\nExecutando 06_modelagem_ridge.py...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "src/06_modelagem_ridge.py",
                "--data-dir", str(PROCESSED_DIR / "integrado_trimestral"),
                "--output-dir", str(OUTPUT_DIR / "modelagem_ridge"),
                "--comparison-dir", str(OUTPUT_DIR),
                "--usar-anomalia", args.usar_anomalia,
                "--usar-lag1", args.usar_lag1,
            ],
            capture_output=False,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("[OK] Ridge concluido com sucesso!")
        else:
            print(f"[ERRO] Ridge falhou com codigo {result.returncode}")
    except Exception as e:
        print(f"[ERRO] Ao executar Ridge: {e}")
else:
    print(
        "\n[COMANDO] python src/06_modelagem_ridge.py "
        f"--data-dir {PROCESSED_DIR / 'integrado_trimestral'} "
        f"--output-dir {OUTPUT_DIR / 'modelagem_ridge'} "
        f"--comparison-dir {OUTPUT_DIR} "
        f"--usar-anomalia {args.usar_anomalia} --usar-lag1 {args.usar_lag1}"
    )

# ============================================================================
# 7. MODELAGEM RANDOM FOREST (NOVO)
# ============================================================================
print("\n" + "=" * 80)
print("[ETAPA 7] MODELAGEM COMPARATIVA RANDOM FOREST")
print("=" * 80)

if args.rodar_random_forest == "sim":
    if EXECUTAR_SCRIPTS:
        print("\nExecutando 07_modelagem_random_forest.py...")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "src/07_modelagem_random_forest.py",
                    "--data-dir", str(PROCESSED_DIR / "integrado_trimestral"),
                    "--output-dir", str(OUTPUT_DIR / "modelagem_random_forest"),
                    "--comparison-dir", str(OUTPUT_DIR),
                    "--usar-anomalia", args.usar_anomalia,
                    "--usar-lag1", args.usar_lag1,
                ],
                capture_output=False,
                text=True,
                check=False
            )
            if result.returncode == 0:
                print("[OK] Random Forest concluido com sucesso!")
            else:
                print(f"[ERRO] Random Forest falhou com codigo {result.returncode}")
        except Exception as e:
            print(f"[ERRO] Ao executar Random Forest: {e}")
    else:
        print(
            "\n[COMANDO] python src/07_modelagem_random_forest.py "
            f"--data-dir {PROCESSED_DIR / 'integrado_trimestral'} "
            f"--output-dir {OUTPUT_DIR / 'modelagem_random_forest'} "
            f"--comparison-dir {OUTPUT_DIR} "
            f"--usar-anomalia {args.usar_anomalia} --usar-lag1 {args.usar_lag1}"
        )
else:
    print("\n[INFO] Random Forest desabilitado via --rodar-random-forest nao")

# ============================================================================
# 8. RESUMO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("RESUMO DO PIPELINE TRIMESTRAL")
print("=" * 80)

# Verificar arquivos gerados
arquivos = {
    "Dados Processados": [
        PROCESSED_DIR / "noaa" / f"noaa_oni_trimestral_{args.ano_inicial - 1}_{args.ano_final}.csv",
        PROCESSED_DIR / "sidra" / "tabela1612_soja.csv",
        PROCESSED_DIR / "inmet_estacoes_combinado.csv"
    ],
    "Dados Integrados": [
        PROCESSED_DIR / "integrado_trimestral" / "dados_integrados_trimestrais.csv",
        PROCESSED_DIR / "integrado_trimestral" / "dados_integrados_safras.csv",
        PROCESSED_DIR / "integrado_trimestral" / "treino_80pct.csv",
        PROCESSED_DIR / "integrado_trimestral" / "teste_20pct.csv",
        PROCESSED_DIR / "integrado_trimestral" / "relatorio_cobertura_municipio_ano.csv",
        PROCESSED_DIR / "integrado_trimestral" / "relatorio_cobertura_municipio_ano.md"
    ],
    "Analise Exploratoria": [
        OUTPUT_DIR / "analise_exploratoria_trimestral" / "analise_produtividade_trimestral.png",
        OUTPUT_DIR / "analise_exploratoria_trimestral" / "analise_clima_trimestral.png",
        OUTPUT_DIR / "analise_exploratoria_trimestral" / "analise_temporal_trimestral.png",
        OUTPUT_DIR / "analise_exploratoria_trimestral" / "relatorio_analise_trimestral.md"
    ],
    "Modelagem": [
        OUTPUT_DIR / "modelagem_trimestral" / "importancia_features_trimestral.png",
        OUTPUT_DIR / "modelagem_gradient_boosting" / "metricas_gradient_boosting.json",
        OUTPUT_DIR / "modelagem_ridge" / "metricas_ridge.json",
        OUTPUT_DIR / "modelagem_random_forest" / "metricas_random_forest.json",
        OUTPUT_DIR / "comparacao_modelos.csv",
        OUTPUT_DIR / "comparacao_modelos.md"
    ]
}

print("\n[INFO] Verificando arquivos gerados:\n")
for categoria, lista in arquivos.items():
    print(f"  {categoria}:")
    for arquivo in lista:
        caminho = Path(arquivo)
        if caminho.exists():
            tamanho = caminho.stat().st_size / 1024
            print(f"    [OK] {caminho.name} ({tamanho:.1f} KB)")
        else:
            print(f"    [FALTA] {caminho.name}")

# ============================================================================
# 9. EXIBIR COMPARACAO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("COMPARACAO FINAL DOS MODELOS")
print("=" * 80)

comparacao_csv = OUTPUT_DIR / "comparacao_modelos.csv"
if comparacao_csv.exists():
    with open(comparacao_csv, newline="", encoding="utf-8") as arquivo_csv:
        linhas = list(_csv.DictReader(arquivo_csv))
        if linhas:
            # Identificar melhor modelo por R² e acuracia
            linhas_com_r2 = [l for l in linhas if l.get('r2') and l['r2'] != 'nan']
            linhas_com_acc = [l for l in linhas if l.get('acuracia') and l['acuracia'] != 'nan']
            
            melhor_regressao = min(linhas_com_r2, key=lambda linha: float(linha['r2'])) if linhas_com_r2 else None
            melhor_regressao_r2 = max(linhas_com_r2, key=lambda linha: float(linha['r2'])) if linhas_com_r2 else None
            melhor_classificacao = max(linhas_com_acc, key=lambda linha: float(linha['acuracia'])) if linhas_com_acc else None
            
            print("\n[RESULTADOS POR MODELO]\n")
            for linha in linhas:
                print(f"  {linha['modelo']}:")
                r2_val = float(linha['r2']) if linha.get('r2') and linha['r2'] != 'nan' else None
                rmse_val = float(linha['rmse']) if linha.get('rmse') and linha['rmse'] != 'nan' else None
                mae_val = float(linha['mae']) if linha.get('mae') and linha['mae'] != 'nan' else None
                acc_val = float(linha['acuracia']) if linha.get('acuracia') and linha['acuracia'] != 'nan' else None
                
                if r2_val is not None:
                    print(f"    - R²: {r2_val:.3f}")
                if rmse_val is not None:
                    unidade = "kg/ha" if args.target == "absoluto" else "%"
                    print(f"    - RMSE: {rmse_val:.1f} {unidade}")
                if mae_val is not None:
                    unidade = "kg/ha" if args.target == "absoluto" else "%"
                    print(f"    - MAE: {mae_val:.1f} {unidade}")
                if acc_val is not None:
                    print(f"    - Acuracia (classificacao): {acc_val:.3f}")
                print()
            
            if melhor_regressao_r2:
                print(f"   Melhor regressao (maior R²): {melhor_regressao_r2['modelo']} (R²={float(melhor_regressao_r2['r2']):.3f})")
            if melhor_regressao:
                print(f"   Menor R² (pior): {melhor_regressao['modelo']} (R²={float(melhor_regressao['r2']):.3f})")
            if melhor_classificacao:
                print(f"   Melhor classificacao (maior acuracia): {melhor_classificacao['modelo']} (acc={float(melhor_classificacao['acuracia']):.3f})")
        else:
            print("  [AVISO] comparacao_modelos.csv esta vazio.")
else:
    print("  [AVISO] comparacao_modelos.csv nao encontrado; execute a modelagem primeiro.")

# Exibir cobertura climatica
cobertura_md = PROCESSED_DIR / "integrado_trimestral" / "relatorio_cobertura_municipio_ano.md"
if cobertura_md.exists():
    print("\n[COBERTURA CLIMATICA POR MUNICIPIO]\n")
    print(cobertura_md.read_text(encoding="utf-8"))

# ============================================================================
# 10. COMANDOS RAPIDOS E ARQUIVOS DE SAIDA
# ============================================================================
print(f"""
[COMANDOS RAPIDOS]
Para executar apenas uma etapa:
  Coleta:                     python src/01_coleta.py
  Integracao:                 python src/02_integrador_safra_clima.py
  Analise:                    python src/03_analise_exploratoria.py
  Modelagem LightGBM:         python src/04_modelagem_lightgbm.py
  Modelagem Gradient Boost:   python src/05_modelagem_Gradient_Boosting.py
  Modelagem Ridge:            python src/06_modelagem_ridge.py
  Modelagem Random Forest:    python src/07_modelagem_random_forest.py

[ARQUIVOS DE SAIDA DESTA RODADA]
  Dados:
    {PROCESSED_DIR / 'integrado_trimestral' / 'dados_integrados_safras.csv'}
    {PROCESSED_DIR / 'integrado_trimestral' / 'treino_80pct.csv'}
    {PROCESSED_DIR / 'integrado_trimestral' / 'teste_20pct.csv'}
  
  Graficos e relatorios:
    {OUTPUT_DIR / 'analise_exploratoria_trimestral'}
    {OUTPUT_DIR / 'modelagem_trimestral'}
    {OUTPUT_DIR / 'modelagem_gradient_boosting'}
    {OUTPUT_DIR / 'modelagem_ridge'}
    {OUTPUT_DIR / 'modelagem_random_forest'}
  
  Comparacao entre modelos:
    {OUTPUT_DIR / 'comparacao_modelos.md'}
""")

# ============================================================================
# 11. MODO INTERATIVO (OPCIONAL)
# ============================================================================
if not EXECUTAR_SCRIPTS:
    print("\n[INFO] Para executar o pipeline completo, altere EXECUTAR_SCRIPTS = True")
    print("   Ou execute os comandos individualmente conforme listado acima.")

print("\n" + "=" * 80)
print("PIPELINE CONCLUIDO!")
print("=" * 80)