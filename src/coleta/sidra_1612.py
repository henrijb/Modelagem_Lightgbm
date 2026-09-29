"""
Módulo para coleta de dados da Tabela 1612 do SIDRA/PAM (Produção Agrícola Municipal)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def carregar_tabela_1612(
    force_atualizacao: bool = True,
    raw_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
    anos_filtro: Optional[List[int]] = None,
    salvar: bool = True
) -> pd.DataFrame:
    """
    Carrega dados da Tabela 1612 do SIDRA (Produção Agrícola Municipal)
    
    Args:
        force_atualizacao: Se True, recarrega do arquivo original
        raw_dir: Diretório com dados brutos
        processed_dir: Diretório para salvar processados
        anos_filtro: Lista de anos para filtrar
        salvar: Se True, salva dados processados
        
    Returns:
        DataFrame com dados processados de produção agrícola
    """
    # Configurar diretórios
    if raw_dir is None:
        raw_dir = Path("data/coletados/sidra/rs")
    if processed_dir is None:
        processed_dir = Path("data/processed/sidra")
    
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    # Arquivo de cache
    cache_file = processed_dir / "tabela1612_completo.csv"
    
    # Se não forçar atualização e cache existir, carrega do cache
    if not force_atualizacao and cache_file.exists():
        logger.info(f"Carregando do cache: {cache_file}")
        df = pd.read_csv(cache_file)
        if anos_filtro:
            df = df[df['Ano'].isin(anos_filtro)]
        return df
    
    # Procurar arquivo original
    arquivo_original = raw_dir / "tabela1612.csv"
    
    if not arquivo_original.exists():
        logger.error(f"Arquivo não encontrado: {arquivo_original}")
        logger.info("Verifique se o arquivo existe em data/coletados/sidra/rs/")
        return pd.DataFrame()
    
    logger.info(f"Processando: {arquivo_original}")
    
    try:
        # Ler o arquivo com encoding correto
        with open(arquivo_original, 'r', encoding='utf-8-sig') as f:
            linhas = f.readlines()
        
        # Encontrar a linha que contém os anos (1974, 1975, etc.)
        linha_anos = None
        inicio_dados = None
        
        for i, linha in enumerate(linhas):
            if '1974' in linha and '1975' in linha and '1976' in linha:
                linha_anos = linha
                inicio_dados = i + 1
                break
        
        if linha_anos is None:
            # Tentar encontrar apenas 1974
            for i, linha in enumerate(linhas):
                if '1974' in linha:
                    linha_anos = linha
                    inicio_dados = i + 1
                    break
        
        if linha_anos is None:
            logger.error("Linha de anos não encontrada")
            return pd.DataFrame()
        
        # Extrair os anos da linha
        partes = linha_anos.split(';')
        anos_colunas = []
        for p in partes:
            p_clean = p.strip('"').strip()
            if p_clean.isdigit() and len(p_clean) == 4:
                anos_colunas.append(int(p_clean))
        
        logger.info(f"Anos encontrados: {anos_colunas[:5]}... (total {len(anos_colunas)})")
        
        # Lista de variáveis que queremos extrair
        # Procurar onde começa cada variável
        variaveis = {
            "Área plantada (Hectares)": "Area_Plantada_ha",
            "Área colhida (Hectares)": "Area_Colhida_ha",
            "Quantidade produzida (Toneladas)": "Quantidade_Produzida_t",
            "Rendimento médio da produção (Quilogramas por Hectare)": "Rendimento_Medio_kg_ha",
            "Valor da produção": "Valor_Producao_1000_R$"
        }
        
        dados_por_variavel = {}
        
        for nome_variavel, nome_coluna in variaveis.items():
            logger.info(f"Processando: {nome_variavel}")
            
            # Encontrar onde começa esta variável
            inicio_var = None
            for i, linha in enumerate(linhas):
                if nome_variavel in linha:
                    inicio_var = i + 2  # Pular linha de cabeçalho
                    break
            
            if inicio_var is None:
                logger.warning(f"Variável {nome_variavel} não encontrada")
                continue
            
            # Encontrar onde termina esta variável
            fim_var = len(linhas)
            for i in range(inicio_var, len(linhas)):
                if "Variável -" in linhas[i] or "Fonte:" in linhas[i]:
                    fim_var = i
                    break
            
            # Processar linhas da variável
            registros_var = []
            
            for i in range(inicio_var, fim_var):
                linha = linhas[i].strip()
                if not linha:
                    continue
                
                partes = linha.split(';')
                if len(partes) < 3:
                    continue
                
                # Verificar se é linha de município
                nivel = partes[0].strip('"')
                if nivel != 'MU':
                    continue
                
                codigo = partes[1].strip('"')
                municipio = partes[2].strip('"')
                
                # Pular linha do estado
                if municipio == 'Rio Grande do Sul':
                    continue
                
                # Para cada ano, extrair o valor da soja
                for j, ano in enumerate(anos_colunas):
                    # Calcular índice da coluna do ano
                    # Os anos estão em colunas alternadas: Total e Soja
                    idx_ano = 3 + (j * 2)
                    idx_soja = idx_ano + 1
                    
                    if idx_soja >= len(partes):
                        continue
                    
                    valor_soja = partes[idx_soja].strip('"').replace(',', '.').strip()
                    
                    if valor_soja and valor_soja not in ['...', '-', '..', '', 'X']:
                        try:
                            valor = float(valor_soja)
                            registros_var.append({
                                'Codigo': codigo,
                                'Municipio': municipio,
                                'Ano': ano,
                                'Produto': 'Soja (em grao)',
                                nome_coluna: valor
                            })
                        except:
                            continue
            
            if registros_var:
                dados_por_variavel[nome_coluna] = pd.DataFrame(registros_var)
                logger.info(f"  {len(registros_var)} registros extraidos")
        
        if not dados_por_variavel:
            logger.error("Nenhuma variável foi processada")
            return pd.DataFrame()
        
        # Combinar todos os DataFrames
        logger.info("Combinando dados...")
        df_completo = None
        
        for nome_coluna, df_var in dados_por_variavel.items():
            if df_completo is None:
                df_completo = df_var
            else:
                df_completo = df_completo.merge(
                    df_var, 
                    on=['Codigo', 'Municipio', 'Ano', 'Produto'],
                    how='outer'
                )
        
        # Ordenar
        df_completo = df_completo.sort_values(['Municipio', 'Ano', 'Produto']).reset_index(drop=True)
        
        # Salvar cache
        if salvar:
            processed_dir.mkdir(parents=True, exist_ok=True)
            df_completo.to_csv(cache_file, index=False, encoding='utf-8-sig')
            logger.info(f"Cache salvo em: {cache_file}")
        
        # Salvar datasets específicos
        df_soja = df_completo[df_completo['Produto'] == 'Soja (em grao)'].copy()
        df_total = df_completo[df_completo['Produto'] == 'Total'].copy()
        
        # Salvar
        df_completo.to_csv(processed_dir / "tabela1612_completo.csv", index=False, encoding='utf-8-sig')
        df_soja.to_csv(processed_dir / "tabela1612_soja.csv", index=False, encoding='utf-8-sig')
        df_total.to_csv(processed_dir / "tabela1612_total.csv", index=False, encoding='utf-8-sig')
        
        logger.info(f"Salvo: {processed_dir / 'tabela1612_completo.csv'}")
        logger.info(f"Salvo: {processed_dir / 'tabela1612_soja.csv'}")
        logger.info(f"Salvo: {processed_dir / 'tabela1612_total.csv'}")
        
        print(f"\n{'='*70}")
        print("RESUMO DOS DADOS")
        print(f"{'='*70}")
        print(f"  Municipios: {df_completo['Municipio'].nunique()}")
        print(f"  Anos: {df_completo['Ano'].min()} a {df_completo['Ano'].max()}")
        print(f"  Registros: {len(df_completo)}")
        
        # Aplicar filtro de anos se necessário
        if anos_filtro:
            df_completo = df_completo[df_completo['Ano'].isin(anos_filtro)]
            logger.info(f"Filtrado para anos: {min(anos_filtro)} a {max(anos_filtro)}")
        
        return df_completo
        
    except Exception as e:
        logger.error(f"Erro ao processar: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def coletar_soja_sidra_rs(
    force_atualizacao: bool = True,
    raw_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
    anos_filtro: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Coleta específica para dados de soja do Rio Grande do Sul
    
    Args:
        force_atualizacao: Se True, força recarregamento
        raw_dir: Diretório com dados brutos
        processed_dir: Diretório para salvar dados processados
        anos_filtro: Lista de anos para filtrar
        
    Returns:
        DataFrame com dados de soja do RS
    """
    # Carregar dados completos
    df_completo = carregar_tabela_1612(
        force_atualizacao=force_atualizacao,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        anos_filtro=anos_filtro
    )
    
    if df_completo.empty:
        return df_completo
    
    # Filtrar apenas soja
    df_soja = df_completo[df_completo['Produto'] == 'Soja (em grao)'].copy()
    
    # Renomear colunas para padronização
    df_soja = df_soja.rename(columns={
        'Municipio': 'municipio',
        'Codigo': 'codigo_municipio',
        'Ano': 'ano',
        'Area_Plantada_ha': 'area_plantada_hectares',
        'Area_Colhida_ha': 'area_colhida_hectares',
        'Quantidade_Produzida_t': 'producao_toneladas',
        'Rendimento_Medio_kg_ha': 'rendimento_kg_ha',
        'Valor_Producao_1000_R$': 'valor_producao_1000_reais'
    })
    
    return df_soja


def listar_municipios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lista todos os municípios com dados disponíveis
    
    Args:
        df: DataFrame com dados da tabela 1612
        
    Returns:
        DataFrame com lista de municípios
    """
    if df.empty:
        return pd.DataFrame()
    
    municipios = df[['codigo_municipio', 'municipio']].drop_duplicates().sort_values('municipio')
    return municipios


def resumo_por_municipio(df: pd.DataFrame, ano: int = None) -> pd.DataFrame:
    """
    Gera resumo de produção por município
    
    Args:
        df: DataFrame com dados da tabela 1612
        ano: Ano específico para filtrar (opcional)
        
    Returns:
        DataFrame com resumo por município
    """
    if df.empty:
        return pd.DataFrame()
    
    if ano:
        df = df[df['ano'] == ano]
    
    resumo = df.groupby(['codigo_municipio', 'municipio']).agg({
        'area_plantada_hectares': 'sum',
        'area_colhida_hectares': 'sum',
        'producao_toneladas': 'sum',
        'rendimento_kg_ha': 'mean'
    }).reset_index()
    
    return resumo


if __name__ == "__main__":
    # Teste rápido
    print("Testando carregamento da Tabela 1612...")
    df = carregar_tabela_1612(anos_filtro=[2000, 2001, 2002])
    
    if not df.empty:
        print(f"\n✅ Sucesso! DataFrame shape: {df.shape}")
        print("\nPrimeiros registros:")
        print(df.head(10))
        
        # Mostrar estatísticas
        print(f"\n📊 Estatísticas:")
        print(f"   Municípios: {df['Municipio'].nunique()}")
        print(f"   Anos: {df['Ano'].min()} a {df['Ano'].max()}")
        
        # Mostrar colunas disponíveis
        colunas_disp = [c for c in df.columns if c not in ['Codigo', 'Municipio', 'Ano', 'Produto']]
        print(f"   Variáveis: {colunas_disp}")
    else:
        print("\n❌ Falha no carregamento")