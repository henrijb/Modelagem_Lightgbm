"""
Analise Exploratoria dos Dados de Safra e Clima - TCC
VERSAO TRIMESTRAL - Analise por fases do ciclo da soja
Autor: TCC
Data: 2025

CORRECOES APLICADAS:
  - Removido JJA (pos-colheita) de todas as analises
  - Corrigido bug NameError na variavel 'prod'
  - Adicionada analise do target_desvio_pct
  - Corrigido boxplot de produtividade (anual, nao trimestral)
  - Justificativa metodologica nos relatorios
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class AnaliseExploratoriaTrimestral:
    def __init__(self, data_dir: str = "data/processed/integrado_trimestral",
                 output_dir: str = "output/analise_exploratoria_trimestral"):
        self.data_dir = Path(data_dir)
        self.dados = None
        self.dados_safras = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Criar subpastas
        self.temporal_dir = self.output_dir / "analise_temporal_trimestral"
        self.temporal_dir.mkdir(parents=True, exist_ok=True)
        self.produtividade_dir = self.output_dir / "analise_produtividade_trimestral"
        self.produtividade_dir.mkdir(parents=True, exist_ok=True)
        self.clima_dir = self.output_dir / "analise_clima_trimestral"
        self.clima_dir.mkdir(parents=True, exist_ok=True)
        self.avancadas_dir = self.output_dir / "analise_features_avancadas"
        self.avancadas_dir.mkdir(parents=True, exist_ok=True)

        # CORRECAO CRITICA: Apenas trimestres da safra (SON, DJF, MAM)
        # JJA e excluido por ser periodo pos-colheita (pousio)
        self.trimestres_safra = ['SON', 'DJF', 'MAM']
        self.cores_trim = {'SON': '#4ecdc4', 'DJF': '#ff6b6b', 'MAM': '#2ecc71'}

    def carregar_dados(self) -> bool:
        logger.info("=" * 60)
        logger.info("CARREGANDO DADOS TRIMESTRAIS")
        logger.info("=" * 60)
        try:
            # Dataset trimestral longo
            caminho = self.data_dir / "dados_integrados_trimestrais.csv"
            if caminho.exists():
                self.dados = pd.read_csv(caminho)
                logger.info(f"Dataset trimestral carregado: {caminho.name}")
                logger.info(f"   Shape: {self.dados.shape}")
                logger.info(f"   Municipios: {self.dados['municipio'].nunique()}")
                logger.info(f"   Trimestres: {sorted(self.dados['trimestre'].unique())}")

                # CORRECAO: Filtrar apenas trimestres da safra (excluir JJA)
                if 'trimestre' in self.dados.columns:
                    antes = len(self.dados)
                    self.dados = self.dados[self.dados['trimestre'].isin(self.trimestres_safra)].copy()
                    depois = len(self.dados)
                    if antes != depois:
                        logger.info(f"   Filtrados {antes - depois} registros de JJA (pos-colheita)")
            else:
                logger.error("Dataset trimestral nao encontrado")
                return False

            # Dataset por safra (pivotado) - opcional
            caminho_safras = self.data_dir / "dados_integrados_safras.csv"
            if caminho_safras.exists():
                self.dados_safras = pd.read_csv(caminho_safras)
                logger.info(f"Dataset por safra carregado: {caminho_safras.name}")
                logger.info(f"   Shape: {self.dados_safras.shape}")
                logger.info(f"   Features avancadas detectadas: {len(self._detectar_features_avancadas())}")
            else:
                logger.warning("Dataset por safra nao encontrado; features avancadas nao serao analisadas")

            return True
        except Exception as e:
            logger.error(f"Erro: {e}")
            return False

    def _detectar_features_avancadas(self) -> list:
        """Detecta quais features avancadas estao disponiveis no dataset por safra."""
        if self.dados_safras is None:
            return []
        prefixos_avancados = [
            'gdd_', 'dias_secos_', 'dias_secos_consecutivos_',
            'temp_desvio_c_', 'temp_amplitude_c_', 'meses_seca_',
            'anom_precipitacao_mm_', 'anom_temp_media_c_'
        ]
        features = []
        for prefixo in prefixos_avancados:
            cols = [col for col in self.dados_safras.columns if col.startswith(prefixo)]
            features.extend(cols)
        return features

    def visao_geral(self):
        print("\n" + "=" * 60)
        print("VISAO GERAL DOS DADOS TRIMESTRAIS")
        print("=" * 60)
        print(f"\nRegistros: {len(self.dados)}")
        print(f"Municipios: {self.dados['municipio'].nunique()}")
        print(f"Trimestres analisados: {self.trimestres_safra}")
        print(f"  (JJA excluido - periodo pos-colheita)")

        prod = self.dados['produtividade_kg_ha'].dropna()
        print(f"\nProdutividade da Soja:")
        print(f"  Media: {prod.mean():.0f} kg/ha")
        print(f"  Mediana: {prod.median():.0f} kg/ha")
        print(f"  Desvio: {prod.std():.0f} kg/ha")

        # NOVO: Analisar target_desvio_pct se disponivel
        if 'target_desvio_pct' in self.dados.columns:
            desvio = self.dados['target_desvio_pct'].dropna()
            print(f"\nDesvio Percentual de Produtividade (target):")
            print(f"  Media: {desvio.mean():.2f}%")
            print(f"  Mediana: {desvio.median():.2f}%")
            print(f"  Desvio: {desvio.std():.2f}%")
            print(f"  Safras 'Bom' (>=+10%): {(desvio >= 10).sum()} observacoes")
            print(f"  Safras 'Regular' (-10% a +10%): {((desvio >= -10) & (desvio < 10)).sum()} observacoes")
            print(f"  Safras 'Ruim' (<-10%): {(desvio < -10).sum()} observacoes")

        print(f"\nClima por Trimestre:")
        for trim in self.trimestres_safra:
            df_trim = self.dados[self.dados['trimestre'] == trim]
            print(f"\n  {trim}:")
            if 'precipitacao_mm' in df_trim.columns:
                print(f"    Precipitacao: {df_trim['precipitacao_mm'].mean():.0f} mm")
            if 'temp_media_c' in df_trim.columns:
                print(f"    Temperatura: {df_trim['temp_media_c'].mean():.1f} °C")
            if 'oni_index' in df_trim.columns:
                print(f"    ONI: {df_trim['oni_index'].mean():.2f}")

    def analise_produtividade(self):
        """Analise de produtividade - Graficos separados"""
        logger.info("\n" + "=" * 60)
        logger.info("ANALISE DE PRODUTIVIDADE")
        logger.info("=" * 60)

        # CORRECAO: A produtividade e anual (municipio/ano), nao varia por trimestre
        # Entao nao faz sentido boxplot por trimestre

        # GRAFICO 1: Media e variabilidade temporal da produtividade
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        dados_anuais = self.dados.drop_duplicates(['municipio', 'ano'])
        estatisticas_anuais = dados_anuais.groupby('ano')['produtividade_kg_ha'].agg(
            media='mean', desvio='std'
        ).sort_index()
        ax1.plot(estatisticas_anuais.index, estatisticas_anuais['media'],
                 color='#2f7f7b', linewidth=2.5, label='Media anual')
        ax1.fill_between(
            estatisticas_anuais.index,
            estatisticas_anuais['media'] - estatisticas_anuais['desvio'],
            estatisticas_anuais['media'] + estatisticas_anuais['desvio'],
            color='#4ecdc4', alpha=0.25, label='Desvio padrao'
        )

        ax1.set_xlabel('Ano', fontsize=12)
        ax1.set_ylabel('Produtividade Media (kg/ha)', fontsize=12)
        ax1.set_title('Evolucao Temporal da Produtividade Anual (Media +/- DP)', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.produtividade_dir / 'evolucao_produtividade_por_municipio.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Grafico salvo: {self.produtividade_dir / 'evolucao_produtividade_por_municipio.png'}")

        # GRAFICO 2: Produtividade media anual (barras)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        prod_por_ano = self.dados.groupby('ano')['produtividade_kg_ha'].mean().sort_index()
        bars = ax2.bar(prod_por_ano.index.astype(str), prod_por_ano.values,
                       color='#4ecdc4', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Ano', fontsize=12)
        ax2.set_ylabel('Produtividade Media (kg/ha)', fontsize=12)
        ax2.set_title('Produtividade Media Anual (RS)', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=60)

        for bar, val in zip(bars, prod_por_ano.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.produtividade_dir / 'media_produtividade_anual.png', dpi=150)
        plt.close()
        logger.info(f"Grafico salvo: {self.produtividade_dir / 'media_produtividade_anual.png'}")

        # NOVO GRAFICO 3: Distribuicao do desvio percentual (se disponivel)
        if 'target_desvio_pct' in self.dados.columns:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            desvio = self.dados['target_desvio_pct'].dropna()
            ax3.hist(desvio, bins=30, alpha=0.7, color='#ff6b6b', edgecolor='black')
            ax3.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Media historica')
            ax3.axvline(x=10, color='green', linestyle='--', linewidth=2, label='Bom (+10%)')
            ax3.axvline(x=-10, color='red', linestyle='--', linewidth=2, label='Ruim (-10%)')
            ax3.set_xlabel('Desvio Percentual (%)', fontsize=12)
            ax3.set_ylabel('Frequencia', fontsize=12)
            ax3.set_title('Distribuicao do Desvio Percentual de Produtividade', fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.produtividade_dir / 'distribuicao_desvio_percentual.png', dpi=150)
            plt.close()
            logger.info(f"Grafico salvo: {self.produtividade_dir / 'distribuicao_desvio_percentual.png'}")

        # GRAFICO 4: Produtividade por municipio
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        prod_mun = self.dados.groupby('municipio')['produtividade_kg_ha'].mean().sort_values()
        cores_mun = plt.cm.Greens(np.linspace(0.4, 0.8, len(prod_mun)))
        ax4.barh(range(len(prod_mun)), prod_mun.values, color=cores_mun, alpha=0.7, edgecolor='black')
        ax4.set_yticks(range(len(prod_mun)))
        ax4.set_yticklabels(prod_mun.index, fontsize=10)
        ax4.set_xlabel('Produtividade Media (kg/ha)', fontsize=12)
        ax4.set_title('Produtividade Media por Municipio', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.produtividade_dir / 'produtividade_por_municipio.png', dpi=150)
        plt.close()
        logger.info(f"Grafico salvo: {self.produtividade_dir / 'produtividade_por_municipio.png'}")

        # CORRECAO: usar variavel local 'prod_media' em vez de 'prod'
        prod_media = self.dados['produtividade_kg_ha'].dropna()
        print(f"\nProdutividade Media: {prod_media.mean():.0f} kg/ha")
        print(f"Graficos salvos em: {self.produtividade_dir}")

    def analise_clima(self):
        """Analise clima vs produtividade - Graficos separados"""
        logger.info("\n" + "=" * 60)
        logger.info("ANALISE CLIMA vs PRODUTIVIDADE POR TRIMESTRE")
        logger.info("=" * 60)

        # CORRECAO: usar apenas trimestres da safra
        # GRAFICO 1: Temperatura vs Produtividade (por trimestre)
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        for trim in self.trimestres_safra:
            if 'temp_media_c' not in self.dados.columns:
                continue
            df_trim = self.dados[
                self.dados['trimestre'] == trim
            ][['temp_media_c', 'produtividade_kg_ha']].apply(pd.to_numeric, errors='coerce')
            df_trim = df_trim[np.isfinite(df_trim).all(axis=1)]
            if df_trim.empty:
                continue
            ax1.scatter(df_trim['temp_media_c'], df_trim['produtividade_kg_ha'],
                        alpha=0.15, label=trim, s=8, color=self.cores_trim[trim])
            if len(df_trim) > 1 and df_trim['temp_media_c'].nunique() > 1:
                z = np.polyfit(df_trim['temp_media_c'].to_numpy(),
                               df_trim['produtividade_kg_ha'].to_numpy(), 1)
                p = np.poly1d(z)
                x_range = np.linspace(df_trim['temp_media_c'].min(),
                                      df_trim['temp_media_c'].max(), 100)
                ax1.plot(x_range, p(x_range), '--', alpha=0.5, color=self.cores_trim[trim], linewidth=1)

        ax1.set_xlabel('Temperatura (°C)', fontsize=12)
        ax1.set_ylabel('Produtividade (kg/ha)', fontsize=12)
        ax1.set_title('Temperatura vs Produtividade por Trimestre', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.clima_dir / 'temperatura_vs_produtividade.png', dpi=150)
        plt.close()
        logger.info(f"Grafico salvo: {self.clima_dir / 'temperatura_vs_produtividade.png'}")

        # GRAFICO 2: Precipitacao vs Produtividade
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        for trim in self.trimestres_safra:
            if 'precipitacao_mm' not in self.dados.columns:
                continue
            df_trim = self.dados[
                self.dados['trimestre'] == trim
            ][['precipitacao_mm', 'produtividade_kg_ha']].apply(pd.to_numeric, errors='coerce')
            df_trim = df_trim[np.isfinite(df_trim).all(axis=1)]
            if df_trim.empty:
                continue
            ax2.scatter(df_trim['precipitacao_mm'], df_trim['produtividade_kg_ha'],
                        alpha=0.15, label=trim, s=8, color=self.cores_trim[trim])
            if len(df_trim) > 1 and df_trim['precipitacao_mm'].nunique() > 1:
                z = np.polyfit(df_trim['precipitacao_mm'].to_numpy(),
                               df_trim['produtividade_kg_ha'].to_numpy(), 1)
                p = np.poly1d(z)
                x_range = np.linspace(df_trim['precipitacao_mm'].min(),
                                      df_trim['precipitacao_mm'].max(), 100)
                ax2.plot(x_range, p(x_range), '--', alpha=0.5, color=self.cores_trim[trim], linewidth=1)

        ax2.set_xlabel('Precipitacao (mm)', fontsize=12)
        ax2.set_ylabel('Produtividade (kg/ha)', fontsize=12)
        ax2.set_title('Precipitacao vs Produtividade por Trimestre', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.clima_dir / 'precipitacao_vs_produtividade.png', dpi=150)
        plt.close()
        logger.info(f"Grafico salvo: {self.clima_dir / 'precipitacao_vs_produtividade.png'}")

        # NOVO GRAFICO 3: Clima vs Desvio Percentual (target real)
        if 'target_desvio_pct' in self.dados.columns:
            fig3, ax3 = plt.subplots(figsize=(12, 8))
            for trim in self.trimestres_safra:
                if 'precipitacao_mm' not in self.dados.columns:
                    continue
                df_trim = self.dados[
                    self.dados['trimestre'] == trim
                ][['precipitacao_mm', 'target_desvio_pct']].apply(pd.to_numeric, errors='coerce')
                df_trim = df_trim[np.isfinite(df_trim).all(axis=1)]
                if df_trim.empty:
                    continue
                ax3.scatter(df_trim['precipitacao_mm'], df_trim['target_desvio_pct'],
                            alpha=0.15, label=trim, s=8, color=self.cores_trim[trim])

            ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Media historica')
            ax3.set_xlabel('Precipitacao (mm)', fontsize=12)
            ax3.set_ylabel('Desvio Percentual (%)', fontsize=12)
            ax3.set_title('Precipitacao vs Desvio Percentual de Produtividade', fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.clima_dir / 'precipitacao_vs_desvio_percentual.png', dpi=150)
            plt.close()
            logger.info(f"Grafico salvo: {self.clima_dir / 'precipitacao_vs_desvio_percentual.png'}")

        # GRAFICO 4: Correlacoes por trimestre (barras)
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        correlacoes = []
        for trim in self.trimestres_safra:
            df_trim = self.dados[self.dados['trimestre'] == trim]
            corr_temp = df_trim['temp_media_c'].corr(df_trim['produtividade_kg_ha']) if 'temp_media_c' in df_trim.columns else np.nan
            corr_precip = df_trim['precipitacao_mm'].corr(df_trim['produtividade_kg_ha']) if 'precipitacao_mm' in df_trim.columns else np.nan
            correlacoes.append({'trimestre': trim, 'temperatura': corr_temp, 'precipitacao': corr_precip})

        df_corr = pd.DataFrame(correlacoes)
        x = range(len(df_corr))
        width = 0.35
        bars1 = ax4.bar([i - width / 2 for i in x], df_corr['temperatura'], width,
                        label='Temperatura', color='#ff6b6b')
        bars2 = ax4.bar([i + width / 2 for i in x], df_corr['precipitacao'], width,
                        label='Precipitacao', color='#4ecdc4')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels(df_corr['trimestre'])
        ax4.set_ylabel('Correlacao com Produtividade', fontsize=12)
        ax4.set_title('Correlacoes por Trimestre', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        valores_corr = pd.concat([df_corr['temperatura'], df_corr['precipitacao']]).dropna()
        limite_corr = max(0.2, np.abs(valores_corr).max() * 1.35) if not valores_corr.empty else 0.2
        ax4.set_ylim(-limite_corr, limite_corr)

        for bar, val in zip(bars1, df_corr['temperatura']):
            if not np.isnan(val):
                deslocamento = limite_corr * 0.04
                valor_exibicao = 0.0 if abs(val) < 0.0005 else val
                ax4.text(bar.get_x() + bar.get_width() / 2,
                         val + (deslocamento if val >= 0 else -deslocamento),
                         f'{valor_exibicao:.3f}', ha='center',
                         va='bottom' if val > 0 else 'top', fontsize=9)
        for bar, val in zip(bars2, df_corr['precipitacao']):
            if not np.isnan(val):
                deslocamento = limite_corr * 0.04
                valor_exibicao = 0.0 if abs(val) < 0.0005 else val
                ax4.text(bar.get_x() + bar.get_width() / 2,
                         val + (deslocamento if val >= 0 else -deslocamento),
                         f'{valor_exibicao:.3f}', ha='center',
                         va='bottom' if val > 0 else 'top', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.clima_dir / 'correlacoes_por_trimestre.png', dpi=150)
        plt.close()
        logger.info(f"Grafico salvo: {self.clima_dir / 'correlacoes_por_trimestre.png'}")

        # GRAFICO 5: ONI por trimestre
        if 'oni_index' in self.dados.columns:
            fig5, ax5 = plt.subplots(figsize=(14, 8))
            for trim in self.trimestres_safra:
                df_trim = self.dados[self.dados['trimestre'] == trim]
                oni_anual = df_trim.groupby('ano')['oni_index'].mean()
                ax5.plot(oni_anual.index, oni_anual.values, marker='^', linewidth=2,
                         label=trim, color=self.cores_trim[trim])
            ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='El Nino')
            ax5.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5, label='La Nina')
            ax5.set_xlabel('Ano', fontsize=12)
            ax5.set_ylabel('Indice ONI', fontsize=12)
            ax5.set_title('Evolucao do ONI por Trimestre', fontsize=14)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.clima_dir / 'oni_por_trimestre.png', dpi=150)
            plt.close()
            logger.info(f"Grafico salvo: {self.clima_dir / 'oni_por_trimestre.png'}")

        # GRAFICO 6: Boxplot da temperatura por trimestre
        if 'temp_media_c' in self.dados.columns:
            fig6, ax6 = plt.subplots(figsize=(12, 7))
            df_temp = self.dados[self.dados['trimestre'].isin(self.trimestres_safra)]
            df_temp.boxplot(column='temp_media_c', by='trimestre', ax=ax6, grid=False)
            ax6.set_title('Temperatura por Trimestre', fontsize=14)
            ax6.set_xlabel('Trimestre', fontsize=12)
            ax6.set_ylabel('Temperatura (°C)', fontsize=12)
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(self.clima_dir / 'boxplot_temperatura_por_trimestre.png', dpi=150)
            plt.close()
            logger.info(f"Grafico salvo: {self.clima_dir / 'boxplot_temperatura_por_trimestre.png'}")

        # GRAFICO 7: Boxplot da precipitacao por trimestre
        if 'precipitacao_mm' in self.dados.columns:
            fig7, ax7 = plt.subplots(figsize=(12, 7))
            df_precip = self.dados[self.dados['trimestre'].isin(self.trimestres_safra)]
            df_precip.boxplot(column='precipitacao_mm', by='trimestre', ax=ax7, grid=False)
            ax7.set_title('Precipitacao por Trimestre', fontsize=14)
            ax7.set_xlabel('Trimestre', fontsize=12)
            ax7.set_ylabel('Precipitacao (mm)', fontsize=12)
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(self.clima_dir / 'boxplot_precipitacao_por_trimestre.png', dpi=150)
            plt.close()
            logger.info(f"Grafico salvo: {self.clima_dir / 'boxplot_precipitacao_por_trimestre.png'}")

        print("\nCorrelacoes com Produtividade por Trimestre:")
        for trim in self.trimestres_safra:
            df_trim = self.dados[self.dados['trimestre'] == trim]
            corr_temp = df_trim['temp_media_c'].corr(df_trim['produtividade_kg_ha']) if 'temp_media_c' in df_trim.columns else np.nan
            corr_precip = df_trim['precipitacao_mm'].corr(df_trim['produtividade_kg_ha']) if 'precipitacao_mm' in df_trim.columns else np.nan
            print(f"\n  {trim}:")
            if not np.isnan(corr_temp):
                print(f"    Temperatura: {corr_temp:.3f}")
            if not np.isnan(corr_precip):
                print(f"    Precipitacao: {corr_precip:.3f}")

        print(f"\nGraficos climaticos salvos em: {self.clima_dir}")

    def analise_features_avancadas(self):
        """Analise das features avancadas (GDD, dias secos, anomalias)"""
        if self.dados_safras is None:
            logger.info("Dataset por safra nao disponivel; pulando analise de features avancadas")
            return

        features_avancadas = self._detectar_features_avancadas()
        if not features_avancadas:
            logger.info("Nenhuma feature avancada detectada; pulando analise")
            return

        logger.info("\n" + "=" * 60)
        logger.info("ANALISE DE FEATURES AVANCADAS")
        logger.info("=" * 60)
        logger.info(f"Features detectadas: {len(features_avancadas)}")

        tipos_features = {
            'gdd_': 'Graus-Dia de Crescimento (GDD)',
            'dias_secos_consecutivos_': 'Maior Sequencia de Dias Secos',
            'dias_secos_': 'Dias Secos',
            'temp_desvio_c_': 'Desvio Padrao da Temperatura',
            'temp_amplitude_c_': 'Amplitude Termica',
            'meses_seca_': 'Meses Secos (< 30mm)',
            'anom_precipitacao_mm_': 'Anomalia de Precipitacao',
            'anom_temp_media_c_': 'Anomalia de Temperatura',
        }

        for prefixo, descricao in tipos_features.items():
            cols = [col for col in features_avancadas if col.startswith(prefixo)]
            if not cols:
                continue

            logger.info(f"\nAnalisando: {descricao}")

            # GRAFICO: Correlacao com produtividade por trimestre
            fig, ax = plt.subplots(figsize=(10, 6))
            correlacoes = []
            for trim in self.trimestres_safra:
                col_trim = f'{prefixo}{trim}'
                if col_trim in self.dados_safras.columns:
                    corr = self.dados_safras[col_trim].corr(
                        self.dados_safras['produtividade_kg_ha']
                    )
                    correlacoes.append({'trimestre': trim, 'correlacao': corr})

            if correlacoes:
                df_corr = pd.DataFrame(correlacoes)
                x = range(len(df_corr))
                bars = ax.bar(
                    x, df_corr['correlacao'],
                    color=[self.cores_trim[t] for t in df_corr['trimestre']],
                    alpha=0.7, edgecolor='black'
                )
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(df_corr['trimestre'])
                ax.set_ylabel('Correlacao com Produtividade', fontsize=12)
                ax.set_title(f'{descricao} - Correlacao por Trimestre', fontsize=14)
                ax.grid(True, alpha=0.3, axis='y')

                valores_corr = df_corr['correlacao'].dropna()
                limite_corr = max(0.2, np.abs(valores_corr).max() * 1.35) if not valores_corr.empty else 0.2
                ax.set_ylim(-limite_corr, limite_corr)

                for bar, val in zip(bars, df_corr['correlacao']):
                    if not np.isnan(val):
                        deslocamento = limite_corr * 0.04
                        valor_exibicao = 0.0 if abs(val) < 0.0005 else val
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            val + (deslocamento if val >= 0 else -deslocamento),
                            f'{valor_exibicao:.3f}', ha='center',
                            va='bottom' if val > 0 else 'top',
                            fontsize=10, fontweight='bold'
                        )

                plt.tight_layout()
                nome_arquivo = f'correlacao_{prefixo.rstrip("_")}_por_trimestre.png'
                plt.savefig(self.avancadas_dir / nome_arquivo, dpi=150)
                plt.close()
                logger.info(f"Grafico salvo: {self.avancadas_dir / nome_arquivo}")

                print(f"\n{descricao} - Correlacoes com Produtividade:")
                for _, row in df_corr.iterrows():
                    if not np.isnan(row['correlacao']):
                        print(f"  {row['trimestre']}: {row['correlacao']:.3f}")

        # GRAFICO: Matriz de correlacao das features avancadas
        if len(features_avancadas) > 1:
            fig, ax = plt.subplots(figsize=(12, 10))
            cols_para_corr = ['produtividade_kg_ha'] + features_avancadas[:15]
            correlacao = self.dados_safras[cols_para_corr].corr()
            sns.heatmap(correlacao, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f',
                        square=True, cbar_kws={"shrink": 0.8})
            ax.set_title('Matriz de Correlacao - Features Avancadas', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.avancadas_dir / 'matriz_correlacao_features_avancadas.png', dpi=150)
            plt.close()
            logger.info(f"Grafico salvo: {self.avancadas_dir / 'matriz_correlacao_features_avancadas.png'}")

        print(f"\nGraficos de features avancadas salvos em: {self.avancadas_dir}")

    def analise_temporal_trimestral(self):
        """Analise temporal trimestral - Graficos separados"""
        logger.info("\n" + "=" * 60)
        logger.info("ANALISE TEMPORAL TRIMESTRAL")
        logger.info("=" * 60)

        # Temperatura
        if 'temp_media_c' in self.dados.columns:
            fig1, ax1 = plt.subplots(figsize=(14, 8))
            for trim in self.trimestres_safra:
                df_trim = self.dados[self.dados['trimestre'] == trim]
                temp_anual = df_trim.groupby('ano')['temp_media_c'].mean()
                ax1.plot(temp_anual.index, temp_anual.values, marker='o', linewidth=2,
                         label=trim, color=self.cores_trim[trim])
            ax1.set_xlabel('Ano', fontsize=12)
            ax1.set_ylabel('Temperatura (°C)', fontsize=12)
            ax1.set_title('Evolucao da Temperatura por Trimestre', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.temporal_dir / 'temperatura_por_trimestre.png', dpi=150)
            plt.close()

        # Precipitacao
        if 'precipitacao_mm' in self.dados.columns:
            fig2, ax2 = plt.subplots(figsize=(14, 8))
            for trim in self.trimestres_safra:
                df_trim = self.dados[self.dados['trimestre'] == trim]
                precip_anual = df_trim.groupby('ano')['precipitacao_mm'].mean()
                ax2.plot(precip_anual.index, precip_anual.values, marker='s', linewidth=2,
                         label=trim, color=self.cores_trim[trim])
            ax2.set_xlabel('Ano', fontsize=12)
            ax2.set_ylabel('Precipitacao (mm)', fontsize=12)
            ax2.set_title('Evolucao da Precipitacao por Trimestre', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.temporal_dir / 'precipitacao_por_trimestre.png', dpi=150)
            plt.close()

        # ONI
        if 'oni_index' in self.dados.columns:
            fig3, ax3 = plt.subplots(figsize=(14, 8))
            for trim in self.trimestres_safra:
                df_trim = self.dados[self.dados['trimestre'] == trim]
                oni_anual = df_trim.groupby('ano')['oni_index'].mean()
                ax3.plot(oni_anual.index, oni_anual.values, marker='^', linewidth=2,
                         label=trim, color=self.cores_trim[trim])
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='El Nino')
            ax3.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5, label='La Nina')
            ax3.set_xlabel('Ano', fontsize=12)
            ax3.set_ylabel('Indice ONI', fontsize=12)
            ax3.set_title('Evolucao do ONI por Trimestre', fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.temporal_dir / 'oni_por_trimestre.png', dpi=150)
            plt.close()

        # Comparacao das variaveis que possuem valor trimestral
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        graficos = [
            ('Temperatura (°C)', 'temp_media_c'),
            ('Precipitacao (mm)', 'precipitacao_mm'),
            ('Indice ONI', 'oni_index'),
        ]
        for idx, (ylabel, var) in enumerate(graficos):
            ax = axes[idx]
            if var not in self.dados.columns:
                ax.set_visible(False)
                continue
            for trim in self.trimestres_safra:
                df_trim = self.dados[self.dados['trimestre'] == trim]
                valores_anual = df_trim.groupby('ano')[var].mean()
                ax.plot(valores_anual.index, valores_anual.values, marker='o', linewidth=2,
                        label=trim, color=self.cores_trim[trim])
            ax.set_xlabel('Ano', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{var.replace("_", " ").title()} por Trimestre', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.temporal_dir / 'comparacao_trimestres.png', dpi=150)
        plt.close()

        print(f"\nGraficos temporais salvos em: {self.temporal_dir}")

    def gerar_relatorio(self):
        logger.info("\n" + "=" * 60)
        logger.info("GERANDO RELATORIO")
        logger.info("=" * 60)
        with open(self.output_dir / 'relatorio_analise_trimestral.md', 'w', encoding='utf-8') as f:
            f.write("# Relatorio de Analise Exploratoria Trimestral\n\n")
            f.write("## 1. Visao Geral\n\n")
            f.write(f"- **Registros:** {len(self.dados)}\n")
            f.write(f"- **Municipios:** {self.dados['municipio'].nunique()}\n")
            f.write(f"- **Trimestres analisados:** {self.trimestres_safra}\n")
            f.write(f"- **JJA excluido:** periodo pos-colheita (pousio)\n\n")

            prod = self.dados['produtividade_kg_ha'].dropna()
            f.write("## 2. Produtividade\n\n")
            f.write(f"- **Media:** {prod.mean():.0f} kg/ha\n")
            f.write(f"- **Mediana:** {prod.median():.0f} kg/ha\n")
            f.write(f"- **Desvio:** {prod.std():.0f} kg/ha\n\n")
            f.write("**Observacao:** a produtividade e anual (municipio/ano) "
                    "e aparece repetida entre trimestres no dataset integrado.\n\n")

            # NOVO: Secao do desvio percentual
            if 'target_desvio_pct' in self.dados.columns:
                desvio = self.dados['target_desvio_pct'].dropna()
                f.write("## 3. Desvio Percentual de Produtividade (Target)\n\n")
                f.write(f"- **Media:** {desvio.mean():.2f}%\n")
                f.write(f"- **Mediana:** {desvio.median():.2f}%\n")
                f.write(f"- **Desvio:** {desvio.std():.2f}%\n")
                f.write(f"- **Safra Bom (>=+10%):** {(desvio >= 10).sum()} observacoes\n")
                f.write(f"- **Safra Regular (-10% a +10%):** {((desvio >= -10) & (desvio < 10)).sum()} observacoes\n")
                f.write(f"- **Safra Ruim (<-10%):** {(desvio < -10).sum()} observacoes\n\n")

            f.write("## 4. Correlacoes por Trimestre\n\n")
            f.write("| Trimestre | Temperatura | Precipitacao |\n")
            f.write("|-----------|-------------|--------------|\n")
            for trim in self.trimestres_safra:
                df_trim = self.dados[self.dados['trimestre'] == trim]
                corr_temp = df_trim['temp_media_c'].corr(
                    df_trim['produtividade_kg_ha']
                ) if 'temp_media_c' in df_trim.columns else np.nan
                corr_precip = df_trim['precipitacao_mm'].corr(
                    df_trim['produtividade_kg_ha']
                ) if 'precipitacao_mm' in df_trim.columns else np.nan
                f.write(f"| {trim} | {corr_temp:.3f} | {corr_precip:.3f} |\n")

            # Features avancadas
            if self.dados_safras is not None:
                features_avancadas = self._detectar_features_avancadas()
                if features_avancadas:
                    f.write("\n## 5. Features Avancadas Disponiveis\n\n")
                    f.write(f"O dataset por safra contem {len(features_avancadas)} "
                            "features avancadas:\n\n")
                    tipos_features = {
                        'gdd_': 'Graus-Dia de Crescimento (GDD)',
                        'dias_secos_consecutivos_': 'Maior Sequencia de Dias Secos',
                        'dias_secos_': 'Dias Secos',
                        'temp_desvio_c_': 'Desvio Padrao da Temperatura',
                        'temp_amplitude_c_': 'Amplitude Termica',
                        'meses_seca_': 'Meses Secos (< 30mm)',
                        'anom_precipitacao_mm_': 'Anomalia de Precipitacao',
                        'anom_temp_media_c_': 'Anomalia de Temperatura',
                    }
                    for prefixo, descricao in tipos_features.items():
                        cols = [col for col in features_avancadas if col.startswith(prefixo)]
                        if cols:
                            f.write(f"### {descricao}\n")
                            f.write(f"- **Colunas:** {', '.join(cols)}\n")
                            f.write("- **Correlacoes com produtividade:**\n")
                            for col in cols:
                                corr = self.dados_safras[col].corr(
                                    self.dados_safras['produtividade_kg_ha']
                                )
                                f.write(f"  - {col}: {corr:.3f}\n")
                            f.write("\n")

            f.write("\n## 6. Conclusoes\n\n")
            f.write("A analise exploratoria confirma que:\n")
            f.write("1. Os trimestres SON, DJF e MAM correspondem ao ciclo da soja no RS\n")
            f.write("2. JJA foi excluido por ser periodo pos-colheita\n")
            f.write("3. O desvio percentual de produtividade e o target adequado para isolar o efeito climatico\n")
            f.write("4. Features avancadas (GDD, dias secos, anomalias) complementam as variaveis basicas\n")

        logger.info(f"Relatorio gerado: {self.output_dir / 'relatorio_analise_trimestral.md'}")

    def executar(self):
        print("\n" + "=" * 60)
        print("ANALISE EXPLORATORIA TRIMESTRAL")
        print("=" * 60)
        if not self.carregar_dados():
            logger.error("Falha ao carregar dados")
            return
        self.visao_geral()
        self.analise_produtividade()
        self.analise_clima()
        self.analise_features_avancadas()
        self.analise_temporal_trimestral()
        self.gerar_relatorio()
        print("\n" + "=" * 60)
        print("ANALISE EXPLORATORIA TRIMESTRAL CONCLUIDA!")
        print(f"Graficos de produtividade: {self.produtividade_dir}")
        print(f"Graficos climaticos: {self.clima_dir}")
        print(f"Graficos de features avancadas: {self.avancadas_dir}")
        print(f"Graficos temporais: {self.temporal_dir}")
        print(f"Relatorio: {self.output_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Executa a analise exploratoria trimestral")
    parser.add_argument("--data-dir", default="data/processed/integrado_trimestral")
    parser.add_argument("--output-dir", default="output/analise_exploratoria_trimestral")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ANALISE EXPLORATORIA - Safra de Soja x Clima (TRIMESTRAL)")
    print("=" * 60)

    analise = AnaliseExploratoriaTrimestral(args.data_dir, args.output_dir)
    analise.executar()


if __name__ == "__main__":
    main()