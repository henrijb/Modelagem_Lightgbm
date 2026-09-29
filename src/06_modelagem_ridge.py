"""
Modelagem comparativa com Ridge para produtividade da soja.
VERSAO CORRIGIDA: split 80/20 aleatorio, target desvio percentual,
sem feature 'ano' e sem features JJA.
Usa os datasets treino_80pct.csv e teste_20pct.csv gerados pelo integrador.
Suporta tanto valores brutos (precipitacao_mm_, temp_media_c_) quanto
anomalias (anom_precipitacao_mm_, anom_temp_media_c_) e as features
avancadas do ERA5-Land (gdd_, dias_secos_, temp_desvio_c_*, etc.).
"""
import argparse
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prefixos de features climaticas reconhecidos pelo modelo.
# Quando --usar-anomalia sim (padrao), o integrador gera colunas do tipo
# anom_precipitacao_mm_<trimestre> e anom_temp_media_c_<trimestre>, que
# representam o desvio da media historica do proprio municipio.
# Quando --usar-anomalia nao, usamos os valores brutos, incluindo as
# features avancadas calculadas no integrador:
# - temp_desvio_c_<trim>        : desvio padrao da temperatura no trim.
# - temp_amplitude_c_<trim>     : amplitude termica (max - min) no trim.
# - meses_seca_<trim>           : numero de meses secos (< 30 mm).
# - gdd_<trim>                  : graus-dia de crescimento (ERA5 diario).
# - dias_secos_<trim>           : dias com precip < 1 mm (ERA5 diario).
# - dias_secos_consecutivos_<trim> : maior sequencia de dias secos.
PREFIXOS_CLIMA_BRUTO = (
    'precipitacao_mm_', 'temp_media_c_', 'oni_index_',
    'temp_desvio_c_', 'temp_amplitude_c_', 'meses_seca_',
    'gdd_', 'dias_secos_', 'dias_secos_consecutivos_',
)
PREFIXOS_CLIMA_ANOMALIA = (
    'anom_precipitacao_mm_', 'anom_temp_media_c_', 'oni_index_',
)


class ModelagemRidge:
    def __init__(self, data_dir, output_dir, comparison_dir,
                 usar_anomalia=True, usar_lag1=True):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.comparison_dir = Path(comparison_dir)
        self.usar_anomalia = usar_anomalia
        self.usar_lag1 = usar_lag1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        self.dados_treino = None
        self.dados_teste = None
        self.dados = None
        self.features = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def carregar_dados(self):
        """Carrega os datasets treino/teste 80/20 gerados pelo integrador."""
        caminho_treino = self.data_dir / "treino_80pct.csv"
        caminho_teste = self.data_dir / "teste_20pct.csv"
        
        if not (caminho_treino.exists() and caminho_teste.exists()):
            raise FileNotFoundError(
                f'Datasets treino_80pct.csv e teste_20pct.csv nao encontrados em {self.data_dir}. '
                f'Execute o 02_integrador_safra_clima.py primeiro.'
            )
        
        self.dados_treino = pd.read_csv(caminho_treino)
        self.dados_teste = pd.read_csv(caminho_teste)
        
        # Combina para preparar features consistentemente
        self.dados = pd.concat([self.dados_treino, self.dados_teste], ignore_index=True)
        
        logger.info('Dados carregados: treino=%d, teste=%d (total=%d linhas)',
                    len(self.dados_treino), len(self.dados_teste), len(self.dados))
        logger.info('Municipios: %d', self.dados['municipio'].nunique())

    def preparar_dados(self):
        """Prepara features removendo 'ano' e JJA, usando target_desvio_pct."""
        dados = self.dados.copy()
        
        # Seleciona prefixos climaticos conforme flag
        prefixos_clima = PREFIXOS_CLIMA_ANOMALIA if self.usar_anomalia else PREFIXOS_CLIMA_BRUTO
        features = [
            coluna for coluna in dados.columns
            if coluna.startswith(prefixos_clima)
        ]
        
        # One-hot encoding para municipio
        dummies = pd.get_dummies(dados['municipio'], prefix='mun', dtype=float)
        dados = pd.concat([dados, dummies], axis=1)
        features.extend(dummies.columns.tolist())
        
        # REMOVIDO: 'ano' (causa vazamento temporal - importancia 55 no modelo anterior)
        # REMOVIDO: features de JJA (pos-colheita - nao faz sentido agronomico)
        features = [f for f in features if not f.endswith('_JJA')]
        
        # Lag de produtividade (ano anterior)
        if self.usar_lag1 and 'produtividade_kg_ha_lag1' in dados.columns:
            features.append('produtividade_kg_ha_lag1')
        
        # Drop NaNs nas features
        dados = dados.dropna(subset=features + ['target_desvio_pct']).copy()
        
        # Re-separa em treino e teste mantendo os indices originais
        n_treino = len(self.dados_treino)
        dados_treino_limpo = dados.iloc[:n_treino]
        dados_teste_limpo = dados.iloc[n_treino:]
        
        # Verifica se ainda temos dados suficientes
        if len(dados_treino_limpo) == 0 or len(dados_teste_limpo) == 0:
            raise ValueError("Nenhum dado valido apos filtrar NaNs.")
        
        self.dados = dados
        self.features = features
        
        # Target: target_desvio_pct (desvio percentual) em vez de produtividade_kg_ha
        self.X_train = dados_treino_limpo[features].to_numpy(dtype=float)
        self.X_test = dados_teste_limpo[features].to_numpy(dtype=float)
        self.y_train = dados_treino_limpo['target_desvio_pct'].to_numpy(dtype=float)
        self.y_test = dados_teste_limpo['target_desvio_pct'].to_numpy(dtype=float)
        
        logger.info('Amostras validas: treino=%d, teste=%d; features: %d',
                    len(self.X_train), len(self.X_test), len(features))
        logger.info('Features climaticas detectadas: %d',
                    sum(1 for f in features if f.startswith(prefixos_clima)))
        logger.info('Target: target_desvio_pct (desvio percentual da media historica)')

    def treinar_regressao(self):
        """Treina Ridge com target_desvio_pct."""
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=10.0)),
        ])
        modelo.fit(self.X_train, self.y_train)
        previsoes = modelo.predict(self.X_test)
        
        metricas = {
            'r2': float(r2_score(self.y_test, previsoes)),
            'rmse': float(np.sqrt(mean_squared_error(self.y_test, previsoes))),
            'mae': float(mean_absolute_error(self.y_test, previsoes)),
        }
        
        self.modelo_regressao = modelo
        self.metricas_regressao = metricas
        
        joblib.dump(modelo, self.output_dir / 'modelo_ridge.pkl')
        
        print("\n" + "=" * 50)
        print("RESULTADOS - RIDGE (REGRESSAO)")
        print("=" * 50)
        print(f"  R²: {metricas['r2']:.3f}")
        print(f"  RMSE: {metricas['rmse']:.1f}%")
        print(f"  MAE: {metricas['mae']:.1f}%")
        
        return metricas

    def treinar_classificacao(self):
        """Treina RidgeClassifier com classes baseadas em desvio percentual."""
        # Classificação baseada em target_desvio_pct (±10%)
        limiar_bom = 10.0    # +10%
        limiar_ruim = -10.0  # -10%
        
        def classificar(valor):
            if valor >= limiar_bom:
                return 'Bom'
            if valor < limiar_ruim:
                return 'Ruim'
            return 'Regular'
        
        # Aplica classificação em treino e teste
        classes_treino = pd.Series(self.y_train).apply(classificar)
        classes_teste = pd.Series(self.y_test).apply(classificar)
        
        encoder = LabelEncoder()
        # Fit no treino + teste para garantir todas as classes
        todas_classes = pd.concat([classes_treino, classes_teste])
        encoder.fit(todas_classes)
        
        y_train_c = encoder.transform(classes_treino)
        y_test_c = encoder.transform(classes_teste)
        
        modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeClassifier(alpha=10.0, class_weight='balanced')),
        ])
        modelo.fit(self.X_train, y_train_c)
        previsoes = modelo.predict(self.X_test)
        
        metricas = {
            'acuracia': float(accuracy_score(y_test_c, previsoes)),
            'report': classification_report(
                y_test_c,
                previsoes,
                labels=list(range(len(encoder.classes_))),
                target_names=encoder.classes_,
                zero_division=0,
            ),
            'limiar_bom': float(limiar_bom),
            'limiar_ruim': float(limiar_ruim),
        }
        
        self.metricas_classificacao = metricas
        self.encoder = encoder
        
        joblib.dump(modelo, self.output_dir / 'modelo_ridge_classificacao.pkl')
        joblib.dump(encoder, self.output_dir / 'label_encoder_classes.pkl')
        
        print("\n" + "=" * 50)
        print("RESULTADOS - RIDGE (CLASSIFICACAO)")
        print("=" * 50)
        print(f"  Acuracia: {metricas['acuracia']:.3f}")
        print(f"\n{metricas['report']}")
        
        return metricas

    def gerar_relatorios(self):
        """Gera relatórios e comparação com outros modelos."""
        # Métricas Ridge
        with open(self.output_dir / 'metricas_ridge.json', 'w', encoding='utf-8') as arquivo:
            json.dump({
                'regressao': self.metricas_regressao,
                'classificacao': {
                    key: value for key, value in self.metricas_classificacao.items() if key != 'report'
                },
                'amostras_treino': len(self.X_train),
                'amostras_teste': len(self.X_test),
                'features': len(self.features),
            }, arquivo, indent=2)
        
        # Relatório Ridge
        with open(self.output_dir / 'relatorio_ridge.md', 'w', encoding='utf-8') as arquivo:
            arquivo.write('# Relatorio do modelo Ridge\n\n')
            arquivo.write('O modelo usa split 80/20 aleatorio (estratificado por municipio),\n')
            arquivo.write('target desvio percentual (target_desvio_pct), sem feature "ano" e sem features JJA.\n\n')
            arquivo.write('## Regressao\n\n')
            arquivo.write(f"- R2 (teste): {self.metricas_regressao['r2']:.3f}\n")
            arquivo.write(f"- RMSE: {self.metricas_regressao['rmse']:.1f}%\n")
            arquivo.write(f"- MAE: {self.metricas_regressao['mae']:.1f}%\n")
            arquivo.write(f"- Amostras de treino: {len(self.X_train)}\n")
            arquivo.write(f"- Amostras de teste: {len(self.X_test)}\n")
            arquivo.write(f"- Total de features: {len(self.features)}\n\n")
            arquivo.write('## Classificacao\n\n')
            arquivo.write(f"- Acuracia: {self.metricas_classificacao['acuracia']:.3f}\n\n")
            arquivo.write('```text\n')
            arquivo.write(self.metricas_classificacao['report'])
            arquivo.write('```\n')
        
        # Comparação com outros modelos
        modelos = {
            'LightGBM': self._ler_metricas('modelagem_trimestral', 'metricas_lightgbm.json'),
            'Gradient Boosting': self._ler_metricas('modelagem_gradient_boosting', 'metricas_gradient_boosting.json'),
            'Ridge': {
                'r2': self.metricas_regressao['r2'],
                'rmse': self.metricas_regressao['rmse'],
                'mae': self.metricas_regressao['mae'],
                'acuracia': self.metricas_classificacao['acuracia'],
            },
        }
        comparacao = pd.DataFrame([
            {'modelo': nome, **metricas} for nome, metricas in modelos.items()
        ])
        comparacao.to_csv(self.comparison_dir / 'comparacao_modelos.csv', index=False)
        
        with open(self.comparison_dir / 'comparacao_modelos.md', 'w', encoding='utf-8') as arquivo:
            arquivo.write('# Comparacao entre modelos\n\n')
            arquivo.write('Todos os modelos usam split 80/20 aleatorio (estratificado por municipio),\n')
            arquivo.write('target desvio percentual (target_desvio_pct), sem feature "ano" e sem features JJA.\n\n')
            arquivo.write(comparacao.to_markdown(index=False, floatfmt='.3f'))
            arquivo.write('\n\nMelhor regressao por R²: **')
            arquivo.write(comparacao.loc[comparacao['r2'].idxmax(), 'modelo'])
            arquivo.write('**. Melhor classificacao por acuracia: **')
            arquivo.write(comparacao.loc[comparacao['acuracia'].idxmax(), 'modelo'])
            arquivo.write('**.\n')

    def _ler_metricas(self, pasta, arquivo):
        """Carrega métricas de outro modelo para comparação."""
        caminho = self.comparison_dir / pasta / arquivo
        if not caminho.exists():
            logger.warning('Metricas nao encontradas: %s', caminho)
            return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'acuracia': np.nan}
        with open(caminho, encoding='utf-8') as entrada:
            dados = json.load(entrada)
        return {
            'r2': dados['regressao']['r2'],
            'rmse': dados['regressao']['rmse'],
            'mae': dados['regressao']['mae'],
            'acuracia': dados['classificacao']['acuracia'],
        }

    def executar(self):
        """Executa o pipeline completo."""
        self.carregar_dados()
        self.preparar_dados()
        self.treinar_regressao()
        self.treinar_classificacao()
        self.gerar_relatorios()
        logger.info('Ridge concluido. Resultados em %s', self.output_dir)


def main():
    parser = argparse.ArgumentParser(description='Executa a modelagem Ridge')
    parser.add_argument('--data-dir', default='data/processed/integrado_trimestral')
    parser.add_argument('--output-dir', default='output/modelagem_ridge')
    parser.add_argument('--comparison-dir', default='output')
    parser.add_argument(
        '--usar-anomalia',
        choices=['sim', 'nao'],
        default='sim',
        help='sim (padrao): anomalia climatica por municipio (anom_*); nao: valores brutos + features avancadas',
    )
    parser.add_argument(
        '--usar-lag1',
        choices=['sim', 'nao'],
        default='sim',
        help='sim (padrao): inclui produtividade_kg_ha_lag1; nao: nao usa autoregressivo',
    )
    args = parser.parse_args()
    
    ModelagemRidge(
        args.data_dir, args.output_dir, args.comparison_dir,
        args.usar_anomalia == 'sim', args.usar_lag1 == 'sim',
    ).executar()


if __name__ == '__main__':
    main()