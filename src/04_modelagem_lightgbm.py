"""
Modelagem Preditiva com LightGBM - TCC
VERSAO TRIMESTRAL - Predicao da produtividade da soja
Autor: Henrique J. Berto
Data: 2025
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, classification_report)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import logging
import warnings
import argparse
import json
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prefixos de features climaticas reconhecidos pelo modelo.
PREFIXOS_CLIMA_BRUTO = (
    'precipitacao_mm_', 'temp_media_c_', 'oni_index_',
    'temp_desvio_c_', 'temp_amplitude_c_', 'meses_seca_',
    'gdd_', 'dias_secos_', 'dias_secos_consecutivos_',
)
PREFIXOS_CLIMA_ANOMALIA = (
    'anom_precipitacao_mm_', 'anom_temp_media_c_', 'oni_index_',
)

class ModelagemLightGBMTrimestral:
    def __init__(self, data_dir: str = "data/processed/integrado_trimestral",
                 output_dir: str = "output/modelagem_trimestral",
                 usar_anomalia: bool = True,
                 usar_lag1: bool = True):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.usar_anomalia = usar_anomalia
        self.usar_lag1 = usar_lag1
        self.dados = None
        self.dados_treino = None
        self.dados_teste = None
        self.modelo = None
        self.modelo_clf = None
        self.scaler = StandardScaler()
        self.features = []
        self.metricas = {}
        self.metricas_clf = {}
        self.importance_df = None
        self.le_clf = None

    def carregar_dados(self) -> bool:
        """Carrega os datasets treino/teste gerados pelo integrador (split 80/20)"""
        logger.info("=" * 60)
        logger.info("CARREGANDO DATASETS TREINO/TESTE (80/20)")
        logger.info("=" * 60)
        try:
            caminho_treino = self.data_dir / "treino_80pct.csv"
            caminho_teste = self.data_dir / "teste_20pct.csv"
            
            if not (caminho_treino.exists() and caminho_teste.exists()):
                logger.error("Datasets treino_80pct.csv e teste_20pct.csv não encontrados.")
                logger.error("Execute o 02_integrador_safra_clima.py primeiro.")
                return False
            
            self.dados_treino = pd.read_csv(caminho_treino)
            self.dados_teste = pd.read_csv(caminho_teste)
            
            logger.info(f"Treino: {len(self.dados_treino)} amostras")
            logger.info(f"Teste: {len(self.dados_teste)} amostras")
            logger.info(f"Municípios: {self.dados_treino['municipio'].nunique()}")
            
            return True
        except Exception as e:
            logger.error(f"Erro: {e}")
            return False

    def preparar_features(self):
        """Prepara features removendo 'ano' e JJA, usando target_desvio_pct"""
        logger.info("\n" + "=" * 60)
        logger.info("PREPARANDO FEATURES TRIMESTRAIS")
        logger.info("=" * 60)
        
        # Combinar treino e teste para preparar features consistentemente
        self.dados = pd.concat([self.dados_treino, self.dados_teste], ignore_index=True)
        
        features = []
        
        # Seleciona os prefixos climaticos conforme a flag --usar-anomalia.
        prefixos_clima = PREFIXOS_CLIMA_ANOMALIA if self.usar_anomalia else PREFIXOS_CLIMA_BRUTO
        features_climaticas = [
            coluna for coluna in self.dados.columns
            if coluna.startswith(prefixos_clima)
        ]
        if features_climaticas:
            features.extend(features_climaticas)
        
        # One-hot encoding para municipio
        dummies = pd.get_dummies(self.dados['municipio'], prefix='mun')
        self.dados = pd.concat([self.dados, dummies], axis=1)
        for col in dummies.columns:
            features.append(col)
        
        # REMOVIDO: 'ano' (causa vazamento temporal - importância 55 no modelo anterior)
        # REMOVIDO: features de JJA (pós-colheita - não faz sentido agronômico)
        features = [f for f in features if not f.endswith('_JJA')]
        
        # Produtividade do mesmo municipio no ano anterior (autoregressivo)
        if self.usar_lag1 and 'produtividade_kg_ha_lag1' in self.dados.columns:
            features.append('produtividade_kg_ha_lag1')
        
        self.features = features
        
        # Separar novamente em treino e teste
        self.dados_treino = self.dados.iloc[:len(self.dados_treino)]
        self.dados_teste = self.dados.iloc[len(self.dados_treino):]
        
        # Target: target_desvio_pct (desvio percentual) em vez de produtividade_kg_ha
        self.y_treino = self.dados_treino['target_desvio_pct'].values.astype(float)
        self.y_teste = self.dados_teste['target_desvio_pct'].values.astype(float)
        
        # Features
        self.X_treino = self.dados_treino[self.features].values.astype(float)
        self.X_teste = self.dados_teste[self.features].values.astype(float)
        
        # Tratar NaNs
        self.X_treino = np.nan_to_num(self.X_treino, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_teste = np.nan_to_num(self.X_teste, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Features: {len(features)}")
        logger.info(f"Features climáticas: {len(features_climaticas)}")
        logger.info(f"Target: target_desvio_pct (desvio percentual)")
        logger.info(f"Treino: {len(self.X_treino)} amostras")
        logger.info(f"Teste: {len(self.X_teste)} amostras")
        
        return self.X_treino, self.X_teste, self.y_treino, self.y_teste

    def treinar_modelo(self):
        """Treina LightGBM para regressão usando split 80/20"""
        logger.info("\n" + "=" * 60)
        logger.info("TREINANDO LIGHTGBM - REGRESSÃO TRIMESTRAL")
        logger.info("=" * 60)
        
        X_train, X_test, y_train, y_test = self.X_treino, self.X_teste, self.y_treino, self.y_teste
        
        logger.info(f"Treino: {len(X_train)} amostras")
        logger.info(f"Teste: {len(X_test)} amostras")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Hiperparâmetros com regularização explícita para evitar overfitting
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15, # Controla a complexidade da árvore
            'learning_rate': 0.05,  # Taxa de aprendizado reduzida para convergência estável, reduzido para evitar overfit
            'feature_fraction': 0.8,  # Subamostragem de features por iteração
            'bagging_fraction': 0.8,  # Subamostragem de dados
            'bagging_freq': 5,  # Frequência da subamostragem
            'verbose': -1,
            'random_state': 42,
            'reg_alpha': 0.1,  # Regularização L1
            'reg_lambda': 0.1,  # Regularização L2
        }
        
        logger.info("Treinando LightGBM...")
        # O early stopping monitora o RMSE no conjunto de validação e interrompe
        # o treinamento quando não há melhoria por 20 iterações consecutivas

        modelo = lgb.train(
            params, train_data, valid_sets=[test_data],
            num_boost_round=200, callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print("\n" + "=" * 50)
        print("RESULTADOS DO MODELO TRIMESTRAL")
        print("=" * 50)
        print(f"  R²: {r2:.3f}")
        print(f"  RMSE: {rmse:.1f}%")
        print(f"  MAE: {mae:.1f}%")
        
        self.modelo = modelo
        self.metricas = {'r2': r2, 'rmse': rmse, 'mae': mae}
        
        # Importancia das features
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': modelo.feature_importance()
        }).sort_values('importance', ascending=False)
        self.importance_df = importance_df
        
        print("\nTOP 10 FEATURES MAIS IMPORTANTES:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.0f}")
        
        # Salvar importancia
        importance_df.to_csv(self.output_dir / 'importancia_features.csv', index=False)
        
        return modelo, importance_df

    def plotar_resultados(self, importance_df):
        """Gera gráficos de importância das features"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Grafico 1: Importancia por trimestre
        ax1 = axes[0]
        trim_importance = importance_df[importance_df['feature'].str.contains('_(SON|DJF|MAM)$', regex=True)]
        if len(trim_importance) > 0:
            trim_importance = trim_importance.copy()
            trim_importance['trimestre'] = trim_importance['feature'].str.extract(r'_(SON|DJF|MAM)$')[0]
            cores = ['#4ecdc4', '#ff6b6b', '#ffb347']
            ax1.bar(trim_importance['trimestre'], trim_importance['importance'], color=cores)
            ax1.set_xlabel('Trimestre')
            ax1.set_ylabel('Importancia')
            ax1.set_title('Importancia por Trimestre')
            ax1.grid(True, alpha=0.3)
        
        # Grafico 2: Top features
        ax2 = axes[1]
        top_features = importance_df.head(8)
        colors = plt.cm.Greens(np.linspace(0.3, 0.8, len(top_features)))
        ax2.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'])
        ax2.set_xlabel('Importancia')
        ax2.set_title('Top 8 Features')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'importancia_features_trimestral.png', dpi=150)
        plt.close()
        logger.info(f"Grafico salvo: {self.output_dir / 'importancia_features_trimestral.png'}")

    # -------------------------------------
    # CLASSIFICACAO (Bom / Regular / Ruim) 
    # -------------------------------------
    def criar_classes_safra(self) -> pd.Series:
        """
        Classifica cada observacao conforme o desvio percentual:
          - Bom    : desvio >= +10% (acima da média histórica)
          - Regular: -10% <= desvio < +10%
          - Ruim   : desvio < -10% (abaixo da média histórica)
        """
        # Usar target_desvio_pct em vez de produtividade_kg_ha
        desvio = self.dados['target_desvio_pct']
        
        self.limiar_bom = 10.0   # +10%
        self.limiar_ruim = -10.0  # -10%
        self.media_desvio = 0.0
        self.desvio_desvio = desvio.std()
        
        logger.info(f"Limiares de classificação (desvio percentual):")
        logger.info(f"  Limiar Bom    : >= {self.limiar_bom:.1f}%")
        logger.info(f"  Limiar Ruim   : <  {self.limiar_ruim:.1f}%")
        
        def _classificar(v):
            if v >= self.limiar_bom:
                return 'Bom'
            elif v < self.limiar_ruim:
                return 'Ruim'
            return 'Regular'
        
        classes = desvio.apply(_classificar)
        logger.info(f"  Distribuição: {classes.value_counts().to_dict()}")
        
        return classes

    def treinar_classificador(self):
        """Treina LightGBM de classificacao multiclasse (Bom/Regular/Ruim)"""
        logger.info("\n" + "=" * 60)
        logger.info("TREINANDO LIGHTGBM - CLASSIFICACAO (Bom/Regular/Ruim)")
        logger.info("=" * 60)
        
        classes = self.criar_classes_safra()
        self.dados['classe_safra'] = classes.values
        
        # Split 80/20 (mesmos indices da regressao)
        mask_treino = np.zeros(len(self.dados), dtype=bool)
        mask_treino[:len(self.dados_treino)] = True
        mask_teste = ~mask_treino
        
        X_all = self.dados[self.features].values.astype(float)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        
        le = LabelEncoder()
        y_all = le.fit_transform(self.dados['classe_safra'])
        
        X_train_c = X_all[mask_treino]
        X_test_c = X_all[mask_teste]
        y_train_c = y_all[mask_treino]
        y_test_c = y_all[mask_teste]
        
        logger.info(f"Treino: {len(X_train_c)} amostras")
        logger.info(f"Teste: {len(X_test_c)} amostras")
        logger.info(f"Classes: {le.classes_.tolist()}")
        
        n_classes = len(le.classes_)
        # Configuração para classificação multiclasse (Bom, Regular, Ruim)
        params_clf = {
            'objective': 'multiclass',
            'num_class': n_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
        
        train_data_c = lgb.Dataset(X_train_c, label=y_train_c)
        test_data_c = lgb.Dataset(X_test_c, label=y_test_c, reference=train_data_c)
        
        logger.info("Treinando classificador LightGBM...")
        modelo_clf = lgb.train(
            params_clf, train_data_c, valid_sets=[test_data_c],
            num_boost_round=200,
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        y_prob = modelo_clf.predict(X_test_c)
        y_pred_c = np.argmax(y_prob, axis=1)
        acc = accuracy_score(y_test_c, y_pred_c)
        
        treino_dist = pd.Series(y_train_c).value_counts().reindex(range(n_classes), fill_value=0)
        teste_dist = pd.Series(y_test_c).value_counts().reindex(range(n_classes), fill_value=0)
        missing_classes = [le.classes_[idx] for idx, count in teste_dist.items() if count == 0]
        
        report = classification_report(
            y_test_c,
            y_pred_c,
            labels=list(range(n_classes)),
            target_names=le.classes_,
            zero_division=0,
        )
        
        print("\n" + "=" * 50)
        print("RESULTADOS - CLASSIFICACAO (Bom/Regular/Ruim)")
        print("=" * 50)
        print(f"  Acuracia: {acc:.3f}")
        print(f"\n{report}")
        
        if missing_classes:
            print(f"Classes ausentes no teste: {', '.join(missing_classes)}")
        
        self.modelo_clf = modelo_clf
        self.le_clf = le
        self.metricas_clf = {
            'acuracia': acc,
            'report': report,
            'treino_dist': treino_dist.to_dict(),
            'teste_dist': teste_dist.to_dict(),
            'missing_classes': missing_classes,
        }
        
        # Persistir modelo
        import joblib
        joblib.dump(modelo_clf, self.output_dir / 'modelo_lightgbm_classificacao.pkl')
        joblib.dump(le, self.output_dir / 'label_encoder_classes.pkl')
        logger.info(f"Modelo classificador salvo em: {self.output_dir}")
        
        self._plotar_distribuicao_classes()
        
        return modelo_clf, acc, report

    def _plotar_distribuicao_classes(self):
        """Gera o grafico global de distribuicao das classes de safra."""
        fig, ax1 = plt.subplots(figsize=(8, 6))
        cores_classe = {'Bom': '#2ecc71', 'Regular': '#f39c12', 'Ruim': '#e74c3c'}
        
        # Grafico 1: distribuicao global
        contagem = (self.dados['classe_safra']
                    .value_counts()
                    .reindex(['Bom', 'Regular', 'Ruim'], fill_value=0))
        bars = ax1.bar(contagem.index, contagem.values,
                       color=[cores_classe[c] for c in contagem.index],
                       edgecolor='black', alpha=0.8)
        for bar, val in zip(bars, contagem.values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     str(val), ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
        ax1.set_xlabel('Classe de Safra', fontsize=12)
        ax1.set_ylabel('Frequencia', fontsize=12)
        ax1.set_title('Distribuicao Global das Classes de Safra\n'
                      f'(Limiares: Bom >= +10%, Ruim < -10%)', fontsize=13)
        ax1.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        caminho = self.output_dir / 'distribuicao_classes_safra.png'
        plt.savefig(caminho, dpi=150)
        plt.close()
        logger.info(f"Grafico salvo: {caminho}")

    def gerar_relatorio_features(self):
        """Gera um relatorio detalhado das features e dos dois modelos."""
        logger.info("\n" + "=" * 60)
        logger.info("GERANDO RELATORIO DAS FEATURES")
        logger.info("=" * 60)
        relatorio_path = self.output_dir / 'relatorio_features_modelo.md'
        
        n_treino = len(self.dados_treino)
        n_teste = len(self.dados_teste)
        
        with open(relatorio_path, 'w', encoding='utf-8') as f:
            f.write("# Relatorio das Features e Modelos LightGBM\n\n")
            
            # --- Regressao ---
            f.write("## 1. Modelo de Regressao\n\n")
            f.write("**Objetivo:** Estimar o desvio percentual de produtividade (%).\n\n")
            f.write(f"- **R² (teste):** {self.metricas['r2']:.3f}\n")
            f.write(f"- **RMSE:** {self.metricas['rmse']:.1f}%\n")
            f.write(f"- **MAE:** {self.metricas['mae']:.1f}%\n")
            f.write(f"- **Amostras de treino:** {n_treino}\n")
            f.write(f"- **Amostras de teste:** {n_teste}\n")
            f.write(f"- **Total de features:** {len(self.features)}\n")
            f.write(f"- **Split:** 80/20 aleatorio (estratificado por municipio)\n")
            f.write(f"- **Target:** target_desvio_pct (desvio percentual da media historica)\n\n")
            
            # --- Classificacao ---
            if hasattr(self, 'metricas_clf'):
                f.write("## 2. Modelo de Classificacao\n\n")
                f.write("**Objetivo:** Categorizar safra em Bom / Regular / Ruim\n\n")
                f.write(f"- **Criterio Bom    :** desvio >= +10%\n")
                f.write(f"- **Criterio Regular:** -10% <= desvio < +10%\n")
                f.write(f"- **Criterio Ruim   :** desvio < -10%\n\n")
                f.write(f"- **Acuracia (teste):** {self.metricas_clf['acuracia']:.3f}\n\n")
                
                f.write("### Distribuicao das Classes\n\n")
                f.write("| Classe | Treino | Teste |\n")
                f.write("|--------|--------|-------|\n")
                for idx, classe in enumerate(self.le_clf.classes_):
                    f.write(
                        f"| {classe} | "
                        f"{self.metricas_clf['treino_dist'].get(idx, 0)} | "
                        f"{self.metricas_clf['teste_dist'].get(idx, 0)} |\n"
                    )
                
                if self.metricas_clf['missing_classes']:
                    faltantes = ', '.join(self.metricas_clf['missing_classes'])
                    f.write(f"\n**Observacao:** classes ausentes no teste: {faltantes}.\n\n")
                
                f.write("### Relatorio de Classificacao\n\n")
                f.write("```\n")
                f.write(self.metricas_clf['report'])
                f.write("```\n\n")
            
            # --- Features ---
            f.write("## 3. Lista de Features\n\n")
            f.write("| Feature | Tipo | Descricao |\n")
            f.write("|---------|------|-----------|\n")
            for feat in self.features:
                desc = self._descrever_feature(feat)
                f.write(f"| {feat} | {desc['tipo']} | {desc['descricao']} |\n")
            
            f.write("\n## 4. Importancia das Features (Regressao)\n\n")
            f.write("| Feature | Importancia | Percentual |\n")
            f.write("|---------|-------------|------------|\n")
            total_imp = self.importance_df['importance'].sum()
            for _, row in self.importance_df.iterrows():
                pct = (row['importance'] / total_imp) * 100 if total_imp > 0 else 0
                f.write(f"| {row['feature']} | {row['importance']:.0f} | {pct:.1f}% |\n")
            
            f.write("\n## 5. Conclusao\n\n")
            f.write(f"O modelo de regressao LightGBM obteve R²={self.metricas['r2']:.3f} "
                    f"com RMSE={self.metricas['rmse']:.1f}%. ")
            if hasattr(self, 'metricas_clf'):
                f.write(f"O classificador Bom/Regular/Ruim atingiu acuracia de "
                        f"{self.metricas_clf['acuracia']:.3f}. ")
            f.write("O split 80/20 aleatorio (estratificado por municipio) foi aplicado. ")
            f.write("A feature 'ano' foi removida para evitar vazamento temporal, ")
            f.write("e as features de JJA foram excluidas por corresponderem ao periodo pos-colheita.\n")
        
        logger.info(f"Relatorio salvo: {relatorio_path}")

    @staticmethod
    def _descrever_feature(feat: str) -> dict:
        """Retorna tipo e descricao para uma feature"""
        if feat == 'precipitacao_mm':
            return {'tipo': 'Numerica', 'descricao': 'Precipitacao total no trimestre (mm)'}
        if feat == 'temp_media_c':
            return {'tipo': 'Numerica', 'descricao': 'Temperatura media no trimestre (C)'}
        if feat == 'oni_index':
            return {'tipo': 'Numerica', 'descricao': 'Indice ONI (Oceanic Nino Index)'}
        if feat == 'produtividade_kg_ha_lag1':
            return {'tipo': 'Numerica', 'descricao': 'Produtividade do municipio no ano anterior (autoregressivo)'}
        if feat.startswith('mun_'):
            return {'tipo': 'Categorica', 'descricao': f"Municipio {feat.replace('mun_', '')} (One-Hot)"}
        if feat.startswith('precipitacao_mm_'):
            trim = feat.replace('precipitacao_mm_', '')
            return {'tipo': 'Numerica', 'descricao': f'Precipitacao acumulada no trimestre {trim} (mm)'}
        if feat.startswith('temp_media_c_'):
            trim = feat.replace('temp_media_c_', '')
            return {'tipo': 'Numerica', 'descricao': f'Temperatura media no trimestre {trim} (C)'}
        if feat.startswith('oni_index_'):
            trim = feat.replace('oni_index_', '')
            return {'tipo': 'Numerica', 'descricao': f'Indice ONI no trimestre {trim}'}
        if feat.startswith('anom_precipitacao_mm_'):
            trim = feat.replace('anom_precipitacao_mm_', '')
            return {'tipo': 'Numerica', 'descricao': f'Anomalia de precipitacao no trimestre {trim} (mm)'}
        if feat.startswith('anom_temp_media_c_'):
            trim = feat.replace('anom_temp_media_c_', '')
            return {'tipo': 'Numerica', 'descricao': f'Anomalia de temperatura no trimestre {trim} (C)'}
        if feat.startswith('temp_desvio_c_'):
            trim = feat.replace('temp_desvio_c_', '')
            return {'tipo': 'Numerica', 'descricao': f'Desvio padrao da temperatura no trimestre {trim} (C)'}
        if feat.startswith('temp_amplitude_c_'):
            trim = feat.replace('temp_amplitude_c_', '')
            return {'tipo': 'Numerica', 'descricao': f'Amplitude termica no trimestre {trim} (C)'}
        if feat.startswith('meses_seca_'):
            trim = feat.replace('meses_seca_', '')
            return {'tipo': 'Numerica', 'descricao': f'Numero de meses secos (<30mm) no trimestre {trim}'}
        if feat.startswith('gdd_'):
            trim = feat.replace('gdd_', '')
            return {'tipo': 'Numerica', 'descricao': f'Graus-dia de crescimento no trimestre {trim} (base 10C)'}
        if feat.startswith('dias_secos_consecutivos_'):
            trim = feat.replace('dias_secos_consecutivos_', '')
            return {'tipo': 'Numerica', 'descricao': f'Maior sequencia de dias secos (<1mm) no trimestre {trim}'}
        if feat.startswith('dias_secos_'):
            trim = feat.replace('dias_secos_', '')
            return {'tipo': 'Numerica', 'descricao': f'Total de dias secos (<1mm) no trimestre {trim}'}
        return {'tipo': 'Numerica', 'descricao': feat}

    def executar(self):
        """Executa o pipeline completo de modelagem"""
        logger.info("\n" + "=" * 60)
        logger.info("INICIANDO MODELAGEM TRIMESTRAL")
        logger.info("=" * 60)
        
        if not self.carregar_dados():
            return
        
        self.preparar_features()
        modelo, importance_df = self.treinar_modelo()
        self.plotar_resultados(importance_df)
        
        # Classificacao Bom/Regular/Ruim (metodologia TCC)
        self.treinar_classificador()
        self.gerar_relatorio_features()
        
        with open(self.output_dir / 'metricas_lightgbm.json', 'w', encoding='utf-8') as arquivo:
            json.dump({
                'regressao': self.metricas,
                'classificacao': {
                    'acuracia': self.metricas_clf['acuracia'],
                },
            }, arquivo, indent=2)
        
        logger.info(f"Resultados salvos em: {self.output_dir}")
        logger.info("Modelagem trimestral concluida!")


def main():
    parser = argparse.ArgumentParser(description="Executa a modelagem LightGBM trimestral")
    parser.add_argument("--data-dir", default="data/processed/integrado_trimestral")
    parser.add_argument("--output-dir", default="output/modelagem_trimestral")
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
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MODELAGEM PREDITIVA TRIMESTRAL - Safra de Soja x Clima")
    print("=" * 60)
    
    modelagem = ModelagemLightGBMTrimestral(
        args.data_dir, args.output_dir,
        args.usar_anomalia == "sim", args.usar_lag1 == "sim",
    )
    modelagem.executar()


if __name__ == "__main__":
    main()