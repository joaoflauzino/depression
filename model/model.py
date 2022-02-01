# Manipulação de dados
from multiprocessing import Pipe
from re import X
import pandas as pd
import numpy as np 

# Modelos
from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model 
import sklearn

# Busca de parâmetros
from sklearn.model_selection import GridSearchCV

# Shap
import shap

# Pipeline
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as pipe_imb 
import imblearn
import joblib

# Pré-Processamento
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

# Avaliação
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Visualização de dados
import seaborn as sns 
import matplotlib.pyplot as plt


class FrameWorkFlauzino(object):

    def __init__(self, X, y) -> None:
        self.split = {"X": X, "y": y}
        self.sample = {"oversample": RandomOverSampler(), "undersample": RandomUnderSampler()}
        self.padronizacao = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()}
        self.estrategia_categoricas = {"OneHot": OneHotEncoder(), "TargetEncoder": TargetEncoder()}
        self.metricas = {

                        "roc": lambda y_teste, y_predicao: roc_auc_score(y_teste, y_predicao),
                        "precisao": lambda y_teste, y_predicao: precision_score(y_teste, y_predicao),
                        "recall": lambda y_teste, y_predicao: recall_score(y_teste, y_predicao),
                        "f1_score": lambda y_teste, y_predicao: f1_score(y_teste, y_predicao),
                        "matriz": lambda y_teste, y_predicao: confusion_matrix(y_teste, y_predicao),

                        }
        self.modelos = {}


    def reamostragem(self, estrategia: str) -> None:
        """Método responsável por aplicar oversample ou undersample"""
        X_reamostrado, y_reamostrado = pipe_imb(steps=[(estrategia, self.sample[estrategia])]).fit_resample(self.X, self.Y)
        X_reamostrado_renomeado = pd.DataFrame(X_reamostrado, columns=self.X.columns)
        self.split["X"] = X_reamostrado_renomeado
        self.split["y"] = y_reamostrado
        return None

    def treino_teste(self, pct: float) -> None:
        """Método para quebrar base em treino e teste"""
        self.x_treino, self.x_teste, self.y_treino, self.y_teste = train_test_split(self.split["X"], self.split["y"], test_size=pct)
        return None

    def criacao_pipeline(self, variaveis_numericas: list,
                               variaveis_categoricas: list,
                               padronizacao: str,
                               estrategia_variaveis_categoricas: str) -> None:
        """Função para criar pipeline de transformação das variáveis"""
        # Criando pipeline para variáveis numéricas
        transformador_numerico = Pipeline(steps=[
                                                ('padronização', self.padronizacao[padronizacao])
                                        ])
        
        # Criando pipeline para variáveis categóricos
        transformador_categorico = Pipeline(steps=[
                                                ("estratégia_categóricas", self.estrategia_categoricas[estrategia_variaveis_categoricas])
                                            ])
        
        # Concatenando pipelines
        self.pre_processamento = ColumnTransformer(transformers=[
                                                        ('categorical', transformador_categorico, variaveis_categoricas),
                                                        ('num', transformador_numerico, variaveis_numericas)
                                        ])

        return None

    def execucao_pipeline(self, params: dict, 
                                algoritmos: list, 
                                k: int,
                                avaliadores: dict) -> None:
        """Função para execução do pipeline"""
        
        steps_pre_processamento = [('pre-processamento', self.pre_processamento)]
        pipeline_pre_processamento = Pipeline(steps=steps_pre_processamento)
        pipeline_pre_processamento_treino = pipeline_pre_processamento.fit(self.x_treino, self.y_treino)
        # Transformando resultado do pipeline em dataframe novamente
        variaveis_categoricas = pipeline_pre_processamento_treino.named_steps['pre-processamento'].transformers_[0][2]
        categoricas = pipeline_pre_processamento_treino.named_steps['pre-processamento'].transformers_[0][1]['categoricas'].get_feature_names()
        variaveis_numericas = pipeline_pre_processamento_treino.named_steps['pre-processamento'].transformers_[1][2]
        variaveis_modelo = list(categoricas) + variaveis_numericas
        X_treino_pre_processado = pd.DataFrame(pipeline_pre_processamento_treino.transform(self.x_treino), columns = variaveis_modelo)
        X_teste_pre_processado = pd.DataFrame(pipeline_pre_processamento_treino.transform(self.x_teste), columns = variaveis_modelo)

        # Rodando cada algoritmo para o dado pré-processado
        for alg in algoritmos:
            # Criando dicionário para cada modelo
            self.modelos[alg.__class__.__name__] = {}
            # Criando pipeline final
            steps = [('modelos', alg)]
            model_kfold = Pipeline(steps=steps) 
            # Executando busca dos melhores parâmetros
            print(f'Treinando {alg.__class__.__name__}...')
            model_grid = GridSearchCV(estimator=model_kfold, param_grid=params[alg.__class__.__name__], cv=k, refit=True, scoring='precision')
            model_grid.fit(X_treino_pre_processado, y_treino)
            print(f'Melhores parâmetros: {model_grid.best_params_}')
            print(f'Melhores resultados: {model_grid.best_score_}')
            # Capturando melhor modelo
            model = model_grid.best_estimator_
            # Aplicando shap
            # if alg.__class__.__name__ in ['GradientBoostingClassifier']:
            #     relatorio_shap(model[0], X_treino_pre_processado, alg.__class__.__name__)
            # # Registrando modelo
            # registra_modelos(model, alg.__class__.__name__)
            # Testando modelo
            predicoes = model.predict(X_teste_pre_processado)
            # Registrando resultados
            for j in avaliadores:
                self.modelos[alg.__class__.__name__][j] = self.metricas[j](self.y_teste, predicoes)
        
        return None
            






