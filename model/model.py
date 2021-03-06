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
from sklearn.metrics import roc_auc_score
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


    def relatorio_shap(self, model: GridSearchCV, X_treino: pd.DataFrame, algoritmo: str) -> None:
        """Função responsável por executar o shap"""
        explainer = shap.Explainer(model)
        shap_values = explainer(X_treino)
        print(f' Exibindo shap values para {algoritmo}')
        shap.plots.beeswarm(shap_values)

    def registra_modelos(self, model: sklearn.pipeline.Pipeline, modelo: str) -> None:
        """Função responsável por salvar modelo"""
        try:
            joblib.dump(model, f'modelos_salvos/{modelo}.pkl')
            print(f'Registrando {modelo} pelo job lib na pasta modelos_salvos')
        except:
            raise('Falha ao salvar modelo')
        return None


    def reamostragem(self, estrategia: str) -> None:
        """Método responsável por aplicar oversample ou undersample"""
        print(f'Distribuição antes da reamostragem: \n')
        print(self.split["y"].value_counts())
        X_reamostrado, y_reamostrado = pipe_imb(steps=[(estrategia, self.sample[estrategia])]).fit_resample(self.split["X"], self.split["y"])
        X_reamostrado_renomeado = pd.DataFrame(X_reamostrado, columns=self.split["X"].columns)
        print(f'Distribuição da target reamostrada com {estrategia}: \n')
        print(f'{y_reamostrado.value_counts()}')
        self.split["X"] = X_reamostrado_renomeado
        self.split["y"] = y_reamostrado
        return None

    def treino_teste(self, pct: float) -> None:
        """Método para quebrar base em treino e teste"""
        print(f'Quantidade de registros antes da separação de treino e teste: \n')
        print(f'{self.split["X"].shape[0]} \n')
        self.x_treino, self.x_teste, self.y_treino, self.y_teste = train_test_split(self.split["X"], self.split["y"], test_size=pct)
        print(f'Quantidade de registros após separação de treino e teste: \n')
        print(f'X_treino: {self.x_treino.shape[0]} \n')
        print(f'x_teste: {self.x_teste.shape[0]} \n')
        print(f'y_treino: {self.y_treino.shape[0]} \n')
        print(f'y_teste: {self.y_teste.shape[0]} \n')
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
                                                ("estrategia_categoricas", self.estrategia_categoricas[estrategia_variaveis_categoricas])
                                            ])
        
        # Concatenando pipelines
        self.pre_processamento = ColumnTransformer(transformers=[
                                                        ('categorical', transformador_categorico, variaveis_categoricas),
                                                        ('num', transformador_numerico, variaveis_numericas)
                                        ])

        print(f'Pipeline criado:\n {self.pre_processamento}')
        
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
        categoricas = pipeline_pre_processamento_treino.named_steps['pre-processamento'].transformers_[0][1]['estrategia_categoricas'].get_feature_names()
        variaveis_numericas = pipeline_pre_processamento_treino.named_steps['pre-processamento'].transformers_[1][2]
        variaveis_modelo = list(categoricas) + variaveis_numericas
        self.X_treino_pre_processado = pd.DataFrame(pipeline_pre_processamento_treino.transform(self.x_treino), columns = variaveis_modelo)
        self.X_teste_pre_processado = pd.DataFrame(pipeline_pre_processamento_treino.transform(self.x_teste), columns = variaveis_modelo)

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
            model_grid.fit(self.X_treino_pre_processado, self.y_treino)
            print(f'Melhores parâmetros: {model_grid.best_params_}')
            print(f'Melhores resultados: {model_grid.best_score_}')
            # Capturando melhor modelo
            model = model_grid.best_estimator_
            # Aplicando shap
            if alg.__class__.__name__ in ['GradientBoostingClassifier']:
                self.relatorio_shap(model[0], self.X_treino_pre_processado, alg.__class__.__name__)
            # Registrando modelo
            self.registra_modelos(model, alg.__class__.__name__)
            # Testando modelo
            predicoes = model.predict(self.X_teste_pre_processado)
            # Registrando resultados
            for j in avaliadores:
                self.modelos[alg.__class__.__name__][j] = self.metricas[j](self.y_teste, predicoes)
        
        return None

    def gerando_relatorio(self) -> None:
        """Função responsável por gerar comparativos entre modelos"""
        fig, axs = plt.subplots(nrows=2, ncols =2 , figsize=(20,10))
        plt.subplots_adjust(hspace = 0.3)
        
        # Transformando em um dataframe
        resultados_dataframe = pd.melt(pd.DataFrame.from_dict(self.modelos).reset_index(), id_vars=['index'], 
                                    value_vars=['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression']
                                    )
        
        metricas = [i for i in resultados_dataframe['index'].unique() if i != 'matriz']
        eixo_x = resultados_dataframe['variable'].unique()
        posicao_inicial = 0
        count = 0
        posicao_final = 0
        for i, metrica in enumerate(metricas):
            chart = sns.barplot(data=resultados_dataframe.query(f'index == "{metrica}"'), x = 'variable', y = 'value', color="cornflowerblue", ax = axs[posicao_inicial,posicao_final])
            chart.set_xticklabels(labels=eixo_x, rotation=0)
            chart.set_title(metrica)
            count +=1
            posicao_final += 1
            if count == 2:
                count = 0
                posicao_inicial +=1
                posicao_final = 0
                
        plt.savefig('resultados.png')
            






