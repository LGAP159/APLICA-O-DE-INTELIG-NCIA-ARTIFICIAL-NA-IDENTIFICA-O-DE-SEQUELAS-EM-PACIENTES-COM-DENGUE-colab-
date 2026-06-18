# Predicao de Risco de Gravidade em Casos de Dengue com IA

Projeto academico de Machine Learning para apoiar a identificacao de risco de gravidade em pacientes com dengue, utilizando dados clinicos, sociodemograficos e comorbidades.

> Aviso: este projeto nao substitui avaliacao medica. O modelo deve ser usado apenas como apoio academico, exploratorio e analitico.

## Sumario

- [Sobre o projeto](#sobre-o-projeto)
- [Objetivo](#objetivo)
- [Dataset](#dataset)
- [Variavel-alvo](#variavel-alvo)
- [Metodologia](#metodologia)
- [Modelos utilizados](#modelos-utilizados)
- [Metricas de avaliacao](#metricas-de-avaliacao)
- [Explicabilidade](#explicabilidade)
- [Predicao de novos pacientes](#predicao-de-novos-pacientes)
- [Arquivos gerados](#arquivos-gerados)
- [Como executar](#como-executar)
- [Limitacoes](#limitacoes)
- [Melhorias futuras](#melhorias-futuras)

## Sobre o projeto

A dengue e uma doenca de grande importancia para a saude publica, especialmente em paises tropicais como o Brasil. Embora muitos casos evoluam de forma favoravel, alguns pacientes podem desenvolver sinais de alarme ou formas graves da doenca.

Este projeto utiliza algoritmos de Inteligencia Artificial para analisar registros de pacientes e estimar o risco de gravidade com base em informacoes como:

- idade;
- sexo;
- sintomas;
- comorbidades;
- classificacao final do caso.

De forma simples, o modelo aprende com casos anteriores e tenta identificar padroes associados a maior risco.

## Objetivo

Desenvolver e avaliar modelos de Machine Learning capazes de apoiar a predicao de risco de gravidade em casos de dengue.

O modelo classifica os pacientes em dois grupos:

- `0`: dengue sem sinais de alarme;
- `1`: dengue com sinais de alarme ou dengue grave.

## Dataset

O projeto utiliza o arquivo:

```text
dados_analise.csv
```

A base utilizada possui aproximadamente:

```text
1.583.651 linhas
13 colunas
```

Principais variaveis:

| Coluna | Descricao |
|---|---|
| `CS_SEXO` | Sexo do paciente |
| `FEBRE` | Presenca de febre |
| `MIALGIA` | Presenca de dor muscular |
| `CEFALEIA` | Presenca de dor de cabeca |
| `NAUSEA` | Presenca de nausea |
| `VOMITO` | Presenca de vomito |
| `DIABETES` | Indicacao de diabetes |
| `HIPERTENSA` | Indicacao de hipertensao |
| `RENAL` | Indicacao de doenca renal |
| `CLASSI_FIN` | Classificacao final do caso |
| `EVOLUCAO` | Evolucao final do caso |
| `TP_IDADE` | Tipo de idade registrada |
| `IDADE_REAL` | Idade real do paciente |

Para sintomas e comorbidades, os codigos foram tratados assim:

| Valor original | Significado | Valor usado |
|---|---|---|
| `1` | Sim | `1` |
| `2` | Nao | `0` |
| `9` | Ignorado | `NaN` |

## Variavel-alvo

A variavel-alvo criada no projeto se chama:

```python
risco_grave
```

Ela foi derivada da coluna `CLASSI_FIN`:

| `CLASSI_FIN` | Significado | `risco_grave` |
|---|---|---|
| `10` | Dengue sem sinais de alarme | `0` |
| `11` | Dengue com sinais de alarme | `1` |
| `12` | Dengue grave | `1` |

A distribuicao observada foi fortemente desbalanceada:

| Classe | Quantidade aproximada | Percentual |
|---|---:|---:|
| Sem risco grave | 1.328.832 | 97,41% |
| Com risco grave | 35.305 | 2,59% |

### Observacao metodologica importante

Como `risco_grave` foi criado a partir de `CLASSI_FIN`, o modelo deve ser interpretado como uma ferramenta de apoio a classificacao automatizada de gravidade da dengue.

Caso `CLASSI_FIN` seja definida oficialmente a partir dos mesmos sintomas usados como entrada do modelo, a IA pode estar reaprendendo criterios clinicos ja existentes. Por isso, os resultados devem ser interpretados com cautela.

## Metodologia

O notebook segue as seguintes etapas:

1. Carregamento do dataset.
2. Criacao da variavel-alvo `risco_grave`.
3. Tratamento de valores ausentes.
4. Conversao de sintomas e comorbidades para formato binario.
5. Separacao entre variaveis preditoras (`X`) e alvo (`Y`).
6. Divisao entre treino e teste.
7. Pre-processamento com `Pipeline` e `ColumnTransformer`.
8. Treinamento de diferentes modelos.
9. Otimizacao de hiperparametros com `RandomizedSearchCV`.
10. Avaliacao com metricas de classificacao.
11. Analise de erros e falsos negativos.
12. Explicabilidade com importancia por permutacao e SHAP.
13. Predicao de multiplos pacientes.

## Modelos utilizados

Foram avaliados os seguintes modelos:

| Modelo | Descricao |
|---|---|
| `DummyClassifier` | Modelo baseline para comparacao |
| `LogisticRegression` | Modelo linear simples e interpretavel |
| `DecisionTreeClassifier` | Modelo baseado em regras de decisao |
| `RandomForestClassifier` | Conjunto de arvores de decisao |
| `XGBClassifier` | Modelo de boosting para dados tabulares |

O SVM foi considerado inicialmente, mas removido da versao rapida por conta do alto custo computacional em uma base com mais de 1,5 milhao de registros.

## Versao rapida para Google Colab

Como a base e grande, a versao final usada no Colab foi otimizada para reduzir o tempo de execucao:

```python
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
n_iter = 3
cenarios = {
    'Sem SMOTE': None
}
```

Essas escolhas reduzem o tempo de treinamento sem remover a comparacao entre modelos principais.

## Desbalanceamento das classes

A classe de risco grave representa apenas cerca de 2,59% dos registros. Por isso, o projeto utiliza estrategias como:

- `class_weight='balanced'`;
- `scale_pos_weight` no XGBoost;
- analise de recall/sensibilidade;
- analise de falsos negativos.

O SMOTE foi discutido, mas nao foi usado na versao rapida porque aumentaria muito o custo computacional.

## Metricas de avaliacao

As metricas usadas foram:

| Metrica | Significado |
|---|---|
| Acuracia | Percentual geral de acertos |
| Precisao | Entre os previstos como graves, quantos eram graves |
| Recall/Sensibilidade | Entre os graves reais, quantos foram identificados |
| Especificidade | Entre os nao graves reais, quantos foram identificados |
| F1-Score | Equilibrio entre precisao e recall |
| AUC | Capacidade geral de separar as classes |

Em saude, o recall e especialmente importante, pois falsos negativos podem representar pacientes graves classificados como nao graves.

## Explicabilidade

O projeto utiliza duas abordagens para interpretar os modelos:

### Importancia por permutacao

Mede quanto o desempenho do modelo piora quando uma variavel e embaralhada.

### SHAP

Ajuda a entender como cada variavel contribui para a predicao. A implementacao foi ajustada para usar os dados ja transformados pelo pre-processamento, evitando erros com variaveis categoricas em texto.

## Analise de erros

O notebook identifica especialmente os falsos negativos:

```text
real = 1
predito = 0
```

Esses casos sao importantes porque representam pacientes com risco grave que o modelo classificou como sem risco grave.

## Analise por grupos

Tambem e feita uma analise de desempenho por:

- sexo;
- faixa etaria.

Essa etapa ajuda a verificar se o modelo apresenta desempenho diferente entre grupos de pacientes.

## Calibracao

O modelo gera probabilidades de risco. Para avaliar se essas probabilidades sao confiaveis, o projeto utiliza:

- curva de calibracao;
- Brier Score.

Isso ajuda a responder se uma probabilidade prevista de 80%, por exemplo, se comporta de fato como um risco proximo de 80%.

## Predicao de novos pacientes

A ultima etapa permite analisar varios pacientes de uma vez.

Exemplo:

```python
pacientes_exemplo = [
    {
        'ID': 'Paciente 1',
        'CS_SEXO': 'M',
        'FEBRE': 1,
        'MIALGIA': 1,
        'CEFALEIA': 1,
        'NAUSEA': 2,
        'VOMITO': 2,
        'DIABETES': 2,
        'HIPERTENSA': 2,
        'RENAL': 2,
        'IDADE_REAL': 22
    },
    {
        'ID': 'Paciente 2',
        'CS_SEXO': 'F',
        'FEBRE': 1,
        'MIALGIA': 1,
        'CEFALEIA': 2,
        'NAUSEA': 1,
        'VOMITO': 1,
        'DIABETES': 1,
        'HIPERTENSA': 1,
        'RENAL': 2,
        'IDADE_REAL': 78
    }
]
```

A saida apresenta:

- identificacao do paciente;
- predicao binaria;
- probabilidade de risco grave;
- faixa de risco;
- recomendacao textual;
- grafico comparativo entre pacientes.

As faixas de risco foram definidas assim:

| Probabilidade | Faixa |
|---|---|
| Menor que 30% | Baixo risco |
| Entre 30% e 70% | Medio risco |
| Maior ou igual a 70% | Alto risco |

## Arquivos gerados

Ao final da execucao, o notebook gera:

| Arquivo | Funcao |
|---|---|
| `modelo_ia_dengue_gravidade.pkl` | Modelo treinado |
| `resultados_modelos_dengue_gravidade.csv` | Tabela de metricas dos modelos |
| `metadata_modelo_dengue_gravidade.json` | Informacoes sobre treinamento, versao e metricas |

O arquivo `metadata` inclui:

- data do treinamento;
- objetivo do modelo;
- melhor modelo selecionado;
- metricas principais;
- variaveis usadas;
- versao do `scikit-learn`.

## Como executar

### No Google Colab

1. Abra o notebook no Google Colab.
2. Execute as celulas em ordem.
3. Faca upload do arquivo `dados_analise.csv`.
4. Aguarde o carregamento e o pre-processamento.
5. Execute as celulas de treinamento.
6. Analise as metricas e os graficos.
7. Execute a celula final para testar novos pacientes.

### Dependencias

Principais bibliotecas usadas:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
joblib
```

## Estrutura sugerida

```text
projeto-dengue-ia/
├── dados_analise.csv
├── notebook_dengue_ia.ipynb
├── modelo_ia_dengue_gravidade.pkl
├── resultados_modelos_dengue_gravidade.csv
├── metadata_modelo_dengue_gravidade.json
└── README.md
```

## Limitacoes

- O projeto depende da qualidade dos dados preenchidos.
- A base possui forte desbalanceamento de classes.
- O alvo foi derivado de `CLASSI_FIN`.
- O modelo pode reaprender criterios clinicos ja existentes.
- Nao ha validacao externa em outra base.
- Nao substitui avaliacao medica.
- O desempenho pode variar em dados de outros anos, estados ou municipios.

## Melhorias futuras

- Investigar oficialmente como `CLASSI_FIN` e definida.
- Testar `EVOLUCAO` como alvo alternativo.
- Testar obito ou hospitalizacao, caso essas variaveis estejam disponiveis.
- Fazer validacao temporal com colunas de data.
- Testar SMOTE em amostra menor.
- Avaliar o modelo em bases de outros anos.
- Criar uma versao em scripts para VS Code.
- Criar uma interface simples para simulacao de pacientes.
- Validar os resultados com profissionais da area da saude.

## Conclusao

Este projeto demonstra como tecnicas de Machine Learning podem ser aplicadas a dados epidemiologicos de dengue para apoiar a identificacao de risco de gravidade.

O modelo analisa idade, sexo, sintomas e comorbidades para estimar a probabilidade de um paciente pertencer ao grupo de dengue com sinais de alarme ou dengue grave.

Apesar dos resultados gerados, a interpretacao deve ser feita com cautela, respeitando as limitacoes dos dados e o contexto clinico.
