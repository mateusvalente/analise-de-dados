# Análise preditiva no Power BI

## Abrir o projeto

Abra `Analise_Preditiva_Tutorial.pbix` no Power BI Desktop. O relatório contém oito páginas: **Indicadores**, **Resumo mensal**, **Diagnostico**, **Regressao Python**, **Classificacao Python**, **Clusters Python**, **Apriori Python** e **Decisao Python**.

O arquivo `Dataset_Analise_Preditiva_PowerBI_Alunos.xlsx` é a origem dos dados. O modelo contém `Eventos` (1.600 linhas) e `Itens_Pedido` (3.263 linhas), relacionados por `registro_id`.

## Integração com Python

Em **Transformar dados**, há seis consultas que executam Python durante a atualização: `ML_Regressao_Python`, `ML_Classificacao_Python`, `ML_Metricas_Python`, `ML_Clusters_Python`, `ML_Apriori_Python` e `ML_Decisao_Python`. Elas chamam os arquivos em `scripts/` e carregam os resultados no modelo. As cinco páginas Python exibem esses resultados; a página de classificação inclui métricas, F1 por modelo e matriz de confusão.

Para atualizar, mantenha esta pasta em `C:\Users\mateusv.dti\Desktop\analise-preditiva` e use **Atualizar** no Power BI Desktop. O Power BI pode pedir autorização para executar os scripts Python. Se a pasta for movida, altere o caminho `base` nas seis consultas do Power Query. A instalação Python configurada neste computador contém `pandas`, `scikit-learn` e `mlxtend`.

`scripts/00_exportar_resultados.py` também permite gerar `resultados/*.csv` para inspeção fora do Power BI. Esses CSVs não são a origem das seis consultas Python do PBIX.

## Interpretação dos modelos

Na classificação, o conjunto de teste tem 400 registros. Com limiar de 0,5, Árvore e Random Forest não classificaram nenhum registro como aprovado; por isso o recall e o F1 da classe `sim` são zero. A página **Classificacao Python** permite comparar isso com a regressão logística.
