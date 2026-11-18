import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = dataset.copy().dropna(subset=["registro_id", "pagamento_aprovado"])
numericas = ["preco_produto", "valor_frete", "desconto_pct", "quantidade_cliques", "tempo_site_seg", "itens_carrinho"]
categoricas = ["dispositivo", "origem_trafego", "canal", "categoria_produto", "cupom_utilizado"]
for coluna in numericas:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

alvo = df.pagamento_aprovado.astype(str).str.lower().eq("sim").astype(int)
atributos = df[numericas + categoricas]
indice_treino, indice_teste = train_test_split(df.index, test_size=0.25, stratify=alvo, random_state=42)
preparo = ColumnTransformer([
    ("numericas", Pipeline([("faltantes", SimpleImputer(strategy="median")), ("escala", StandardScaler())]), numericas),
    ("categoricas", Pipeline([("faltantes", SimpleImputer(strategy="most_frequent")),
                               ("indicadoras", OneHotEncoder(handle_unknown="ignore"))]), categoricas),
])
modelos = {
    "Logistica": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Arvore": DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=8, random_state=42),
}

resultado = df.loc[indice_teste, ["registro_id", "pagamento_aprovado"] + numericas + categoricas].copy()
linhas = []
for nome, algoritmo in modelos.items():
    pipeline = Pipeline([("preparo", preparo), ("modelo", algoritmo)])
    pipeline.fit(atributos.loc[indice_treino], alvo.loc[indice_treino])
    probabilidade = pipeline.predict_proba(atributos.loc[indice_teste])[:, 1]
    previsao = (probabilidade >= 0.5).astype(int)
    resultado[f"prob_{nome}"] = probabilidade
    resultado[f"prev_{nome}"] = ["sim" if valor else "nao" for valor in previsao]
    linhas.append([
        nome,
        accuracy_score(alvo.loc[indice_teste], previsao),
        precision_score(alvo.loc[indice_teste], previsao, zero_division=0),
        recall_score(alvo.loc[indice_teste], previsao, zero_division=0),
        f1_score(alvo.loc[indice_teste], previsao, zero_division=0),
        roc_auc_score(alvo.loc[indice_teste], probabilidade),
    ])
metricas = pd.DataFrame(linhas, columns=["modelo", "acuracia", "precisao", "recall", "f1", "auc_roc"])
