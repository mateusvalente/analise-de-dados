import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = dataset.copy()
for coluna in ("quantidade_cliques", "tempo_site_seg"):
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
df = df.dropna(subset=["registro_id", "quantidade_cliques", "tempo_site_seg"])
df = df[(df.quantidade_cliques <= 100) & (df.tempo_site_seg <= 1500)]

treino, teste = train_test_split(df, test_size=0.25, random_state=42)
modelo = LinearRegression().fit(treino[["quantidade_cliques"]], treino.tempo_site_seg)
previsao = modelo.predict(teste[["quantidade_cliques"]])

resultado = teste[["registro_id", "quantidade_cliques", "tempo_site_seg"]].copy()
resultado["tempo_previsto"] = previsao
resultado["residuo"] = resultado.tempo_site_seg - resultado.tempo_previsto
resultado["MAE_teste"] = mean_absolute_error(teste.tempo_site_seg, previsao)
resultado["R2_teste"] = r2_score(teste.tempo_site_seg, previsao)
resultado["intercepto"] = modelo.intercept_
resultado["coeficiente_cliques"] = modelo.coef_[0]
