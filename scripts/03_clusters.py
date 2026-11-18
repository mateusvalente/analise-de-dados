import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = dataset.copy()
campos = ["preco_produto", "valor_frete", "desconto_pct", "quantidade_cliques", "tempo_site_seg", "itens_carrinho"]
for coluna in campos:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
df = df.dropna(subset=["registro_id"] + campos)
df = df[(df.quantidade_cliques <= 100) & (df.tempo_site_seg <= 1500)]

normalizados = StandardScaler().fit_transform(df[campos])
grupos = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(normalizados)
projecao = PCA(n_components=2, random_state=42).fit_transform(normalizados)

resultado = df[["registro_id", "pagamento_aprovado"] + campos].copy()
resultado["cluster"] = [f"Grupo {indice + 1}" for indice in grupos]
resultado["pca_x"] = projecao[:, 0]
resultado["pca_y"] = projecao[:, 1]
resultado["silhueta_global"] = silhouette_score(normalizados, grupos)
resultado["tipo_ponto"] = "Registro"
resultado["tamanho_ponto"] = 1

# A projeção PCA é linear: a média projetada de cada grupo é seu centroide no gráfico.
centros = (resultado.groupby("cluster", as_index=False)
           [[*campos, "pca_x", "pca_y"]].mean())
centros["registro_id"] = "CENTROIDE " + centros.cluster
centros["pagamento_aprovado"] = ""
centros["tipo_ponto"] = "Centroide"
centros["tamanho_ponto"] = 35
centros["silhueta_global"] = resultado.silhueta_global.iloc[0]
centros["cluster"] = "Centroide " + centros.cluster
resultado = pd.concat([resultado, centros[resultado.columns]], ignore_index=True)

# Perfil de cada centroide em desvios padrão da amostra: permite comparar
# características com unidades diferentes na mesma linha.
perfil = pd.DataFrame(normalizados, columns=campos).assign(cluster=[f"Grupo {i + 1}" for i in grupos])
perfil = perfil.groupby("cluster", as_index=False)[campos].mean().melt(
    id_vars="cluster", var_name="caracteristica", value_name="desvio_padrao"
)
perfil["ordem"] = perfil.caracteristica.map({nome: i + 1 for i, nome in enumerate(campos)})
