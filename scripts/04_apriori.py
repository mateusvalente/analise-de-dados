import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = dataset.copy()
df = df[df.pagamento_aprovado.astype(str).str.lower().eq("sim")]
cestas = pd.crosstab(df.registro_id, df.produto).gt(0)
frequentes = apriori(cestas, min_support=0.025, use_colnames=True)
regras = association_rules(frequentes, num_itemsets=len(cestas), metric="confidence", min_threshold=0.20)
regras = regras[(regras.lift > 1.0)
                & regras.antecedents.apply(lambda itens: len(itens) == 1)
                & regras.consequents.apply(lambda itens: len(itens) == 1)].copy()
regras["antecedentes"] = regras.antecedents.apply(lambda itens: ", ".join(sorted(itens)))
regras["consequentes"] = regras.consequents.apply(lambda itens: ", ".join(sorted(itens)))
resultado = (regras[["antecedentes", "consequentes", "support", "confidence", "lift"]]
             .sort_values(["antecedentes", "confidence"], ascending=[True, False])
             .reset_index(drop=True))
resultado["suporte_pct"] = resultado.support * 100
resultado["confianca_pct"] = resultado.confidence * 100
