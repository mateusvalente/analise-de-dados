import pandas as pd

df = dataset.copy()
for coluna in ("valor_total", "valor_frete"):
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
df = df[df.pagamento_aprovado.eq("nao") & df.valor_total.notna()].copy()

# Hipóteses didáticas; os incrementos não são estimativas causais.
custo_contato = 8.0
margem = 0.25
incremento = df.valor_frete.apply(lambda frete: 0.08 if frete > 40 else 0.04)
df["ganho_esperado_bruto"] = df.valor_total * margem * incremento
df["ganho_esperado_liquido"] = df.ganho_esperado_bruto - custo_contato
df["custo_acao"] = custo_contato
df = df[df.ganho_esperado_liquido > 0].sort_values("ganho_esperado_liquido", ascending=False)
df["custo_acumulado"] = df.custo_acao.cumsum()
df["ganho_liquido_acumulado"] = df.ganho_esperado_liquido.cumsum()
df["ordem_prioridade"] = range(1, len(df) + 1)
df["selecionado"] = df.custo_acumulado.le(500).map({True: "sim", False: "nao"})

resultado = df[["registro_id", "origem_trafego", "valor_total", "valor_frete",
                "ganho_esperado_bruto", "custo_acao", "ganho_esperado_liquido",
                "custo_acumulado", "ganho_liquido_acumulado", "ordem_prioridade", "selecionado"]]
