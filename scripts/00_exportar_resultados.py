"""Gera CSVs das práticas usando apenas os pacotes de análise do tutorial.

Uso: python scripts/00_exportar_resultados.py
"""

from pathlib import Path
import re
import runpy
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd


BASE = Path(__file__).resolve().parent.parent
ARQUIVO = BASE / "Dataset_Analise_Preditiva_PowerBI_Alunos.xlsx"
SAIDA = BASE / "resultados"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def ler_planilhas():
    with zipfile.ZipFile(ARQUIVO) as pacote:
        livro = ET.fromstring(pacote.read("xl/workbook.xml"))
        relacoes = ET.fromstring(pacote.read("xl/_rels/workbook.xml.rels"))
        caminhos = {item.attrib["Id"]: item.attrib["Target"].lstrip("/") for item in relacoes}
        compartilhadas = []
        if "xl/sharedStrings.xml" in pacote.namelist():
            raiz = ET.fromstring(pacote.read("xl/sharedStrings.xml"))
            compartilhadas = ["".join(t.text or "" for t in item.findall(".//m:t", NS))
                             for item in raiz.findall("m:si", NS)]
        tabelas = {}
        for aba in livro.findall("m:sheets/m:sheet", NS):
            caminho = caminhos[aba.attrib[REL]]
            if not caminho.startswith("xl/"):
                caminho = "xl/" + caminho
            raiz = ET.fromstring(pacote.read(caminho))
            linhas = []
            for linha in raiz.findall(".//m:sheetData/m:row", NS):
                valores = {}
                for celula in linha.findall("m:c", NS):
                    letras = re.match(r"[A-Z]+", celula.attrib["r"]).group()
                    indice = 0
                    for letra in letras:
                        indice = indice * 26 + ord(letra) - 64
                    valor_xml = celula.find("m:v", NS)
                    tipo = celula.attrib.get("t")
                    if tipo == "inlineStr":
                        valor = "".join(t.text or "" for t in celula.findall(".//m:t", NS))
                    elif valor_xml is None:
                        valor = None
                    elif tipo == "s":
                        valor = compartilhadas[int(valor_xml.text)]
                    elif tipo in ("str", "e"):
                        valor = valor_xml.text
                    else:
                        valor = float(valor_xml.text) if "." in valor_xml.text else int(valor_xml.text)
                    valores[indice - 1] = valor
                linhas.append(valores)
            largura = max((max(linha, default=-1) for linha in linhas), default=-1) + 1
            dados = [[linha.get(indice) for indice in range(largura)] for linha in linhas]
            tabelas[aba.attrib["name"]] = pd.DataFrame(dados[1:], columns=dados[0])
    return tabelas


def main():
    tabelas = ler_planilhas()
    SAIDA.mkdir(exist_ok=True)
    praticas = [
        ("01_regressao.py", "Eventos", "ML_Regressao"),
        ("02_classificacao.py", "Eventos", "ML_Classificacao"),
        ("03_clusters.py", "Eventos", "ML_Clusters"),
        ("04_apriori.py", "Itens_Pedido", "ML_Apriori"),
        ("05_decisao.py", "Eventos", "ML_Decisao"),
    ]
    for script, origem, nome in praticas:
        dados = runpy.run_path(str(BASE / "scripts" / script), init_globals={"dataset": tabelas[origem]})
        dados["resultado"].to_csv(SAIDA / f"{nome}.csv", index=False, encoding="utf-8-sig")
        print(f"{nome}: {len(dados['resultado'])} linhas")
        if script == "02_classificacao.py":
            dados["metricas"].to_csv(SAIDA / "ML_Metricas.csv", index=False, encoding="utf-8-sig")
            print(dados["metricas"].to_string(index=False))
        if script == "03_clusters.py":
            dados["perfil"].to_csv(SAIDA / "ML_Perfil_Clusters.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
