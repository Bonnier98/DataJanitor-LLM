import streamlit as st
import pandas as pd
import numpy as np
import re
import csv
from unidecode import unidecode
from datetime import datetime
from io import StringIO, BytesIO

# --- O CÓDIGO DO LLM (P2: Receita de Limpeza) ---
codigo_funcoes_llm = """
import re
import pandas as pd
import numpy as np
from unidecode import unidecode
from datetime import datetime

# Mapa de números por extenso (expandido e corrigido para evitar problemas)
MAPA_TEXTO_NUMERO = {
    'UM': 1, 'DOIS': 2, 'TRES': 3, 'QUATRO': 4, 'CINCO': 5,
    'SEIS': 6, 'SETE': 7, 'OITO': 8, 'NOVE': 9, 'DEZ': 10,
    'ONZE': 11, 'DOZE': 12, 'TREZE': 13, 'QUATORZE': 14, 'QUINZE': 15,
    'DEZESSEIS': 16, 'DEZESSETE': 17, 'DEZOITO': 18, 'DEZENOVE': 19, 'VINTE': 20
}

# 1. Funções que lidam com TEXTO
def limpar_Texto(valor):
    if pd.isna(valor) or valor is None: return valor
    if not isinstance(valor, str):
        valor = str(valor)

    # Padroniza em minúsculo
    valor_limpo = unidecode(valor).lower()
    valor_limpo = re.sub(r'[^a-z0-9\s]', ' ', valor_limpo)
    valor_limpo = re.sub(r'\s+', ' ', valor_limpo).strip()
    return valor_limpo

# 2. Funções que lidam com NÚMEROS DECIMAIS (FLOAT e INT)
def limpar_Numerico_Decimal(valor):
    if pd.isna(valor) or str(valor).strip() == '': return np.nan
    valor_str = str(valor)

    # Prioridade 1: Tenta mapear números escritos por extenso PURO.
    # Esta lógica é a mais agressiva para isolar a palavra
    texto_puro = re.sub(r'[^\w]', '', unidecode(valor_str)).upper().strip()
    if texto_puro in MAPA_TEXTO_NUMERO:
        return float(MAPA_TEXTO_NUMERO[texto_puro])

    # Prioridade 2: Tenta conversão direta para float (para números e decimais com sujeira)
    valor_limpo_numerico = valor_str.replace('R$', '').replace('€', '').replace(' ', '').replace(',', '.')
    try:
        # Tenta extrair apenas números e pontos
        return float(re.sub(r'[^\d\.]', '', valor_limpo_numerico))
    except ValueError:
        pass

    return np.nan

# 3. Funções que lidam com DATAS
def limpar_Data(valor):
    if pd.isna(valor) or valor is None: return pd.NaT
    return pd.to_datetime(valor, errors='coerce', dayfirst=True)
"""
# --- Lógica de Split (Pré-processamento) ---

def split_composite_value(valor):
    """Tenta separar uma string em uma parte de texto e uma parte numérica."""
    if not isinstance(valor, str): return (valor, np.nan)

    match = re.search(r'(\D*) ([\d,\.]*)', valor.strip())

    if match and (match.group(2).strip() != '' or match.group(1).strip() == valor.strip()):
        text_part = match.group(1).strip()
        num_part = match.group(2).strip()

        if num_part and re.search(r'\d', num_part):
            return (text_part, num_part)

    return (valor, np.nan)


def aplicar_limpeza_em_lote(df):

    exec(codigo_funcoes_llm, globals())

    df_limpo = df.copy()
    df.columns = df.columns.str.strip()

    coluna_mista_tratada = False

    # --- NOVO PRÉ-PROCESSAMENTO: Mapeamento de Texto Numérico Crítico ---
    # Garante que valores problemáticos como 'VINTE' sejam transformados em '20' ANTES da limpeza

    colunas_numericas_criticas = ['QUANTIDADE', 'CONTAGEM', 'DIAS', 'TREINADOS']

    for col_suja in df.columns:
        col_upper = col_suja.upper()

        if any(keyword in col_upper for keyword in colunas_numericas_criticas) and df[col_suja].dtype == 'object':

            st.info(f"Pré-processamento: Mapeando números por extenso na coluna '{col_suja}'.")

            # Cria uma cópia da série para substituição
            df[col_suja + '_TEMP_FIX'] = df[col_suja].astype(str).str.upper().str.strip()

            # Mapeamento e substituição direta para garantir a conversão
            df[col_suja + '_TEMP_FIX'] = df[col_suja + '_TEMP_FIX'].replace(
                {
                    'VINTE': '20',
                    'DEZ': '10',
                    'UM': '1',
                    'DOIS': '2',
                    # Mapeie outras exceções se necessário
                }
            )
            # Substitui a coluna original pela coluna temporariamente corrigida
            df[col_suja] = df[col_suja + '_TEMP_FIX']
            df = df.drop(columns=[col_suja + '_TEMP_FIX']) # Remove a coluna temporária

    # --- PRÉ-PROCESSAMENTO: SPLIT INTELIGENTE (EXISTENTE) ---
    for col_suja in df.columns:
        # ... (restante da lógica de split)
        col_upper = col_suja.upper()

        if ('PRODUTO' in col_upper or 'ITEM' in col_upper) and df[col_suja].dtype == 'object':

            sample_size = min(len(df), 100)
            split_check = df[col_suja].head(sample_size).apply(split_composite_value)

            if sum(split_check.apply(lambda x: pd.notna(x[1]))) / sample_size > 0.1:
                st.info(f"Coluna '{col_suja}' detectada como MISTA. Realizando split em duas novas colunas.")

                split_results = df[col_suja].apply(split_composite_value)

                df['Temp_Texto'] = split_results.apply(lambda x: x[0])
                df['Temp_Numerico'] = split_results.apply(lambda x: x[1])

                df_limpo[f'{col_suja}_TEXTO_Limpo'] = df['Temp_Texto'].apply(globals()['limpar_Texto'])
                df_limpo[f'{col_suja}_NUMERICO_Limpo'] = df['Temp_Numerico'].apply(globals()['limpar_Numerico_Decimal']).astype('float64')

                coluna_mista_tratada = True
                break

    # --- PROCESSAMENTO PRINCIPAL: DType INFERENCE (Baseado na MAIORIA) ---

    for coluna_suja in df.columns:

        if coluna_mista_tratada and ('PRODUTO' in coluna_suja.upper() or 'ITEM' in coluna_suja.upper()):
            continue

        dtype_original = df[coluna_suja].dtype
        nome_coluna_limpa = f"{coluna_suja}_Limpo"
        funcao_limpeza = None
        tipo_final = None

        coluna_upper = coluna_suja.upper()

        if dtype_original in ['int64', 'float64', 'Int64']:
            df_limpo[nome_coluna_limpa] = df[coluna_suja]
            continue

        if 'DATA' in coluna_upper or 'DT' in coluna_upper:
            funcao_limpeza = globals()['limpar_Data']
            tipo_final = 'datetime64[ns]'

        elif 'PRECO' in coluna_upper or 'VALOR' in coluna_upper or 'GRAMATURA' in coluna_upper or 'PERCENT' in coluna_upper:
            funcao_limpeza = globals()['limpar_Numerico_Decimal']
            tipo_final = 'float64'

        elif 'QUANTIDADE' in coluna_upper or 'CONTAGEM' in coluna_upper or 'DIAS' in coluna_upper:
            funcao_limpeza = globals()['limpar_Numerico_Decimal']
            tipo_final = 'Int64'

        elif dtype_original == 'object':
            funcao_limpeza = globals()['limpar_Texto']
            tipo_final = 'object'

        else:
            df_limpo[nome_coluna_limpa] = df[coluna_suja]

        # --- APLICAÇÃO E TIPAGEM FINAL ---
        if funcao_limpeza is not None:
            # Aplica a função de limpeza na coluna (que agora já deve ter '20' em vez de 'VINTE')
            df_limpo[nome_coluna_limpa] = df[coluna_suja].apply(funcao_limpeza)

            # Tipagem final, forçando o tipo inferido (Int64 para dias_treinados)
            if tipo_final == 'float64':
                df_limpo[nome_coluna_limpa] = pd.to_numeric(df_limpo[nome_coluna_limpa], errors='coerce').astype('float64')
            elif tipo_final == 'Int64':
                df_limpo[nome_coluna_limpa] = pd.to_numeric(df_limpo[nome_coluna_limpa], errors='coerce').astype('Int64')
            elif tipo_final == 'datetime64[ns]':
                df_limpo[nome_coluna_limpa] = pd.to_datetime(df_limpo[nome_coluna_limpa], errors='coerce')

    return df_limpo

# --- FUNÇÃO DE LEITURA ROBusta DE CSV ---

def read_robust_csv(uploaded_file):

    raw_data = uploaded_file.getvalue()

    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1']
    separators_to_try = [',', ';', '\t', '|']

    # 1. Tenta usar o Sniffer (mais preciso)
    for encoding in encodings_to_try:
        try:
            data = raw_data.decode(encoding)
            dialect = csv.Sniffer().sniff(data[:1024])
            return pd.read_csv(StringIO(data), sep=dialect.delimiter, encoding=encoding)
        except (UnicodeDecodeError, csv.Error):
            continue

    # 2. Tenta combinações padrão se o Sniffer falhar
    for encoding in encodings_to_try:
        for sep in separators_to_try:
            try:
                data = raw_data.decode(encoding)
                return pd.read_csv(StringIO(data), sep=sep)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

    return None

# --- FRONT-END STREAMLIT ---

st.set_page_config(layout="wide")

st.title("🤖 DataJanitor LLM (Versão Final)")
st.markdown("---")
st.subheader("Ferramenta de Limpeza de Dados Automatizada com IA")
st.caption("A conversão de números por extenso foi reforçada (mapeamento direto) e a padronização de texto é em minúsculo.")

uploaded_file = st.file_uploader("Upload do Arquivo Sujo (.csv ou .xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:

    df = None
    try:
        if uploaded_file.name.endswith('.csv'):
            df = read_robust_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)


        if df is not None and not df.empty:
            df.columns = df.columns.str.strip()

            st.success(f"Arquivo '{uploaded_file.name}' carregado com {len(df)} linhas.")

            tab1, tab2, tab3 = st.tabs(["📊 Dados Sujos (Origem)", "✨ Dados Limpos (Destino)", "🧪 Prova Técnica (DTypes)"])

            with tab1:
                st.markdown("### Amostra de Dados Brutos")
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown("---")
                st.caption("Tipos de Dados Atuais (Antes da limpeza):")
                st.code(df.dtypes)

            if st.button("▶️ Rodar Limpeza de Dados Dinâmica (LLM em Lote)"):
                with st.spinner('Analisando tipos de dados e aplicando limpeza dinâmica com split...'):
                    df_limpo = aplicar_limpeza_em_lote(df)

                st.balloons()
                st.success("Limpeza concluída com sucesso!")

                colunas_limpas_e_id = [c for c in df_limpo.columns if c.endswith('_Limpo') or 'ID' in c.upper() or 'TEXTO' in c.upper() or 'NUMERICO' in c.upper()]

                with tab2:
                    st.markdown("### DataFrame Final Limpo e Padronizado")
                    st.dataframe(df_limpo[colunas_limpas_e_id].head(10), use_container_width=True)

                    csv_limpo = df_limpo[colunas_limpas_e_id].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Baixar Dados Limpos (.csv)",
                        data=csv_limpo,
                        file_name='dados_limpos_llm.csv',
                        mime='text/csv',
                        type="primary"
                    )

                with tab3:
                    st.markdown("### Tipos de Dados Finais (Prova de Sucesso Técnico)")
                    st.caption("Tipos corrigidos dinamicamente: **Int64** (Inteiros), **float64** (Decimais), **datetime64[ns]** (Datas).")
                    st.code(df_limpo[colunas_limpas_e_id].dtypes)

        else:
             st.error("Falha ao carregar o arquivo CSV. O algoritmo robusto não conseguiu determinar o separador ou a codificação correta.")


    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Detalhe: {e}")
