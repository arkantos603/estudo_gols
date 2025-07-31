import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Função auxiliar para calcular mediana corretamente
def calc_median(values):
    arr = sorted(values)
    n = len(arr)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    else:
        return (arr[mid - 1] + arr[mid]) / 2

# Carregar dados com cache para melhorar performance
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

st.set_page_config(layout="wide", page_title="Análise de Gols (2016-2024)")
st.title("Análise Profunda de Gols dos Jogadores (2016-2024)")

# Carregar dados do CSV padrão
df = load_data("gols_jogadores_2016_2024_com_filtros.csv")

# Sidebar - configurações do usuário
st.sidebar.header("Configurações de Filtro")
anos = [str(y) for y in range(2016, 2025)]
selected_years = st.sidebar.multiselect("Selecione os anos", anos, default=anos)
top_n = st.sidebar.slider("Número de principais artilheiros", 5, 20, 10)
players = df['Nome'].unique().tolist()
selected_players = st.sidebar.multiselect("Jogadores para análise específica", players)

st.markdown("---")
# Preparar dados long-form completos
df_long_full = df.melt(id_vars=['Nome'], value_vars=anos, var_name='Ano', value_name='Gols')
df_long_full['Ano'] = df_long_full['Ano'].astype(int)
df_long_full = df_long_full.dropna(subset=['Gols'])  # remove anos sem dado
# Aplica filtro de anos
df_long = df_long_full[df_long_full['Ano'].astype(str).isin(selected_years)]

# Determinar dataset atual (geral ou específico)
def current_data():
    if selected_players:
        return df_long[df_long['Nome'].isin(selected_players)]
    return df_long

data = current_data()
context = 'Selecionados' if selected_players else 'Geral'

# 1) Top Artilheiros
st.subheader(f"Top {top_n} Artilheiros ({context})")
resumo = data.groupby('Nome')['Gols'].sum().reset_index()
resumo_sorted = resumo.sort_values('Gols', ascending=False).head(top_n)
st.dataframe(resumo_sorted.rename(columns={'Gols':'Total Selecionado'}))
fig, ax = plt.subplots()
sns.barplot(data=resumo_sorted, x='Gols', y='Nome', ax=ax)
ax.set_xlabel('Gols'); ax.set_ylabel('Jogador')
st.pyplot(fig)

# 2) Série Temporal de Gols por Ano
st.subheader(f"Série Temporal de Gols por Ano ({context})")
fig2, ax2 = plt.subplots()
if selected_players:
    sns.lineplot(data=data, x='Ano', y='Gols', hue='Nome', marker='o', ax=ax2)
else:
    sns.lineplot(data=data, x='Ano', y='Gols', estimator='mean', marker='o', ax=ax2)
ax2.set_xlabel('Ano'); ax2.set_ylabel('Gols')
st.pyplot(fig2)

# 3) Média e Mediana de Gols por Temporada
st.subheader(f"Média e Mediana de Gols por Temporada ({context})")
# Coletar gols do dataset atual
gols_values = data['Gols'].tolist()
media = np.mean(gols_values) if gols_values else None
mediana = calc_median(gols_values) if gols_values else None
# Exibir valores gerais
if selected_players:
    # Valores individuais por jogador
    st.subheader("Valores por Jogador")
    for nome in selected_players:
        gols_jog = df_long[df_long['Nome'] == nome]['Gols'].dropna().tolist()
        media_j = np.mean(gols_jog) if gols_jog else None
        mediana_j = calc_median(gols_jog) if gols_jog else None
        st.write(f"**{nome}** – Média: {media_j:.2f} | Mediana: {mediana_j:.2f}")
    # Valor geral entre selecionados
    st.subheader("Valores Gerais (Selecionados)")
    st.write(f"Média: {media:.2f} | Mediana: {mediana:.2f}")
else:
    st.write(f"Média: {media:.2f} | Mediana: {mediana:.2f}")

# 4) Regressão à Média
st.subheader(f"Regressão à Média ({context})")
reg_df = data.copy()
reg_df['Next_Gols'] = reg_df.groupby('Nome')['Gols'].shift(-1)
reg_df = reg_df.dropna(subset=['Next_Gols'])
fig4, ax4 = plt.subplots()
sns.regplot(data=reg_df, x='Gols', y='Next_Gols', scatter_kws={'alpha':0.5}, ax=ax4)
ax4.set_xlabel('(x) Gols temporada N'); ax4.set_ylabel('(y) Gols temporada N+1')
st.pyplot(fig4)

# 5) Variação Percentual da Melhor Temporada
st.subheader(f"Variação Percentual da Melhor Temporada ({context})")
pct_dict = {}
base_group = data.groupby('Nome') if selected_players else df_long_full.groupby('Nome')
for nome, grp in base_group:
    gols_list = grp['Gols'].dropna().tolist()
    if not gols_list:
        continue
    ano_best = grp.loc[grp['Gols'].idxmax()]['Ano']
    gols_best = max(gols_list)
    next_goals = grp[grp['Ano'] == ano_best+1]['Gols'].values
    if next_goals.size and gols_best != 0:
        pct_dict[nome] = (next_goals[0] / gols_best - 1) * 100
if pct_dict:
    avg_pct = np.mean(list(pct_dict.values()))
    st.write(f"Variação média: {avg_pct:.2f}%")
    fig5, ax5 = plt.subplots()
    sns.histplot(list(pct_dict.values()), kde=True, ax=ax5)
    ax5.set_xlabel('Variação (%)')
    st.pyplot(fig5)
    if selected_players:
        st.subheader("Variação Percentual por Jogador")
        for nome, pct in pct_dict.items():
            st.write(f"**{nome}**: {pct:.2f}%")
else:
    st.write("Sem dados suficientes para calcular variação percentual.")

st.markdown("---")
st.write("Desenvolvido com 🐍 Python, 📊 Seaborn/Matplotlib e ⚡ Streamlit.")
