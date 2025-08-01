import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

# Função auxiliar para calcular mediana corretamente
def calc_median(values):
    arr = sorted(values)
    n = len(arr)
    if n == 0:
        return None
    mid = n // 2
    return arr[mid] if n % 2 == 1 else (arr[mid-1] + arr[mid]) / 2

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
# Preparar dados
df_long_full = df.melt(id_vars=['Nome'], value_vars=anos, var_name='Ano', value_name='Gols')
df_long_full['Ano'] = df_long_full['Ano'].astype(int)
df_long_full = df_long_full.dropna(subset=['Gols'])
df_long = df_long_full[df_long_full['Ano'].astype(str).isin(selected_years)]

def current_data():
    return df_long[df_long['Nome'].isin(selected_players)] if selected_players else df_long

data = current_data()
context = 'Selecionados' if selected_players else 'Geral'

##### 1) Top Artilheiros #####
st.subheader(f"Top {top_n} Artilheiros ({context})")
resumo = data.groupby('Nome')['Gols'].sum().reset_index()
resumo_sorted = resumo.sort_values('Gols', ascending=False).head(top_n)
st.dataframe(resumo_sorted.rename(columns={'Gols':'Total Selecionado'}))
fig, ax = plt.subplots(figsize=(3,2))
sns.barplot(data=resumo_sorted, x='Gols', y='Nome', ax=ax)
ax.set_xlabel('Gols'); ax.set_ylabel('Jogador')
st.pyplot(fig)

##### 2) Série Temporal #####
st.subheader(f"Série Temporal de Gols por Ano ({context})")
fig2, ax2 = plt.subplots(figsize=(3,2))
if selected_players:
    sns.lineplot(data=data, x='Ano', y='Gols', hue='Nome', marker='o', ax=ax2)
else:
    sns.lineplot(data=data, x='Ano', y='Gols', estimator='mean', marker='o', ax=ax2)
ax2.set_xlabel('Ano'); ax2.set_ylabel('Gols')
st.pyplot(fig2)

##### 3) Média e Mediana #####
st.subheader(f"Média e Mediana de Gols por Temporada ({context})")
vals = data['Gols'].tolist()
media = np.mean(vals) if vals else np.nan
mediana = calc_median(vals) if vals else np.nan
if selected_players:
    for nome in selected_players:
        v = df_long[df_long['Nome']==nome]['Gols'].dropna().tolist()
        st.write(f"**{nome}** – Média: {np.mean(v):.2f} | Mediana: {calc_median(v):.2f}")
    st.write(f"**Geral (sel)** – Média: {media:.2f} | Mediana: {mediana:.2f}")
else:
    st.write(f"Média: {media:.2f} | Mediana: {mediana:.2f}")

##### 4) Regressão à Média #####
st.subheader(f"Regressão à Média ({context})")
reg_df = data.copy()
reg_df['Next_Gols'] = reg_df.groupby('Nome')['Gols'].shift(-1)
reg_df = reg_df.dropna(subset=['Next_Gols'])
fig4, ax4 = plt.subplots(figsize=(3,2))
sns.regplot(data=reg_df, x='Gols', y='Next_Gols', scatter_kws={'alpha':0.5}, ax=ax4)
ax4.set_xlabel('(x) Gols N'); ax4.set_ylabel('(y) Gols N+1')
st.pyplot(fig4)

##### 5) Variação Percentual #####
st.subheader(f"Variação Percentual da Melhor Temporada ({context})")
pct = {}
gs = data.groupby('Nome') if selected_players else df_long_full.groupby('Nome')
for nome,grp in gs:
    b = grp.loc[grp['Gols'].idxmax()]
    n = grp[grp['Ano']==b['Ano']+1]
    if not n.empty and b['Gols']>0:
        pct[nome] = (n['Gols'].iloc[0]/b['Gols']-1)*100
if pct:
    df_pct = pd.DataFrame([{'Nome':n,'Max':grp['Gols'].max(),'Pct':p}
                            for n,p in pct.items()
                            for _,grp in [(n,gs.get_group(n))]])
    st.write(f"Variação média: {df_pct['Pct'].mean():.2f}%")
    fig_cdf,ax_cdf=plt.subplots(figsize=(3,2))
    sns.histplot(df_pct['Pct'],stat='density',cumulative=True,ax=ax_cdf)
    st.pyplot(fig_cdf)
    fig_sc,ax_sc=plt.subplots(figsize=(3,2))
    sns.scatterplot(data=df_pct,x='Max',y='Pct',ax=ax_sc)
    st.pyplot(fig_sc)
    fig_box,ax_box=plt.subplots(figsize=(3,2))
    df_pct['Bin']=pd.cut(df_pct['Max'],bins=[0,10,20,30,df_pct['Max'].max()+1],
                         labels=['0-10','11-20','21-30','>30'])
    sns.boxplot(data=df_pct,x='Bin',y='Pct',ax=ax_box)
    st.pyplot(fig_box)
    rel=[]
    for nome,grp in df_long_full.groupby('Nome'):
        idx=grp['Gols'].idxmax();y=grp.loc[idx,'Ano']
        for off in range(-2,3):
            v=grp[grp['Ano']==y+off]['Gols']
            rel.append({'Off':off,'Gols':v.values[0] if not v.empty else np.nan})
    df_rel=pd.DataFrame(rel).dropna()
    fig_rel,ax_rel=plt.subplots(figsize=(3,2))
    sns.lineplot(data=df_rel,x='Off',y='Gols',estimator='mean',marker='o',ax=ax_rel)
    st.pyplot(fig_rel)
    fig_hm,ax_hm=plt.subplots(figsize=(3,2))
    qt=pd.qcut(df_pct['Pct'],4)
    pt=pd.pivot_table(df_pct,index='Bin',columns=qt,aggfunc='size',fill_value=0)
    sns.heatmap(pt,annot=True,fmt='d',ax=ax_hm)
    st.pyplot(fig_hm)
    st.plotly_chart(px.scatter(df_pct,x='Max',y='Pct',hover_data=['Nome'],width=300,height=200))
else:
    st.write("Sem dados para variações.")

##### Declínio após Temporadas Excepcionais #####
st.markdown("---")
st.subheader("Declínio após Temporadas Excepcionais")
thrs=[15,20,25,30];res=[]
for th in thrs:
    p=[]
    for nome, grp in df_long_full.groupby('Nome'):
        br=grp.loc[grp['Gols'].idxmax()]
        if br['Gols']>=th:
            ns=grp[grp['Ano']==br['Ano']+1]
            if not ns.empty:
                p.append((ns['Gols'].iloc[0]/br['Gols']-1)*100)
    if p:
        decline_rate=np.mean([1 for x in p if x<0])/len(p)*100
        res.append({'Thr':th,'%Declínio':decline_rate})
res_df=pd.DataFrame(res)
fig6,ax6=plt.subplots(figsize=(6,4))
sns.barplot(data=res_df,x='Thr',y='%Declínio',ax=ax6)
ax6.set_ylabel('% de jogadores com declínio após temporada >= Threshold')
st.pyplot(fig6)

##### Superação após temporadas ≥10 gols #####
st.markdown("---")
st.subheader("Superação após temporadas com ≥10 gols")
# Identificar jogadores que tiveram pico >=10 e se superaram no ano seguinte
tested_players=set()
success_players=set()
for nome,grp in df_long_full.groupby('Nome'):
    for _,row in grp.iterrows():
        if row['Gols']>=10:
            nxt=grp[grp['Ano']==row['Ano']+1]['Gols']
            if not nxt.empty:
                tested_players.add(nome)
                if nxt.values[0]>row['Gols']:
                    success_players.add(nome)
# Preparar DataFrame
count_success=len(success_players)
count_fail=len(tested_players-success_players)
df_supr=pd.DataFrame({
    'Resultado':['Superaram','Não superaram'],
    'Jogadores':[count_success,count_fail]
})
# Gráfico de barras
# Gráfico interativo com tooltip
fig7 = px.bar(
    df_supr,
    x='Resultado',
    y='Jogadores',
    hover_data={'Jogadores':True},
    labels={'Jogadores':'Número de jogadores'},
    title='Jogadores com ≥10 gols na liga durante a temporada',
    height=300
)
st.plotly_chart(fig7)

st.markdown("---")
st.write("Desenvolvido com 🐍 Python, 📊 Seaborn/Matplotlib e ⚡ Streamlit.")
