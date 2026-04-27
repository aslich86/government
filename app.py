import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(page_title="GovApp Sentiment AI", page_icon="🏛️", layout="wide")

@st.cache_resource
def load_model():
    try:
        with open('nlp_gov_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

data_pack = load_model()

if data_pack is None:
    st.error("⚠️ File 'nlp_gov_model.pkl' tidak ditemukan!")
    st.stop()

model_nlp = data_pack['model']
df = data_pack['data_sampel']

# --- HEADER ---
st.title("🏛️ GovApp Sentiment Intelligence")
st.markdown("Platform Analisis Ulasan Aplikasi Pemerintah Berbasis *Natural Language Processing* (NLP).")
st.markdown("---")

# --- FITUR 1: TESTER SENTIMEN LANGSUNG ---
st.subheader("🔍 Uji Coba AI Analis")
st.markdown("Ketikkan simulasi ulasan dari masyarakat, biarkan AI menebak apakah ini pujian atau keluhan.")

ulasan_baru = st.text_area("Masukkan teks ulasan:", placeholder="Contoh: Aplikasinya sering error pas mau login, tolong diperbaiki servernya!")

if st.button("Analisis Sentimen", type="primary"):
    if ulasan_baru:
        hasil = model_nlp.predict([ulasan_baru])[0]
        probabilitas = model_nlp.predict_proba([ulasan_baru])[0]
        
        # Ambil probabilitas persentase kepastian AI
        prob_max = max(probabilitas) * 100
        
        if hasil == 'Positif':
            st.success(f"🟢 **Sentimen POSITIF** (Tingkat Keyakinan AI: {prob_max:.1f}%)")
        else:
            st.error(f"🔴 **Sentimen NEGATIF** (Tingkat Keyakinan AI: {prob_max:.1f}%)")
    else:
        st.warning("Ketik ulasannya dulu dong!")

st.markdown("---")

# --- FITUR 2: DASHBOARD ANALITIK ---
st.subheader("📊 Analitik Sampel Ulasan Aplikasi")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribusi Sentimen Keseluruhan**")
    # Asumsi nama kolomnya 'app_name' (disesuaikan otomatis dari Colab)
    df_sentimen = df['Sentimen'].value_counts().reset_index()
    df_sentimen.columns = ['Sentimen', 'Jumlah']
    
    fig1 = px.pie(df_sentimen, values='Jumlah', names='Sentimen', hole=0.4, 
                  color='Sentimen', color_discrete_map={'Positif':'#10b981', 'Negatif':'#ef4444'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("**Proporsi Sentimen per Aplikasi**")
    # Memeriksa nama kolom aplikasi yang disimpan dari Colab
    kolom_app = [col for col in df.columns if col not in ['content', 'score', 'Sentimen']][0]
    
    df_app = df.groupby([kolom_app, 'Sentimen']).size().reset_index(name='Jumlah')
    fig2 = px.bar(df_app, x='Jumlah', y=kolom_app, color='Sentimen', orientation='h',
                  color_discrete_map={'Positif':'#10b981', 'Negatif':'#ef4444'})
    fig2.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
with st.expander("Lihat Data Mentah Ulasan (Disensor sebagian untuk privasi)"):
    st.dataframe(df.head(100), use_container_width=True)