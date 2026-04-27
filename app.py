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

# --- FITUR BARU: PEMETAAN KATEGORI INSTANSI ---
# Kita cari tau dulu apa nama kolom aplikasinya (dari Colab kemarin)
kolom_app = [col for col in df.columns if col not in ['content', 'score', 'Sentimen']][0]

# Mapping/Kategorisasi Aplikasi ke Instansi Induk
kategori_map = {
    'Digital Korlantas': 'Kepolisian & Samsat',
    'Signal Samsat': 'Kepolisian & Samsat',
    'Satu Sehat': 'Kementerian Kesehatan',
    'Info BMKG': 'Badan Pusat (Non-Kementerian)'
}

# Membuat kolom baru bernama 'Kategori_Instansi'
df['Kategori_Instansi'] = df[kolom_app].map(kategori_map).fillna('Instansi Lainnya')


# --- HEADER ---
st.title("🏛️ GovApp Sentiment Intelligence")
st.markdown("Platform Analisis Ulasan Aplikasi Pemerintah Berbasis *Natural Language Processing* (NLP).")
st.markdown("---")

# --- FITUR 1: TESTER SENTIMEN LANGSUNG ---
with st.expander("🔍 Uji Coba AI Analis (Klik untuk membuka)"):
    st.markdown("Ketikkan simulasi ulasan dari masyarakat, biarkan AI menebak apakah ini pujian atau keluhan.")
    ulasan_baru = st.text_area("Masukkan teks ulasan:", placeholder="Contoh: Aplikasinya sering error pas mau login!")

    if st.button("Analisis Sentimen", type="primary"):
        if ulasan_baru:
            hasil = model_nlp.predict([ulasan_baru])[0]
            probabilitas = model_nlp.predict_proba([ulasan_baru])[0]
            prob_max = max(probabilitas) * 100
            
            if hasil == 'Positif':
                st.success(f"🟢 **Sentimen POSITIF** (Keyakinan AI: {prob_max:.1f}%)")
            else:
                st.error(f"🔴 **Sentimen NEGATIF** (Keyakinan AI: {prob_max:.1f}%)")
        else:
            st.warning("Ketik ulasannya dulu dong!")

# --- FITUR 2: DASHBOARD ANALITIK (REVISI DOSEN) ---
st.subheader("📊 Analitik Sentimen Publik (Hierarki Instansi)")

# Tab untuk merapikan tampilan
tab1, tab2 = st.tabs(["🏛️ Rekap per Instansi Induk", "📱 Detail per Aplikasi"])

with tab1:
    st.markdown("**Perbandingan Sentimen Berdasarkan Kelompok Instansi**")
    # Menghitung sentimen berdasarkan Kategori Instansi
    df_instansi = df.groupby(['Kategori_Instansi', 'Sentimen']).size().reset_index(name='Jumlah')
    
    # Bar Chart Grouped
    fig_instansi = px.bar(df_instansi, x='Jumlah', y='Kategori_Instansi', color='Sentimen', 
                          orientation='h', barmode='group',
                          color_discrete_map={'Positif':'#10b981', 'Negatif':'#ef4444'},
                          title="Total Kritik & Pujian per Sektor Instansi")
    fig_instansi.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_instansi, use_container_width=True)

with tab2:
    col_pie, col_bar = st.columns([1, 2])
    with col_pie:
        st.markdown("**Rasio Keseluruhan**")
        df_sentimen = df['Sentimen'].value_counts().reset_index()
        df_sentimen.columns = ['Sentimen', 'Jumlah']
        fig_pie = px.pie(df_sentimen, values='Jumlah', names='Sentimen', hole=0.5, 
                      color='Sentimen', color_discrete_map={'Positif':'#10b981', 'Negatif':'#ef4444'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.markdown("**Proporsi Sentimen Spesifik per Aplikasi**")
        df_app = df.groupby([kolom_app, 'Sentimen']).size().reset_index(name='Jumlah')
        # Bar Chart Stacked untuk melihat rasio per aplikasi
        fig_app = px.bar(df_app, x='Jumlah', y=kolom_app, color='Sentimen', orientation='h',
                      color_discrete_map={'Positif':'#10b981', 'Negatif':'#ef4444'})
        fig_app.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_app, use_container_width=True)

st.markdown("---")
with st.expander("Lihat Data Mentah Ulasan (Disensor sebagian untuk privasi)"):
    st.dataframe(df.head(100), use_container_width=True)
