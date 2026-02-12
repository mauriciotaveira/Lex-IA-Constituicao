import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURAÇÃO E ESTILO ---
st.set_page_config(page_title="Lex-IA 2.0 Pro", page_icon="⚖️", layout="wide")

# Inicializa o histórico na memória da sessão
if 'historico' not in st.session_state:
    st.session_state.historico = []

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .titulo-moderno {
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
    }
    .stButton>button {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: white; border: none; border-radius: 12px; font-weight: bold;
    }
    .card-historico {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #4facfe;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARREGAMENTO ---
@st.cache_data
def carregar_dados():
    try: return pd.read_excel("Constituicao_Mestra_V2.xlsx")
    except: return None

df = carregar_dados()

# --- 3. BARRA LATERAL (CONFIGS + HISTÓRICO) ---
with st.sidebar:
    st.markdown("### 🛠️ Lab de IA")
    api_key = st.text_input("Sua Gemini Key", type="password")
    top_k = st.slider("Profundidade", 1, 5, 3)
    
    st.divider()
    st.markdown("### 📜 Histórico de Consultas")
    if not st.session_state.historico:
        st.caption("Nenhuma consulta realizada ainda.")
    else:
        for idx, item in enumerate(reversed(st.session_state.historico)):
            with st.expander(f"🔍 {item['pergunta'][:30]}..."):
                st.write(item['resposta'])
                if st.button("Limpar este", key=f"del_{idx}"):
                    st.session_state.historico.pop(len(st.session_state.historico) - 1 - idx)
                    st.rerun()

# --- 4. INTERFACE PRINCIPAL ---
st.markdown('<p class="titulo-moderno">Lex-IA 2.0 Pro</p>', unsafe_allow_html=True)

if df is not None and api_key:
    genai.configure(api_key=api_key)
    try:
        modelos = [m.name for m in genai.list_models() if "gemini" in m.name.lower()]
        modelo_escolhido = st.selectbox("Motor da IA:", modelos)
        
        st.divider()
        pergunta = st.text_input("O que deseja decifrar na Constituição?")

        if st.button("Analisar Agora 🚀") and pergunta:
            # Busca RAG Turbo
            vectorizer = TfidfVectorizer(
                stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'as', 'no', 'na', 'artigo', 'parágrafo', 'inciso', 'constituição', 'regras', 'sobre'],
                max_df=0.2, ngram_range=(1, 2), sublinear_tf=True
            )
            tfidf_matrix = vectorizer.fit_transform(df['Conteúdo'].fillna(''))
            pergunta_vec = vectorizer.transform([pergunta])
            similares = cosine_similarity(pergunta_vec, tfidf_matrix).flatten()
            indices = similares.argsort()[-10:][::-1]
            contexto = "\n".join([f"Artigo: {df.iloc[i]['Conteúdo']}" for i in indices[:top_k]])

            with st.spinner('Elaborando parecer...'):
                model = genai.GenerativeModel(modelo_escolhido)
                prompt = f"Você é o Lex-IA 2.0, consultor jurídico sênior. Use tom cordial e executivo, bullet points e negrito. Contexto: {contexto}. Pergunta: {pergunta}"
                response = model.generate_content(prompt)
                
                # Salva no Histórico
                st.session_state.historico.append({"pergunta": pergunta, "resposta": response.text})
                
                # Exibe Resposta
                st.markdown("### 📝 Parecer Técnico:")
                st.write(response.text)
                
                # FUNÇÃO COPIAR (Usando componente de código para facilitar cópia nativa)
                st.info("💡 Você pode copiar o parecer abaixo clicando no ícone no canto superior direito:")
                st.code(response.text, language="text")

                with st.expander("🔗 Fontes Originais"):
                    for i in indices[:top_k]: st.caption(df.iloc[i]['Conteúdo'])
                    
    except Exception as e: st.error(f"Erro: {e}")
else:
    st.info("👋 Insira sua API Key para começar.")