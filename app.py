import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Lex-IA 2.0", page_icon="⚖️", layout="wide")

# --- 2. ESTILO CSS (O BANHO DE LOJA) ---
st.markdown("""
    <style>
    /* Fundo e Fonte */
    .main { background-color: #0e1117; color: #ffffff; }
    
    /* Título com Degradê */
    .titulo-moderno {
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    /* Botão estilizado */
    .stButton>button {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 15px 32px;
        font-weight: bold;
        border-radius: 12px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 5px 15px rgba(79, 172, 254, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CARREGAMENTO E MOTOR ---
@st.cache_data
def carregar_dados():
    try:
        return pd.read_excel("Constituicao_Mestra_V2.xlsx")
    except: return None

df = carregar_dados()

# --- 4. BARRA LATERAL ---
with st.sidebar:
    st.markdown("### 🛠️ Lab de IA")
    api_key = st.text_input("Sua Gemini Key", type="password")
    top_k = st.slider("Profundidade da análise", 1, 5, 3)
    st.divider()
    st.write("🤖 **Versão:** 2.5 Flash Ativa")

# --- 5. INTERFACE ---
st.markdown('<p class="titulo-moderno">Lex-IA 2.0</p>', unsafe_allow_html=True)
st.markdown("#### Seu Consultor Jurídico Ágil e Inteligente")

if df is not None and api_key:
    # Diagnóstico de Modelos
    try:
        genai.configure(api_key=api_key)
        modelos = [m.name for m in genai.list_models() if "gemini" in m.name.lower()]
        modelo_escolhido = st.selectbox("Escolha o motor da IA:", modelos)
        
        st.divider()
        pergunta = st.text_input("O que você quer decifrar na Constituição hoje?", placeholder="Ex: Direitos trabalhistas de forma resumida...")

        if st.button("Analisar Agora 🚀") and pergunta:
            # Busca RAG
            vectorizer = TfidfVectorizer(max_df=0.4, min_df=2)
            tfidf_matrix = vectorizer.fit_transform(df['Conteúdo'].fillna(''))
            pergunta_vec = vectorizer.transform([pergunta])
            similares = cosine_similarity(pergunta_vec, tfidf_matrix).flatten()
            indices = similares.argsort()[-top_k:][::-1]
            contexto = "\n".join([f"Artigo: {df.iloc[i]['Conteúdo']}" for i in indices])
            
            # IA - PROMPT MODERNO (PERSONALIDADE)
            with st.spinner('Traduzindo o "juridiquês" para você...'):
                model = genai.GenerativeModel(modelo_escolhido)
                
                prompt_moderno = f"""
                Você é o Lex-IA 2.0, um consultor jurídico moderno, direto e amigável.
                Sua missão é explicar a Constituição sem ser chato ou "careta".

                DIRETRIZES:
                1. NÃO use "Prezado", "Outrossim", "Insculpida" ou saudações formais.
                2. Responda em tópicos (bullet points) para ser rápido de ler.
                3. Use **negrito** para destacar os pontos cruciais.
                4. Use emojis moderadamente para tornar o texto leve.
                5. Foque no que o usuário REALMENTE quer saber.

                CONTEXTO CONSTITUCIONAL:
                {contexto}

                PERGUNTA:
                {pergunta}
                """
                
                response = model.generate_content(prompt_moderno)
                
                st.markdown("### 📝 O que eu encontrei:")
                st.markdown(response.text)
                
                with st.expander("🔗 Ver fontes originais"):
                    for i in indices:
                        st.caption(df.iloc[i]['Conteúdo'])
                        
    except Exception as e:
        st.error(f"Erro na conexão: {e}")
else:
    st.info("👋 Olá! Insira sua API Key na esquerda para começarmos a consulta.")