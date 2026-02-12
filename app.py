import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURAÇÃO E ESTILO ---
st.set_page_config(page_title="Lex-IA 2.0 Pro", page_icon="⚖️", layout="wide")

# Inicializa estados de sessão
if 'historico' not in st.session_state: st.session_state.historico = []
if 'ultima_resposta' not in st.session_state: st.session_state.ultima_resposta = None

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .titulo-moderno {
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem; font-weight: 800;
    }
    .stButton>button {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: white; border: none; border-radius: 12px; font-weight: bold; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTOR DE DADOS ---
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
    st.markdown("### 📜 Histórico")
    for item in reversed(st.session_state.historico):
        with st.expander(f"🔍 {item['pergunta'][:20]}..."):
            st.write(item['resposta'])

# --- 4. INTERFACE PRINCIPAL ---
st.markdown('<p class="titulo-moderno">Lex-IA 2.0 Pro</p>', unsafe_allow_html=True)

if df is not None and api_key:
    genai.configure(api_key=api_key)
    modelos = [m.name for m in genai.list_models() if "gemini" in m.name.lower()]
    modelo_escolhido = st.selectbox("Motor da IA:", modelos)
    
    pergunta = st.text_input("O que deseja decifrar na Constituição?")

    # --- LÓGICA DE PROCESSAMENTO (NÃO EXIBE NADA AQUI) ---
    if st.button("Analisar Agora 🚀") and pergunta:
        with st.spinner('O Lex-IA está elaborando o parecer...'):
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

            model = genai.GenerativeModel(modelo_escolhido)
            prompt = f"Você é o Lex-IA 2.0, consultor sênior. Responda de forma executiva, polida e em tópicos. Contexto: {contexto}. Pergunta: {pergunta}"
            response = model.generate_content(prompt)
            
            # SALVAMENTO SILENCIOSO
            st.session_state.ultima_resposta = response.text
            st.session_state.ultimos_indices = indices[:top_k]
            st.session_state.historico.append({"pergunta": pergunta, "resposta": response.text})
            st.rerun() # Força o app a limpar a tela e mostrar o resultado novo

    # --- ÁREA ÚNICA DE EXIBIÇÃO (ESTÁVEL E SEM REPETIÇÃO) ---
    if st.session_state.get('ultima_resposta'):
        st.divider()
        st.markdown("### 📝 Parecer Técnico")
        
        # Orientação visual: retira o "mistério" de onde está o botão
        st.info("💡 **Dica:** Para copiar o parecer, use o botão que aparece no canto superior direito da caixa cinza abaixo.")
        
        # Exibição segura e estável. O botão 'Copy' aparece nativamente aqui.
        st.code(st.session_state.ultima_resposta, language="text")
        
        st.divider()
        with st.expander("🔗 Ver Fontes Originais"):
            indices_para_exibir = st.session_state.get('ultimos_indices', [])
            for i in indices_para_exibir:
                st.caption(df.iloc[i]['Conteúdo'])
else:
    st.info("👋 Insira sua API Key para começar.")