import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Lex-IA 2.0 Pro", page_icon="⚖️", layout="wide")

# Inicialização do Histórico e Estados
if 'historico' not in st.session_state:
    st.session_state.historico = []
if 'ultima_resposta' not in st.session_state:
    st.session_state.ultima_resposta = None
if 'indices_fontes' not in st.session_state:
    st.session_state.indices_fontes = []

# --- 2. ESTILO CSS ---
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

# --- 3. CARREGAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    try:
        return pd.read_excel("Constituicao_Mestra_V2.xlsx")
    except Exception as e:
        st.error(f"Erro ao carregar Excel: {e}")
        return None

df = carregar_dados()

# --- 4. BARRA LATERAL ---
with st.sidebar:
    st.markdown("### 🛠️ Lab de IA")
    api_key = st.text_input("Sua Gemini Key", type="password")
    top_k = st.slider("Profundidade da Análise", 1, 5, 3)
    
    st.divider()
    st.markdown("### 📜 Histórico de Consultas")
    if not st.session_state.historico:
        st.caption("Nenhuma consulta realizada.")
    else:
        for item in reversed(st.session_state.historico):
            with st.expander(f"🔍 {item['pergunta'][:30]}..."):
                st.write(item['resposta'])

# --- 5. INTERFACE PRINCIPAL ---
st.markdown('<p class="titulo-moderno">Lex-IA 2.0 Pro</p>', unsafe_allow_html=True)

if df is not None and api_key:
    genai.configure(api_key=api_key)
    
    try:
        # Seleção do Modelo
        modelos = [m.name for m in genai.list_models() if "gemini" in m.name.lower()]
        modelo_escolhido = st.selectbox("Escolha o motor da IA:", modelos)
        
        st.divider()
        pergunta = st.text_input("O que você quer decifrar na Constituição hoje?")

        # BLOCO DE PROCESSAMENTO
        if st.button("Analisar Agora 🚀") and pergunta:
            with st.spinner('O Lex-IA está elaborando o parecer técnico...'):
                # Motor de Busca RAG Turbo
                vectorizer = TfidfVectorizer(
                    stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'as', 'no', 'na', 'artigo', 'parágrafo', 'inciso', 'constituição', 'regras', 'sobre'],
                    max_df=0.2, ngram_range=(1, 2), sublinear_tf=True
                )
                
                content_col = df['Conteúdo'].fillna('')
                tfidf_matrix = vectorizer.fit_transform(content_col)
                pergunta_vec = vectorizer.transform([pergunta])
                similares = cosine_similarity(pergunta_vec, tfidf_matrix).flatten()
                
                indices = similares.argsort()[-10:][::-1]
                contexto = "\n".join([f"Artigo: {df.iloc[i]['Conteúdo']}" for i in indices[:top_k]])

                # Geração com Gemini
                model = genai.GenerativeModel(modelo_escolhido)
                prompt = f"Você é o Lex-IA 2.0, consultor jurídico sênior. Use tom cordial, executivo e bullet points. Contexto: {contexto}. Pergunta: {pergunta}"
                response = model.generate_content(prompt)
                
                # Salva os dados na sessão e reinicia para limpar a tela
                st.session_state.ultima_resposta = response.text
                st.session_state.indices_fontes = indices[:top_k]
                st.session_state.historico.append({"pergunta": pergunta, "resposta": response.text})
                st.rerun()

        # BLOCO DE EXIBIÇÃO ÚNICA (Fora do botão Analisar)
        if st.session_state.ultima_resposta:
            st.divider()
            st.markdown("### 📝 Parecer Técnico")
            
            # Guia de Cópia
            st.info("💡 **Dica:** Para copiar o parecer, use o botão no canto superior direito da caixa abaixo.")
            
            # Caixa de Código estável com botão de cópia
            st.code(st.session_state.ultima_resposta, language="text")
            
            # Fontes Originais
            st.divider()
            with st.expander("🔗 Ver Fontes Originais"):
                for i in st.session_state.indices_fontes:
                    st.caption(df.iloc[i]['Conteúdo'])

    except Exception as e:
        st.error(f"Houve um problema técnico: {e}")

else:
    st.info("👋 Olá! Por favor, insira sua API Key na barra lateral para começar.")

# --- 6. RODAPÉ (SEMPRE VISÍVEL) ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 20px;'>
        Desenvolvido por <b>Maurício Taveira</b> | 2026 <br>
        <span style='color: #4facfe;'>Lex-IA 2.0 Pro</span> - Inteligência Artificial aplicada ao Direito
    </div>
    """,
    unsafe_allow_html=True
)