import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURAÇÃO E IDENTIDADE ---
st.set_page_config(page_title="Lex-IA 2.0 Pro", page_icon="⚖️", layout="wide")

# Inicialização de Estados de Sessão
if 'historico' not in st.session_state: st.session_state.historico = []
if 'ultima_resposta' not in st.session_state: st.session_state.ultima_resposta = None
if 'primeiro_acesso' not in st.session_state: st.session_state.primeiro_acesso = True

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
    /* Estilo para o Parecer para garantir leitura fluida */
    .parecer-texto { font-size: 1.1rem; line-height: 1.6; color: #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SEGURANÇA (SECRETS) ---
# O sistema busca a chave nas configurações do Streamlit Cloud
api_key = st.secrets.get("GEMINI_API_KEY")

# --- 3. CARREGAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    try: return pd.read_excel("Constituicao_Mestra_V2.xlsx")
    except: return None

df = carregar_dados()

# --- 4. BOAS-VINDAS (SÓ NO PRIMEIRO ACESSO) ---
if st.session_state.primeiro_acesso:
    st.balloons()
    st.toast("Bem-vindo, Maurício! Lex-IA 2.0 Pro pronto para o serviço.", icon="⚖️")
    st.session_state.primeiro_acesso = False

# --- 5. BARRA LATERAL (LAB E HISTÓRICO) ---
with st.sidebar:
    st.markdown("### 🛠️ Lab de IA")
    if not api_key:
        api_key = st.text_input("Insira sua Gemini Key", type="password")
        st.warning("⚠️ Chave manual. Para automação, use os 'Secrets'.")
    else:
        st.success("🔒 Conexão Segura Ativa")
    
    top_k = st.slider("Profundidade da Análise", 1, 5, 3)
    st.divider()
    st.markdown("### 📜 Histórico")
    for item in reversed(st.session_state.historico):
        with st.expander(f"🔍 {item['pergunta'][:20]}..."):
            st.write(item['resposta'])

# --- 6. INTERFACE PRINCIPAL ---
st.markdown('<p class="titulo-moderno">Lex-IA 2.0 Pro</p>', unsafe_allow_html=True)

if df is not None and api_key:
    genai.configure(api_key=api_key)
    try:
        modelos = [m.name for m in genai.list_models() if "gemini" in m.name.lower()]
        modelo_escolhido = st.selectbox("Escolha o motor da IA:", modelos)
        st.divider()
        pergunta = st.text_input("O que você quer decifrar na Constituição hoje?")

        if st.button("Analisar Agora 🚀") and pergunta:
            with st.spinner('O Lex-IA está elaborando o parecer técnico...'):
                # Motor de Busca RAG
                vectorizer = TfidfVectorizer(
                    stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'as', 'no', 'na', 'artigo', 'parágrafo', 'inciso'],
                    max_df=0.2, ngram_range=(1, 2), sublinear_tf=True
                )
                tfidf_matrix = vectorizer.fit_transform(df['Conteúdo'].fillna(''))
                pergunta_vec = vectorizer.transform([pergunta])
                similares = cosine_similarity(pergunta_vec, tfidf_matrix).flatten()
                indices = similares.argsort()[-10:][::-1]
                contexto = "\n".join([f"Artigo: {df.iloc[i]['Conteúdo']}" for i in indices[:top_k]])

                # IA Multilingue e Executiva
                model = genai.GenerativeModel(modelo_escolhido)
                prompt = (
                    f"Você é o Lex-IA 2.0, consultor jurídico sênior. Responda obrigatoriamente "
                    f"no MESMO IDIOMA da pergunta do usuário. Use tom executivo e cordial. "
                    f"Contexto: {contexto}. Pergunta: {pergunta}"
                )
                response = model.generate_content(prompt)
                
                # Salva e Reinicia
                st.session_state.ultima_resposta = response.text
                st.session_state.indices_fontes = indices[:top_k]
                st.session_state.historico.append({"pergunta": pergunta, "resposta": response.text})
                st.rerun()

        # --- ÁREA DE EXIBIÇÃO ÚNICA E FORMATADA ---
        if st.session_state.ultima_resposta:
            st.divider()
            st.markdown("### 📝 Parecer Técnico")
            
            # 1. TEXTO PARA LEITURA (Markdown faz quebra de linha automática)
            st.markdown(f'<div class="parecer-texto">{st.session_state.ultima_resposta}</div>', unsafe_allow_html=True)
            
            # 2. ÁREA DE CÓPIA (Dentro de um expander para não atrapalhar a leitura)
            with st.expander("📋 Clique aqui para copiar o texto formatado"):
                st.code(st.session_state.ultima_resposta, language="text")
            
            st.divider()
            with st.expander("🔗 Ver Fontes Originais"):
                for i in st.session_state.indices_fontes:
                    st.caption(df.iloc[i]['Conteúdo'])

    except Exception as e:
        st.error(f"Erro: {e}")
else:
    st.info("👋 Olá! Insira sua API Key para começar.")

# --- 7. RODAPÉ ---
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