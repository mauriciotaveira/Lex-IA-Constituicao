import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="Guia Cidadão. Constituição descomplicada", page_icon="⚖️", layout="wide")

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
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTÃO DE SEGURANÇA (HÍBRIDA) ---
# Tenta capturar a chave automaticamente do cofre do servidor (Secrets)
api_key = st.secrets.get("GEMINI_API_KEY")

# Se a chave não for encontrada no cofre, oferece o campo manual na barra lateral
if not api_key:
    with st.sidebar:
        st.markdown("### 🔑 Acesso")
        api_key = st.text_input("Insira sua Gemini Key:", type="password")
        if not api_key:
            st.warning("⚠️ Chave de API não detectada. Por favor, insira uma para operar o sistema.")
            st.stop() # Interrompe a execução até que a chave seja inserida

# --- 3. DADOS ---
@st.cache_data
def carregar_dados():
    try: return pd.read_excel("Constituicao_Mestra_V2.xlsx")
    except: return None

df = carregar_dados()

# --- 4. BOAS-VINDAS ---
if st.session_state.primeiro_acesso:
    st.balloons()
    st.toast("Habite-se concedido! Guia Cidadão. Constituição descomplicada.", icon="🚀")
    st.session_state.primeiro_acesso = False

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### 🛠️ Lab de IA")
    if not api_key:
        api_key = st.text_input("Insira sua Gemini Key", type="password")
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
                vectorizer = TfidfVectorizer(
                    stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'as', 'no', 'na', 'artigo', 'parágrafo', 'inciso'],
                    max_df=0.2, ngram_range=(1, 2), sublinear_tf=True
                )
                tfidf_matrix = vectorizer.fit_transform(df['Conteúdo'].fillna(''))
                pergunta_vec = vectorizer.transform([pergunta])
                similares = cosine_similarity(pergunta_vec, tfidf_matrix).flatten()
                indices = similares.argsort()[-10:][::-1]
                contexto = "\n".join([f"Artigo: {df.iloc[i]['Conteúdo']}" for i in indices[:top_k]])

                model = genai.GenerativeModel(modelo_escolhido)
                prompt = (
                    f"Você é o Lex-IA 2.0, consultor jurídico sênior. Responda no MESMO IDIOMA da pergunta. "
                    f"Use tom executivo e cordial. Use negrito para dar ênfase. Contexto: {contexto}. Pergunta: {pergunta}"
                )
                response = model.generate_content(prompt)
                
                st.session_state.ultima_resposta = response.text
                st.session_state.indices_fontes = indices[:top_k]
                st.session_state.historico.append({"pergunta": pergunta, "resposta": response.text})
                st.rerun()

        # --- EXIBIÇÃO ORGANIZADA (VENCENDO A INVISIBILIDADE) ---
        if st.session_state.ultima_resposta:
            st.divider()
            st.markdown("### 📝 Parecer Técnico")
            
            # Exibição principal em Markdown Puro (Contraste e Quebra de Linha Automática)
            st.markdown(st.session_state.ultima_resposta)
            
            # Ferramenta de Cópia isolada (para evitar scroll horizontal na leitura)
            with st.expander("📋 Clique aqui para copiar o texto"):
                st.code(st.session_state.ultima_resposta, language="text")
            
            st.divider()
            with st.expander("🔗 Ver Fontes Originais"):
                for i in st.session_state.indices_fontes:
                    st.caption(df.iloc[i]['Conteúdo'])

    except Exception as e:
        st.error(f"Erro: {e}")
else:
    st.info("👋 Olá! Insira sua API Key para começar.")

# --- 7. RODAPÉ COM AVISO LEGAL ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

# Aviso de Isenção de Responsabilidade (Disclaimer)
st.markdown(
    """
    <div style='text-align: center; color: #555; font-size: 0.8rem; padding: 0 20px; font-style: italic;'>
        <b>Aviso Legal:</b> O Lex-IA 2.0 Pro é uma ferramenta de apoio técnico baseada em Inteligência Artificial. 
        Suas respostas têm caráter estritamente informativo e não constituem consulta jurídica, 
        parecer vinculante ou orientação legal oficial. Esta ferramenta não substitui, em hipótese alguma, 
        a análise e o aconselhamento de um advogado devidamente inscrito na OAB.
    </div>
    """,
    unsafe_allow_html=True
)

# Sua Assinatura
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9rem; padding: 10px 20px;'>
        Desenvolvido por <b>Maurício Taveira</b> | 2026 <br>
        <span style='color: #4facfe;'>Lex-IA 2.0 Pro</span> - Inteligência Artificial aplicada ao Direito
    </div>
    """,
    unsafe_allow_html=True
)
