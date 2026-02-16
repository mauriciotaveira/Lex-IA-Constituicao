import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="Guia Cidadão", page_icon="⚖️", layout="wide")

if 'historico' not in st.session_state: st.session_state.historico = []
if 'ultima_resposta' not in st.session_state: st.session_state.ultima_resposta = None
if 'primeiro_acesso' not in st.session_state: st.session_state.primeiro_acesso = True

# --- 2. CSS FINAL (SOLUÇÃO DE VISIBILIDADE) ---
st.markdown("""
    <style>
    /* Força fundo branco */
    .stApp {
        background-color: #ffffff !important;
    }

    /* --- BOTÃO (FUNDO PRETO, TEXTO BRANCO) --- */
    div.stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;   /* Texto Branco */
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        height: 50px !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    div.stButton > button:active {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    div.stButton > button p {
        color: #ffffff !important; /* Garante texto interno branco */
    }

    /* --- TEXTOS GERAIS (PRETO) --- */
    h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li, label, div {
        color: #000000 !important;
    }

    /* --- INPUTS (ONDE DIGITA) --- */
    .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        border: 1px solid #ccc !important;
    }

    /* --- SUGESTÃO (PLACEHOLDER) --- */
    ::placeholder {
        color: #888888 !important;
        font-style: italic !important;
        opacity: 1 !important;
    }

    /* --- ESTILO PERSONALIZADO DO PROJETO --- */
    .titulo-cidadao {
        font-family: 'Helvetica', 'Arial', sans-serif;
        color: #000000 !important;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0px;
        line-height: 1.1;
    }
    
    .subtitulo-cidadao {
        color: #444444 !important;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 25px;
    }

    .convite-pesquisa {
        background-color: #f0f2f6 !important;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #000;
        color: #333333 !important;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Responsivo */
    @media (max-width: 768px) { 
        .titulo-cidadao { font-size: 2.5rem !important; } 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DADOS ---
@st.cache_data
def carregar_constitucional():
    arquivos = [f for f in os.listdir() if f.endswith('.xlsx')]
    if not arquivos: return None
    try:
        df = pd.read_excel(arquivos[0])
        return df.fillna("")
    except: return None

df = carregar_constitucional()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Configurações")
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("🔒 IA Conectada")
    else:
        st.error("🔑 Sem Chave API")

    temperatura = st.slider("Criatividade da Resposta", 0.0, 1.0, 0.3)
    
    st.divider()
    st.header("📜 Histórico")
    if st.button("Limpar Histórico"):
        st.session_state.historico = []
        st.session_state.ultima_resposta = None
        st.rerun()
        
    for i, (p, r) in enumerate(reversed(st.session_state.historico[-5:])):
        with st.expander(f"Q: {p[:30]}..."):
            st.write(r)

# --- 5. BOAS-VINDAS ---
if st.session_state.primeiro_acesso:
    st.balloons()
    st.toast("Bem-vindo ao Guia Cidadão!", icon="🇧🇷")
    st.session_state.primeiro_acesso = False

# --- 6. LÓGICA RAG ---
def buscar_resposta(pergunta):
    if df is None: return "Erro: Base de dados não carregada."
    if not api_key: return "Erro: Configure a API Key."
    
    # Retrieval
    palavras = pergunta.lower().split()
    mask = df.astype(str).apply(lambda x: x.str.lower()).apply(lambda x: any(p in x.values for p in palavras if len(p)>3), axis=1)
    contexto = df[mask].head(10).to_string()
    
    # Generation
    modelo = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    Você é um advogado constitucionalista especialista em linguagem simples.
    Use EXCLUSIVAMENTE estes trechos da Constituição para responder:
    {contexto}
    Pergunta do Cidadão: {pergunta}
    Regras:
    1. Cite o artigo/inciso.
    2. Explique de forma que qualquer pessoa entenda.
    3. Se não estiver no contexto, diga que a Constituição não cita isso explicitamente nesses trechos.
    """
    
    try:
        res = modelo.generate_content(prompt, generation_config={'temperature': temperatura})
        return res.text
    except Exception as e:
        return f"Erro na IA: {e}"

# --- 7. INTERFACE PRINCIPAL ---
st.markdown('<div class="titulo-cidadao">Guia Cidadão</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo-cidadao">Constituição Descomplicada</div>', unsafe_allow_html=True)

st.markdown('<div class="convite-pesquisa">💡 <b>Dica:</b> Pergunte coisas como "tenho direito a férias?" ou "o que é liberdade de expressão?".</div>', unsafe_allow_html=True)

st.markdown("<small style='color:#666; font-style:italic;'>Digite sua dúvida abaixo:</small>", unsafe_allow_html=True)
pgt = st.text_input("Pergunta", placeholder="Ex: O que é Habeas Corpus?", label_visibility="collapsed")

if st.button("Consultar Constituição"):
    if pgt:
        with st.spinner("Consultando a lei..."):
            resposta = buscar_resposta(pgt)
            st.session_state.historico.append((pgt, resposta))
            st.session_state.ultima_resposta = resposta

if st.session_state.ultima_resposta:
    st.markdown("### 🏛️ Análise:")
    st.markdown(st.session_state.ultima_resposta)
