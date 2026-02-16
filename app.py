import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="Guia Cidadão", page_icon="⚖️", layout="wide")

# Inicialização de Variáveis
if 'historico' not in st.session_state: st.session_state.historico = []
if 'ultima_resposta' not in st.session_state: st.session_state.ultima_resposta = None
if 'primeiro_acesso' not in st.session_state: st.session_state.primeiro_acesso = True

# --- 2. CSS FINAL (SOLUÇÃO DE VISIBILIDADE) ---
st.markdown("""
    <style>
    /* Força fundo branco */
    .stApp { background-color: #ffffff !important; }

    /* --- BOTÃO (FUNDO PRETO, TEXTO BRANCO) --- */
    div.stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;
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
    div.stButton > button p { color: #ffffff !important; }

    /* --- TEXTOS GERAIS (PRETO) --- */
    h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li, label, div {
        color: #000000 !important;
    }

    /* --- INPUTS --- */
    .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        border: 1px solid #ccc !important;
    }
    ::placeholder { color: #888888 !important; font-style: italic !important; opacity: 1 !important; }

    /* --- ESTILO PERSONALIZADO --- */
    .titulo-cidadao {
        font-family: 'Helvetica', 'Arial', sans-serif;
        color: #000000 !important;
        font-size: 3.5rem; font-weight: 900; margin-bottom: 0px; line-height: 1.1;
    }
    .subtitulo-cidadao {
        color: #444444 !important; font-size: 1.5rem; font-weight: 600; margin-bottom: 25px;
    }
    .convite-pesquisa {
        background-color: #f0f2f6 !important; padding: 15px; border-radius: 8px;
        border-left: 6px solid #000; color: #333333 !important;
        font-size: 1.1rem; margin-bottom: 30px;
    }
    .resposta-box {
        background-color: #f8f9fa !important; border-left: 5px solid #000;
        padding: 20px; border-radius: 5px; margin-top: 15px; color: #000 !important;
    }
    
    @media (max-width: 768px) { .titulo-cidadao { font-size: 2.5rem !important; } }
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

# --- 4. SIDEBAR (LIMPA E AUTOMÁTICA) ---
with st.sidebar:
    st.header("⚖️ Status do Sistema")
    
    # Tenta pegar a chave automaticamente
    api_key = st.secrets.get("GOOGLE_API_KEY")
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("✅ IA Conectada e Pronta")
    else:
        st.error("🔴 Aguardando Chave de Acesso")
        st.info("Configure a 'GOOGLE_API_KEY' nos Secrets do Streamlit.")

    st.divider()
    st.header("📜 Histórico Recente")
    if st.button("Limpar Histórico"):
        st.session_state.historico = []
        st.session_state.ultima_resposta = None
        st.rerun()
        
    for i, (p, r) in enumerate(reversed(st.session_state.historico[-5:])):
        with st.expander(f"❓ {p[:25]}..."):
            st.caption(r[:100] + "...")

# --- 5. LÓGICA RAG (CALIBRADA PARA EXCELÊNCIA) ---
def buscar_resposta(pergunta):
    if df is None: return "⚠️ Erro: A Constituição (Excel) não foi carregada."
    if not api_key: return "⚠️ Erro: Chave de segurança não configurada."
    
    # 1. Recuperação (Retrieval)
    palavras = pergunta.lower().split()
    # Filtro simples para achar artigos relevantes
    mask = df.astype(str).apply(lambda x: x.str.lower()).apply(lambda x: any(p in x.values for p in palavras if len(p)>3), axis=1)
    
    # Pega até 15 artigos para dar bastante contexto
    contexto = df[mask].head(15).to_string()
    
    # 2. Geração (Generation) - PROMPT "PREMIUM"
    modelo = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    Atue como um Consultor Jurídico Sênior, especialista em Direito Constitucional Brasileiro.
    Seu objetivo é explicar a lei de forma didática, completa e acolhedora para um cidadão comum.
    
    Use EXCLUSIVAMENTE estes trechos da Constituição para embasar sua resposta:
    {contexto}
    
    Pergunta do Cidadão: {pergunta}
    
    Estrutura da Resposta:
    1. **Resumo Direto:** Responda a dúvida de forma clara em um parágrafo.
    2. **O que diz a Lei:** Cite os artigos ou incisos encontrados (use o contexto acima).
    3. **Explicação Descomplicada:** Traduza o "juridiquês" para uma linguagem do dia a dia.
    4. **Conclusão:** Finalize com uma orientação prática, se houver.
    
    Se o assunto não estiver nos trechos fornecidos, diga honestamente: "Não encontrei um artigo específico sobre isso nos trechos da Constituição que consultei agora, mas posso analisar outro tema."
    """
    
    try:
        # Temperatura 0.4: Equilíbrio perfeito entre precisão (lei) e fluidez (texto bom)
        res = modelo.generate_content(prompt, generation_config={'temperature': 0.4})
        return res.text
    except Exception as e:
        return f"Erro na IA: {e}"

# --- 6. INTERFACE PRINCIPAL ---
st.markdown('<div class="titulo-cidadao">Guia Cidadão</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo-cidadao">Constituição Descomplicada</div>', unsafe_allow_html=True)

st.markdown('''
<div class="convite-pesquisa">
    💡 <b>Dica:</b> A IA analisa a Constituição em tempo real. 
    Pergunte: "Tenho direito a férias?", "O que é liberdade de expressão?" ou "Quais são os direitos dos trabalhadores?".
</div>
''', unsafe_allow_html=True)

pgt = st.text_input("Sua dúvida:", placeholder="Digite aqui sua pergunta sobre seus direitos...", label_visibility="collapsed")

if st.button("Consultar Constituição"):
    if pgt:
        if not api_key:
            st.error("🔒 O sistema precisa da Chave de Acesso (API Key) para funcionar.")
        else:
            with st.spinner("Consultando a Constituição Federal..."):
                resposta = buscar_resposta(pgt)
                st.session_state.historico.append((pgt, resposta))
                st.session_state.ultima_resposta = resposta

if st.session_state.ultima_resposta:
    st.markdown(f'<div class="resposta-box">{st.session_state.ultima_resposta}</div>', unsafe_allow_html=True)
