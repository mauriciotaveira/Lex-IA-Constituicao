import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import unicodedata

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

# --- 3. DADOS E FUNÇÕES AUXILIARES ---
@st.cache_data
def carregar_constitucional():
    arquivos = [f for f in os.listdir() if f.endswith('.xlsx')]
    if not arquivos: return None
    try:
        df = pd.read_excel(arquivos[0])
        # Garante que tudo vire texto para a busca não falhar
        return df.fillna("").astype(str)
    except: return None

def normalizar_texto(texto):
    # Remove acentos e deixa minúsculo (para busca funcionar: é = e)
    if not isinstance(texto, str): return str(texto).lower()
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower()

df = carregar_constitucional()

# --- 4. SIDEBAR (STATUS) ---
with st.sidebar:
    st.header("⚖️ Status do Sistema")
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

# --- 5. LÓGICA RAG INTELIGENTE (RANKING) ---
def buscar_resposta(pergunta):
    if df is None: return "⚠️ Erro: A Constituição (Excel) não foi carregada."
    if not api_key: return "⚠️ Erro: Chave de segurança não configurada."
    
    # 1. Recuperação Inteligente (Smart Retrieval)
    
    # Lista de palavras para IGNORAR (Stopwords)
    ignorar = ['quais', 'sao', 'os', 'as', 'de', 'do', 'da', 'em', 'que', 'para', 'com', 'conforme', 'constituição', 'artigo', 'lei']
    
    pergunta_norm = normalizar_texto(pergunta)
    palavras_chave = [p for p in pergunta_norm.split() if p not in ignorar and len(p) > 2]
    
    if not palavras_chave:
        # Se a pessoa digitou só palavras comuns, usa tudo
        palavras_chave = pergunta_norm.split()

    # Função de Pontuação (Ranking)
    # Conta quantas palavras-chave aparecem em cada linha do Excel
    def pontuar_linha(linha):
        texto_linha = normalizar_texto(str(linha))
        pontos = 0
        for p in palavras_chave:
            if p in texto_linha:
                pontos += 1
        return pontos

    # Aplica a pontuação em todas as linhas
    df['score'] = df.apply(lambda row: pontuar_linha(row.values), axis=1)
    
    # Pega as 10 melhores linhas (maior pontuação)
    # Se ninguém pontuar, pega as 5 primeiras por garantia
    melhores = df[df['score'] > 0].sort_values('score', ascending=False).head(15)
    
    if melhores.empty:
        contexto = df.head(5).to_string() # Fallback
    else:
        contexto = melhores.to_string()
    
    # 2. Geração (Generation) - PROMPT PREMIUM
    modelo = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    Atue como um Consultor Jurídico Sênior, especialista em Direito Constitucional Brasileiro.
    Seu objetivo é explicar a lei de forma didática, completa e acolhedora.
    
    Use EXCLUSIVAMENTE estes trechos da Constituição para embasar sua resposta:
    {contexto}
    
    Pergunta do Cidadão: {pergunta}
    
    Estrutura da Resposta:
    1. **Resumo Direto:** Responda a dúvida de forma clara em um parágrafo.
    2. **O que diz a Lei:** Cite o Artigo/Inciso exato (Ex: Art. 5º, Art. 6º) encontrado no contexto.
    3. **Explicação Descomplicada:** Traduza o termo jurídico para linguagem simples.
    4. **Conclusão:** Finalize com uma orientação prática.
    
    Se não encontrar a resposta no contexto, diga: "Não encontrei esse tema específico nos trechos analisados."
    """
    
    try:
        # Temperatura 0.3 para ser mais fiel ao texto da lei
        res = modelo.generate_content(prompt, generation_config={'temperature': 0.3})
        return res.text
    except Exception as e:
        return f"Erro na IA: {e}"

# --- 6. INTERFACE PRINCIPAL ---
st.markdown('<div class="titulo-cidadao">Guia Cidadão</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo-cidadao">Constituição Descomplicada</div>', unsafe_allow_html=True)

st.markdown('''
<div class="convite-pesquisa">
    💡 <b>Dica:</b> A IA analisa a Constituição em tempo real. 
    Pergunte: "Tenho direito a férias?", "O que é liberdade de expressão?" ou "Quais são os direitos sociais?".
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
