import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from openai import OpenAI
import pickle
import jieba

st.set_page_config(page_title="HiSi 智能业务大脑", page_icon="✈️")
st.title("✈️ HiSi-G.I.D.S. V2.0 (双引擎记忆版)")
st.caption("已挂载：语义向量引擎 + BM25 精确匹配引擎。随便拷问极度生僻的设备型号！")

# ================= 1. 挂载双引擎 =================
@st.cache_resource
def load_dual_engines():
    print("⏳ 正在启动文科生：语义向量引擎...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vector_db = Chroma(persist_directory="hisi_vdb", embedding_function=embedding_model)
    # 包装成检索器
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    print("⏳ 正在启动理科生：BM25 精确匹配引擎...")
    # 把之前抽真空的字典解冻拿出来
    with open("hisi_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
        
    # 💡 核心：教 BM25 怎么切分中文词语
    def jieba_tokenizer(text):
        return list(jieba.cut(text))
        
    bm25_retriever = BM25Retriever.from_texts(chunks, preprocess_func=jieba_tokenizer)
    bm25_retriever.k = 3
    
    print("🤝 正在融合：大堂经理就位...")
    # 权重各占 50%，你可以根据业务需要随时调配，比如精确匹配要求高，可以改成 [0.7, 0.3]
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

# 获取集成检索器
ensemble_retriever = load_dual_engines()

client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"], 
    base_url="https://api.deepseek.com"
)

# ================= 2. 网页记事本 (保持不变) =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= 3. 带记忆的交互流水线 =================
if prompt := st.chat_input("请输入您的问题（试着搜一个极其精确的型号）..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # 💡 核心变化：以前是 db.similarity_search，现在直接呼叫大堂经理！
        docs = ensemble_retriever.invoke(prompt)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        api_messages = [
            {"role": "system", "content": "你是极其严谨的机场业务AI专家。请结合我提供的【参考知识库】和【历史对话记录】来回答最新问题。如果知识库未提及，直接回答不知道。"}
        ]
        
        for msg in st.session_state.messages[:-1]:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
            
        latest_prompt_with_context = f"【参考知识库】\n{context}\n\n【最新问题】\n{prompt}"
        api_messages.append({"role": "user", "content": latest_prompt_with_context})
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages,
            stream=False
        )
        
        answer = response.choices[0].message.content
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})



