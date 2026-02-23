import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import os

st.set_page_config(page_title="HiSi 智能业务大脑", page_icon="✈️")
st.title("✈️ HiSi-G.I.D.S. V2.0 业务大脑 (带连续记忆版)")
st.caption("现在你可以尽情追问了！试试问完参数后，直接问：‘那它的另一个型号呢？’")

# ================= 1. 挂载物理 U 盘 =================
@st.cache_resource
def load_knowledge_base():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    db = Chroma(persist_directory="hisi_vdb", embedding_function=embedding_model)
    return db

db = load_knowledge_base()
client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"], 
    base_url="https://api.deepseek.com"
)

# ================= 2. 激活网页记事本 (Session State) =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 把历史记忆渲染到屏幕上 (只显示纯净的对话，不显示后台偷偷加的知识库)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= 3. 核心：带记忆的交互流水线 =================
if prompt := st.chat_input("请输入您的问题（支持连续追问）..."):
    
    # A. 屏幕上显示用户的原始问题，并存入前端记忆
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # B. 拿当前问题去 U 盘搜知识 
        # (注：这里目前是初级检索，未来进阶可加入"Query Rewrite"问题重写技术)
        docs = db.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # C. 组装发给大模型的【终极记忆包裹】
        api_messages = [
            {"role": "system", "content": "你是极其严谨的机场业务AI专家。请结合我提供的【参考知识库】和【历史对话记录】来回答最新问题。如果知识库未提及，直接回答不知道。"}
        ]
        
        # 核心魔法 1：把之前的聊天记录全部按顺序装进去（不包括刚说的最后一句）
        for msg in st.session_state.messages[:-1]:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
            
        # 核心魔法 2：在最后一句最新问题里，强行注入刚刚查到的知识库！
        latest_prompt_with_context = f"【参考知识库】\n{context}\n\n【最新问题】\n{prompt}"
        api_messages.append({"role": "user", "content": latest_prompt_with_context})
        
        # D. 呼叫大模型进行推理
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages,
            stream=False
        )
        
        answer = response.choices[0].message.content
        st.markdown(answer)
        
        # E. 把大模型的回答存入前端记忆，形成闭环
        st.session_state.messages.append({"role": "assistant", "content": answer})
