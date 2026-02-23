import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from openai import OpenAI
import pickle
import jieba

st.set_page_config(page_title="HiSi & 维保 智能调度大脑", page_icon="✈️")
st.title("✈️ 企业级多路由业务大脑 (完全体)")
st.caption("已挂载：调度 Agent + 航显双引擎 + 维保向量库 + 连续记忆。")

# ================= 1. 挂载所有 U 盘与引擎 =================
@st.cache_resource
def load_all_engines():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # --- 抽屉 A：航显系统 (双引擎满配版) ---
    hisi_vector_db = Chroma(persist_directory="hisi_vdb", embedding_function=embedding_model)
    hisi_vector_retriever = hisi_vector_db.as_retriever(search_kwargs={"k": 3})
    
    with open("hisi_chunks.pkl", "rb") as f:
        hisi_chunks = pickle.load(f)
    def jieba_tokenizer(text): return list(jieba.cut(text))
    hisi_bm25_retriever = BM25Retriever.from_texts(hisi_chunks, preprocess_func=jieba_tokenizer)
    hisi_bm25_retriever.k = 3
    
    hisi_ensemble = EnsembleRetriever(
        retrievers=[hisi_bm25_retriever, hisi_vector_retriever],
        weights=[0.5, 0.5]
    )
    
    # --- 抽屉 B：车辆与桥载设备维保系统 (纯向量版) ---
    bridge_vector_db = Chroma(persist_directory="bridge_vdb", embedding_function=embedding_model)
    bridge_retriever = bridge_vector_db.as_retriever(search_kwargs={"k": 3})
    
    return hisi_ensemble, bridge_retriever

hisi_retriever, bridge_retriever = load_all_engines()

client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"], 
    base_url="https://api.deepseek.com"
)

# ================= 2. 网页记事本 =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= 3. 核心调度与检索流水线 =================
def classify_intent(question):
prompt = f"""你是一个极其聪明的机场业务总调度员。
    请判断下面这个问题，属于哪个业务领域：
    A: 航显系统、屏幕参数、软件功能 (HiSi-G.I.D.S)
    B: 车辆维修、登机桥、理赔、洗车、工时分配、桥载设备、施工管理、施工现场、维保
    
    你只能回答一个大写字母 'A' 或 'B'，绝对不要输出任何其他标点或废话。
    用户问题：{question}"""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

if prompt := st.chat_input("尝试跨界拷问（如：先问屏幕型号，再问洗车工时）..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # 💡 步骤 1：呼叫大堂调度员进行分发！
        intent = classify_intent(prompt)
        
        # 💡 步骤 2：根据分类去不同的抽屉拿资料，并在网页上实时播报！
        if "A" in intent:
            st.caption("🤖 *调度员思考中... -> 锁定为【航显业务】，已前往抽屉 A (启动双引擎检索)*")
            docs = hisi_retriever.invoke(prompt)
        elif "B" in intent:
            st.caption("🤖 *调度员思考中... -> 锁定为【维保业务】，已前往抽屉 B (启动专有知识库)*")
            docs = bridge_retriever.invoke(prompt)
        else:
            st.caption("🤖 *调度员遇到未知领域，默认前往抽屉 A 碰碰运气...*")
            docs = hisi_retriever.invoke(prompt)
            
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 💡 步骤 3：带着特定抽屉的资料，结合历史记忆，生成最终回答
        api_messages = [
            {"role": "system", "content": "你是极其严谨的业务AI专家。请结合我提供的【参考知识库】和【历史对话记录】来回答最新问题。如果知识库未提及，直接回答不知道。"}
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





