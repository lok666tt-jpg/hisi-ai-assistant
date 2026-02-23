import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from openai import OpenAI
import pickle
import jieba

st.set_page_config(page_title="小徐の业务调度大脑", page_icon="✈️")
st.title("✈️ 小徐の业务调度大脑")
st.caption("全领域挂载：汇集我目前阶段所有整理和参与的项目资料！")

@st.cache_resource
def load_all_engines():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # --- 抽屉 A：机场项目 (airport_vdb 双引擎) ---
    airport_vector_db = Chroma(persist_directory="airport_vdb", embedding_function=embedding_model)
    airport_vector_retriever = airport_vector_db.as_retriever(search_kwargs={"k": 3})
    
    with open("airport_chunks.pkl", "rb") as f: airport_chunks = pickle.load(f)
    def jieba_tokenizer(text): return list(jieba.cut(text))
    airport_bm25_retriever = BM25Retriever.from_texts(airport_chunks, preprocess_func=jieba_tokenizer)
    airport_bm25_retriever.k = 3
    
    airport_ensemble = EnsembleRetriever(retrievers=[airport_bm25_retriever, airport_vector_retriever], weights=[0.5, 0.5])
    
    # --- 抽屉 B：地服维修项目 (ground_vdb 纯向量引擎) ---
    ground_vector_db = Chroma(persist_directory="ground_vdb", embedding_function=embedding_model)
    ground_retriever = ground_vector_db.as_retriever(search_kwargs={"k": 3})
    
    return airport_ensemble, ground_retriever

airport_retriever, ground_retriever = load_all_engines()
client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

def classify_intent(question):
    # 💡 极其硬核的 Prompt：彻底分开航显和维修开单！
    prompt = f"""你是一个极其聪明的机场业务总调度员。
    请判断下面这个问题，属于哪个业务领域：
    A: 机场项目、航显系统 (HiSi-G.I.D.S)、综合显示系统、屏幕参数、接口规范
    B: 地服公司项目、车辆维修、维修开单、理赔、登机桥维保、施工管理、工时分配
    
    你只能回答一个大写字母 'A' 或 'B'，绝对不要输出任何其他标点或废话。
    用户问题：{question}"""
    
    response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
    return response.choices[0].message.content.strip()

if prompt := st.chat_input("跨界拷问（例：先问航显参数，再问维修开单工时）..."):
    with st.chat_message("user"): st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        intent = classify_intent(prompt)
        
        if "A" in intent:
            st.caption("🤖 *调度员 -> 锁定为【机场项目/航显】，启动抽屉 A 双引擎检索...*")
            docs = airport_retriever.invoke(prompt)
        elif "B" in intent:
            st.caption("🤖 *调度员 -> 锁定为【地服项目/维保】，启动抽屉 B 专属检索...*")
            docs = ground_retriever.invoke(prompt)
        else:
            st.caption("🤖 *调度员遇到未知领域，默认去抽屉 A 碰运气...*")
            docs = airport_retriever.invoke(prompt)
            
        context = "\n\n".join([doc.page_content for doc in docs])
        
        api_messages = [{"role": "system", "content": "你是极其严谨的业务专家。请结合【参考知识库】回答。如未提及直接回答不知道。"}]
        for msg in st.session_state.messages[:-1]: api_messages.append({"role": msg["role"], "content": msg["content"]})
        api_messages.append({"role": "user", "content": f"【参考知识库】\n{context}\n\n【最新问题】\n{prompt}"})
        
        response = client.chat.completions.create(model="deepseek-chat", messages=api_messages, stream=False)
        answer = response.choices[0].message.content
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})










