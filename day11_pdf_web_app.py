import streamlit as st
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ================= 1. UI 皮囊配置 =================
st.set_page_config(page_title="HiSi 白皮书专家", page_icon="📖")
st.title("📖 HiSi-G.I.D.S. 白皮书智能检索")
st.caption("已挂载本地 PDF 知识库，基于企业级物理硬盘秒级检索")

# ================= 2. 核心架构变更 (PM 请注意！) =================
@st.cache_resource
def load_local_knowledge_base():
    # 1. 请出翻译官
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # 2. 💡 魔法变更：不要再用 from_texts 从头建了！
    # 直接告诉程序：去读取现成的 hisi_vdb 文件夹
    db = Chroma(persist_directory="hisi_vdb", embedding_function=embedding_model)
    return db

# 挂载“记忆 U 盘”并唤醒大模型
db = load_local_knowledge_base()
client = OpenAI(
    # 💡 核心安全变更：不再写死秘钥！
    # 告诉程序：“等上了云服务器，去服务器的安全保险箱(secrets)里拿秘钥”
    api_key=st.secrets["DEEPSEEK_API_KEY"],  
    base_url="https://api.deepseek.com"
)

# ================= 3. 会话记忆初始化 =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= 4. 检索流水线 =================
user_input = st.chat_input("关于《产品白皮书》，您想查阅什么核心功能或条款？")

if user_input:
    # A. 记录问题
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # B. 开始在多维宇宙中寻找答案
    with st.spinner("🔍 正在秒级检索《产品白皮书》核心条款..."):
        
        # 💡 策略调优：因为 PDF 内容多，我们把 k=1 改成 k=3
        # 也就是一次性抓取 3 个最相关的段落，给大模型更充足的上下文
        results = db.similarity_search(user_input, k=3)
        
        # 把找到的 3 个段落拼装起来
        retrieved_knowledge = ""
        for i, res in enumerate(results):
            retrieved_knowledge += f"【参考段落 {i+1}】:\n{res.page_content}\n\n"
            
        # 强行注入系统提示词
        system_prompt = f"""
        你是 HiSi-G.I.D.S. 系统的资深产品专家。
        请【严格基于以下白皮书提取的内容】回答用户问题。如果下面提供的内容里没写，请诚实地回答“白皮书中未提及此信息”，禁止瞎编！
        
        【白皮书提取内容】：
        {retrieved_knowledge}
        """
        
        api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
        
        # C. 生成回答
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages
        )
        ai_reply = response.choices[0].message.content
        
    # D. 展示回答
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})