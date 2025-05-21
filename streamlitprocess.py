# app.py
import streamlit as st
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
from modules.reranking import cross_encoder_rerank
from modules.dialogue import DialogueManager

# ----------------------
# 初始化组件
# ----------------------
try:
    # 连接 Milvus
    connections.connect(host="localhost", port="19530")

    # 验证集合存在性
    if not utility.has_collection("medical_rag"):
        raise RuntimeError("集合不存在")

    # 加载集合
    collection = Collection("medical_rag")
    collection.load()

    # 加载模型
    model = SentenceTransformer("BAAI/bge-m3")

    # 初始化对话管理器
    if "dialogue" not in st.session_state:
        st.session_state.dialogue = DialogueManager()

except Exception as e:
    st.error(f"系统初始化失败: {str(e)}")
    st.stop()

# ----------------------
# 构建界面
# ----------------------
st.title("白血病知识问答系统")
query = st.text_input("请输入您的问题：", key="query_input")

# 显示历史对话
if st.session_state.dialogue.history:
    st.subheader("对话历史")
    for q, a in st.session_state.dialogue.history[-3:]:
        st.markdown(f"**用户**: {q}  \n**系统**: {a}")

if query:
    try:
        # 多轮对话增强查询
        context = st.session_state.dialogue.get_context()
        augmented_query = st.session_state.dialogue.refine_query(query)

        # 直接使用增强后的查询，无需图增强
        query_vec = model.encode([augmented_query])[0]

        # 执行搜索
        raw_results = collection.search(
            data=[query_vec.tolist()],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": 50}},
            limit=5,  # 扩大召回数量供Re-ranking
            output_fields=["text", "source_file"]
        )

        # Re-ranking优化
        hits = [
            {
                "text": hit.entity.get("text"),
                "score": hit.score  # 保留原始分数
            }
            for hit in raw_results[0]
        ]
        candidate_texts = [hit["text"] for hit in hits]
        reranked_indices = cross_encoder_rerank(query, candidate_texts, top_k=3)
        reranked_results = [hits[idx] for idx in reranked_indices]

        # 生成回答（示例）
        response_lines = []
        for idx, res in enumerate(reranked_results):
            # 明确访问 "text" 字段并切片
            truncated_text = res["text"][:100] + "..." if len(res["text"]) > 100 else res["text"]
            response_lines.append(f"{idx + 1}. {truncated_text} (相关性: {res['score']:.2f})")

        response = "优化后的结果：\n" + "\n".join(response_lines)

        # 更新对话历史
        st.session_state.dialogue.update(query, response)

        # 显示结果
        st.subheader("优化后的回答")
        st.write(response)

    except Exception as e:
        st.error(f"搜索失败: {str(e)}")