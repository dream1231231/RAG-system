import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType



os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像
MILVUS_COLLECTION_NAME = "medical_rag"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

with open("processed_data.json", "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

texts = [item["abstract"] for item in chunks_data]
ids = [item["id"] for item in chunks_data]
source_files=[item["source_file"] for item in chunks_data]

# Step 2: 生成向量嵌入
print("正在加载嵌入模型...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("生成文本向量...")
embeddings = model.encode(texts, normalize_embeddings=True)

# Step 3: 连接Milvus
print("\n连接Milvus数据库...")
connections.connect(host="localhost", port="19530")

if utility.has_collection(MILVUS_COLLECTION_NAME):  # 正确检查集合是否存在
    utility.drop_collection(MILVUS_COLLECTION_NAME)  # 安全删除集合

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=300),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
]

# if Collection(MILVUS_COLLECTION_NAME).exists():
#     Collection(MILVUS_COLLECTION_NAME).drop()
schema = CollectionSchema(fields, description="医疗知识库")
collection = Collection(MILVUS_COLLECTION_NAME, schema)

# Step 4: 插入数据
print("\n插入数据到Milvus...")
entities = [
    ids,                            # 主键
    texts,                          # 原始文本
    source_files,
    embeddings.tolist()             # 向量数据
]

insert_result = collection.insert(entities)
collection.flush()

# Step 5: 创建索引
print("创建HNSW索引...")
index_params = {
    "index_type": "HNSW",
    "metric_type": "IP",           # 因向量已归一化，使用内积（等效余弦相似度）
    "params": {"M": 16, "efConstruction": 200}
}

collection.create_index(field_name="vector", index_params=index_params)
print(f"数据插入完成！总计 {len(ids)} 条记录")