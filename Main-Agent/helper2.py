import os
import time
import chromadb
import pymysql
from dotenv import load_dotenv
from openai import OpenAI
import json

# ========== 导出函数 ==========
def load_env():
    load_dotenv()

def setup_mysql_database():
    """首次运行一键建库 + 两张表：travel_agent_memory（事实记忆） + travel_snapshot（版本快照）"""
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        charset="utf8mb4"
    )
    cur = conn.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS travel_agent_memory_db DEFAULT CHARACTER SET utf8mb4;")
    cur.execute("USE travel_agent_memory_db;")

    # 事实记忆表，带上 is_valid！！
    create_memory_table = """
    CREATE TABLE IF NOT EXISTS travel_agent_memory (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(128) NOT NULL,
        memory_type VARCHAR(64) NOT NULL,
        content TEXT NOT NULL,
        is_valid TINYINT NOT NULL DEFAULT 1 COMMENT '1有效 0失效',
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    cur.execute(create_memory_table)

    # 快照表（方案B必备）
    create_snapshot_table = """
    CREATE TABLE IF NOT EXISTS travel_snapshot (
        id INT PRIMARY KEY AUTO_INCREMENT,
        session_id VARCHAR(100) NOT NULL,
        version_no INT NOT NULL COMMENT '版本号',
        travel_info_json JSON NOT NULL COMMENT '整套出行信息',
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_session_version (session_id, version_no)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    cur.execute(create_snapshot_table)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ 数据库、记忆表、快照表初始化完成")

def connect_to_mysql():
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
        charset="utf8mb4"
    )
    return conn

# =====================================================
load_dotenv()

# ---------------- 百炼 Embedding 客户端 ----------------
def _get_bailian_client():
    client = OpenAI(
        api_key=os.getenv("BAILIAN_API_KEY"),
        base_url=os.getenv("BAILIAN_BASE_URL")
    )
    return client

def get_embedding(text: str):
    client = _get_bailian_client()
    resp = client.embeddings.create(
        input=text,
        model="text-embedding-v2"
    )
    return resp.data[0].embedding

# ---------------- Chroma 持久化向量库 ----------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="travel_agent_memory_index")

# ---------------- MySQL 连接 ----------------
def get_mysql_conn():
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
        charset="utf8mb4"
    )
    return conn

# ---------------- 新增记忆：分place/time/money/people/preference，软删除同type旧记录 ----------------
def save_memory(session_id: str, memory_text: str, memory_type: str):
    conn = get_mysql_conn()
    cur = conn.cursor()
    # 同session+同type全部失效
    cur.execute("UPDATE travel_agent_memory SET is_valid=0 WHERE session_id=%s AND memory_type=%s",(session_id, memory_type))
    sql = """
    INSERT INTO travel_agent_memory(session_id, memory_type, content, is_valid)
    VALUES (%s, %s, %s, 1)
    """
    cur.execute(sql, (session_id, memory_type, memory_text))
    mysql_id = cur.lastrowid
    conn.commit()
    cur.close()
    conn.close()

    emb = get_embedding(memory_text)
    mem_id = f"mem_{mysql_id}"
    collection.add(
        ids=[mem_id],
        embeddings=[emb],
        metadatas=[{"mysql_id": mysql_id, "session_id": session_id}]
    )
    print(f"✅ 新增记忆 | mysql_id={mysql_id}, type={memory_type}")

# ---------------- 召回逻辑 ----------------
def recall_memory(session_id: str, query: str, top_k=3):
    query_emb = get_embedding(query)
    result = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where={"session_id": session_id}
    )
    if not result["metadatas"] or len(result["metadatas"][0]) == 0:
        return []

    mysql_id_list = [meta["mysql_id"] for meta in result["metadatas"][0]]
    conn = get_mysql_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    fmt = ",".join(["%s"] * len(mysql_id_list))
    sql = f"SELECT id, session_id, memory_type, content FROM travel_agent_memory WHERE id IN ({fmt}) AND is_valid=1"
    cur.execute(sql, mysql_id_list)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# ====================== 一键重建Chroma索引 ======================
def rebuild_chroma_from_mysql():
    all_data = collection.get()
    if all_data["ids"]:
        collection.delete(ids=all_data["ids"])

    conn = get_mysql_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT id, session_id, memory_type, content FROM travel_agent_memory WHERE is_valid=1")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    count = 0
    for row in rows:
        mysql_id = row["id"]
        session_id = row["session_id"]
        content = row["content"]
        emb = get_embedding(content)
        mem_id = f"mem_{mysql_id}"
        collection.add(
            ids=[mem_id],
            embeddings=[emb],
            metadatas=[{"mysql_id": mysql_id, "session_id": session_id}]
        )
        count += 1
    print(f"✅ Chroma重建完成！一共加载 {count} 条记忆，向量全部是最新文本生成")

# 快照相关
def save_snapshot(session_id: str, version_no: int, travel_info: dict):
    conn = get_mysql_conn()
    cur = conn.cursor()
    sql = """
    INSERT INTO travel_snapshot(session_id, version_no, travel_info_json)
    VALUES (%s, %s, %s)
    """
    json_str = json.dumps(travel_info, ensure_ascii=False)
    cur.execute(sql, (session_id, version_no, json_str))
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ 快照保存成功 session={session_id}, version={version_no}")

def get_snapshot(session_id: str, version_no: int) -> dict|None:
    conn = get_mysql_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    sql = "SELECT travel_info_json FROM travel_snapshot WHERE session_id=%s AND version_no=%s"
    cur.execute(sql, (session_id, version_no))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return None
    return json.loads(row["travel_info_json"])

def get_max_version(session_id: str) -> int:
    conn = get_mysql_conn()
    cur = conn.cursor()
    sql = "SELECT MAX(version_no) as max_v FROM travel_snapshot WHERE session_id=%s"
    cur.execute(sql, (session_id,))
    res = cur.fetchone()
    cur.close()
    conn.close()
    if res[0] is None:
        return 0
    return int(res[0])
