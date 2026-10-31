from helper2 import save_memory, recall_memory, rebuild_chroma_from_mysql

SESSION_ID = "thread-001"

# 1. 新增记忆（首次录入的时候打开，录入完即注释）
# save_memory(SESSION_ID, "用户喜欢自然风光，预算3000，4个人出行", "fact")

# 2.改完MySQL之后，运行这一行重建向量
rebuild_chroma_from_mysql()

