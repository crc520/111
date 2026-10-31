# 新建数据库时使用
from helper2 import load_env, setup_mysql_database, connect_to_mysql
import warnings

# Warning control
warnings.filterwarnings("ignore")

load_env()

# 第一次运行打开，后续注释！
setup_mysql_database()

# 连接到数据库
database_connection = connect_to_mysql()

print("Using user:", database_connection.user)
