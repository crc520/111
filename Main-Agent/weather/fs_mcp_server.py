from fastmcp import FastMCP
import os
from pathlib import Path

mcp = FastMCP("filesystem-server")

# 沙箱目录：新建 workspace 文件夹，专门用来读写文件
WORKSPACE_ROOT = Path(r"../workspace").resolve()
# 自动创建这个文件夹，如果不存在
WORKSPACE_ROOT.mkdir(exist_ok=True)

def safe_path(user_path: str) -> Path:
    # 拼接沙箱根目录 + 用户传入的相对路径
    target = (WORKSPACE_ROOT / user_path).resolve()
    # 校验：必须在沙箱目录内部
    if not str(target).startswith(str(WORKSPACE_ROOT)):
        raise PermissionError(f"禁止访问沙箱外部路径：{user_path}")
    return target

# 1. 列出目录文件
@mcp.tool()
async def list_dir(subpath: str = ".") -> str:
    """
    列出指定目录下的文件和子文件夹
    Args:
        subpath: 相对工作区的子目录，默认 "." 代表根目录
    """
    try:
        p = safe_path(subpath)
        if not p.exists():
            return f"目录不存在: {subpath}"
        if not p.is_dir():
            return f"错误：{subpath} 不是目录"
        items = []
        for entry in os.scandir(p):
            typ = "[DIR]" if entry.is_dir() else "[FILE]"
            items.append(f"{typ} {entry.name}")
        return "\n".join(items)
    except Exception as e:
        return f"list_dir错误: {str(e)}"

# 2. 读取文件
@mcp.tool()
async def read_file(file_path: str) -> str:
    """
    读取文本文件内容
    Args:
        file_path: 相对工作区的文件路径
    """
    try:
        p = safe_path(file_path)
        if not p.exists():
            return f"文件不存在: {file_path}"
        if not p.is_file():
            return f"错误：{file_path} 是目录，不是文件，无法读取"
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"read_file错误: {str(e)}"

# 3. 写入/覆盖文件
@mcp.tool()
async def write_file(file_path: str, content: str) -> str:
    """
    写入文本文件，存在则覆盖，不存在新建
    Args:
        file_path: 相对工作区的文件路径
        content: 要写入的文本内容
    """
    try:
        p = safe_path(file_path)
        # 创建父文件夹，多级目录自动生成
        p.parent.mkdir(parents=True, exist_ok=True)
        # 写入文件，utf-8编码
        p.write_text(content, encoding="utf-8")
        return f"✅ 文件写入成功：{file_path}"
    except Exception as e:
        return f"write_file错误: {str(e)}"

# 4. 在目录内搜索包含关键词的文件内容
@mcp.tool()
async def search_content(keyword: str, subpath: str = ".") -> str:
    """
    在指定目录递归搜索文本内容，返回包含关键词的文件
    Args:
        keyword: 需要检索的文本关键词
        subpath: 相对工作区的目录，默认 "." 代表根目录
    """
    try:
        base_dir = safe_path(subpath)
        if not base_dir.exists():
            return f"目录不存在: {subpath}"
        if not base_dir.is_dir():
            return f"错误：{subpath} 不是目录，无法搜索"

        match_files = []
        # 递归遍历目录所有文件
        for file in base_dir.rglob("*"):
            # 只处理文件，跳过文件夹
            if not file.is_file():
                continue
            try:
                text = file.read_text(encoding="utf-8")
                if keyword in text:
                    # 返回相对于沙箱根目录的路径
                    rel_path = file.relative_to(WORKSPACE_ROOT)
                    match_files.append(str(rel_path))
            except:
                # 二进制文件/编码异常直接跳过，不中断搜索
                continue

        if match_files:
            return "找到包含关键词的文件：\n" + "\n".join(match_files)
        else:
            return "未找到包含该关键词的文件"
    except Exception as e:
        return f"search_content错误: {str(e)}"


if __name__ == "__main__":
    mcp.run()
