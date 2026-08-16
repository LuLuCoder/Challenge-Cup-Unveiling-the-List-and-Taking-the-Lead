"""文本读取工具：自动尝试多种编码。"""


def read_text_auto(path):
    """自动尝试多种编码读取文本文件，避免中文乱码。"""
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception as e:
            last_error = e

    raise RuntimeError(f"无法读取文件：{path}\n{last_error}")
