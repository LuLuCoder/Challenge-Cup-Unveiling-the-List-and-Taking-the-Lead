"""数字解析工具。"""
import re

_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def numeric_tokens(line):
    """提取一行中的数字，支持科学计数法（如 3.246e+006）。"""
    return re.findall(_NUMBER_PATTERN, line)


def is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def clean_number_string(value):
    """去除数字两侧的引号等无关字符。"""
    return str(value).strip().replace('"', "").replace("'", "")
