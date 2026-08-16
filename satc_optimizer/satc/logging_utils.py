"""日志工具：同时输出到控制台和日志文件。"""

from datetime import datetime


class Logger:
    """简单文件 + 控制台日志器。"""

    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("SATC-NSGA-II 运行日志\n")
            f.write("=" * 90 + "\n")
            f.write(
                "开始时间："
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + "\n"
            )
            f.write("=" * 90 + "\n\n")
            f.flush()

    def write(self, text=""):
        text = str(text)
        print(text, flush=True)
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(text + "\n")
            f.flush()

    def section(self, title):
        self.write()
        self.write("=" * 90)
        self.write(title)
        self.write("=" * 90)
