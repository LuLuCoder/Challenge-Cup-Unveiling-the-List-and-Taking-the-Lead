"""子应用嵌入适配器。"""

import tkinter as tk


class EmbeddableRoot(tk.Frame):
    """把以 Tk 根窗口为宿主的子应用嵌入到标签页中的适配器。

    原有两个子应用（路径规划、参数优化）的构造函数会直接调用
    ``root.title()`` / ``root.geometry()`` / ``root.protocol()``，
    这些调用对嵌入用的 Frame 没有意义，因此在这里全部静默忽略；
    其余所有 Tk 接口（pack/grid/after/update/destroy 等）照常工作。
    """

    def title(self, *args, **kwargs):
        pass

    def geometry(self, *args, **kwargs):
        pass

    def protocol(self, *args, **kwargs):
        pass
