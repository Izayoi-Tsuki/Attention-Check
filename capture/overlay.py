"""全屏、置顶、鼠标穿透的透明窗口，只画一个视线点和所在窗口名。"""
import tkinter as tk

from . import screen

TRANSPARENT = "#ff00ff"


class GazeOverlay:
    def __init__(self, root, sw, sh, radius):
        self.r = radius
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.geometry(f"{sw}x{sh}+0+0")
        self.win.attributes("-topmost", True)
        self.win.attributes("-transparentcolor", TRANSPARENT)
        self.cv = tk.Canvas(self.win, bg=TRANSPARENT, highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        r = radius
        self.dot = self.cv.create_oval(-r, -r, r, r, outline="red", width=3)
        self.center = self.cv.create_oval(-3, -3, 3, 3, fill="red", outline="")
        self.label_bg = self.cv.create_rectangle(0, 0, 0, 0, fill="black", outline="")
        self.label = self.cv.create_text(0, 0, text="", fill="yellow", anchor="nw",
                                         font=("Microsoft YaHei UI", 10))
        screen.make_click_through(self.win)
        self.visible = True

    def update(self, xy, text):
        if xy is None:
            for item in (self.dot, self.center, self.label, self.label_bg):
                self.cv.itemconfigure(item, state="hidden")
            return
        x, y = xy
        r = self.r
        self.cv.coords(self.dot, x - r, y - r, x + r, y + r)
        self.cv.coords(self.center, x - 3, y - 3, x + 3, y + 3)
        self.cv.coords(self.label, x + r + 6, y + r + 2)
        self.cv.itemconfigure(self.label, text=text)
        bbox = self.cv.bbox(self.label)
        if bbox:
            self.cv.coords(self.label_bg, bbox[0] - 3, bbox[1] - 1, bbox[2] + 3, bbox[3] + 1)
        for item in (self.dot, self.center, self.label, self.label_bg):
            self.cv.itemconfigure(item, state="normal")

    def set_visible(self, on):
        self.visible = on
        (self.win.deiconify if on else self.win.withdraw)()
