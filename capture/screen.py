"""Windows 屏幕相关工具：DPI 感知、屏幕尺寸、查询某点下的顶层窗口、鼠标穿透窗口。"""
import ctypes
import os
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
dwmapi = ctypes.windll.dwmapi

GW_HWNDNEXT, GW_CHILD = 2, 5
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x20
WS_EX_TOOLWINDOW = 0x80
WS_EX_LAYERED = 0x80000
WS_EX_NOACTIVATE = 0x08000000
DWMWA_CLOAKED = 14
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

user32.GetDesktopWindow.restype = wintypes.HWND
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
user32.GetWindow.restype = wintypes.HWND
user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetParent.restype = wintypes.HWND
user32.GetParent.argtypes = [wintypes.HWND]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]

SPECIAL_CLASSES = {"Shell_TrayWnd": "任务栏", "Progman": "桌面", "WorkerW": "桌面"}
_OWN_PID = os.getpid()
_exe_cache: dict = {}


def set_dpi_awareness():
    """必须在创建任何窗口之前调用，保证 tkinter / pynput / 摄像头坐标都是物理像素。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    except Exception:
        user32.SetProcessDPIAware()


def screen_size() -> tuple:
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _text(fn, hwnd, n=512) -> str:
    buf = ctypes.create_unicode_buffer(n)
    fn(hwnd, buf, n)
    return buf.value


def _exe_name(pid: int) -> str:
    # 缓存 60 秒，避免 PID 被系统复用后长期显示旧进程名
    hit = _exe_cache.get(pid)
    if hit and time.monotonic() - hit[1] < 60:
        return hit[0]
    name = "?"
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if h:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            name = Path(buf.value).name
        kernel32.CloseHandle(h)
    _exe_cache[pid] = (name, time.monotonic())
    return name


def _is_cloaked(hwnd) -> bool:
    v = ctypes.c_int(0)
    dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(v), ctypes.sizeof(v))
    return v.value != 0


def window_at(x: float, y: float) -> Optional[dict]:
    """按 Z 序从上到下找到包含 (x, y) 的第一个可见顶层窗口。"""
    rect = wintypes.RECT()
    hwnd = user32.GetWindow(user32.GetDesktopWindow(), GW_CHILD)
    while hwnd:
        if (user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd)
                and not (user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TRANSPARENT)):
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if rect.left <= x < rect.right and rect.top <= y < rect.bottom and not _is_cloaked(hwnd):
                title = _text(user32.GetWindowTextW, hwnd)
                cls = _text(user32.GetClassNameW, hwnd, 256)
                if title or cls in SPECIAL_CLASSES:
                    pid = wintypes.DWORD()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    if pid.value == _OWN_PID:
                        return dict(exe="本程序", title=title, cls=cls)
                    return dict(exe=_exe_name(pid.value), title=SPECIAL_CLASSES.get(cls, title), cls=cls)
        hwnd = user32.GetWindow(hwnd, GW_HWNDNEXT)
    return None


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


def idle_seconds() -> float:
    """距离最后一次键盘/鼠标输入的秒数（系统级）。"""
    info = _LASTINPUTINFO(ctypes.sizeof(_LASTINPUTINFO), 0)
    user32.GetLastInputInfo(ctypes.byref(info))
    return ((kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0


def foreground_window() -> Optional[dict]:
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    cls = _text(user32.GetClassNameW, hwnd, 256)
    title = _text(user32.GetWindowTextW, hwnd)
    return dict(
        hwnd=hwnd,
        exe="本程序" if pid.value == _OWN_PID else _exe_name(pid.value),
        title=SPECIAL_CLASSES.get(cls, title),
        cls=cls,
        rect=(rect.left, rect.top, rect.right, rect.bottom),
    )


def set_foreground(hwnd):
    if hwnd:
        user32.SetForegroundWindow(hwnd)


def force_foreground(tk_window):
    """把 tkinter 窗口切到前台。后台进程直接 SetForegroundWindow 会被系统拦截，
    先把本线程的输入队列挂到当前前台线程上再切换。"""
    tk_window.update_idletasks()
    hwnd = user32.GetParent(tk_window.winfo_id()) or tk_window.winfo_id()
    fg = user32.GetForegroundWindow()
    fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    me = kernel32.GetCurrentThreadId()
    attached = bool(fg_thread and fg_thread != me and user32.AttachThreadInput(me, fg_thread, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(me, fg_thread, False)


def make_click_through(tk_window):
    """让 tkinter 顶层窗口鼠标穿透、不抢焦点、不出现在 Alt+Tab 中。"""
    tk_window.update_idletasks()
    hwnd = user32.GetParent(tk_window.winfo_id())
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                          style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
