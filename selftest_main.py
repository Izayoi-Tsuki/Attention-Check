"""主程序冒烟测试：自动确认任务对话框，写入测试库，依次触发自评弹窗、自察热键、暂停/继续、今日报告，然后退出。

运行：  .venv\\Scripts\\python.exe selftest_main.py
"""
import sys
import time
import webbrowser

import main
from settings import ROOT, load_config

DB = ROOT / "data" / "selftest" / "smoke.db"


def patched_config():
    cfg = load_config()
    cfg["storage"]["db_path"] = str(DB)
    cfg["storage"]["reports_dir"] = str(ROOT / "data" / "selftest")
    return cfg


class AutoSession:
    def __init__(self, root, last_task, last_apps, on_done):
        root.after(200, lambda: on_done("冒烟测试", "Code.exe, Qoder.exe"))


def run():
    main.load_config = patched_config
    main.SessionDialog = AutoSession
    webbrowser.open = lambda *a, **k: True
    app = main.App()
    log = []
    t0 = time.time()

    def step(delay, name, fn):
        def go():
            try:
                fn()
                log.append(f"{time.time() - t0:5.1f}s  {name}")
            except Exception as e:
                log.append(f"{time.time() - t0:5.1f}s  {name} 失败：{type(e).__name__}: {e}")
        app.root.after(int(delay * 1000), go)

    step(3, "随机自评弹窗", lambda: app._show_probe(0))
    step(5, "回答 1", lambda: app.popup._answer(1))
    step(6, "自察热键", lambda: app.ui_queue.put("self_report"))
    step(8, "暂停", app.toggle_pause)
    step(10, "继续", app.toggle_pause)
    step(12, "今日报告", app.open_report)
    step(25, "状态", lambda: (log.append("      状态：" + app.status.get().replace("\n", " | ")),
                                 log.append("      info：" + app.info.get())))
    step(26, "退出", app.quit)
    app.run()

    for line in log:
        print(line)
    rows = main.Database(str(DB)).query(
        "SELECT (SELECT COUNT(*) FROM context), (SELECT COUNT(*) FROM probes), (SELECT COUNT(*) FROM events), "
        "(SELECT group_concat(kind) FROM events), (SELECT COUNT(*) FROM sessions WHERE end IS NOT NULL)")[0]
    print(f"context {rows[0]} 行  probes {rows[1]} 行  events {rows[2]} 行  已结束会话 {rows[4]}")
    print(f"events: {rows[3]}")
    failed = any("失败" in s for s in log) or rows[0] < 10 or rows[1] < 2
    print("冒烟测试", "未通过" if failed else "通过")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    run()
