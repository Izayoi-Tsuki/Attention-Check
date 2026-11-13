"""SQLite 存储：后台线程批量写入，采集线程只往队列里放数据，不会被磁盘 IO 阻塞。

时间戳统一为 Unix 时间（秒，float）。
"""
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    id INTEGER PRIMARY KEY, start REAL, end REAL, task TEXT, task_apps TEXT, calib_id INTEGER);
CREATE TABLE IF NOT EXISTS calibrations(
    id INTEGER PRIMARY KEY, ts REAL, val_error_px REAL, cv_error_px REAL, alpha REAL,
    screen_w INTEGER, screen_h INTEGER, model BLOB);
CREATE TABLE IF NOT EXISTS gaze(
    ts REAL, session_id INTEGER, face INTEGER, valid INTEGER, x REAL, y REAL,
    blink_l REAL, blink_r REAL, yaw REAL, pitch REAL, roll REAL, tx REAL, ty REAL, tz REAL,
    calib_id INTEGER, feat BLOB);
CREATE TABLE IF NOT EXISTS context(
    ts REAL, session_id INTEGER, exe TEXT, title TEXT, category TEXT, label TEXT, on_task INTEGER,
    win_l INTEGER, win_t INTEGER, win_r INTEGER, win_b INTEGER,
    keys INTEGER, clicks INTEGER, mouse_moves INTEGER, scrolls INTEGER, idle_s REAL);
CREATE TABLE IF NOT EXISTS probes(
    ts REAL, session_id INTEGER, source TEXT, answer INTEGER, response_ms INTEGER, deferred_ms INTEGER);
CREATE TABLE IF NOT EXISTS events(
    ts REAL, session_id INTEGER, kind TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_gaze_ts ON gaze(ts);
CREATE INDEX IF NOT EXISTS idx_context_ts ON context(ts);
CREATE INDEX IF NOT EXISTS idx_probes_ts ON probes(ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""

COLUMNS = {
    "gaze": ["ts", "session_id", "face", "valid", "x", "y", "blink_l", "blink_r",
             "yaw", "pitch", "roll", "tx", "ty", "tz", "calib_id", "feat"],
    "context": ["ts", "session_id", "exe", "title", "category", "label", "on_task",
                "win_l", "win_t", "win_r", "win_b", "keys", "clicks", "mouse_moves", "scrolls", "idle_s"],
    "probes": ["ts", "session_id", "source", "answer", "response_ms", "deferred_ms"],
    "events": ["ts", "session_id", "kind", "detail"],
}

# 自评答案编码
ANSWERS = {1: "专注当前任务", 2: "在想别的事（走神）", 3: "被别的东西吸引", 4: "主动休息", 5: "犯困 / 疲劳"}
ANSWER_SKIPPED, ANSWER_TIMEOUT = 0, -1


def connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


class Database:
    def __init__(self, path: str, flush_seconds: float = 1.0):
        self.path = path
        connect(path).close()  # 建表
        self._q: queue.Queue = queue.Queue()
        self._flush_seconds = flush_seconds
        self._stop = threading.Event()
        self.rows_written = 0
        self.error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---------- 异步写入（任意线程） ----------
    def insert(self, table: str, row: tuple):
        self._q.put((table, row))

    def event(self, session_id, kind: str, detail: str = ""):
        self.insert("events", (time.time(), session_id, kind, detail))

    # ---------- 同步操作（调用线程自己的连接） ----------
    def execute(self, sql: str, params: tuple = ()) -> int:
        conn = connect(self.path)
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def query(self, sql: str, params: tuple = ()) -> list:
        conn = connect(self.path)
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        conn = connect(self.path)
        sql = {t: f"INSERT INTO {t}({','.join(c)}) VALUES({','.join('?' * len(c))})" for t, c in COLUMNS.items()}
        while True:
            stopping = self._stop.wait(self._flush_seconds)
            batch: dict = {}
            while True:
                try:
                    table, row = self._q.get_nowait()
                except queue.Empty:
                    break
                batch.setdefault(table, []).append(row)
            if batch:
                try:
                    with conn:
                        for table, rows in batch.items():
                            conn.executemany(sql[table], rows)
                    self.rows_written += sum(len(r) for r in batch.values())
                except sqlite3.Error as e:
                    self.error = f"写入数据库失败：{e}"
            if stopping:
                break
        conn.close()
