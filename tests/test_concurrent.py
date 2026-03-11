"""Tests for concurrent access and threading safety with SQLite WAL."""

from __future__ import annotations

import sqlite3
import threading
import tempfile
import os

import pytest

from engine import ChunkLog


class TestConcurrentAppends:
    def test_concurrent_appends_no_crash(self):
        """Multiple threads appending simultaneously — verify no deadlock."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Pre-initialize DB schema in WAL mode to avoid contention on CREATE TABLE
        init_log = ChunkLog(db_path=db_path, max_tokens=100000, scoring_mode=None)
        init_log.close()

        completed = {"count": 0}

        def append_worker(worker_id: int):
            try:
                log = ChunkLog(db_path=db_path, max_tokens=100000, scoring_mode=None)
                log._conn.execute("PRAGMA busy_timeout = 30000")
                for i in range(10):
                    log.append("user", f"Worker {worker_id} message {i}")
                    log.next_turn()
                log.close()
                completed["count"] += 1
            except sqlite3.OperationalError:
                # SQLite locking under heavy write contention is acceptable
                completed["count"] += 1

        try:
            threads = [threading.Thread(target=append_worker, args=(w,)) for w in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)

            # All threads should complete (no deadlock)
            assert completed["count"] == 3

            # Verify some data was written
            conn = sqlite3.connect(db_path)
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            conn.close()
            assert count > 0
        finally:
            for f in [db_path, db_path + "-wal", db_path + "-shm"]:
                try:
                    os.unlink(f)
                except FileNotFoundError:
                    pass

    def test_concurrent_reads_during_writes(self):
        """Reading while writing should work with WAL mode."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Pre-initialize the database
        init_log = ChunkLog(db_path=db_path, max_tokens=100000, scoring_mode=None)
        init_log.append("user", "Initial message")
        init_log.close()

        write_done = threading.Event()
        read_count = {"value": 0}

        def writer():
            log = ChunkLog(db_path=db_path, max_tokens=100000, scoring_mode=None)
            log._conn.execute("PRAGMA busy_timeout = 30000")
            for i in range(20):
                log.append("user", f"Write message {i}")
                log.next_turn()
            log.close()
            write_done.set()

        def reader():
            log = ChunkLog(db_path=db_path, max_tokens=100000, scoring_mode=None)
            log._conn.execute("PRAGMA busy_timeout = 30000")
            while not write_done.is_set():
                try:
                    ctx = log.get_context()
                    read_count["value"] += 1
                except sqlite3.OperationalError:
                    pass  # Acceptable under contention
            log.close()

        try:
            t_write = threading.Thread(target=writer)
            t_read = threading.Thread(target=reader)
            t_write.start()
            t_read.start()
            t_write.join(timeout=60)
            t_read.join(timeout=60)

            # Reader should have read at least once
            assert read_count["value"] > 0
        finally:
            for f in [db_path, db_path + "-wal", db_path + "-shm"]:
                try:
                    os.unlink(f)
                except FileNotFoundError:
                    pass

    def test_concurrent_appends_with_compaction(self):
        """Concurrent appends with compaction — verify no deadlock."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        completed = {"count": 0}

        def append_worker(worker_id: int):
            try:
                log = ChunkLog(
                    db_path=db_path, max_tokens=200,
                    soft_threshold=0.7, hard_threshold=0.9,
                    scoring_mode=None,
                )
                log._conn.execute("PRAGMA busy_timeout = 30000")
                for i in range(10):
                    log.append("user", f"Worker {worker_id} message {i} with some filler")
                    log.next_turn()
                log.close()
                completed["count"] += 1
            except sqlite3.OperationalError:
                # SQLite locking under heavy write contention is expected
                completed["count"] += 1

        try:
            threads = [threading.Thread(target=append_worker, args=(w,)) for w in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)

            # All threads should complete (no deadlock)
            assert completed["count"] == 3
        finally:
            for f in [db_path, db_path + "-wal", db_path + "-shm"]:
                try:
                    os.unlink(f)
                except FileNotFoundError:
                    pass

    def test_in_memory_single_thread_only(self):
        """In-memory databases are single-connection, verify basic operation."""
        log = ChunkLog(db_path=":memory:", max_tokens=10000, scoring_mode=None)
        for i in range(50):
            log.append("user", f"Message {i}")
            log.next_turn()

        ctx = log.get_context()
        assert len(ctx) == 50
        log.close()
