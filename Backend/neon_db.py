import os
import logging
import threading
import sqlite3
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Use SQLite for local database
# Use SQLite for local database


def _open_connection(database_url: str):
    """Open SQLite connection."""
    return sqlite3.connect(database_url.replace('sqlite:///', ''))


def _cursor_as_dict(conn):
    """Return a dict-row cursor for SQLite."""
    conn.row_factory = sqlite3.Row
    return conn.cursor()


# ---------------------------------------------------------------------------
# Minimal thread-safe connection pool
# ---------------------------------------------------------------------------
class _SimplePool:
    def __init__(self, database_url: str, maxconn: int = 5):
        self._url = database_url
        self._maxconn = maxconn
        self._pool: list = []
        self._lock = threading.Lock()

    def getconn(self):
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return _open_connection(self._url)

    def putconn(self, conn):
        try:
            conn.rollback()
            with self._lock:
                if len(self._pool) < self._maxconn:
                    self._pool.append(conn)
                    return
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    def closeall(self):
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self._pool.clear()


# ---------------------------------------------------------------------------
# NeonDB class
# ---------------------------------------------------------------------------
class NeonDB:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.conn = None

        if not self.database_url:
            logger.warning("DATABASE_URL not set – database disabled.")
        else:
            try:
                self.conn = _open_connection(self.database_url)
                logger.info("SQLite database ready")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.conn = None

    # ------------------------------------------------------------------
    def _get_conn(self):
        if not self.conn:
            raise RuntimeError("Database not configured.")
        return self.conn

    def _put_conn(self, conn):
        # For SQLite, no need to return to pool
        pass

    # ------------------------------------------------------------------
    def execute_query(self, query: str, params=None) -> list:
        """Run a SELECT; return list of dicts."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = _cursor_as_dict(conn)
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            if cursor:
                cursor.close()
            self._put_conn(conn)

    def execute_update(self, query: str, params=None) -> int:
        """Run INSERT / UPDATE / DELETE; return rowcount."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            conn.rollback()
            logger.error(f"DB update error: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            self._put_conn(conn)

    def execute_query_single(self, query: str, params=None):
        """Return first row or None."""
        rows = self.execute_query(query, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    def create_tables(self):
        """Create schema if it doesn't exist."""
        ddl = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp
                ON detection_logs(timestamp)
            """,
        ]
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            for stmt in ddl:
                cursor.execute(stmt)
            conn.commit()
            logger.info("Database tables ready.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating tables: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            self._put_conn(conn)

    # ------------------------------------------------------------------
    def get_detection_logs(self, limit: int = 100, offset: int = 0) -> list:
        return self.execute_query(
            "SELECT * FROM detection_logs ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

    def save_detection_log(self, filename: str, prediction: str,
                           confidence: float, user_id=None):
        query = """
            INSERT INTO detection_logs (filename, prediction, confidence, user_id)
            VALUES (?, ?, ?, ?)
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(query, (filename, prediction, confidence, user_id))
            log_id = cursor.lastrowid
            conn.commit()
            # Get the inserted row
            cursor.execute("SELECT id, timestamp FROM detection_logs WHERE id = ?", (log_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving detection log: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            self._put_conn(conn)

    def close(self):
        if self.conn:
            self.conn.close()


# Module-level singleton
db = NeonDB()
