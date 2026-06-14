import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import bcrypt
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)

_database_url = os.getenv("DATABASE_URL")
if not _database_url:
    raise RuntimeError("DATABASE_URL is not defined in .env")

_DB_CONNECT_TIMEOUT_SECONDS = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "5"))
_DB_STATEMENT_TIMEOUT_MS = int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "10000"))


def _connect_options(extra: Optional[dict] = None) -> dict:
    options = {
        "connect_timeout": _DB_CONNECT_TIMEOUT_SECONDS,
        "options": f"-c statement_timeout={_DB_STATEMENT_TIMEOUT_MS}",
    }
    if extra:
        options.update(extra)
    return options


def _now():
    return datetime.now(timezone.utc)


def _json_date(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _json_dump(value):
    return json.dumps(value)


def _json_load(value, default=None):
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def get_conn():
    return psycopg.connect(_database_url, row_factory=dict_row, **_connect_options())


def ensure_database_exists():
    url = psycopg.conninfo.conninfo_to_dict(_database_url)
    database_name = url.get("dbname")
    if not database_name:
        return

    maintenance = dict(url)
    maintenance["dbname"] = "postgres"
    try:
        with psycopg.connect(**maintenance, autocommit=True, **_connect_options()) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (database_name,),
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{database_name.replace(chr(34), chr(34) * 2)}"')
    except psycopg.errors.DuplicateDatabase:
        pass


def ensure_schema():
    ensure_database_exists()
    with get_conn() as conn:
        conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'enum_users_role') THEN
                    CREATE TYPE enum_users_role AS ENUM ('admin', 'user');
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'enum_sessions_status') THEN
                    CREATE TYPE enum_sessions_status AS ENUM ('pending', 'running', 'completed', 'failed', 'paused');
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'enum_messages_role') THEN
                    CREATE TYPE enum_messages_role AS ENUM ('user', 'agent');
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'enum_messages_messagetype') THEN
                    CREATE TYPE enum_messages_messagetype AS ENUM ('text', 'action', 'result', 'error', 'screenshot');
                END IF;
            END $$;
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                role enum_users_role DEFAULT 'user',
                "isActive" BOOLEAN DEFAULT TRUE,
                "createdAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                "userId" UUID NOT NULL REFERENCES users(id) ON UPDATE CASCADE ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                description TEXT NOT NULL,
                "systemContext" TEXT NOT NULL,
                "isActive" BOOLEAN DEFAULT TRUE,
                metadata JSONB DEFAULT '{}'::JSONB,
                "createdAt" TIMESTAMPTZ DEFAULT NOW(),
                "updatedAt" TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                "sessionId" VARCHAR(255) UNIQUE NOT NULL,
                "userId" UUID NOT NULL REFERENCES users(id) ON UPDATE CASCADE ON DELETE CASCADE,
                "agentId" UUID NULL REFERENCES agent(id) ON UPDATE CASCADE ON DELETE SET NULL,
                task TEXT NOT NULL,
                "toolsUsed" TEXT[] DEFAULT ARRAY[]::TEXT[],
                status enum_sessions_status DEFAULT 'pending',
                result JSONB DEFAULT NULL,
                "urlsVisited" TEXT[] DEFAULT ARRAY[]::TEXT[],
                "formsFilled" JSONB DEFAULT '[]'::JSONB,
                "lastUrl" TEXT DEFAULT NULL,
                "lastScreenshot" TEXT DEFAULT NULL,
                "startedAt" TIMESTAMPTZ DEFAULT NOW(),
                "endedAt" TIMESTAMPTZ DEFAULT NULL,
                metadata JSONB DEFAULT '{}'::JSONB
            )
        """)
        # Migrate existing DBs
        conn.execute('ALTER TABLE sessions ADD COLUMN IF NOT EXISTS "agentId" UUID NULL REFERENCES agent(id) ON UPDATE CASCADE ON DELETE SET NULL')
        conn.execute('ALTER TABLE sessions ADD COLUMN IF NOT EXISTS "lastUrl" TEXT DEFAULT NULL')
        conn.execute('ALTER TABLE sessions ADD COLUMN IF NOT EXISTS "lastScreenshot" TEXT DEFAULT NULL')
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                "sessionId" UUID NOT NULL REFERENCES sessions(id) ON UPDATE CASCADE ON DELETE CASCADE,
                "userId" UUID NOT NULL REFERENCES users(id) ON UPDATE CASCADE ON DELETE CASCADE,
                role enum_messages_role NOT NULL,
                content TEXT NOT NULL,
                "messageType" enum_messages_messagetype DEFAULT 'text',
                attachments JSONB DEFAULT '[]'::JSONB,
                "generatedBy" VARCHAR(255) DEFAULT NULL,
                "isLastScreenshot" BOOLEAN DEFAULT FALSE,
                "pageUrl" VARCHAR(255) DEFAULT NULL,
                "pageState" JSONB DEFAULT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.execute('CREATE INDEX IF NOT EXISTS messages_session_timestamp_idx ON messages ("sessionId", timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_agent_user_active ON agent("userId", "isActive")')
        
        # Tables pour les workflows générés
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generated_workflows (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                "userId" UUID NOT NULL REFERENCES users(id) ON UPDATE CASCADE ON DELETE CASCADE,
                "workflowName" VARCHAR(255) NOT NULL,
                "promptPattern" TEXT NOT NULL,
                "scriptCode" TEXT NOT NULL,
                parameters JSONB DEFAULT '{}'::JSONB,
                "filePath" TEXT,
                language VARCHAR(50) DEFAULT 'python',
                "isActive" BOOLEAN DEFAULT TRUE,
                "executionCount" INTEGER DEFAULT 0,
                "lastExecutedAt" TIMESTAMPTZ,
                "createdAt" TIMESTAMPTZ DEFAULT NOW(),
                "updatedAt" TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_actions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                "workflowId" UUID NOT NULL REFERENCES generated_workflows(id) ON UPDATE CASCADE ON DELETE CASCADE,
                "stepNumber" INTEGER NOT NULL,
                "actionType" VARCHAR(50) NOT NULL,
                "actionData" JSONB NOT NULL,
                "pageUrl" TEXT,
                selector TEXT,
                "inputValue" TEXT,
                success BOOLEAN DEFAULT TRUE,
                "errorMessage" TEXT,
                "executionTimeMs" INTEGER,
                "createdAt" TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_executions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                "workflowId" UUID NOT NULL REFERENCES generated_workflows(id) ON UPDATE CASCADE ON DELETE CASCADE,
                "parametersUsed" JSONB,
                success BOOLEAN,
                "executionTimeMs" INTEGER,
                result JSONB,
                "errorMessage" TEXT,
                "executedAt" TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        conn.execute('CREATE INDEX IF NOT EXISTS idx_workflow_prompt ON generated_workflows("promptPattern")')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_workflow_user ON generated_workflows("userId", "isActive")')
        conn.execute('ALTER TABLE workflow_actions ADD COLUMN IF NOT EXISTS selector TEXT')
        conn.execute('ALTER TABLE workflow_actions ADD COLUMN IF NOT EXISTS "pageUrl" TEXT')
        conn.execute('ALTER TABLE workflow_actions ADD COLUMN IF NOT EXISTS success BOOLEAN DEFAULT TRUE')
        conn.execute('ALTER TABLE workflow_actions ADD COLUMN IF NOT EXISTS "errorMessage" TEXT')
        conn.execute('ALTER TABLE workflow_actions ADD COLUMN IF NOT EXISTS "createdAt" TIMESTAMPTZ DEFAULT NOW()')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_actions_workflow ON workflow_actions("workflowId", "stepNumber")')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_actions_page_url ON workflow_actions("pageUrl")')
        
        conn.commit()


ensure_schema()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, stored_password: str) -> bool:
    if not stored_password:
        return False
    if stored_password.startswith("$2"):
        return bcrypt.checkpw(password.encode("utf-8"), stored_password.encode("utf-8"))
    try:
        algorithm, rounds, salt, digest = stored_password.split("$", 3)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), int(rounds)).hex()
    return hmac.compare_digest(candidate, digest)


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise RuntimeError("JWT_SECRET is not defined in .env")
    return secret


def _jwt_expires_seconds() -> int:
    raw = os.getenv("JWT_EXPIRES_IN", "7d")
    if raw.endswith("d"):
        return int(raw[:-1]) * 24 * 60 * 60
    if raw.endswith("h"):
        return int(raw[:-1]) * 60 * 60
    return int(raw)


def create_token(user: dict) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(user["id"]),
        "email": user["email"],
        "role": user.get("role", "user"),
        "exp": int(time.time()) + _jwt_expires_seconds(),
    }
    encoded_header = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(_jwt_secret().encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{encoded_header}.{encoded_payload}.{_b64url(signature)}"


def verify_token(token: str) -> dict:
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        expected = hmac.new(_jwt_secret().encode("utf-8"), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url(expected), encoded_signature):
            raise ValueError("Invalid token signature")
        payload = json.loads(_b64url_decode(encoded_payload))
        if payload.get("exp", 0) < int(time.time()):
            raise ValueError("Token expired")
        return payload
    except Exception as exc:
        raise ValueError("Invalid token") from exc


def public_user(user: dict) -> dict:
    return {
        "id": str(user["id"]),
        "username": user.get("username", ""),
        "email": user.get("email", ""),
        "role": user.get("role", "user"),
    }


def public_session(session: dict, last_message: Optional[dict] = None) -> dict:
    return {
        "id": session.get("sessionId", ""),
        "task": session.get("task", ""),
        "agentId": str(session.get("agentId")) if session.get("agentId") else None,
        "agentName": session.get("agentName") or session.get("agent_name") or None,
        "status": session.get("status", "pending"),
        "toolsUsed": session.get("toolsUsed", []),
        "result": _json_load(session.get("result")),
        "lastUrl": session.get("lastUrl"),
        "lastScreenshot": session.get("lastScreenshot"),
        "startedAt": _json_date(session.get("startedAt")),
        "endedAt": _json_date(session.get("endedAt")),
        "lastMessage": last_message.get("content", "") if last_message else "",
        "lastMessageAt": _json_date(last_message.get("timestamp")) if last_message else _json_date(session.get("startedAt")),
    }


def public_agent(agent_row: dict) -> dict:
    return {
        "id": str(agent_row["id"]),
        "name": agent_row.get("name", ""),
        "description": agent_row.get("description", ""),
        "systemContext": agent_row.get("systemContext", ""),
        "isActive": agent_row.get("isActive", True),
        "metadata": _json_load(agent_row.get("metadata"), {}),
        "createdAt": _json_date(agent_row.get("createdAt")),
        "updatedAt": _json_date(agent_row.get("updatedAt")),
    }


def public_message(message: dict) -> dict:
    return {
        "id": str(message["id"]),
        "role": message.get("role", ""),
        "content": message.get("content", ""),
        "messageType": message.get("messageType", "text"),
        "generatedBy": message.get("generatedBy"),
        "isLastScreenshot": message.get("isLastScreenshot", False),
        "pageUrl": message.get("pageUrl", ""),
        "timestamp": _json_date(message.get("timestamp")),
    }


def register_user(username: str, email: str, password: str) -> dict:
    try:
        with get_conn() as conn:
            now = _now()
            user = conn.execute(
                """
                INSERT INTO users (id, username, email, password, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (str(uuid4()), username.strip(), email.strip().lower(), hash_password(password), now, now),
            ).fetchone()
            conn.commit()
            return user
    except psycopg.errors.UniqueViolation as exc:
        raise ValueError("Email ou nom d'utilisateur deja utilise") from exc


def login_user(email: str, password: str) -> dict:
    with get_conn() as conn:
        user = conn.execute(
            'SELECT * FROM users WHERE email = %s AND "isActive" = TRUE',
            (email.strip().lower(),),
        ).fetchone()
    if not user or not verify_password(password, user.get("password", "")):
        raise ValueError("Email ou mot de passe invalide")
    return user


def find_user_by_id(user_id: str) -> Optional[dict]:
    with get_conn() as conn:
        return conn.execute(
            'SELECT * FROM users WHERE id = %s AND "isActive" = TRUE',
            (user_id,),
        ).fetchone()


def create_agent(user_id: str, name: str, description: str, system_context: str) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO agent (id, "userId", name, description, "systemContext", "createdAt", "updatedAt")
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (str(uuid4()), user_id, name.strip(), description.strip(), system_context.strip(), _now(), _now()),
        ).fetchone()
        conn.commit()
        return row


def get_user_agents(user_id: str) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            'SELECT * FROM agent WHERE "userId" = %s AND "isActive" = TRUE ORDER BY "updatedAt" DESC',
            (user_id,),
        ).fetchall()


def get_agent_by_id(agent_id: str, user_id: str, include_inactive: bool = False) -> Optional[dict]:
    active_clause = '' if include_inactive else ' AND "isActive" = TRUE'
    with get_conn() as conn:
        return conn.execute(
            f'SELECT * FROM agent WHERE id = %s AND "userId" = %s{active_clause}',
            (agent_id, user_id),
        ).fetchone()


def update_agent(agent_id: str, user_id: str, fields: dict) -> Optional[dict]:
    allowed = {
        "name": "name",
        "description": "description",
        "systemContext": '"systemContext"',
        "metadata": "metadata",
    }
    updates, values = [], []
    for key, column in allowed.items():
        if key not in fields:
            continue
        value = fields[key]
        if value is None:
            continue
        if key == "metadata":
            updates.append(f"{column} = %s::jsonb")
            values.append(_json_dump(value))
        else:
            updates.append(f"{column} = %s")
            values.append(str(value).strip())
    if not updates:
        return get_agent_by_id(agent_id, user_id)
    updates.append('"updatedAt" = %s')
    values.append(_now())
    values.extend([agent_id, user_id])
    with get_conn() as conn:
        row = conn.execute(
            f'UPDATE agent SET {", ".join(updates)} WHERE id = %s AND "userId" = %s AND "isActive" = TRUE RETURNING *',
            values,
        ).fetchone()
        conn.commit()
        return row


def delete_agent(agent_id: str, user_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            'UPDATE agent SET "isActive" = FALSE, "updatedAt" = %s WHERE id = %s AND "userId" = %s AND "isActive" = TRUE RETURNING id',
            (_now(), agent_id, user_id),
        ).fetchone()
        conn.commit()
        return bool(row)


def create_agent_session(user: dict, task: str, agent_id=None, tools_used=None) -> dict:
    with get_conn() as conn:
        session = conn.execute(
            """
            INSERT INTO sessions (id, "sessionId", "userId", "agentId", task, "toolsUsed", status)
            VALUES (%s, %s, %s, %s, %s, %s, 'running')
            RETURNING *
            """,
            (str(uuid4()), str(uuid4()), user["id"], agent_id, task, tools_used or []),
        ).fetchone()
        conn.commit()
        return session


def get_user_sessions(user: dict) -> list[dict]:
    with get_conn() as conn:
        sessions = conn.execute(
            '''SELECT s.id, s."sessionId", s."userId", s."agentId", a.name AS "agentName",
               s.task, s."toolsUsed", s.status, s.result, s."urlsVisited", s."formsFilled",
               s."lastUrl", s."startedAt", s."endedAt", s.metadata
               FROM sessions s
               LEFT JOIN agent a ON a.id = s."agentId"
               WHERE s."userId" = %s
               ORDER BY s."startedAt" DESC''',
            (user["id"],),
        ).fetchall()
        result = []
        for session in sessions:
            last_message = conn.execute(
                'SELECT * FROM messages WHERE "sessionId" = %s AND "messageType" != \'screenshot\' ORDER BY timestamp DESC LIMIT 1',
                (session["id"],),
            ).fetchone()
            result.append(public_session(session, last_message))
        return result


def get_session_for_user(user: dict, session_id: str) -> Optional[dict]:
    with get_conn() as conn:
        return conn.execute(
            '''SELECT s.*, a.name AS "agentName", a.description AS "agentDescription",
               a."systemContext" AS "agentSystemContext"
               FROM sessions s
               LEFT JOIN agent a ON a.id = s."agentId"
               WHERE s."sessionId" = %s AND s."userId" = %s''',
            (session_id, user["id"]),
        ).fetchone()


def get_session_messages_for_user(user: dict, session_id: str) -> tuple[dict, list[dict]]:
    session = get_session_for_user(user, session_id)
    if not session:
        raise ValueError("Session introuvable")
    with get_conn() as conn:
        messages = conn.execute(
            'SELECT * FROM messages WHERE "sessionId" = %s AND "userId" = %s AND "messageType" != \'screenshot\' ORDER BY timestamp ASC',
            (session["id"], user["id"]),
        ).fetchall()
    return public_session(session), [public_message(message) for message in messages]


def mark_agent_session_running(session: Optional[dict]):
    if not session:
        return
    with get_conn() as conn:
        conn.execute(
            'UPDATE sessions SET status = %s, "endedAt" = NULL WHERE id = %s',
            ("running", session["id"]),
        )
        conn.commit()
    session["status"] = "running"
    session["endedAt"] = None


def save_agent_message(session: Optional[dict], user: dict, role: str, content: str, message_type="text", generated_by=None):
    if not session or not content:
        return
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (id, "sessionId", "userId", role, content, "messageType", attachments, "generatedBy")
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            """,
            (str(uuid4()), session["id"], user["id"], role, content, message_type, _json_dump([]), generated_by),
        )
        conn.commit()


def save_screenshot(session: Optional[dict], user: dict, screenshot_b64: str, page_url: str = '') -> None:
    """Save a screenshot as a message and mark it as the latest for this session."""
    if not session or not screenshot_b64:
        return
    with get_conn() as conn:
        # Clear previous isLastScreenshot flags for this session
        conn.execute(
            'UPDATE messages SET "isLastScreenshot" = FALSE WHERE "sessionId" = %s AND "isLastScreenshot" = TRUE',
            (session['id'],),
        )
        conn.execute(
            """
            INSERT INTO messages (id, "sessionId", "userId", role, content, "messageType", attachments, "generatedBy", "isLastScreenshot", "pageUrl")
            VALUES (%s, %s, %s, 'agent', %s, 'screenshot', '[]'::jsonb, 'playwright', TRUE, %s)
            """,
            (str(uuid4()), session['id'], user['id'], screenshot_b64, page_url or ''),
        )
        conn.commit()


def get_last_screenshot(session: Optional[dict]) -> Optional[str]:
    """Return the base64 screenshot marked as last for this session."""
    if not session:
        return None
    with get_conn() as conn:
        row = conn.execute(
            'SELECT content FROM messages WHERE "sessionId" = %s AND "isLastScreenshot" = TRUE ORDER BY timestamp DESC LIMIT 1',
            (session['id'],),
        ).fetchone()
    return row['content'] if row else None


def get_session_screenshots(session: Optional[dict]) -> list[dict]:
    """Return all screenshots for this session with timestamp and URL."""
    if not session:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            'SELECT content, "pageUrl", timestamp FROM messages WHERE "sessionId" = %s AND "messageType" = \'screenshot\' ORDER BY timestamp ASC',
            (session['id'],),
        ).fetchall()
    return [{'content': row['content'], 'pageUrl': row['pageUrl'], 'timestamp': _json_date(row['timestamp'])} for row in rows]


def close_agent_session(session: Optional[dict], status: str, result=None):
    if not session:
        return
    with get_conn() as conn:
        conn.execute(
            'UPDATE sessions SET status = %s, result = %s::jsonb, "endedAt" = %s WHERE id = %s',
            (status, _json_dump(result) if result is not None else None, _now(), session["id"]),
        )
        conn.commit()


def update_session_page_state(session: Optional[dict], url: Optional[str] = None, screenshot_b64: Optional[str] = None) -> None:
    """Persist the last visited URL and screenshot so they survive page reload."""
    if not session:
        return
    fields, values = [], []
    if url is not None:
        fields.append('"lastUrl" = %s')
        values.append(url)
        # Append to urlsVisited array if not already last
        with get_conn() as conn:
            conn.execute(
                'UPDATE sessions SET "urlsVisited" = array_append("urlsVisited", %s) WHERE id = %s AND ("urlsVisited"[array_length("urlsVisited",1)] IS DISTINCT FROM %s)',
                (url, session['id'], url),
            )
            conn.commit()
    if screenshot_b64 is not None:
        fields.append('"lastScreenshot" = %s')
        values.append(screenshot_b64)
    if not fields:
        return
    values.append(session['id'])
    set_clause = ', '.join(fields)
    with get_conn() as conn:
        conn.execute(f'UPDATE sessions SET {set_clause} WHERE id = %s', values)
        conn.commit()


def get_latest_running_session(user: dict) -> Optional[dict]:
    """Return the most recent running or completed session for this user."""
    with get_conn() as conn:
        return conn.execute(
            'SELECT * FROM sessions WHERE "userId" = %s ORDER BY "startedAt" DESC LIMIT 1',
            (user['id'],),
        ).fetchone()


def delete_session_completely(session_id: str, user_id: str) -> bool:
    """Delete a session and all its messages completely from the database."""
    with get_conn() as conn:
        # Verify session belongs to user
        session = conn.execute(
            'SELECT id FROM sessions WHERE "sessionId" = %s AND "userId" = %s',
            (session_id, user_id),
        ).fetchone()
        
        if not session:
            return False
        
        # Delete all messages (including screenshots) for this session
        conn.execute(
            'DELETE FROM messages WHERE "sessionId" = %s',
            (session['id'],),
        )
        
        # Delete the session itself
        conn.execute(
            'DELETE FROM sessions WHERE id = %s',
            (session['id'],),
        )
        
        conn.commit()
        return True
