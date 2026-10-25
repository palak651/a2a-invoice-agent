```python
import os
import json
import sqlite3
import uuid

from datetime import datetime, timezone
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="A2A Invoice Agent")

DATABASE = "tasks.db"
BASE_PATH = os.getenv("A2A_BASE_PATH", "/a2a").rstrip("/")


def connect_db():
    conn = sqlite3.connect(DATABASE, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                principal TEXT NOT NULL,
                context_id TEXT NOT NULL,
                batch_id TEXT,
                state TEXT NOT NULL,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


initialize_db()


def get_principal(authorization):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")

    scheme, separator, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not separator or not token.strip():
        raise HTTPException(status_code=401, detail="Invalid authentication")

    return token.strip()


def load_task(task_id, principal):
    with connect_db() as conn:
        row = conn.execute(
            "SELECT task_json FROM tasks WHERE id=? AND principal=?",
            (task_id, principal)
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return json.loads(row["task_json"])


def save_task(task, principal, batch_id=None):
    with connect_db() as conn:
        conn.execute("""
            INSERT INTO tasks
            (id, principal, context_id, batch_id, state, task_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            task["id"],
            principal,
            task["contextId"],
            batch_id,
            task["status"]["state"],
            json.dumps(task),
            datetime.now(timezone.utc).isoformat()
        ))


@app.get("/")
def home():
    return {
        "service": "A2A Invoice Agent",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/.well-known/agent-card.json")
def agent_card(request: Request):
    origin = str(request.base_url).rstrip("/")
    base_url = origin + BASE_PATH + "/"

    return {
        "name": "A2A Invoice Action Agent",
        "description": "An agent for evidence-based invoice decisions and safe action handling.",
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [
            {
                "url": base_url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0"
            }
        ],
        "defaultInputModes": [
            "application/vnd.ga5.invoice-claim-batch+json"
        ],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ],
        "skills": [
            {
                "id": "invoice
