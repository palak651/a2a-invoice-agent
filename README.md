# A2A Invoice Agent

A FastAPI implementation scaffold for the IITM TDS GA3 A2A Invoice Agent contract.

## Files
- `main.py`: A2A HTTP/JSON API, task storage, invoice decision call, receipts and auth.
- `requirements.txt`: Python dependencies.
- `render.yaml`: Render deployment configuration.

## Important setup
This application requires an AI API key for invoice decisions. Set `AI_API_KEY` in the hosting environment. `AI_BASE_URL` must be an OpenAI-compatible `/v1` API base URL, and `AI_MODEL` must be supported by that provider. The default uses the OpenAI-compatible chat completions API.

Set `A2A_BEARER_TOKEN` to a long random token. The grader must use the exact token expected by the assignment platform; if the platform gives a required token or expects arbitrary bearer tokens, adjust this setting/validation to match the official grader contract.

## Local run
```bash
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Agent Card:
`http://127.0.0.1:8000/.well-known/agent-card.json`

Base URL to submit must be the deployed base URL with `/a2a/`, not just the Render origin, if the assignment form expects the A2A base URL. Check the actual submission field instructions.

## Known caveats
This is a generated starting implementation and must be tested against the official assignment grader before submission. It cannot guarantee full marks. Render free services have an ephemeral filesystem, so SQLite state can be lost after restarts/redeploys. For reliable task persistence across restarts, configure a persistent database or a Render persistent disk on a plan that supports it.
