# Repository Guidelines

## Project Structure & Module Organization
Source lives in `src/` plus `backend/src/` (FastAPI `api/`, Redis/DuckDB `services/`, typed `models/`, runtime `config.py`), while `backend/tests/` mirrors the same tree. The monitoring dashboard sits in `frontend/src/` with CSS/JS colocated for simple deploys. Docs and specs live under `docs/`, `specs/`, and `examples/`; update them whenever APIs or data contracts move. Runtime artifacts (Redis snapshots, DuckDB files, token caches) stay in `data/` and must never be committed.

## Build, Test, and Development Commands
- `pip install -e .` installs the editable package plus extras used by scripts.
- `python data-pipeline.py --host 0.0.0.0 --port 8877` starts the orchestration API and monitoring UI.
 - `python scripts/start_schwab_streamer.py --dry-run` validates Schwab credentials and channel wiring before live ticks.
- `pytest backend/tests` (add `-k pattern` or `-m "not integration"`) drives unit and contract coverage.
- `ruff check . && ruff format .` enforces lint plus formatter parity with CI.

## Coding Style & Naming Conventions
Python 3.11 with 4-space indentation, f-strings, and type hints on public functions is the baseline. Use snake_case for variables/functions, PascalCase for Pydantic models, and suffix response DTOs with `Response`. Keep FastAPI dependency factories prefixed with `get_` and Redis publishers suffixed `_channel` for quick grepping. Run Ruff before committing so import order, docstrings, and formatting stay aligned across modules.

## Testing Guidelines
Prefer pytest fixtures over ad-hoc setup; place shared helpers in `backend/tests/conftest.py`. Mirror package paths in filenames (`test_services_redis_flush.py`) and mark external feeds with `@pytest.mark.integration` so CI can skip them. Aim for ~90% coverage in `services/` and any DuckDB/Polars transforms; each bugfix touching Redis or IO should add a regression test alongside the fix.

## Commit & Pull Request Guidelines
Follow the existing log style: lowercase Conventional Commit prefixes (`fix(redis_flush_worker): ...`, `chore:`) plus imperative summaries. PRs should link relevant specs or issues, summarize API changes, and attach screenshots or log tails for UI/monitoring tweaks. Include the Ruff + pytest command output (or CI links) in the PR body before requesting review, and document new config keys or folders in `docs/` or `specs/` as part of the same change.

## Security & Configuration Tips
Store tokens and client secrets only in `.env` and runtime files under `data/`; rotate Schwab credentials with `python scripts/schwab_token_manager.py rotate` instead of manual editing. When experimenting with Redis, namespace keys with your username (for example, `dev_jane_ticks`) to avoid colliding with shared live channels.
