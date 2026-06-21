# Agent Skill Matrix

Maps the 11 Copilot agents in `.github/agents/` to task types and code domains. Use this to decide which agent to invoke — and in what order.

**Last updated:** 2026-06-20

---

## Tier & Cost at a Glance

| Agent | Tier | Invoke when... |
|---|---|---|
| `system-monitor` | 🟢 Low | First check — anything feels wrong with a service |
| `error-log-analyst` | 🟢 Low | Specific log file or error pattern to parse deeply |
| `data-integrity-checker` | 🟢 Low | Gaps in Parquet/DuckDB/Redis suspected |
| `market-temperature` | 🟢 Low | RTH market outlook / current GEX regime |
| `federal-case-status-tracker` | 🟢 Low | Legal folder status update needed |
| `legal-docs-evidence-reconciler` | 🟢 Low | Cross-file legal discrepancy or timeline check |
| `remediation-planner` | 🔴 High | After triage — diagnose root cause, open GitHub Issue |
| `code-reviewer` | 🔴 High | PR or diff — surfaces bugs, contracts, security issues |
| `downstream-integration-advisor` | 🔴 High | New Redis channel / API endpoint / schema change |
| `discord-bot-refactor-advisor` | 🔴 High | Extracting commands or listeners from `trade_bot.py` |
| `spec-writer` | 🔴 High | Starting a new feature from scratch |

> **Rule of thumb:** always run a low-cost agent first. Escalate to high-cost only when you have a clear finding or decision to make.

---

## By Code Domain

Domains from [`docs/MICRO_AGENT_ARCHITECTURE.md`](MICRO_AGENT_ARCHITECTURE.md).

| Domain | Detect / Triage | Fix / Plan | Review | Change Impact |
|---|---|---|---|---|
| **1 · Market Data Ingestion** (Schwab, TastyTrade) | `system-monitor` → `error-log-analyst` | `remediation-planner` | `code-reviewer` | `downstream-integration-advisor` |
| **2 · GEX Data Services** | `system-monitor` → `data-integrity-checker` | `remediation-planner` | `code-reviewer` | `downstream-integration-advisor` |
| **3 · Redis TimeSeries** | `data-integrity-checker` | `remediation-planner` | `code-reviewer` | `downstream-integration-advisor` |
| **4 · Persistence & Flush** (DuckDB / Parquet) | `data-integrity-checker` | `remediation-planner` | `code-reviewer` | `downstream-integration-advisor` |
| **5 · API Gateway** | `system-monitor` | `remediation-planner` | `code-reviewer` | `downstream-integration-advisor` |
| **6 · Discord Bot** | `error-log-analyst` | `discord-bot-refactor-advisor` | `code-reviewer` | `downstream-integration-advisor` |
| **7 · UW Message Processing** | `error-log-analyst` | `remediation-planner` | `code-reviewer` | `downstream-integration-advisor` |
| **8 · Batch / SCID / Imports** | `data-integrity-checker` | `remediation-planner` | `code-reviewer` | — |
| **9 · Monitoring & Metrics** | `system-monitor` | `remediation-planner` | `code-reviewer` | — |
| **New Features** | — | `spec-writer` | `code-reviewer` | `downstream-integration-advisor` |
| **Market Analysis (RTH)** | `market-temperature` | — | — | — |
| **Legal / Evidence** | `federal-case-status-tracker` | `legal-docs-evidence-reconciler` | — | — |

---

## Standard Escalation Flow

Steps marked 🤖 are automated via GitHub Actions workflows in `.github/workflows/`.

```
Anomaly detected
  └─ system-monitor                  ← health scan, service state, log overview  [manual]
       └─ error-log-analyst          ← deep parse of specific log or error type  [manual / 🤖 auto-remediate.yml]
            └─ remediation-planner   ← root cause diagnosis → opens GitHub Issue [🤖 auto-remediate.yml]
                 └─ code-reviewer    ← PR review after fix is implemented         [🤖 pr-review.yml]
                      └─ downstream-integration-advisor                           [🤖 pr-review.yml on 'breaking-change' label]
```

### Automation triggers

| Workflow | Trigger | Agents invoked |
|---|---|---|
| `scheduled-monitor.yml` | Cron every 6h | lint, pytest, pip-audit → opens `monitoring-alert` issue |
| `auto-remediate.yml` | Issue labeled `monitoring-alert` or `needs-remediation` | `error-log-analyst` → `remediation-planner` |
| `pr-review.yml` | PR opened / ready for review | `code-reviewer` |
| `pr-review.yml` | PR labeled `breaking-change` | `downstream-integration-advisor` |
| Any | Manual `workflow_dispatch` | configurable |

---

## Task-Type Quick Reference

| I want to... | Use |
|---|---|
| Check if services are healthy right now | `system-monitor` |
| Understand a specific traceback or log tail | `error-log-analyst` |
| Find missing Parquet dates or DuckDB gaps | `data-integrity-checker` |
| Get a market bias before the open | `market-temperature` |
| Turn a triage report into a fix plan | `remediation-planner` |
| Review a PR before merging | `code-reviewer` |
| Add a new Redis channel or change an API response | `downstream-integration-advisor` |
| Pull a command out of `trade_bot.py` into a Cog | `discord-bot-refactor-advisor` |
| Start designing a new feature | `spec-writer` |
| Track FBI/DOJ meeting status | `federal-case-status-tracker` |
| Reconcile legal ↔ docs discrepancies | `legal-docs-evidence-reconciler` |

---

## Agent Argument Hints

Copy-paste starting points for common invocations.

```
# Health check — last hour, all services
@system-monitor last hour, all services

# Deep log parse — specific service
@error-log-analyst schwab_streamer tracebacks since 2026-06-20

# Data gaps — specific symbol and range
@data-integrity-checker NQ_NDX Parquet ticks 2026-06-01 to today

# Market outlook — default symbol
@market-temperature NQ_NDX

# Fix plan — paste triage report or issue number
@remediation-planner Issue #42

# PR review
@code-reviewer PR #88

# New Redis channel downstream impact
@downstream-integration-advisor new Redis channel market:temperature:snapshot

# Bot refactor — specific command
@discord-bot-refactor-advisor extract the !position command into a Cog

# New feature spec
@spec-writer <describe the feature in plain English>
```

---

## Adding New Agents

When creating a new agent in `.github/agents/`:

1. Add it to the **Tier & Cost** table above with tier, and one-line trigger condition
2. Add it to the **By Code Domain** table for any domain it serves
3. Add it to the **Task-Type Quick Reference** table
4. Add an argument hint example to the **Argument Hints** section
5. Update `docs/MICRO_AGENT_ARCHITECTURE.md` if it introduces a new domain
