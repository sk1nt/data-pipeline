## Security & Secret Hygiene

This repository must never store secrets (tokens, credentials, refresh tokens) in source control.

If sensitive environment files were accidentally committed (e.g., `.env`, `.env.back`, `.env.local`), follow these steps:

1. Revoke and rotate secrets immediately for any leaked keys.
2. Run `git filter-repo` (or BFG) to remove secrets from Git history.
3. Update `.gitignore` to exclude `.env*` and any backups.
4. Implement secret scanning in CI to detect pre-commit accidental commits.

Contact security or the repository owner for help in removing secret history and rotating tokens.
