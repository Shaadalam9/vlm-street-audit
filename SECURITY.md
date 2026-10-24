# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, **please do not open a public issue**.
Instead, report it directly by emailing:

- **Email:** [md_shadab_alam@outlook.com]

Please include as much detail as possible about the vulnerability and how to reproduce it.

We will acknowledge your report within **48 hours** and aim to give an initial assessment
within **7 days**. Once the issue is resolved, we will tell you before disclosing it publicly.

---

## Supported Versions

| Version               | Supported |
|-----------------------|-----------|
| Latest (main branch)  | ✅        |
| Older versions        | ❌        |

---

## Running models safely

This project downloads open-weight models from Hugging Face and runs them locally.

- **Pin model revisions.** Set `revision` in `configs/models.toml` to a commit hash, so a
  changed upstream checkpoint cannot alter what runs.
- **Remote code.** Some checkpoints (e.g. Molmo2) need `trust_remote_code=True`, which runs
  Python from the model repository. Read that code at the pinned revision before enabling it.
- **Prefer safetensors** weights; do not load pickle (`.bin`, `.pt`) files from untrusted sources.
- **Keep model servers local.** Bind vLLM or Ollama to `127.0.0.1`, not `0.0.0.0`, unless the
  network is trusted: their APIs have no authentication by default.
- **Never commit secrets** (Hugging Face tokens, `.env`). `.gitignore` excludes `.env` and `*.key`.

---

*Thank you for helping us keep this project safe!*
