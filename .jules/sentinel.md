# Security journal

- The optional `fff` search path includes bundled native dependencies through git2. Audit all features before release even when slim hosts do not use that path; an unaffected default build does not clear the full dependency graph.
- RUSTSEC-2026-0544 is fixed by locking libz-sys to 1.1.33 or newer. Keep advisory checks enabled; a maintenance-policy exception requires a separate decision from a vulnerability fix.
