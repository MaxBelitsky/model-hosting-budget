# Security Policy

Report security issues privately rather than filing a public issue.

This tool is offline by design. It should not fetch URLs, execute embedded paths from workload files, contact cloud accounts, or provision infrastructure. Treat workload JSON as untrusted input. Reports escape HTML and neutralize spreadsheet formula prefixes in CSV output, but users should still review externally supplied profiles before relying on them for spending decisions.
