# Security Policy

## Reporting a vulnerability

Please report vulnerabilities through [GitHub private vulnerability reporting](https://github.com/jundot/omlx/security/advisories/new), rather than public issues or pull requests.

Include the affected version or commit, relevant configuration, a reproducible example, and the demonstrated impact. Clearly state the access an attacker needs and any required operator actions. Scanner output alone is not sufficient.

## Supported versions

Security fixes target the latest release and `main`. Older releases may require an upgrade.

## Scope and trust assumptions

oMLX is designed for personal use and deployments managed by a trusted administrator.

- Local users and processes are trusted. Isolation from hostile local accounts or an already compromised host is outside the threat model.
- API and admin authentication bypasses, unauthorized file or data access, and unintended code execution are in scope, including in the admin UI.
- Administrators, installed model code, configured tools, and paired cluster nodes are trusted. Enabling `trust_remote_code` intentionally permits code execution with the server process's privileges.
- oMLX does not provide isolation between mutually untrusted tenants.
- Expected inference resource consumption and model-output quality issues are not, by themselves, security vulnerabilities. Other denial-of-service reports are assessed case by case.

These assumptions do not exclude vulnerabilities that let an attacker bypass authentication or gain privileges they do not already have.

## Deployment

Keep oMLX updated. Configure an API key before exposing the server to a network, including through a reverse proxy or tunnel. Restrict access to intended clients and protect traffic over untrusted networks. Keep cluster communication on a trusted, access-controlled network.

## Disclosure and credit

I review reports on a best-effort basis. Please coordinate disclosure through the private report to allow time for a fix or mitigation.

Reporters of confirmed vulnerabilities are credited in the relevant fix or advisory unless they prefer to remain anonymous. oMLX does not offer a paid bug bounty.
