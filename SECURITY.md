# Security

## Scope

PrismFlow is a **private research repository**. It is not a published package,
it has no users, and nothing in it is deployed. The static site under `web/` is
built and served locally.

## No credentials are stored here

No API key, token, password or other credential is committed to this
repository, in any branch or in its history.

- `.env` holds the real values on the developer's machine only. It is
  gitignored and has never been tracked.
- `.env.template` is the committed file, and it contains **placeholders only**
  (`sk-...`, `ghp_...`, and empty values). Copy it to `.env` and fill it in.
- `web/build_data.py` aborts rather than write a file whose contents match a
  credential pattern, so a key cannot reach `web/data/`.
- `web/build_dist.py` scans the whole deploy build — including inside the
  `.xlsx`, which is a zip — for developer paths, account names, email
  addresses, loopback hosts, IP addresses, `.env` variable names and provider
  key shapes.
- `tools/check_secrets.py` checks for both provider key shapes and the actual
  values in `.env`, and is meant to run before every commit. See the README for
  how to install it as a local hook.

The site itself makes **no network requests**. It reads committed files under
`web/data/` and nothing else, and `web/serve.py` sends a Content-Security-Policy
of `default-src 'self'` so that a stray external request fails loudly instead
of quietly becoming an undeclared dependency. `web/_headers` carries the same
policy plus `frame-ancestors 'none'` for a future deploy.

## Reporting a problem

If you have access to this repository and find a security problem — a
credential that reached a commit, a path that serves a file it should not, or
anything that lets the site reach the network — please report it privately to
the repository owner rather than opening an issue.

**Contact:** the email address on the commits in this repository, or whatever
private channel you already use with the owner.

Please include what you found, the file and line, and how you reproduced it. If
a credential is involved, **do not paste its value** into the report — name the
variable and say where it appeared. That is enough to act on, and it avoids
copying the secret into one more place.

### If a key is ever exposed

Rotate first, clean up second. Revoke the key at the provider, issue a new one,
put it in `.env`, and only then work out how it got committed. A key that is
still valid while the history is being rewritten is still a live key.
