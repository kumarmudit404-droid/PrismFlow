# Vendored dependencies

Local copies, not a CDN. The site has zero external references and a
`connect-src 'self'` / `script-src 'self'` CSP; fetching these at runtime from
anywhere else would break both properties.

| package | version | licence | files |
|---|---|---|---|
| anime.js | 4.5.0 | MIT (`anime/LICENSE.md`) | `anime.esm.min.js` (118 KB) |

Obtained with `npm pack` and copied verbatim; it has not been edited.

## three.js was here, and is not any more

three.js 0.186.1 was vendored for exactly one thing: a ray-traced glass prism
layered over the hero. That hero was built, measured, found to render
incorrectly, and dropped -- so its 2.1 MB went with it. The prism mark is SVG
and always was; see the hero comment in `web/index.html`. Nothing in the site
imports three.js now.
