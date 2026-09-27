# Vendored dependencies

Local copies, not a CDN. The site has zero external references and a
`connect-src 'self'` / `script-src 'self'` CSP; fetching these at runtime from
anywhere else would break both properties.

| package | version | licence | files |
|---|---|---|---|
| anime.js | 4.5.0 | MIT (`anime/LICENSE.md`) | `anime.esm.min.js` (118 KB) |
| three.js | 0.186.1 | MIT (`three/LICENSE`) | `three.module.js` (663 KB) + `three.core.js` (1.46 MB) |

Both obtained with `npm pack` and copied verbatim; neither has been edited.

`three.module.js` imports `./three.core.js`, so the two must stay side by side.
three.js no longer ships a minified build, and 2.1 MB is a lot to spend on one
hero visual — so it is **lazy-loaded**, imported only when the hero scrolls into
view AND WebGL is actually available AND motion is not reduced. A visitor who
never reaches the hero, or who prefers reduced motion, never downloads it.
