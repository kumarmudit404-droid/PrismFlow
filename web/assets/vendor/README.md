# Vendored dependencies

Local copies, not a CDN. The site has zero external references and a
`connect-src 'self'` / `script-src 'self'` CSP; fetching these at runtime from
anywhere else would break both properties.

| package | version | licence | files |
|---|---|---|---|
| three.js | 0.186.1 (r186) | MIT (`three/LICENSE`) | `three.module.js` (667 KB), `three.core.js` (1.45 MB) |

Obtained with `npm pack` and copied verbatim; it has not been edited.

## anime.js: removed 2026-11-05

anime.js was vendored for one thing, the scroll reveals, and they are a CSS
class now: `motion.js` adds `.is-visible` and sets `--i`, and the curve, the
offset and the stagger are declarations in `app.css`. Nothing imported the
library any more, so the 118 KB and its licence file are gone rather than
shipped unused. The reveal guarantee is unaffected -- it never depended on
anime.js; see `reveal-failsafe.js`.

## three.js: removed in phase (c), restored in phase (f)

three.js is vendored for exactly one thing: the glass prism hero. It was
dropped in phase (c) because that hero rendered as an opaque grey slab, and the
2.1 MB went with it. Phase (f) found the cause -- a transmissive material with
no environment map and nothing opaque behind it to refract -- and fixed it, so
the files are back, restored from commit 269d194 rather than re-downloaded.

`three.module.js` imports `./three.core.js`, so the two must stay side by side.

**It is never on the critical path and is never loaded eagerly.** `motion.js`
imports it dynamically only when a WebGL 2 context is actually obtained AND
motion is not reduced. Nothing under `examples/jsm` is
vendored, which is why `hero3d.js` writes out its own room environment instead
of importing `RoomEnvironment`. It used to hand-roll a bloom pass for the same
reason; that is gone, for reasons recorded in `web/README.md`.
