# Vendored dependencies

Local copies, not a CDN. The site has zero external references and a
`connect-src 'self'` / `script-src 'self'` CSP; fetching these at runtime from
anywhere else would break both properties.

| package | version | licence | files |
|---|---|---|---|
| anime.js | 4.5.0 | MIT (`anime/LICENSE.md`) | `anime.esm.min.js` (118 KB) |
| three.js | 0.186.1 (r186) | MIT (`three/LICENSE`) | `three.module.js` (667 KB), `three.core.js` (1.45 MB) |

Obtained with `npm pack` and copied verbatim; it has not been edited.

## three.js: removed in phase (c), restored in phase (f)

three.js is vendored for exactly one thing: the glass prism hero. It was
dropped in phase (c) because that hero rendered as an opaque grey slab, and the
2.1 MB went with it. Phase (f) found the cause -- a transmissive material with
no environment map and nothing opaque behind it to refract -- and fixed it, so
the files are back, restored from commit 269d194 rather than re-downloaded.

`three.module.js` imports `./three.core.js`, so the two must stay side by side.

**It is never on the critical path and is never loaded eagerly.** `motion.js`
imports it dynamically only when the hero is near the viewport AND a WebGL 2
context is actually obtained AND motion is not reduced. A reader who never
scrolls to the hero does not request it. Nothing under `examples/jsm` is
vendored, which is why `hero3d.js` writes out its own room environment and its
own bloom pass instead of importing `RoomEnvironment` and `UnrealBloomPass`.
