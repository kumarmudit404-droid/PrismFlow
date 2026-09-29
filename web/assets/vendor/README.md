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

## liquid-glass: phase (f) step 3, a technique credit rather than a vendored file

`web/assets/js/liquid_glass.js` ports the real-refraction / rippling-base /
blob-shadow / hand-rolled-bloom technique from
[kunal-chaudhary-design/liquid-buttons](https://github.com/kunal-chaudhary-design/liquid-buttons)
(MIT; full text in `liquid-glass/LICENSE`). No file from that repository is
copied here -- it is a Vite/TypeScript project with its own full-window
showcase scene, three named toggle buttons, per-room background crossfade and
webfont labels, none of which fits a data page with one fixed dark theme. The
port keeps the technique (`MeshPhysicalMaterial` transmission glass, the
slope-correct ripple shader on the base plate, blob shadows instead of a
shadow map, the three-beat slam/bounce/flip interaction) and drops the
showcase-specific parts, fitting the scene to a bounded container the same way
`hero3d.js` fits the prism to `.hero__stage`, and drawing its three labels from
the page's own thesis sentence instead of inventing settings. The bloom pass
is a second, independent copy of `hero3d.js`'s hand-rolled version rather than
a shared import, because `hero3d.js` is a previously-verified module this step
does not touch.
