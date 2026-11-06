/* The photorealistic glass prism, anchored in the hero over the background
 * image. It used to be the site-wide fixed background; the five coloured beams
 * it drew are baked into that image now, so they are gone from here -- drawing
 * them twice put a second, differently-angled fan over the photograph's.
 *
 * An earlier attempt rendered as an opaque grey slab. The cause was not the
 * material: a MeshPhysicalMaterial with transmission = 1 shows what is behind
 * it, bent, and that scene had nothing behind the prism and no environment
 * map. Glass with nothing to refract and nothing to reflect is correctly a
 * grey slab. Three things fix that cause:
 *
 *   1. An environment. buildRoom() assembles a small emissive room and
 *      PMREMGenerator pre-filters it into a real radiance map. It is written
 *      out here rather than imported because examples/jsm is not part of the
 *      vendored module and nothing may be fetched at runtime.
 *   2. A backdrop. A plane behind the prism carries a generated gradient and
 *      the five beams start behind the glass, so transmission has something
 *      to bend.
 *   3. Dispersion. ior, thickness, dispersion and a little iridescence, so
 *      the white incident beam breaks into colour.
 *
 * No asset files: no image, video, model or HDR is loaded. The environment,
 * the gradient and the bloom kernel are generated in code, which is what keeps
 * script-src 'self' honest.
 *
 * Every colour is read from tokens.css at runtime; there is no hex literal in
 * this file. A missing token throws, the caller catches, and the static SVG
 * stays -- a guessed colour would be the palette drift we are avoiding.
 */

import * as THREE from "../vendor/three/three.module.js";

/* The SVG's own coordinate system is the world. The triangle is
 * M360 40 L470 190 L250 190 Z, so working in SVG units with the origin moved
 * to the triangle's centroid and y flipped makes the silhouette match by
 * construction instead of by eye.
 *
 * FIT is the SVG fallback's viewBox, "235 35 250 210", which is the triangle
 * plus margin and is centred on the same centroid. Framing GL to the same box
 * is what makes the swap between the two invisible. */
const CEN = { x: 360, y: 140 };              // the triangle's centroid
const FIT = { w: 250, h: 210 };
const toWorld = (x, y) => new THREE.Vector2(x - CEN.x, CEN.y - y);

const TRI = [toWorld(360, 40), toWorld(470, 190), toWorld(250, 190)];
const DEPTH = 130;                           // extrusion, z

/* A long lens at a long distance: near-orthographic, so the silhouette lands
 * where the SVG's does, but still with enough perspective to read as a solid. */
const CAM_DIST = 1400;
const FOV = 2 * Math.atan((FIT.h / 2) / CAM_DIST) * (180 / Math.PI);

export async function mount(host, api) {
  const { palette, token } = api;
  const angles = palette();
  if (angles.length < 5) {
    throw new Error("expected 5 angle tokens, got " + angles.length);
  }

  const col = (name) => {
    const v = token(name);
    if (!v) { throw new Error("missing token " + name); }
    return new THREE.Color(v);
  };

  /* The six graphics-only spectrum stops. A missing one throws, the caller
   * catches, and the static SVG stays -- a guessed colour would be the palette
   * drift this indirection exists to prevent. */
  const spectrum = [1, 2, 3, 4, 5, 6].map((i) => col("--spec-" + i));

  const field = col("--prism-backdrop");
  const surface = col("--surface-raised");
  const accent = col("--accent");
  const incident = col("--beam-incident");

  /* palette() hands back the RAW token strings, because that is what SVG and
   * canvas want. Everything in here wants a THREE.Color, so they are converted
   * once, here, rather than at each use -- mixing the two is what threw
   * "getHexString is not a function" and dropped the whole hero to the SVG
   * fallback on the first real-GPU run. */
  const angleColour = angles.map((a) => new THREE.Color(a.colour));

  /* ---- renderer ------------------------------------------------------ */
  const renderer = new THREE.WebGLRenderer({
    antialias: true, alpha: true, powerPreference: "high-performance",
  });
  renderer.setClearColor(0x000000, 0);
  // NO TONE MAPPING. ACES was right for a near-black page: it compressed the
  // beams' and specular's highlights so they did not clip. There is nothing
  // to compress here and one thing it actively breaks -- ACES maps linear 1.0
  // to about 0.8, so the white backdrop plane, and the glass sampling it
  // through transmission, both came out grey. The old bloom pass was adding
  // that 20% back by accident; with the bloom gone the grey slab was all that
  // was left. Colours come out as tokens.css authored them.
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // Shadows off. Nothing receives one any more: the backdrop plane is unlit
  // so the glass could not cast onto it, and the contact shadow under the
  // prism is a CSS radial gradient in app.css. A shadow map rendered every
  // frame for nothing is the most expensive no-op in the file.
  renderer.shadowMap.enabled = false;

  const canvas = renderer.domElement;
  canvas.className = "prism-bg__gl";
  canvas.setAttribute("aria-hidden", "true");

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 1, 10, 6000);
  camera.position.set(0, 0, CAM_DIST);

  /* ---- 1. the environment, generated ---------------------------------
   * Emissive boxes in a dark room. PMREM turns it into the radiance map the
   * glass reflects and refracts; without it the material has nothing to sample
   * and the prism goes grey, which is the bug this replaces. */
  function buildRoom() {
    const room = new THREE.Scene();
    const box = new THREE.BoxGeometry();

    const lit = (colour, intensity) => new THREE.MeshStandardMaterial({
      color: 0x000000, emissive: colour, emissiveIntensity: intensity,
      side: THREE.DoubleSide,
    });

    const put = (mat, pos, scale) => {
      const m = new THREE.Mesh(box, mat);
      m.position.set(pos[0], pos[1], pos[2]);
      m.scale.set(scale[0], scale[1], scale[2]);
      room.add(m);
      return m;
    };

    // The shell, barely lit. It is a floor for the reflections, not a source:
    // a bright shell fills the whole radiance map evenly, which flattens the
    // specular instead of sharpening it.
    put(lit(surface, 0.45), [0, 0, 0], [2400, 1400, 2400]);

    // THESE INTENSITIES ARE HIGH ON PURPOSE. The page is near-black, so a
    // glass solid is read almost entirely through its edges and its specular
    // highlights; a dim environment gives a dark shape on a dark page, which
    // is the grey-slab failure wearing a different colour. Measured on this
    // laptop's Intel UHD: at the first values the prism was invisible in a
    // real-GPU screenshot.
    // BIG AND CLOSE, not small and bright. Specular sharpness comes from the
    // SOLID ANGLE a source subtends, so a small panel far away contributes an
    // almost invisible highlight however high its intensity is. These are the
    // sizes at which the prism's edges finally read on this laptop's GPU.
    put(lit(incident, 16.0), [-360, 330, 330], [700, 150, 700]);    // key, above-left
    put(lit(accent, 12.0), [-460, -60, 240], [180, 460, 520]);      // warm side wall
    put(lit(angleColour[2], 9.0), [500, 120, 220], [170, 480, 520]); // cool side wall
    put(lit(angleColour[3], 5.0), [0, -420, 300], [900, 150, 420]);  // low bounce
    // Behind, so the back facets light up through the solid.
    put(lit(incident, 9.0), [-200, 40, -560], [420, 300, 30]);
    put(lit(angleColour[1], 6.0), [320, -90, -560], [300, 260, 30]);

    return room;
  }

  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  const room = buildRoom();
  const envRT = pmrem.fromScene(room, 0.035);
  scene.environment = envRT.texture;
  room.traverse((o) => { if (o.geometry) { o.geometry.dispose(); } });
  pmrem.dispose();

  /* ---- 2. the refraction patch, generated -----------------------------
   * Something for the glass to bend: a canvas gradient, not a file.
   *
   * Do not delete this geometry to "show only the prism". three.js builds the
   * transmission render target from the opaque objects only, so a transmissive
   * solid with nothing opaque behind it refracts the clear colour and renders
   * as a grey slab -- the original bug.
   *
   * Instead the plane is cut to roughly the prism's own footprint so it no
   * longer spans the frame, and its field is painted so that it is invisible
   * against the page by construction.
   *
   * That field is --prism-backdrop, which is the pixel value the hero image
   * actually HAS at the anchor: (252,252,252), measured off the committed
   * JPG. --surface's pure white and --base's #F7F8FA were both tried and both
   * showed as a faint card, one lighter than the photograph and one greyer.
   */
  function backdropTexture() {
    const W = 512, H = 256;
    const hex = (x) => "#" + x.getHexString();

    // Where the prism sits on this plane, in texture pixels.
    const px = W * (0.5 + (TRI[0].x / 1400));
    const py = H * 0.52;

    /* The content, on its own transparent canvas so it can be masked without
     * putting alpha into the material. */
    const inner = document.createElement("canvas");
    inner.width = W; inner.height = H;
    const g = inner.getContext("2d");

    const grad = g.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0.00, hex(field));
    grad.addColorStop(0.50, hex(surface));
    grad.addColorStop(1.00, hex(field));
    g.fillStyle = grad;
    g.fillRect(0, 0, W, H);

    /* NO POOLS. Two radial pools used to sit behind the glass to give it
     * something bright to bend. On a near-black page that worked; on this one
     * the field is already white, so the only thing a coloured pool can be is
     * DARKER than its surroundings, and refracted through the prism the two of
     * them came out as a pair of black eyes on the front face. What the glass
     * needs here is structure, not brightness, and the streaks below are that.
     */

    // STREAKS. A blob seen through glass is still a blob; straight lines
    // BEND, and that is what makes a photograph read as glass.
    g.lineWidth = 3;
    const streak = [spectrum[0], spectrum[3], spectrum[5], spectrum[4]];
    for (let i = 0; i < 9; i++) {
      g.strokeStyle = hex(streak[i % streak.length]);
      g.globalAlpha = 0.20 + 0.08 * (i % 3);
      g.beginPath();
      const x0 = (i / 9) * W * 1.4 - W * 0.2;
      g.moveTo(x0, -20);
      g.lineTo(x0 + W * 0.16, H + 20);
      g.stroke();
    }
    g.globalAlpha = 1;

    // Keep only what sits behind the prism. Everything else is just streaks
    // on the page, and it turns the hero canvas into a visible rectangle.
    g.globalCompositeOperation = "destination-in";
    const mask = g.createRadialGradient(px, py, 0, px, py, W * 0.17);
    mask.addColorStop(0.00, "rgba(0,0,0,1)");
    mask.addColorStop(0.55, "rgba(0,0,0,0.92)");
    mask.addColorStop(1.00, "rgba(0,0,0,0)");
    g.fillStyle = mask;
    g.fillRect(0, 0, W, H);

    /* The plane must stay opaque. three.js renders the transmission target from
     * the opaque objects; a transparent backdrop is excluded from it, the glass
     * then has nothing behind it, and the prism goes back to being a grey slab.
     * Making this material transparent to hide the streaks reproduced that
     * failure exactly. The masked content is composited over an opaque field of
     * --prism-backdrop instead, so the plane is invisible where it has no
     * content without ever being transparent. */
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const out = c.getContext("2d");
    out.fillStyle = hex(field);
    out.fillRect(0, 0, W, H);
    out.drawImage(inner, 0, 0);

    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  // 260x200, down from 560x330. The camera frames 250x210 now rather than the
  // old 900x240 strip. Wider than the prism's 220 units -- refraction has to
  // have somewhere to sample from at every tilt in step 5's range -- and
  // inside the 287x241 frustum at its own depth, so no edge of it is ever on
  // screen. Taller than it is tall-enough-for-the-prism on purpose: at 200 the
  // refraction under the bottom bevel bent off the edge of the plane, sampled
  // the clear colour and drew a dark strip along the base of the glass.
  const backdrop = new THREE.Mesh(
    new THREE.PlaneGeometry(270, 228),
    (() => {
      const tex = backdropTexture();
      // UNLIT and OPAQUE. Unlit because the field has to come out at exactly
      // the texture's own white: that is what makes the plane disappear
      // against the hero image's 252-253 grey, and anything that multiplies
      // it by a light rig or by ACES tone mapping renders it as a grey card
      // instead. Opaque because the transmission target is built from the
      // opaque objects only -- see the note in backdropTexture().
      return new THREE.MeshBasicMaterial({ map: tex, toneMapped: false });
    })()
  );
  backdrop.position.z = -210;
  scene.add(backdrop);

  /* ---- 3. the prism --------------------------------------------------- */
  const shape = new THREE.Shape();
  shape.moveTo(TRI[0].x, TRI[0].y);
  shape.lineTo(TRI[1].x, TRI[1].y);
  shape.lineTo(TRI[2].x, TRI[2].y);
  shape.closePath();

  const glassGeo = new THREE.ExtrudeGeometry(shape, {
    depth: DEPTH, bevelEnabled: true, bevelThickness: 11, bevelSize: 9,
    bevelSegments: 5, curveSegments: 1,
  });
  glassGeo.translate(0, 0, -DEPTH / 2);
  glassGeo.computeVertexNormals();

  const glass = new THREE.MeshPhysicalMaterial({
    // A faint cool tint, from --prism-tint. Clear glass on a white page is
    // invisible -- which is exactly how this looked before -- and this is what
    // gives the solid a body without making it a coloured object.
    color: col("--prism-tint"),
    metalness: 0.0,
    roughness: 0.05,
    transmission: 0.95,
    // The brief's "thickness 1.2" is a figure for a unit-scale scene. This one
    // works in SVG units where the prism is 220 across and DEPTH is 130, so
    // the ratio is what carries over, not the number.
    thickness: DEPTH * 1.2,
    ior: 1.52,                 // crown glass
    // The white beam has to come out coloured. Without this the prism is
    // physically a clear block and the whole mark loses its point.
    dispersion: 3.2,
    iridescence: 0.35,
    iridescenceIOR: 1.32,
    iridescenceThicknessRange: [120, 420],
    clearcoat: 1.0,
    clearcoatRoughness: 0.02,
    envMapIntensity: 4.0,
    specularIntensity: 1.0,
    specularColor: new THREE.Color(0xffffff),
    attenuationDistance: 6000,
    attenuationColor: new THREE.Color(0xffffff),
  });

  const prism = new THREE.Mesh(glassGeo, glass);
  const pivot = new THREE.Group();
  pivot.add(prism);

  /* The edges, drawn. A real prism's silhouette and internal edges are the
   * brightest thing about it, but reproducing that from the environment alone
   * needs a larger light rig than a decorative mark can afford at 60fps. This
   * is a thin line along the geometry's own edges, cycling the six --spec-*
   * tokens so the rim reads as an iridescent Fresnel edge rather than one
   * colour. 26 degrees keeps it to the silhouette and the facet breaks rather
   * than wireframing every bevel quad.
   *
   * NOT additive here. Additive blending adds light, and on a near-white page
   * there is no headroom left to add any -- the rim disappeared completely. */
  const edgeMat = new THREE.LineBasicMaterial({
      color: spectrum[0].clone(), transparent: true, opacity: 0.55,
      depthWrite: false, toneMapped: false,
      // depthTest OFF on purpose, and it is the physically right answer as
      // well as the prettier one: through a transparent solid you SEE the far
      // edges. With the test on, the glass occluded its own back edges and
      // the silhouette came out as a dashed line.
      depthTest: false,
  });
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(glassGeo, 26), edgeMat);
  pivot.add(edges);
  scene.add(pivot);

  /* ---- 4. the beams: NOT DRAWN HERE ------------------------------------
   * The five coloured beams and the white incident beam are painted into
   * web/assets/img/bg-spectrum-light.jpg, which this canvas is anchored over.
   * The prism is positioned so the photograph's fan converges inside the
   * glass; a second live fan on top of it read as two overlapping marks. */

  /* ---- 5. lights ------------------------------------------------------ */
  const key = new THREE.DirectionalLight(0xffffff, 4.2);
  key.position.set(-420, 520, 760);
  scene.add(key);
  scene.add(new THREE.AmbientLight(0xffffff, 0.2));
  // A second, grazing light from the left. Glass reads through its EDGES, and
  // a grazing source is what draws them.
  const rim = new THREE.DirectionalLight(0xffffff, 3.0);
  rim.position.set(-900, -140, 260);
  scene.add(rim);

  /* ---- 6. bloom: REMOVED ----------------------------------------------
   * There used to be a hand-rolled bloom here -- bright-pass, two separable
   * blurs, additive composite -- plus a PEAK_LUM cap in the composite that
   * bounded the luminance of every pixel the canvas emitted.
   *
   * Both existed for a near-black page. On this one they are worse than
   * useless. There is no darkness for a glow to read against, and the
   * bright-pass threshold of 0.55 was below the luminance of the backdrop
   * plane's own white field, so bloom lifted the whole plane to 255,255,255
   * and drew it as a visible white card over the photograph. Four extra draws
   * a frame to make the canvas worse.
   *
   * The scene now renders straight to the canvas. If a glow is ever wanted
   * back, it needs a threshold above the field's luminance (0.973 for
   * --prism-backdrop), not the old 0.55.
   */

  /* ---- 7. sizing ------------------------------------------------------ */
  let W = 0, H = 0, DPR = 1;

  function resize() {
    const r = host.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width));
    const h = Math.max(1, Math.round(r.height));
    // Cap the device pixel ratio. A 3x panel would quadruple the fragment
    // cost of a decorative mark for no readable gain.
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (w === W && h === H && dpr === DPR) { return; }
    W = w; H = h; DPR = dpr;

    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // The SVG scales to the stage. Matching its framing means holding the
    // same VISIBLE HEIGHT whatever the container aspect, so the triangle sits
    // exactly where the SVG's does.
    const visibleH = FIT.h * Math.max(1, (FIT.w / FIT.h) / (w / h));
    camera.fov = 2 * Math.atan((visibleH / 2) / CAM_DIST) * (180 / Math.PI);
    camera.updateProjectionMatrix();

  }

  /* ---- 8. motion ------------------------------------------------------ */
  const pointer = { x: 0, y: 0, tx: 0, ty: 0 };
  function onMove(e) {
    const r = host.getBoundingClientRect();
    pointer.tx = ((e.clientX - r.left) / Math.max(1, r.width) - 0.5) * 2;
    pointer.ty = ((e.clientY - r.top) / Math.max(1, r.height) - 0.5) * 2;
  }
  window.addEventListener("pointermove", onMove, { passive: true });

  // The resting pose. Off-axis, so a side face is visible.
  const POSE_Y = -0.30;
  const POSE_X = 0.10;

  let reduced = false;
  function frame(t) {
    resize();
    const s = t * 0.001;

    // A slow continuous Y rotation at 0.15 rad/s with a slight wobble, plus
    // the cursor-reactive tilt, eased. Under prefers-reduced-motion the pose
    // is held still: the tilt is decoration, and nothing is explained by it.
    if (!reduced) {
      pointer.x += (pointer.tx - pointer.x) * 0.06;
      pointer.y += (pointer.ty - pointer.y) * 0.06;
      pivot.rotation.y = POSE_Y + s * 0.15 + Math.sin(s * 0.37) * 0.06 + pointer.x * 0.22;
      pivot.rotation.x = POSE_X + Math.sin(s * 0.17) * 0.05 - pointer.y * 0.12;
      pivot.rotation.z = Math.sin(s * 0.11) * 0.015;

      // The rim cycles the six spectrum stops, one full pass every ~12s.
      const f = (s / 12) % 1 * spectrum.length;
      const i = Math.floor(f);
      edgeMat.color.copy(spectrum[i])
             .lerp(spectrum[(i + 1) % spectrum.length], f - i);
    }

    renderer.setRenderTarget(null);
    renderer.clear();
    renderer.render(scene, camera);
  }

  function setReduced(v) {
    reduced = v;
    if (v) { pivot.rotation.set(POSE_X, POSE_Y, 0); }
  }

  function dispose() {
    window.removeEventListener("pointermove", onMove);
    scene.traverse((o) => {
      if (o.geometry) { o.geometry.dispose(); }
      if (o.material) {
        (Array.isArray(o.material) ? o.material : [o.material])
          .forEach((m) => m.dispose());
      }
    });
    envRT.dispose();
    renderer.dispose();
    canvas.remove();
  }

  pivot.rotation.set(POSE_X, POSE_Y, 0);

  host.appendChild(canvas);
  resize();
  frame(0);

  // Reported back so the harness can prove which adapter drew this, rather
  // than a screenshot being taken on SwiftShader and called a GPU test.
  const gl = renderer.getContext();
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  const info = {
    renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : "unavailable",
    vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : "unavailable",
    version: gl.getParameter(gl.VERSION),
    three: THREE.REVISION,
  };

  return { canvas, frame, dispose, setReduced, info };
}
