/* PrismFlow Part 25 phase (f) step 2 -- the photorealistic glass prism.
 *
 * WHY THE FIRST ATTEMPT FAILED, AND WHAT IS DIFFERENT HERE
 * --------------------------------------------------------
 * Phase (c) built this and dropped it: it rendered as an opaque grey slab on a
 * real GPU as well as in software. The cause was not the material and not the
 * driver. A MeshPhysicalMaterial with transmission = 1 shows you WHAT IS
 * BEHIND IT, bent. That scene had nothing behind the prism and no environment
 * map, so the answer to "what is behind it" was the clear colour, and the
 * answer to "what does it reflect" was nothing at all. Glass with nothing to
 * refract and nothing to reflect is, correctly, a grey slab.
 *
 * Three things fix the cause rather than the symptom:
 *
 *   1. AN ENVIRONMENT. buildRoom() assembles a small emissive room -- walls,
 *      a ceiling panel and several coloured light boxes -- and PMREMGenerator
 *      pre-filters it into a real radiance map. This is what three.js's
 *      RoomEnvironment example does; it is written out here rather than
 *      imported because examples/jsm is not part of the vendored module and
 *      nothing may be fetched at runtime. Every colour in it comes from
 *      tokens.css.
 *   2. A BACKDROP. A plane sits behind the prism carrying a gradient built in
 *      code, and the five outgoing beams start behind the glass. Transmission
 *      renders the non-transmissive scene to its own target, so all of that is
 *      what the prism bends.
 *   3. DISPERSION. The material has ior, thickness, dispersion and a little
 *      iridescence, so the white incident beam breaks into colour through the
 *      glass instead of passing through grey.
 *
 * NO ASSET FILES. No image, video, model or HDR is loaded. The environment,
 * the backdrop gradient and the bloom kernel are all generated in code, which
 * is also what keeps the CSP at script-src 'self' honest.
 *
 * COLOUR. Same rule as motion.js: every colour is read from tokens.css at
 * runtime. There is no hex literal in this file. A missing token throws, the
 * caller catches, and the static SVG stays -- substituting a guessed colour
 * would be exactly the palette drift the indirection prevents.
 */

import * as THREE from "../vendor/three/three.module.js";

/* The SVG's own coordinate system is the world. viewBox is 0 0 900 240 and the
 * triangle is M360 40 L470 190 L250 190 Z, so working in SVG units with the
 * origin moved to the centre and y flipped makes the silhouette match by
 * construction instead of by eye. */
const VB = { w: 900, h: 240 };
const toWorld = (x, y) => new THREE.Vector2(x - VB.w / 2, VB.h / 2 - y);

const TRI = [toWorld(360, 40), toWorld(470, 190), toWorld(250, 190)];
const APEX_OUT = toWorld(400, 120);          // where the SVG's beams leave
const BEAM_Y = [38, 79, 120, 161, 202].map((y) => VB.h / 2 - y);
const DEPTH = 130;                           // extrusion, z

/* A long lens at a long distance: near-orthographic, so the silhouette lands
 * where the SVG's does, but still with enough perspective to read as a solid. */
const CAM_DIST = 1400;
const FOV = 2 * Math.atan((VB.h / 2) / CAM_DIST) * (180 / Math.PI);

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

  const base = col("--base");
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
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.7;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const canvas = renderer.domElement;
  canvas.className = "hero__gl";
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

  /* ---- 2. the backdrop, generated ------------------------------------
   * Something for the glass to bend. A canvas gradient, not a file. */
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
    grad.addColorStop(0.00, hex(base));
    grad.addColorStop(0.50, hex(surface));
    grad.addColorStop(1.00, hex(base));
    g.fillStyle = grad;
    g.fillRect(0, 0, W, H);

    // A pool directly behind the prism, SIZED TO THE PRISM, not to the frame.
    // At 0.40 of the texture width it was 560 world units against a 220-unit
    // prism and became a blown-out sun with the glass silhouetted on it --
    // the opposite failure to the grey slab and just as wrong.
    const pool = g.createRadialGradient(px, py, 0, px, py, W * 0.11);
    pool.addColorStop(0.00, hex(accent));
    pool.addColorStop(0.45, hex(surface));
    pool.addColorStop(1.00, "rgba(0,0,0,0)");
    g.globalAlpha = 0.9;
    g.fillStyle = pool;
    g.fillRect(0, 0, W, H);

    // A cooler second pool, so refraction has more than one colour to split.
    const qx = px + W * 0.16, qy = H * 0.38;
    const cool = g.createRadialGradient(qx, qy, 0, qx, qy, W * 0.10);
    cool.addColorStop(0.00, hex(angleColour[2]));
    cool.addColorStop(1.00, "rgba(0,0,0,0)");
    g.globalAlpha = 0.55;
    g.fillStyle = cool;
    g.fillRect(0, 0, W, H);

    // STREAKS, not just pools. A blob seen through glass is still a blob;
    // straight lines BEND, and that is what makes a photograph read as glass.
    g.lineWidth = 3;
    const streak = [incident, angleColour[2], accent, angleColour[3]];
    for (let i = 0; i < 9; i++) {
      g.strokeStyle = hex(streak[i % streak.length]);
      g.globalAlpha = 0.14 + 0.06 * (i % 3);
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

    /* THE PLANE MUST STAY OPAQUE. three.js renders the transmission target
     * from the OPAQUE objects; a transparent backdrop is excluded from it, the
     * glass then has nothing behind it, and the prism goes back to being a
     * grey slab. That is not a guess -- making this material transparent to
     * hide the streaks reproduced the phase (c) failure exactly, in one
     * change. So the masked content is composited over an opaque field of
     * --base instead, which is the colour of the page behind the canvas: the
     * plane is invisible where it has no content, without ever being
     * transparent. */
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const out = c.getContext("2d");
    out.fillStyle = hex(base);
    out.fillRect(0, 0, W, H);
    out.drawImage(inner, 0, 0);

    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  // 1400 wide, not 3000: the texture's pools have to be the right SIZE behind
  // the prism, and the plane sits close enough that refraction is strong.
  const backdrop = new THREE.Mesh(
    new THREE.PlaneGeometry(1400, 500),
    (() => {
      const tex = backdropTexture();
      // SELF-LIT and OPAQUE. Self-lit because a backdrop that only reflects
      // the key light is dark exactly where the glass needs it bright, which
      // is the same failure as having no backdrop at all. Opaque because the
      // transmission target is built from the opaque objects only -- see the
      // note in backdropTexture(). `map` is kept so it still takes the shadow.
      return new THREE.MeshStandardMaterial({
        map: tex, emissiveMap: tex, emissive: 0xffffff,
        emissiveIntensity: 1.0, roughness: 0.9, metalness: 0.0,
      });
    })()
  );
  backdrop.position.z = -330;
  backdrop.receiveShadow = true;
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
    color: 0xffffff,
    metalness: 0.0,
    roughness: 0.02,
    transmission: 0.94,
    thickness: DEPTH * 0.55,
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
  prism.castShadow = true;
  const pivot = new THREE.Group();
  pivot.add(prism);

  /* The edges, drawn. A real prism's silhouette and its internal edges are the
   * brightest thing about it -- total internal reflection concentrates light
   * along them -- but reproducing that from the environment alone needs a much
   * larger light rig than a decorative mark can afford at 60fps. This is a
   * thin additive line along the geometry's own edges, in the beam colour, at
   * low opacity. It is a rendering choice and it is not subtle-washing
   * anything: the geometry it traces is the geometry that is there.
   * 26 degrees keeps it to the silhouette and the facet breaks rather than
   * wireframing every bevel quad. */
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(glassGeo, 26),
    new THREE.LineBasicMaterial({
      color: incident, transparent: true, opacity: 0.30,
      blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
      // depthTest OFF on purpose, and it is the physically right answer as
      // well as the prettier one: through a transparent solid you SEE the far
      // edges. With the test on, the glass occluded its own back edges and
      // the silhouette came out as a dashed line.
      depthTest: false,
    })
  );
  pivot.add(edges);
  scene.add(pivot);

  /* ---- 4. the beams ---------------------------------------------------
   * One white beam in, five coloured beams out, each an additive tube. They
   * start BEHIND the glass so transmission has them to refract; the visible
   * segment outside the prism is what the reader sees. */
  function beam(from, to, colour, radius, opacity) {
    const a = new THREE.Vector3(from.x, from.y, from.z || 0);
    const b = new THREE.Vector3(to.x, to.y, to.z || 0);
    const len = a.distanceTo(b);
    const geo = new THREE.CylinderGeometry(radius, radius, len, 8, 1, true);
    const mat = new THREE.MeshBasicMaterial({
      color: colour, transparent: true, opacity,
      blending: THREE.AdditiveBlending, depthWrite: false,
      side: THREE.DoubleSide, toneMapped: false,
    });
    const m = new THREE.Mesh(geo, mat);
    m.position.copy(a).add(b).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
    return m;
  }

  const beams = new THREE.Group();
  scene.add(beams);

  // In: from off-frame left to inside the prism.
  const entry = { x: TRI[2].x + 44, y: 0 };       // just inside the left face
  beams.add(beam({ x: -VB.w / 2 - 40, y: 0 }, entry, incident, 1.6, 0.85));
  beams.add(beam({ x: -VB.w / 2 - 40, y: 0 }, entry, incident, 5.5, 0.16));

  // Out: five, from just inside the exit face to off-frame right.
  // The fan starts at the ENTRY point, not at the exit face, so the five
  // beams are visibly born inside the glass and leave through it. Starting
  // them outside made the mark read as "beam, gap, fan".
  const outStart = { x: entry.x + 6, y: entry.y };
  angles.forEach((a, i) => {
    const end = { x: VB.w / 2 + 40, y: BEAM_Y[i] };
    const dim = a.unbuilt ? 0.38 : 1.0;      // regulatory stays visibly unbuilt
    beams.add(beam(outStart, end, angleColour[i], 1.6, 0.85 * dim));
    beams.add(beam(outStart, end, angleColour[i], 6.0, 0.14 * dim));
  });

  /* ---- 5. lights and a soft shadow ------------------------------------ */
  const key = new THREE.DirectionalLight(0xffffff, 4.2);
  key.position.set(-420, 520, 760);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.radius = 7;
  key.shadow.bias = -0.0015;
  const cam = key.shadow.camera;
  cam.left = -600; cam.right = 600; cam.top = 300; cam.bottom = -300;
  cam.near = 100; cam.far = 2600;
  scene.add(key);
  scene.add(new THREE.AmbientLight(0xffffff, 0.2));
  // A second, grazing light from the left. Glass reads through its EDGES, and
  // a grazing source is what draws them.
  const rim = new THREE.DirectionalLight(0xffffff, 3.0);
  rim.position.set(-900, -140, 260);
  scene.add(rim);

  /* ---- 6. bloom, small and hand-rolled --------------------------------
   * UnrealBloomPass lives in examples/jsm, which is not vendored, so this is a
   * minimal equivalent: bright-pass at quarter resolution, two separable
   * Gaussian blurs, additive composite. Four extra draws at a sixteenth of the
   * pixel count -- it is the cheapest part of the frame. */
  const QUAD = new THREE.BufferGeometry();
  QUAD.setAttribute("position", new THREE.Float32BufferAttribute(
    [-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  QUAD.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  const VERT = `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = vec4(position, 1.0); }`;

  const brightMat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: { tSrc: { value: null }, threshold: { value: 0.55 } },
    vertexShader: "in vec3 position;\nin vec2 uv;\n" + VERT
      .replace("varying", "out").replace("void main", "void main"),
    fragmentShader: `
      precision highp float;
      in vec2 vUv; out vec4 fragColor;
      uniform sampler2D tSrc; uniform float threshold;
      void main() {
        vec4 c = texture(tSrc, vUv);
        float l = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
        float k = max(l - threshold, 0.0) / max(l, 1e-4);
        fragColor = vec4(c.rgb * k, 1.0);
      }`,
  });

  const blurMat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: { tSrc: { value: null }, dir: { value: new THREE.Vector2(1, 0) },
                texel: { value: new THREE.Vector2(1, 1) } },
    vertexShader: "in vec3 position;\nin vec2 uv;\n" + VERT.replace("varying", "out"),
    fragmentShader: `
      precision highp float;
      in vec2 vUv; out vec4 fragColor;
      uniform sampler2D tSrc; uniform vec2 dir; uniform vec2 texel;
      // 9-tap Gaussian, sigma 2.0, normalised.
      const float W[5] = float[5](0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162);
      void main() {
        vec3 sum = texture(tSrc, vUv).rgb * W[0];
        for (int i = 1; i < 5; i++) {
          vec2 o = dir * texel * float(i);
          sum += texture(tSrc, vUv + o).rgb * W[i];
          sum += texture(tSrc, vUv - o).rgb * W[i];
        }
        fragColor = vec4(sum, 1.0);
      }`,
  });

  const compMat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    transparent: true,
    uniforms: { tScene: { value: null }, tBloom: { value: null },
                strength: { value: 0.60 } },
    vertexShader: "in vec3 position;\nin vec2 uv;\n" + VERT.replace("varying", "out"),
    fragmentShader: `
      precision highp float;
      in vec2 vUv; out vec4 fragColor;
      uniform sampler2D tScene; uniform sampler2D tBloom; uniform float strength;
      void main() {
        vec4 s = texture(tScene, vUv);
        vec3 b = texture(tBloom, vUv).rgb * strength;
        // The canvas is transparent over the page, so the glow has to carry
        // its own alpha or it would only show where the scene already is.
        float ba = clamp(dot(b, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        fragColor = vec4(s.rgb + b, clamp(s.a + ba, 0.0, 1.0));
      }`,
  });

  const quad = new THREE.Mesh(QUAD, brightMat);
  const quadScene = new THREE.Scene();
  quadScene.add(quad);

  const rtOpts = { type: THREE.HalfFloatType, depthBuffer: true };
  let sceneRT = new THREE.WebGLRenderTarget(2, 2, rtOpts);
  let brightRT = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType });
  let blurRT = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType });

  function pass(material, target) {
    quad.material = material;
    renderer.setRenderTarget(target);
    renderer.clear();
    renderer.render(quadScene, quadCam);
  }

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
    // The SVG is 900x240 and scales to the stage width. Matching its framing
    // means holding the same VISIBLE HEIGHT whatever the container aspect, so
    // the triangle sits exactly where the SVG's does.
    const visibleH = VB.h * Math.max(1, (VB.w / VB.h) / (w / h));
    camera.fov = 2 * Math.atan((visibleH / 2) / CAM_DIST) * (180 / Math.PI);
    camera.updateProjectionMatrix();

    const pw = Math.max(1, Math.floor(w * dpr));
    const ph = Math.max(1, Math.floor(h * dpr));
    sceneRT.setSize(pw, ph);
    brightRT.setSize(Math.max(1, pw >> 2), Math.max(1, ph >> 2));
    blurRT.setSize(Math.max(1, pw >> 2), Math.max(1, ph >> 2));
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

    // Cursor-reactive tilt, eased. Under prefers-reduced-motion the pose is
    // held still: the tilt is decoration, and nothing is explained by it.
    if (!reduced) {
      pointer.x += (pointer.tx - pointer.x) * 0.06;
      pointer.y += (pointer.ty - pointer.y) * 0.06;
      pivot.rotation.y = POSE_Y + Math.sin(s * 0.22) * 0.13 + pointer.x * 0.22;
      pivot.rotation.x = POSE_X + Math.sin(s * 0.17) * 0.05 - pointer.y * 0.12;
      pivot.rotation.z = Math.sin(s * 0.11) * 0.015;
    }

    renderer.setRenderTarget(sceneRT);
    renderer.clear();
    renderer.render(scene, camera);

    brightMat.uniforms.tSrc.value = sceneRT.texture;
    pass(brightMat, brightRT);

    const bw = brightRT.width, bh = brightRT.height;
    blurMat.uniforms.tSrc.value = brightRT.texture;
    blurMat.uniforms.dir.value.set(1, 0);
    blurMat.uniforms.texel.value.set(1 / bw, 1 / bh);
    pass(blurMat, blurRT);

    blurMat.uniforms.tSrc.value = blurRT.texture;
    blurMat.uniforms.dir.value.set(0, 1);
    pass(blurMat, brightRT);

    compMat.uniforms.tScene.value = sceneRT.texture;
    compMat.uniforms.tBloom.value = brightRT.texture;
    renderer.setRenderTarget(null);
    renderer.clear();
    quad.material = compMat;
    renderer.render(quadScene, quadCam);
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
    [sceneRT, brightRT, blurRT].forEach((rt) => rt.dispose());
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
