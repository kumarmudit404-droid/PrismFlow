/* PrismFlow Part 25 phase (f) step 3 -- liquid glass buttons.
 *
 * WHAT THIS IS AND WHY IT LOOKS LIKE hero3d.js
 * ---------------------------------------------
 * The technique -- real refraction via MeshPhysicalMaterial transmission, a
 * base plate that ripples with a slope-correct normal so the wave actually
 * catches light, blob shadows instead of a shadow map, and a hand-rolled
 * bloom pass -- is adapted from kunal-chaudhary-design/liquid-buttons (MIT;
 * see web/assets/vendor/liquid-glass/LICENSE). That project is a full-window
 * showcase of three specific toggle buttons with per-button rooms, icons and
 * a webfont label. This file ports the GLASS TECHNIQUE, not the showcase: it
 * is fitted to a bounded container like the hero prism is, it draws its three
 * labels from the page's own thesis sentence ("Agreement", "Independence",
 * "Evidence" -- see the hero copy) instead of invented settings, and it drops
 * the per-room background/light crossfade because this site has one dark
 * theme, not a light-room/dark-room pair to cut between.
 *
 * Nothing here is the accessible control. Three real <button data-liquid-idx>
 * elements in index.html are that, always -- clickable and keyboard-operable
 * whether or not this module ever loads. This module only paints a synced
 * visualisation behind/around them and forwards its own pointer hits back
 * through api.onPick, so there is exactly one place (motion.js's
 * chooseLiquid) that owns which button is selected.
 *
 * SAME DISCIPLINE AS THE HERO: examples/jsm is not vendored, so the bloom
 * pass below is written out against three's core API, duplicated from
 * hero3d.js rather than imported from it -- hero3d.js is a previously-
 * verified module and this file does not touch it.
 */

import * as THREE from "../vendor/three/three.module.js";

const BUTTONS = [
  { key: "agreement", label: "Agreement", token: "--angle-market" },
  { key: "independence", label: "Independence", token: "--angle-tech" },
  { key: "evidence", label: "Evidence", token: "--accent" },
];

// ---- geometry constants, carried over from the reference port -----------
const PILL_LEN = 2.5, PILL_WID = 0.8, PILL_THICK = 0.23, PILL_BEVEL = 0.111;
const PILL_GAP_X = 0.46, PILL_GAP_Z = 0.94;
const TRAY_PAD = 0.34, TRAY_THICK = 0.1, TRAY_SINK = 0.05;
const PLATFORM_W = 10, PLATFORM_D = 7.4, PLATFORM_THICK = 0.55;
const GROUP_YAW = -0.44;

const HOVER_DELAY = 0.1, HOVER_LIFT = 0.3, HOVER_TIME = 0.26, HOVER_DROP_TIME = 0.26 / 1.5;
const SLAM_TIME = 0.11, BOUNCE_TIME = 0.22, FLIP_TIME = 0.62;

const RIPPLE_AMP = 0.1, RIPPLE_FREQ = 13, RIPPLE_SPEED = 7.5, RIPPLE_FALLOFF = 1.05, RIPPLE_LIFE = 1.15;

const FIT_W = 8.2, FIT_H = 6.4;
const CAM_FOV = 26;
const VIEW_DIR = new THREE.Vector3(0.02, 0.79, 0.61).normalize();

const GLOW_INTENSITY = 3.2;

const easeInOut = (x) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
const clamp01 = (x) => Math.min(1, Math.max(0, x));

export async function mount(host, api) {
  const { token, onPick } = api;

  const col = (name) => {
    const v = token(name);
    if (!v) { throw new Error("missing token " + name); }
    return new THREE.Color(v);
  };
  const base = col("--base");
  const surface = col("--surface-raised");
  const platTone = col("--hairline");
  const labelInk = col("--text-primary");
  const cfgColour = BUTTONS.map((b) => col(b.token));

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const canvas = renderer.domElement;
  canvas.className = "liquid-stage__gl";
  canvas.setAttribute("aria-hidden", "true");

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(CAM_FOV, 1, 0.1, 100);

  function fitCamera(aspect) {
    const vFov = THREE.MathUtils.degToRad(camera.fov);
    const distH = FIT_H / 2 / Math.tan(vFov / 2);
    const distW = FIT_W / 2 / (Math.tan(vFov / 2) * aspect);
    camera.position.copy(VIEW_DIR).multiplyScalar(Math.max(distH, distW) * 1.04);
    camera.lookAt(0, 0, 0);
  }

  /* ---- a small, cheap environment -------------------------------------
   * Not the hero's 7-box dramatic rig: this sits low in a data page, not at
   * the top of it, and the glass here only needs plausible reflections, not
   * a showpiece. */
  function buildEnv() {
    const room = new THREE.Scene();
    const box = new THREE.BoxGeometry();
    const lit = (colour, intensity) => new THREE.MeshStandardMaterial({
      color: 0x000000, emissive: colour, emissiveIntensity: intensity, side: THREE.DoubleSide,
    });
    const put = (mat, pos, scale) => {
      const m = new THREE.Mesh(box, mat);
      m.position.set(pos[0], pos[1], pos[2]);
      m.scale.set(scale[0], scale[1], scale[2]);
      room.add(m);
    };
    put(lit(surface, 0.5), [0, 0, 0], [40, 24, 40]);
    put(lit(labelInk, 3.0), [-6, 8, 5], [10, 3, 10]);
    put(lit(cfgColour[2], 2.2), [6, 2, 4], [3, 8, 8]);
    return room;
  }
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  const env = buildEnv();
  const envRT = pmrem.fromScene(env, 0.05);
  scene.environment = envRT.texture;
  env.traverse((o) => { if (o.geometry) { o.geometry.dispose(); } });
  pmrem.dispose();

  const key = new THREE.DirectionalLight(0xffffff, 1.55);
  key.position.set(-3.4, 7.5, 3.2);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xffffff, 0.45);
  rim.position.set(4, 3, -4.5);
  scene.add(rim);
  const glowLight = new THREE.PointLight(cfgColour[0].getHex(), 0, 9, 2);
  glowLight.layers.set(1);
  scene.add(glowLight);

  // ---- shapes, ported verbatim (they are the technique, not the demo) ----
  function roundedRectPath(w, h, r, path) {
    const x = -w / 2, y = -h / 2, rr = Math.min(r, h / 2, w / 2);
    path.moveTo(x + rr, y);
    path.lineTo(x + w - rr, y);
    path.quadraticCurveTo(x + w, y, x + w, y + rr);
    path.lineTo(x + w, y + h - rr);
    path.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    path.lineTo(x + rr, y + h);
    path.quadraticCurveTo(x, y + h, x, y + h - rr);
    path.lineTo(x, y + rr);
    path.quadraticCurveTo(x, y, x + rr, y);
    return path;
  }
  function slab(w, h, depth, radius, bevel, curveSeg = 48) {
    const shape = roundedRectPath(w, h, radius, new THREE.Shape());
    const geo = new THREE.ExtrudeGeometry(shape, {
      depth: Math.max(0.001, depth - bevel * 2), bevelEnabled: true,
      bevelThickness: bevel, bevelSize: bevel, bevelSegments: 6, curveSegments: curveSeg,
    });
    geo.rotateX(-Math.PI / 2);
    geo.center();
    geo.computeVertexNormals();
    return geo;
  }

  const group = new THREE.Group();
  group.rotation.y = GROUP_YAW;
  scene.add(group);

  const trayW = PILL_LEN + PILL_GAP_X * 2 + TRAY_PAD * 2;
  const trayD = PILL_WID + PILL_GAP_Z * 2 + TRAY_PAD * 2;

  const platShape = roundedRectPath(PLATFORM_W, PLATFORM_D, 0.4, new THREE.Shape());
  const hole = roundedRectPath(trayW, trayD, 0.5, new THREE.Path());
  platShape.holes.push(hole);
  const platGeo = new THREE.ExtrudeGeometry(platShape, {
    depth: PLATFORM_THICK, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.02,
    bevelSegments: 3, curveSegments: 48,
  });
  platGeo.rotateX(-Math.PI / 2);
  const platform = new THREE.Mesh(platGeo, new THREE.MeshStandardMaterial({
    color: platTone, roughness: 0.72, metalness: 0,
  }));
  group.add(platform);

  function roundedRectSDF(px, pz, hw, hh, r) {
    const qx = Math.abs(px) - (hw - r), qz = Math.abs(pz) - (hh - r);
    const outside = Math.hypot(Math.max(qx, 0), Math.max(qz, 0));
    return outside + Math.min(Math.max(qx, qz), 0) - r;
  }
  function roundedRectDisc(w, h, r, rings, segs) {
    const hw = w / 2, hh = h / 2;
    const pos = [], nor = [], idx = [];
    for (let s = 0; s <= segs; s++) {
      const th = (s / segs) * Math.PI * 2, dx = Math.cos(th), dz = Math.sin(th);
      let lo = 0, hi = Math.max(hw, hh) * 1.6;
      for (let it = 0; it < 26; it++) {
        const mid = (lo + hi) / 2;
        if (roundedRectSDF(dx * mid, dz * mid, hw, hh, r) < 0) { lo = mid; } else { hi = mid; }
      }
      for (let ri = 0; ri <= rings; ri++) {
        const t = ri / rings;
        pos.push(dx * lo * t, 0, dz * lo * t);
        nor.push(0, 1, 0);
      }
    }
    const stride = rings + 1;
    for (let s = 0; s < segs; s++) {
      for (let ri = 0; ri < rings; ri++) {
        const a = s * stride + ri, b = a + 1, c = (s + 1) * stride + ri, d = c + 1;
        idx.push(a, c, b, b, c, d);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
    g.setIndex(idx);
    return g;
  }
  const trayGeo = roundedRectDisc(trayW - 0.03, trayD - 0.03, 0.48, 56, 140);
  const trayMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff, metalness: 0, roughness: 0.12, transmission: 0.96,
    thickness: 0.25, ior: 1.45, clearcoat: 0.7, clearcoatRoughness: 0.1, envMapIntensity: 0.9,
  });
  const rippleUniforms = {
    uRippleCenter: { value: new THREE.Vector2(0, 0) },
    uRippleTime: { value: 99 },
    uRippleAmp: { value: 0 },
  };
  trayMat.onBeforeCompile = (shader) => {
    shader.uniforms.uRippleCenter = rippleUniforms.uRippleCenter;
    shader.uniforms.uRippleTime = rippleUniforms.uRippleTime;
    shader.uniforms.uRippleAmp = rippleUniforms.uRippleAmp;
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>
       uniform vec2 uRippleCenter;
       uniform float uRippleTime;
       uniform float uRippleAmp;
       const float RF = ${RIPPLE_FREQ.toFixed(1)};
       const float RS = ${RIPPLE_SPEED.toFixed(1)};
       const float RK = ${RIPPLE_FALLOFF.toFixed(2)};
       float rippleH(float d) {
         return sin(d * RF - uRippleTime * RS) * exp(-d * RK) * exp(-uRippleTime * 3.0) * uRippleAmp;
       }`)
      .replace("#include <beginnormal_vertex>", `#include <beginnormal_vertex>
       {
         vec2 rel = position.xz - uRippleCenter;
         float d = length(rel);
         if (d > 0.0001 && uRippleAmp > 0.0) {
           float e = 0.012;
           float slope = (rippleH(d + e) - rippleH(d - e)) / (2.0 * e);
           vec2 dir = rel / d;
           objectNormal = normalize(vec3(-dir.x * slope, 1.0, -dir.y * slope));
         }
       }`)
      .replace("#include <begin_vertex>", `#include <begin_vertex>
       transformed.y += rippleH(distance(position.xz, uRippleCenter));`);
  };
  const tray = new THREE.Mesh(trayGeo, trayMat);
  tray.position.y = PLATFORM_THICK - TRAY_SINK - TRAY_THICK / 2;
  tray.layers.enable(1);
  group.add(tray);

  function makeLabelTexture(text, colour) {
    const W = 1024, H = 328;
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#" + colour.getHexString();
    ctx.font = "600 84px " + (token("--font-sans") || "sans-serif");
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, W / 2, H / 2 + 4);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  const pillGeo = slab(PILL_LEN, PILL_WID, PILL_THICK, PILL_WID / 2, PILL_BEVEL, 48);
  const pills = [];
  const restY = tray.position.y + TRAY_THICK / 2 + PILL_THICK / 2;

  BUTTONS.forEach((cfg, i) => {
    const off = i - (BUTTONS.length - 1) / 2;
    const pivot = new THREE.Group();
    pivot.position.set(off * PILL_GAP_X, restY, off * PILL_GAP_Z);
    group.add(pivot);

    const mat = new THREE.MeshPhysicalMaterial({
      color: 0xffffff, metalness: 0, roughness: 0.015, transmission: 1.0,
      thickness: 0.22, ior: 1.5, clearcoat: 0.65, clearcoatRoughness: 0.02,
      reflectivity: 0.5, envMapIntensity: 0.95,
    });
    const mesh = new THREE.Mesh(pillGeo, mat);
    pivot.add(mesh);

    const labelMat = new THREE.MeshBasicMaterial({
      map: makeLabelTexture(cfg.label, labelInk), transparent: true, depthWrite: false, toneMapped: false,
    });
    const label = new THREE.Mesh(new THREE.PlaneGeometry(PILL_LEN * 0.94, PILL_WID * 0.94), labelMat);
    label.rotation.x = -Math.PI / 2;
    label.position.y = PILL_THICK / 2 + 0.005;
    pivot.add(label);

    pills.push({ pivot, mesh, mat, restY, y: 0, hoverT: 0, fill: 0, index: i, cfg, colour: cfgColour[i] });
  });

  const SHADOW_PLANE_W = PILL_LEN * 1.6, SHADOW_PLANE_D = PILL_WID * 3.4;
  function softBlobTexture(tint, blur, spread = 1.32) {
    const S = 512;
    const c = document.createElement("canvas");
    c.width = S; c.height = S;
    const ctx = c.getContext("2d");
    const fw = (PILL_LEN / SHADOW_PLANE_W) * S * spread;
    const fh = (PILL_WID / SHADOW_PLANE_D) * S * spread;
    ctx.filter = `blur(${blur}px)`;
    ctx.fillStyle = tint;
    ctx.beginPath();
    ctx.roundRect((S - fw) / 2, (S - fh) / 2, fw, fh, fh / 2);
    ctx.fill();
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }
  const shadowTex = softBlobTexture("rgba(0,0,0,0.92)", 30);
  const glowTex = softBlobTexture("rgba(255,255,255,1)", 50, 1.0);
  const shadowGeo = new THREE.PlaneGeometry(SHADOW_PLANE_W, SHADOW_PLANE_D);
  const glowGeo = new THREE.PlaneGeometry(SHADOW_PLANE_W * 1.9, SHADOW_PLANE_D * 1.9);
  const shadowY = tray.position.y + 0.008;

  const blobs = pills.map((p) => {
    const sMat = new THREE.MeshBasicMaterial({
      map: shadowTex, transparent: true, depthWrite: false, opacity: 0.62, toneMapped: false,
    });
    const shadow = new THREE.Mesh(shadowGeo, sMat);
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(p.pivot.position.x + 0.16, shadowY, p.pivot.position.z - 0.13);
    shadow.renderOrder = -2;
    group.add(shadow);

    const gMat = new THREE.MeshBasicMaterial({
      map: glowTex, transparent: true, depthWrite: false, opacity: 0,
      blending: THREE.AdditiveBlending, toneMapped: false,
    });
    const glow = new THREE.Mesh(glowGeo, gMat);
    glow.rotation.x = -Math.PI / 2;
    glow.position.set(p.pivot.position.x, shadowY + 0.002, p.pivot.position.z);
    glow.renderOrder = -1;
    group.add(glow);
    return { shadow, glow, sMat, gMat };
  });

  function updateBlobs() {
    pills.forEach((p, i) => {
      const b = blobs[i];
      const lift = p.y / HOVER_LIFT;
      const spread = 1 + lift * 0.6;
      b.shadow.scale.set(spread, spread, 1);
      b.shadow.position.x = p.pivot.position.x + 0.16 + lift * 0.14;
      b.shadow.position.z = p.pivot.position.z - 0.13 - lift * 0.11;
      b.sMat.opacity = 0.55 - lift * 0.2;
      b.gMat.color.copy(p.colour);
      b.gMat.opacity = 0.72 * p.fill;
      const gs = 1 + p.fill * 0.15;
      b.glow.scale.set(gs, gs, 1);
    });
  }

  const WHITE = new THREE.Color(0xffffff);
  function setFill(p, amount) {
    p.fill = amount;
    p.mat.color.copy(WHITE).lerp(p.colour, amount * 0.88);
    p.mat.attenuationColor.copy(WHITE).lerp(p.colour, amount);
    p.mat.attenuationDistance = THREE.MathUtils.lerp(30, 0.28, amount);
    p.mat.transmission = THREE.MathUtils.lerp(1.0, 0.84, amount);
    p.mat.thickness = THREE.MathUtils.lerp(0.22, 0.8, amount);
    p.mat.emissive.copy(p.colour);
    p.mat.emissiveIntensity = 0.9 * amount;
  }
  setFill(pills[0], 1);
  for (let i = 1; i < pills.length; i++) { setFill(pills[i], 0); }

  // ---- state machine, ported verbatim minus the room crossfade ----------
  let activeIndex = 0, phase = "idle", tPhase = 0, fromIndex = 0, toIndex = 0;
  let hoverIndex = -1, hoverHeld = 0, rippleClock = 99;

  function requestSwitch(target) {
    if (phase !== "idle" || target === activeIndex) { return; }
    fromIndex = activeIndex; toIndex = target; tPhase = 0; phase = "slam";
    for (const q of pills) { q.hoverT = 0; }
    hoverHeld = 0;
    if (onPick) { onPick(target); }
  }

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  function pick(clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    if (!r.width || !r.height) { return -1; }
    pointer.x = ((clientX - r.left) / r.width) * 2 - 1;
    pointer.y = -((clientY - r.top) / r.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(pills.map((p) => p.mesh), false);
    return hits.length ? pills.findIndex((q) => q.mesh === hits[0].object) : -1;
  }
  function onPointerDown(e) {
    const i = pick(e.clientX, e.clientY);
    if (i >= 0) { requestSwitch(i); }
  }
  function onPointerMove(e) {
    const i = pick(e.clientX, e.clientY);
    if (i !== hoverIndex) { hoverIndex = i; hoverHeld = 0; }
  }
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove, { passive: true });

  /* ---- bloom, small and hand-rolled -- duplicated from hero3d.js's
   * pipeline rather than imported from it (see file header). */
  const QUAD = new THREE.BufferGeometry();
  QUAD.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  QUAD.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const VERT = "in vec3 position;\nin vec2 uv;\nout vec2 vUv;\nvoid main() { vUv = uv; gl_Position = vec4(position, 1.0); }";
  const brightMat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: { tSrc: { value: null }, threshold: { value: 0.6 } },
    vertexShader: VERT,
    fragmentShader: `precision highp float; in vec2 vUv; out vec4 fragColor;
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
    uniforms: { tSrc: { value: null }, dir: { value: new THREE.Vector2(1, 0) }, texel: { value: new THREE.Vector2(1, 1) } },
    vertexShader: VERT,
    fragmentShader: `precision highp float; in vec2 vUv; out vec4 fragColor;
      uniform sampler2D tSrc; uniform vec2 dir; uniform vec2 texel;
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
    uniforms: { tScene: { value: null }, tBloom: { value: null }, strength: { value: 0.5 } },
    vertexShader: VERT,
    fragmentShader: `precision highp float; in vec2 vUv; out vec4 fragColor;
      uniform sampler2D tScene; uniform sampler2D tBloom; uniform float strength;
      void main() {
        vec4 s = texture(tScene, vUv);
        vec3 b = texture(tBloom, vUv).rgb * strength;
        float ba = clamp(dot(b, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        fragColor = vec4(s.rgb + b, clamp(s.a + ba, 0.0, 1.0));
      }`,
  });
  const quad = new THREE.Mesh(QUAD, brightMat);
  const quadScene = new THREE.Scene();
  quadScene.add(quad);
  function pass(material, target) {
    quad.material = material;
    renderer.setRenderTarget(target);
    renderer.clear();
    renderer.render(quadScene, quadCam);
  }
  const rtOpts = { type: THREE.HalfFloatType, depthBuffer: true };
  let sceneRT = new THREE.WebGLRenderTarget(2, 2, rtOpts);
  let brightRT = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType });
  let blurRT = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType });

  // ---- sizing, fitted to the host container, not the window -------------
  let W = 0, H = 0, DPR = 1;
  function resize() {
    const r = host.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width));
    const h = Math.max(1, Math.round(r.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (w === W && h === H && dpr === DPR) { return; }
    W = w; H = h; DPR = dpr;
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    fitCamera(w / h);
    camera.updateProjectionMatrix();
    const pw = Math.max(1, Math.floor(w * dpr));
    const ph = Math.max(1, Math.floor(h * dpr));
    sceneRT.setSize(pw, ph);
    brightRT.setSize(Math.max(1, pw >> 2), Math.max(1, ph >> 2));
    blurRT.setSize(Math.max(1, pw >> 2), Math.max(1, ph >> 2));
  }

  function frame(t, dt) {
    resize();
    rippleClock += dt;
    rippleUniforms.uRippleTime.value = rippleClock;
    rippleUniforms.uRippleAmp.value = rippleClock < RIPPLE_LIFE ? RIPPLE_AMP : 0;

    const target = pills[phase === "idle" ? activeIndex : toIndex];

    if (phase === "slam") {
      tPhase += dt;
      const u = clamp01(tPhase / SLAM_TIME);
      target.y = THREE.MathUtils.lerp(target.y, 0, u * u);
      if (tPhase >= SLAM_TIME) {
        target.y = 0; phase = "bounce"; tPhase = 0;
        rippleUniforms.uRippleCenter.value.set(target.pivot.position.x, target.pivot.position.z);
        rippleClock = 0;
      }
    } else if (phase === "bounce") {
      tPhase += dt;
      const u = clamp01(tPhase / BOUNCE_TIME);
      target.y = Math.sin(Math.PI * u) * 0.055 * (1 - u);
      const squash = 1 - 0.3 * Math.sin(Math.PI * clamp01(u * 1.6));
      target.mesh.scale.set(1, squash, 1);
      if (tPhase >= BOUNCE_TIME) {
        target.y = 0; target.mesh.scale.set(1, 1, 1); phase = "flip"; tPhase = 0;
      }
    } else if (phase === "flip") {
      tPhase += dt;
      const u = clamp01(tPhase / FLIP_TIME);
      const e = easeInOut(u);
      target.pivot.rotation.x = e * Math.PI * 2;
      const edgeOn = Math.abs(Math.sin(target.pivot.rotation.x));
      target.mesh.scale.set(1, THREE.MathUtils.lerp(1, 0.74, edgeOn), 1);
      setFill(pills[fromIndex], 1 - clamp01(u * 2));
      setFill(target, e);
      if (tPhase >= FLIP_TIME) {
        target.pivot.rotation.x = 0; target.mesh.scale.set(1, 1, 1);
        activeIndex = toIndex; phase = "idle";
      }
    }

    if (phase === "idle") {
      hoverHeld = hoverIndex >= 0 ? hoverHeld + dt : 0;
      pills.forEach((p, i) => {
        const want = i === hoverIndex && i !== activeIndex && hoverHeld > HOVER_DELAY ? 1 : 0;
        p.hoverT = clamp01(p.hoverT + (want ? dt / HOVER_TIME : -dt / HOVER_DROP_TIME));
        p.y = HOVER_LIFT * easeInOut(p.hoverT);
      });
    }
    for (const p of pills) { p.pivot.position.y = p.restY + p.y; }
    updateBlobs();

    const act = pills[phase === "idle" ? activeIndex : toIndex];
    glowLight.color.copy(act.colour);
    const w = act.pivot.position.clone().applyMatrix4(group.matrixWorld);
    glowLight.position.set(w.x, w.y + 0.28, w.z);
    glowLight.intensity = GLOW_INTENSITY * act.fill;

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

  let lastT = 0;
  function tickFrame(t) {
    const dt = lastT ? Math.min((t - lastT) / 1000, 0.05) : 0;
    lastT = t;
    frame(t, dt);
  }

  function select(i) {
    requestSwitch(i);
  }

  function dispose() {
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    scene.traverse((o) => {
      if (o.geometry) { o.geometry.dispose(); }
      if (o.material) {
        (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
      }
    });
    [sceneRT, brightRT, blurRT].forEach((rt) => rt.dispose());
    envRT.dispose();
    renderer.dispose();
    canvas.remove();
  }

  host.appendChild(canvas);
  resize();
  frame(0, 0);

  const gl = renderer.getContext();
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  const info = {
    renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : "unavailable",
    three: THREE.REVISION,
  };

  return { canvas, frame: tickFrame, dispose, select, info };
}
