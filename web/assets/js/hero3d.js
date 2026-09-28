/* PrismFlow Part 25 -- the ONE physically-lit visual on the site.
 *
 * A real glass prism: transmissive material with an index of refraction and
 * dispersion, lit by coloured lights taken from the angle tokens, so the
 * refraction that leaves it is the site's own palette rather than a rainbow.
 * Everything else on this page stays flat and legible, deliberately.
 *
 * This module is imported dynamically and only when three.js can actually be
 * used -- see loadHero() in motion.js. It is never on the critical path.
 *
 * COLOUR RULE, same as motion.js: every colour arrives through the `token`
 * and `palette` helpers, which read tokens.css at runtime. No hex literal.
 */

import * as THREE from "../vendor/three/three.module.js";

export async function mount(host, api) {
  const { palette, token } = api;
  const angles = palette();
  if (!angles.length) { throw new Error("no angle tokens available"); }

  /* Every colour must come from tokens.css. A missing token throws, and
   * motion.js catches it and leaves the flat SVG mark in place, which is a
   * correct fallback. Substituting a literal here -- this file briefly carried
   * a plain white fallback that is not even in the palette -- would quietly put
   * an unapproved colour on screen, the one thing the token rule exists to
   * prevent. */
  const need = (name) => {
    const v = token(name);
    if (!v) { throw new Error("missing required token " + name); }
    return new THREE.Color(v);
  };

  const width = host.clientWidth || 900;
  const height = host.clientHeight || 260;

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: "low-power",
  });
  // DPR capped at 2: this is a decorative surface and a 3x display would cost
  // 2.25x the fragments for no readable gain.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height, false);
  renderer.domElement.setAttribute("aria-hidden", "true");
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, width / height, 0.1, 100);
  camera.position.set(0, 0, 7.2);

  // --- the prism -----------------------------------------------------
  // A true triangular prism: a 3-sided cylinder is exactly that, and it keeps
  // the silhouette identical to the flat SVG mark the site uses elsewhere.
  const geometry = new THREE.CylinderGeometry(1.35, 1.35, 1.5, 3, 1);
  const material = new THREE.MeshPhysicalMaterial({
    color: need("--beam-incident"),
    metalness: 0,
    roughness: 0.03,
    transmission: 1,        // real refraction rather than a fake alpha
    thickness: 1.7,
    ior: 1.62,              // ~ dense flint glass, which disperses visibly
    dispersion: 3.2,
    clearcoat: 1,
    clearcoatRoughness: 0.04,
    attenuationDistance: 6,
    transparent: true,
  });
  const prism = new THREE.Mesh(geometry, material);
  prism.rotation.z = Math.PI;      // apex up, matching the SVG mark
  prism.rotation.x = 0.16;
  scene.add(prism);

  // --- lighting: the palette, not white ------------------------------
  // One warm key standing in for the incident beam, plus one coloured light
  // per angle placed along the fan. The refraction therefore carries the
  // site's own hues out of the glass.
  const key = new THREE.DirectionalLight(
    need("--beam-incident"), 2.6);
  key.position.set(-5, 0.6, 3);
  scene.add(key);

  angles.forEach((a, i) => {
    const spread = (i / Math.max(1, angles.length - 1)) * 2 - 1;
    // The unbuilt angle is dim here too: Regulatory must not look alive in
    // any rendering on this site, including a physically-lit one.
    const intensity = a.unbuilt ? 2.2 : 7.0;
    const light = new THREE.PointLight(new THREE.Color(a.colour), intensity, 18, 2);
    light.position.set(3.4, spread * 2.3, 1.6 + Math.abs(spread) * 0.6);
    scene.add(light);
  });

  scene.add(new THREE.AmbientLight(need("--surface-raised"), 1.4));

  // --- interaction & loop --------------------------------------------
  let targetX = 0, targetY = 0, curX = 0, curY = 0;
  const onMove = (cx, cy) => {
    const r = host.getBoundingClientRect();
    if (!r.width) { return; }
    targetY = ((cx - r.left) / r.width - 0.5) * 0.9;
    targetX = ((cy - r.top) / r.height - 0.5) * 0.5;
  };
  window.addEventListener("pointermove", (e) => onMove(e.clientX, e.clientY), { passive: true });
  window.addEventListener("touchmove", (e) => {
    if (e.touches && e.touches[0]) { onMove(e.touches[0].clientX, e.touches[0].clientY); }
  }, { passive: true });

  let raf = 0, visible = true, running = false;
  const clock = new THREE.Clock();

  function frame() {
    if (!running) { return; }
    const t = clock.getElapsedTime();
    curX += (targetX - curX) * 0.05;
    curY += (targetY - curY) * 0.05;
    prism.rotation.x = 0.16 + curX + Math.sin(t * 0.28) * 0.05;
    prism.rotation.y = curY + t * 0.12;
    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }
  function start() { if (!running && visible && !document.hidden) { running = true; clock.getDelta(); raf = requestAnimationFrame(frame); } }
  function stop() { running = false; cancelAnimationFrame(raf); }

  // Offscreen and hidden-tab pausing, same contract as the 2D layer.
  new IntersectionObserver((en) => {
    visible = en[0].isIntersecting;
    if (visible) { start(); } else { stop(); }
  }, { threshold: 0.01 }).observe(host);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { stop(); } else { start(); }
  });

  window.addEventListener("resize", () => {
    const w = host.clientWidth || width;
    const h = host.clientHeight || height;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }, { passive: true });

  start();
  return { start, stop, renderer };
}
