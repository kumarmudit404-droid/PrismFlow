/* try_it.js -- upload an array, see its dependence structure.
 *
 * THE API IS OPTIONAL AND THE SITE MUST NOT NEED IT.
 * ---------------------------------------------------
 * This page is static and the scoring service is somewhere else. It may be
 * unreachable, asleep, not deployed yet, or blocked by this site's own
 * Content-Security-Policy -- which, as shipped, is `connect-src 'self'` and
 * therefore blocks it by design until the owner edits two files by hand.
 *
 * All four cases look different to JavaScript and must look the SAME to a
 * reader: "Live scoring offline". A CSP violation surfaces as a TypeError
 * from fetch with no response, which is indistinguishable from a dead host
 * and is treated identically here. Nothing else on the page depends on this
 * file, so a failed probe costs this one panel and nothing more.
 *
 * NO NUMBER IS INVENTED WHILE OFFLINE. There is no cached result, no demo
 * payload and no placeholder: offline means the panel says so and shows
 * nothing. The sample file is scored by the same API as an upload, so
 * "use sample data" is a real round trip, not a canned answer.
 *
 * THE CAVEATS ARE NOT COLLAPSIBLE. The three literal fields the API returns
 * are rendered before the numbers, every time, with no control that hides
 * them.
 */
(function (global) {
  "use strict";

  var doc = global.document;

  /* The one config value. An empty string means "no API configured", which
   * is the shipped state and renders exactly like an unreachable one. */
  var API_BASE = (global.PRISMFLOW_API_BASE || "").replace(/\/+$/, "");

  var SAMPLE_URL = "data/sample_ait_views.npz";
  var HEALTH_TIMEOUT_MS = 4000;

  function esc(value) {
    if (value === null || value === undefined) { return ""; }
    return String(value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  /* null is "not measured" and is never printed as a number. The API already
   * maps NaN to null for exactly this reason; this is the other half. */
  function num(value, digits) {
    if (value === null || value === undefined || typeof value !== "number") {
      return '<span class="pill pill--notmeasured">not measured</span>';
    }
    return value.toFixed(digits === undefined ? 4 : digits);
  }

  function el(id) { return doc.getElementById(id); }

  function setStatus(state, text) {
    var node = el("try-status");
    if (!node) { return; }
    node.className = "try__status try__status--" + state;
    node.innerHTML = text;
  }

  function offline(why) {
    setStatus("offline",
      '<span class="pill pill--notmeasured">Live scoring offline</span> ' +
      esc(why));
    var controls = el("try-controls");
    if (controls) {
      Array.prototype.forEach.call(
        controls.querySelectorAll("input, button"),
        function (c) { c.disabled = true; });
    }
  }

  /* fetch with a deadline. Without one a sleeping Space leaves the panel
   * saying "checking" forever, which reads as a bug rather than as offline. */
  function fetchWithTimeout(url, options, ms) {
    var controller = ("AbortController" in global) ? new AbortController() : null;
    var opts = Object.assign({}, options || {});
    if (controller) { opts.signal = controller.signal; }
    var timer = global.setTimeout(function () {
      if (controller) { controller.abort(); }
    }, ms);
    return global.fetch(url, opts).then(function (r) {
      global.clearTimeout(timer);
      return r;
    }, function (e) {
      global.clearTimeout(timer);
      throw e;
    });
  }

  /* ---- rendering: the same components Part 31 uses -------------------- */

  function heatmap(matrix) {
    var values = [];
    matrix.forEach(function (row) {
      row.forEach(function (v) { if (typeof v === "number") { values.push(v); } });
    });
    var hi = values.length ? Math.max.apply(null, values.map(Math.abs)) : 1;

    var out = ['<div class="table-wrap"><table class="heat">' +
      '<caption class="visually-hidden">Measured dependence between every ' +
      'pair of uploaded views.</caption><thead><tr>' +
      '<th scope="col" class="heat__hd">view</th>'];
    matrix.forEach(function (_, i) {
      out.push('<th scope="col" class="heat__hd">' + i + "</th>");
    });
    out.push("</tr></thead><tbody>");
    matrix.forEach(function (row, i) {
      out.push('<tr><th scope="row" class="heat__hd">' + i + "</th>");
      row.forEach(function (v) {
        if (typeof v !== "number") {
          out.push('<td class="heat__cell heat__cell--nm" ' +
            'title="not measured: this pair had no measurable overlap">' +
            "&middot;</td>");
          return;
        }
        var s = hi > 0 ? Math.min(1, Math.abs(v) / hi) : 0;
        out.push('<td class="heat__cell" style="--s:' + s.toFixed(3) +
          '" title="' + v.toFixed(4) + '">' + v.toFixed(2) + "</td>");
      });
      out.push("</tr>");
    });
    out.push("</tbody></table></div>");
    return out.join("");
  }

  function caveats(data) {
    /* First, not last, and with no control that hides them. */
    return '<div class="panel panel--caution"><h4>Read these before the numbers</h4>' +
      "<p><strong>Calibration.</strong> " + esc(data.calibration) + "</p>" +
      "<p><strong>Defence.</strong> " + esc(data.defence) + "</p>" +
      "<p><strong>Estimator.</strong> " + esc(data.estimator_validation) + "</p>" +
      "<p><strong>Conditioning.</strong> " + esc(data.conditioning) + "</p>" +
      "</div>";
  }

  function render(data) {
    var host = el("try-result");
    if (!host) { return; }
    var eigen = data.eniv.eigen;
    var design = data.eniv.design_effect;

    var out = [caveats(data)];
    out.push('<div class="panel"><h4>What was measured</h4><dl class="kv">' +
      "<dt>samples</dt><dd>" + num(data.shape.samples, 0) + "</dd>" +
      "<dt>views</dt><dd>" + num(data.shape.views, 0) + "</dd>" +
      "<dt>features per view</dt><dd>" +
        num(data.shape.features_per_view, 0) + "</dd>" +
      "<dt>ENIV (eigen)</dt><dd><strong>" + num(eigen.effective_views, 3) +
        "</strong> of " + num(eigen.nominal_views, 0) +
        ' <span class="unit">efficiency ' +
        num(eigen.efficiency_ratio, 3) + "</span></dd>" +
      "<dt>ENIV (design effect)</dt><dd>" + num(design.effective_views, 3) +
        " of " + num(design.nominal_views, 0) + "</dd>" +
      "<dt>mean dependence</dt><dd>" + num(eigen.mean_dependence, 4) + "</dd>" +
      "<dt>seed</dt><dd>" + num(data.seed, 0) +
        ' <span class="unit">config ' + esc(data.config_hash) + "</span></dd>" +
      "</dl></div>");

    out.push('<div class="panel"><h4>Dependence between your views</h4>' +
      heatmap(data.dependence_matrix) +
      '<p class="pass-note">A dot is a pair with no measurable overlap. It ' +
      "is not a zero, and it is not drawn as one.</p></div>");

    out.push('<div class="panel"><h4>Per-view discount</h4>' +
      '<div class="table-wrap"><table><caption class="visually-hidden">' +
      'The discount factor soft_cluster_alpha assigns each view.</caption>' +
      '<thead><tr><th scope="col">view</th>' +
      '<th scope="col" class="num">alpha</th></tr></thead><tbody>' +
      data.per_view_alpha.map(function (a, i) {
        return '<tr><th scope="row">' + i + '</th><td class="num">' +
          num(a, 4) + "</td></tr>";
      }).join("") + "</tbody></table></div></div>");

    if (data.fused === null) {
      out.push('<div class="panel"><h4>Fusion and suspicion</h4>' +
        '<p class="nm nm--why">' + esc(data.fused_absent_reason) + "</p></div>");
    } else {
      out.push('<div class="panel"><h4>Fusion and suspicion</h4><dl class="kv">' +
        "<dt>mean uncertainty</dt><dd>" +
          num(data.fused.mean_uncertainty, 4) + "</dd>" +
        "<dt>suspicion, mean</dt><dd>" +
          num(data.suspicion.mean_score, 4) + "</dd>" +
        "<dt>suspicion, max</dt><dd>" +
          num(data.suspicion.max_score, 4) + "</dd>" +
        "<dt>flagged</dt><dd>" +
          num(data.suspicion.flagged_fraction, 4) + "</dd>" +
        "</dl><p class=\"pass-note\">" + esc(data.suspicion.note) +
        "</p></div>");
    }

    host.innerHTML = out.join("");
  }

  /* ---- sending ------------------------------------------------------- */

  function send(blob, filename) {
    setStatus("busy", "Scoring…");
    var form = new FormData();
    form.append("file", blob, filename);
    return fetchWithTimeout(API_BASE + "/score", {
      method: "POST", body: form
    }, 60000).then(function (r) {
      return r.json().then(function (body) { return { ok: r.ok, body: body }; });
    }).then(function (res) {
      if (!res.ok) {
        /* A refusal is a result, not a crash: the service named a rule and
         * the reader is entitled to see which one. */
        setStatus("rejected",
          '<strong>Not scored:</strong> ' + esc(res.body.detail || "") +
          ' <span class="unit">(' + esc(res.body.error || "error") + ")</span>");
        var host = el("try-result");
        if (host) { host.innerHTML = ""; }
        return;
      }
      setStatus("ok", "Scored. Every figure below came from this request.");
      render(res.body);
    }).catch(function () {
      offline("The request did not complete.");
    });
  }

  /* ---- wiring -------------------------------------------------------- */

  function wire() {
    var fileInput = el("try-file");
    var sampleButton = el("try-sample");

    if (fileInput) {
      fileInput.addEventListener("change", function () {
        var f = fileInput.files && fileInput.files[0];
        if (f) { send(f, f.name); }
      });
    }
    if (sampleButton) {
      sampleButton.addEventListener("click", function () {
        setStatus("busy", "Fetching the sample…");
        global.fetch(SAMPLE_URL).then(function (r) { return r.blob(); })
          .then(function (b) { return send(b, "sample_ait_views.npz"); })
          .catch(function () { offline("The sample file could not be read."); });
      });
    }
  }

  function start() {
    if (!el("try-status")) { return; }
    if (!API_BASE) {
      /* The shipped state. Not an error -- the owner has not pointed the
       * site at a service yet, and the page says exactly that. */
      offline("No scoring service is configured for this site.");
      return;
    }
    setStatus("busy", "Checking the scoring service…");
    fetchWithTimeout(API_BASE + "/health", { method: "GET" }, HEALTH_TIMEOUT_MS)
      .then(function (r) {
        if (!r.ok) { throw new Error("health " + r.status); }
        return r.json();
      })
      .then(function (health) {
        setStatus("ok", "Live scoring available — uploads up to " +
          Math.round(health.limits.max_upload_bytes / 1048576) +
          " MB, at least " + health.limits.min_samples + " samples, 2 or " +
          "more views.");
        wire();
      })
      .catch(function () {
        /* Dead host, 5xx, timeout and a CSP block all land here, and all
         * four mean the same thing to a reader. */
        offline("The service did not answer, or this page's " +
          "Content-Security-Policy does not allow it yet.");
      });
  }

  if (doc.readyState === "loading") {
    doc.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})(this);
