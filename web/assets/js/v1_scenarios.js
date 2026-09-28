/* PrismFlow Part 25 -- phase (e), step 2. The V1 app's controls, rebuilt.
 *
 * The Streamlit app has a sidebar: pick a dataset, pick a scenario, press Run.
 * A static site cannot train a model on demand, so web/build_v1.py ran every
 * scenario through app.py's OWN engine functions at a recorded seed and wrote
 * the outputs to web/data/v1_scenarios.json. This file is the picker over those
 * pre-computed outputs.
 *
 * NOTHING HERE IS INTERPOLATED BETWEEN SCENARIOS. Choosing a scenario shows
 * that scenario's committed record or nothing at all. There is no transition, no
 * smoothing, and no derived "what if" -- a slider that produced numbers the
 * engine never computed would be a lie with a nice interaction.
 *
 * Every panel carries the illustrative-run label from the data file, because a
 * 12-epoch single-seed run is not a finding under docs/CONTRACT.md section 5.
 */
(function () {
  "use strict";

  var NOT_MEASURED = '<span class="nm">not measured</span>';

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) { n.className = cls; }
    if (html !== undefined) { n.innerHTML = html; }
    return n;
  }

  function esc(v) {
    if (v === null || v === undefined) { return ""; }
    return String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function num(v, digits) {
    if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) {
      return NOT_MEASURED;
    }
    return Number(v).toFixed(digits === undefined ? 3 : digits);
  }

  function load(path) {
    return fetch(path, { cache: "no-store" }).then(function (r) {
      if (!r.ok) { throw new Error(path + " → HTTP " + r.status); }
      return r.json();
    });
  }

  /* ---- the dependence matrix, as a real heatmap ---------------------------
   * app.py renders this with matplotlib. Here it is a table of cells: a table
   * because the numbers are the content, and colour is only reinforcement. A
   * null cell (too few tail samples to estimate) is dim and labelled, never a
   * zero, because zero means "measured, and independent". */
  function heatmap(m, names) {
    if (!m || !m.length) { return '<p class="notmeasured">dependence matrix not measured</p>'; }
    var out = ['<div class="table-wrap"><table class="heat">' +
      '<caption class="visually-hidden">Cross-view dependence matrix. Row and ' +
      'column headers are view names; each cell is the measured dependence ' +
      'between that pair, or "not measured" where there were too few samples ' +
      'to estimate it.</caption><thead><tr><th></th>'];
    names.forEach(function (n) {
      out.push('<th scope="col" class="heat__hd">' + esc(n) + "</th>");
    });
    out.push("</tr></thead><tbody>");
    m.forEach(function (row, i) {
      out.push('<tr><th scope="row" class="heat__hd">' + esc(names[i]) + "</th>");
      row.forEach(function (v, j) {
        if (v === null) {
          out.push('<td class="heat__cell heat__cell--nm" title="not measured">·</td>');
          return;
        }
        var s = Math.max(0, Math.min(1, v));
        out.push('<td class="heat__cell" style="--s:' + s.toFixed(3) + '" title="' +
          esc(names[i]) + " / " + esc(names[j]) + ": " + v.toFixed(4) + '">' +
          v.toFixed(2) + "</td>");
      });
      out.push("</tr>");
    });
    out.push("</tbody></table></div>");
    return out.join("");
  }

  var STATE_PILL = {
    available: "pill--live", duplicate: "pill--notmeasured",
    missing: "pill--zero", noisy: "pill--zero", compromised: "pill--error"
  };

  function viewTable(rec) {
    var out = ['<div class="table-wrap"><table>' +
      '<caption class="visually-hidden">Per-view status: the state the scenario ' +
      'put each view in, its ENIV alpha weight, and the fraction of samples for ' +
      'which it was present.</caption><thead><tr>' +
      '<th scope="col">view</th><th scope="col">state</th>' +
      '<th scope="col" class="num">alpha</th>' +
      '<th scope="col" class="num">present</th></tr></thead><tbody>'];
    rec.view_names.forEach(function (name, i) {
      var st = rec.view_states[i] || "available";
      out.push('<tr><th scope="row" class="mono">' + esc(name) + "</th>" +
        '<td><span class="pill ' + (STATE_PILL[st] || "pill--live") + '">' + esc(st) + "</span></td>" +
        '<td class="num">' + num(rec.per_view_alpha ? rec.per_view_alpha[i] : null) + "</td>" +
        '<td class="num">' + num(rec.present_fraction ? rec.present_fraction[i] : null) + "</td></tr>");
    });
    out.push("</tbody></table></div>");
    return out.join("");
  }

  function predictionsTable(p) {
    if (!p || !p.examples || !p.examples.length) {
      return '<p class="notmeasured">per-example predictions not recorded</p>';
    }
    var out = ['<p class="axis-note">' + p.n_wrong + " of " + p.n_scored +
      " scored samples were predicted wrongly. " + esc(p.selection) + ".</p>",
      '<div class="table-wrap"><table>' +
      '<caption class="visually-hidden">Individual predictions against the true ' +
      'label, with the confidence the model reported.</caption><thead><tr>' +
      '<th scope="col" class="num">sample</th><th scope="col" class="num">predicted</th>' +
      '<th scope="col" class="num">true</th><th scope="col">outcome</th>' +
      '<th scope="col" class="num">confidence</th>' +
      '<th scope="col" class="num">max prob</th></tr></thead><tbody>'];
    p.examples.forEach(function (e) {
      out.push('<tr class="' + (e.correct ? "" : "ex--wrong") + '">' +
        '<th scope="row" class="num mono">' + e.index + "</th>" +
        '<td class="num">' + e.predicted + "</td><td class=\"num\">" + e.true + "</td>" +
        "<td>" + (e.correct
          ? '<span class="pill pill--live">correct</span>'
          : '<span class="pill pill--error">wrong</span>') + "</td>" +
        '<td class="num">' + num(e.confidence) + "</td>" +
        '<td class="num">' + num(e.max_probability) + "</td></tr>");
    });
    out.push("</tbody></table></div>");
    return out.join("");
  }

  function renderRecord(host, data, dsKey, scenario) {
    var ds = data.datasets[dsKey];
    var rec = ds.scenarios[scenario];
    host.innerHTML = "";

    if (!rec) {
      var why = (ds.errors && ds.errors[scenario]) || "no record was written";
      host.appendChild(el("div", "notmeasured notmeasured--wide",
        '<strong class="nm">this scenario did not run</strong><br>' + esc(why) +
        "<br>No value is shown for it. Nothing was substituted from another " +
        "scenario, seed or dataset."));
      return;
    }

    host.appendChild(el("p", "scenario__desc", esc(rec.description)));

    /* The headline numbers app.py puts in its "Independence budget" tab. */
    var stats = el("div", "grid grid--4 scenario__stats");
    [
      ["nominal views", num(rec.nominal_views, 0), "views the model was given"],
      ["effective independent views", num(rec.effective_views), "ENIV"],
      ["efficiency ratio", num(rec.efficiency_ratio), "effective ÷ nominal"],
      ["batch accuracy", num(rec.accuracy), rec.n_samples_scored + " samples, seed " + rec.seed]
    ].forEach(function (s) {
      stats.appendChild(el("div", "panel stat",
        '<span class="stat__value">' + s[1] + "</span>" +
        '<span class="stat__label">' + esc(s[0]) + "</span>" +
        '<span class="stat__src">' + esc(s[2]) + "</span>"));
    });
    host.appendChild(stats);

    var budget = el("div", "panel");
    budget.appendChild(el("h4", "chart-panel__title", "Independence budget"));
    budget.insertAdjacentHTML("beforeend",
      "<p class=\"axis-note\">most redundant pair: " +
      (rec.most_redundant_pair
        ? "<span class=\"mono\">" + esc(rec.most_redundant_pair[0]) + "</span> and <span class=\"mono\">" +
          esc(rec.most_redundant_pair[1]) + "</span>, tail lambda_u " + num(rec.most_redundant_lambda_u)
        : NOT_MEASURED) +
      " · mean dependence " + num(rec.mean_dependence) + "</p>");
    budget.insertAdjacentHTML("beforeend", heatmap(rec.dependence, rec.view_names));
    host.appendChild(budget);

    var views = el("div", "panel");
    views.appendChild(el("h4", "chart-panel__title", "Per-view status"));
    views.insertAdjacentHTML("beforeend", viewTable(rec));
    host.appendChild(views);

    /* Sample 0, the one app.py shows in its results tab. */
    var s0 = rec.sample_0;
    var pred = el("div", "panel");
    pred.appendChild(el("h4", "chart-panel__title", "Prediction — sample 0"));
    pred.insertAdjacentHTML("beforeend",
      '<dl class="detail__meta detail__meta--num">' +
      "<dt>predicted class</dt><dd>" + num(s0.prediction, 0) + "</dd>" +
      "<dt>true class</dt><dd>" + num(s0.true_label, 0) + "</dd>" +
      "<dt>confidence</dt><dd>" + num(s0.confidence) + "</dd>" +
      "<dt>uncertainty</dt><dd>" + num(s0.uncertainty) + "</dd>" +
      "<dt>max probability</dt><dd>" + num(s0.max_probability) + "</dd>" +
      "</dl>");
    host.appendChild(pred);

    var tail = el("div", "panel");
    tail.appendChild(el("h4", "chart-panel__title", "Tail dependence"));
    var enough = rec.tail_n_min >= rec.tail_min_samples_required;
    tail.insertAdjacentHTML("beforeend",
      '<dl class="detail__meta detail__meta--num">' +
      "<dt>mean lambda_u</dt><dd>" + num(rec.tail_lambda_u_mean) + "</dd>" +
      "<dt>quantile</dt><dd>" + num(rec.tail_quantile, 2) + "</dd>" +
      "<dt>smallest tail n</dt><dd>" + num(rec.tail_n_min, 0) + "</dd>" +
      "<dt>minimum required</dt><dd>" + num(rec.tail_min_samples_required, 0) + "</dd>" +
      "</dl>" +
      (enough ? "" : '<p class="pass-note">The smallest tail sample is below the ' +
        "module's own minimum, so this estimate is reported and should not be " +
        "leaned on.</p>"));
    host.appendChild(tail);

    if (rec.suspicion) {
      var sus = el("div", "panel");
      sus.appendChild(el("h4", "chart-panel__title", "Suspicion detector"));
      sus.insertAdjacentHTML("beforeend",
        '<dl class="detail__meta detail__meta--num">' +
        "<dt>mean score</dt><dd>" + num(rec.suspicion.score_mean, 4) + "</dd>" +
        "<dt>flag rate</dt><dd>" + num(rec.suspicion.flag_rate) + "</dd>" +
        "<dt>threshold</dt><dd>" + num(rec.suspicion.threshold, 4) + "</dd>" +
        "<dt>calibrated at</dt><dd>" + num(rec.suspicion.calibrated_at, 2) + "</dd>" +
        "</dl><p class=\"axis-note\">" + esc(rec.suspicion.threshold_note) + "</p>");
      host.appendChild(sus);
    }

    var ex = el("div", "panel");
    ex.appendChild(el("h4", "chart-panel__title", "Real predictions against the true label"));
    ex.insertAdjacentHTML("beforeend", predictionsTable(rec.predictions));
    host.appendChild(ex);

    /* The label is repeated with every record, not printed once at the top. */
    host.appendChild(el("div", "notmeasured notmeasured--wide",
      '<strong class="nm">not evidence</strong> ' + esc(rec.label)));
  }

  function renderScenarios(host, data) {
    host.className = "";
    host.innerHTML = "";

    var current = { ds: "synthetic", scenario: "clean" };

    var controls = el("div", "picker");
    controls.innerHTML =
      '<div class="picker__group" role="group" aria-label="Dataset">' +
      '<span class="picker__legend">Dataset</span>' +
      Object.keys(data.datasets).map(function (k, i) {
        return '<button type="button" class="picker__btn' + (i === 0 ? " is-on" : "") +
          '" data-ds="' + esc(k) + '" aria-pressed="' + (i === 0) + '">' +
          esc(data.datasets[k].title) + "</button>";
      }).join("") + "</div>" +
      '<div class="picker__group" role="group" aria-label="Scenario">' +
      '<span class="picker__legend">Scenario</span>' +
      data.scenario_order.map(function (s, i) {
        return '<button type="button" class="picker__btn' + (i === 0 ? " is-on" : "") +
          '" data-scenario="' + esc(s) + '" aria-pressed="' + (i === 0) + '">' +
          esc(s) + "</button>";
      }).join("") + "</div>";
    host.appendChild(controls);

    var body = el("div", "scenario__body");
    // A live region: choosing a scenario replaces this subtree, and a keyboard
    // or screen-reader user needs to be told that something changed.
    body.setAttribute("aria-live", "polite");
    host.appendChild(body);

    controls.addEventListener("click", function (e) {
      var btn = e.target.closest ? e.target.closest(".picker__btn") : null;
      if (!btn) { return; }
      var group = btn.parentNode;
      Array.prototype.forEach.call(group.querySelectorAll(".picker__btn"), function (b) {
        b.classList.remove("is-on");
        b.setAttribute("aria-pressed", "false");
      });
      btn.classList.add("is-on");
      btn.setAttribute("aria-pressed", "true");
      if (btn.hasAttribute("data-ds")) { current.ds = btn.getAttribute("data-ds"); }
      else { current.scenario = btn.getAttribute("data-scenario"); }
      renderRecord(body, data, current.ds, current.scenario);
    });

    renderRecord(body, data, current.ds, current.scenario);

    host.insertAdjacentHTML("beforeend",
      '<p class="src">computed by web/build_v1.py through app.py ' +
      '<span class="sha">sha256 ' +
      esc((data.provenance.sources[0].sha256 || "").slice(0, 12)) + "</span> — " +
      "seed " + data.seed + ", " + esc(JSON.stringify(data.training)) + "</p>");
  }

  /* ==================================================================
   * Committed V1 figures
   * ================================================================== */

  function renderFigures(host, data) {
    host.className = "";
    host.innerHTML = "";
    var grid = el("div", "grid grid--2");
    data.figures.forEach(function (f) {
      var panel = el("figure", "panel figure");
      if (!f.file) {
        panel.innerHTML = '<div class="notmeasured">' + esc(f.title) +
          " — not present in results/ at build time</div>";
        grid.appendChild(panel);
        return;
      }
      panel.innerHTML =
        '<img class="figure__img" src="data/' + esc(f.file) + '" loading="lazy" ' +
        'alt="' + esc(f.title) + '. ' + esc(f.caption) + '">' +
        "<figcaption class=\"figure__cap\"><strong>" + esc(f.title) + "</strong>" +
        "<span>" + esc(f.caption) + "</span>" +
        '<span class="src">' + esc(f.source) + ' <span class="sha">sha256 ' +
        esc((f.sha256 || "").slice(0, 12)) + "</span></span>" +
        '<span class="src">numbers: ' + esc(f.numbers_from) + "</span></figcaption>";
      grid.appendChild(panel);
    });
    host.appendChild(grid);
    host.insertAdjacentHTML("beforeend",
      '<p class="pass-note">' + esc(data.note) + " " + esc(data.not_shown) + "</p>");
  }

  function mount(id, path, renderer) {
    var host = document.getElementById(id);
    if (!host) { return; }
    load(path).then(function (data) {
      try { renderer(host, data); }
      catch (e) {
        host.className = "notmeasured";
        host.innerHTML = "could not render — " + esc(e && e.message ? e.message : e);
      }
    }).catch(function (e) {
      host.className = "notmeasured";
      host.innerHTML = "could not load committed data — " + esc(e && e.message ? e.message : e) +
        "<br>run <code>python web/build_v1.py</code>";
    });
  }

  mount("v1-scenarios", "data/v1_scenarios.json", renderScenarios);
  mount("v1-figures", "data/v1_figures.json", renderFigures);
})();
