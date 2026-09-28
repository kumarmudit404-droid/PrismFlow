/* PrismFlow -- the "Scores" page. Every number here is read from web/data/,
 * which build_data.py copied from a committed result file and stamped with a
 * sha256. This file computes nothing except client-side filtering and
 * counting over rows/cells that are already committed numbers -- it never
 * derives a metric a source file doesn't already report, and every null in a
 * source file renders as "not measured", never 0 or blank.
 *
 * Filters are domain / angle / system. Domain and angle apply to the two V2
 * panels (the only ones with those dimensions); system applies to the three
 * V1 panels (the only ones with more than one system). A panel a filter
 * doesn't apply to ignores that filter rather than guessing what it should do
 * with it -- documented once, in the filter-bar note in scores.html, not
 * repeated as a caveat under every panel.
 */
(function () {
  "use strict";

  var NOT_MEASURED = '<span class="nm">not measured</span>';
  var ANGLE_ORDER = ["tech", "market", "financial", "regulatory", "sentiment"];
  var ANGLE_LABEL = {
    tech: "Tech", market: "Market", financial: "Financial",
    regulatory: "Regulatory", sentiment: "Sentiment"
  };
  var SYSTEM_COLOUR = {
    naive: "var(--text-secondary)",
    prismflow: "var(--angle-tech)",
    prismflow_nodiscount: "var(--angle-regulatory)",
    naive_weights_discounted: "var(--angle-financial)"
  };
  var ANGLE_COLOUR = {
    tech: "var(--angle-tech)", market: "var(--angle-market)",
    financial: "var(--angle-financial)", regulatory: "var(--angle-regulatory)",
    sentiment: "var(--angle-sentiment)"
  };

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) { n.className = cls; }
    if (html !== undefined) { n.innerHTML = html; }
    return n;
  }

  function num(value, digits) {
    if (value === null || value === undefined ||
        (typeof value === "number" && !isFinite(value))) {
      return NOT_MEASURED;
    }
    return Number(value).toFixed(digits === undefined ? 3 : digits);
  }

  function pm(mean, std, digits) {
    if (mean === null || mean === undefined) { return NOT_MEASURED; }
    var out = num(mean, digits);
    if (std !== null && std !== undefined && isFinite(std)) {
      out += ' <span class="pm">± ' + num(std, digits) + "</span>";
    }
    return out;
  }

  function esc(value) {
    if (value === null || value === undefined) { return ""; }
    return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function load(path) {
    return fetch(path, { cache: "no-store" }).then(function (r) {
      if (!r.ok) { throw new Error(path + " -> HTTP " + r.status); }
      return r.json();
    });
  }

  function provenanceLine(prov) {
    if (!prov || !prov.sources) { return ""; }
    return prov.sources.map(function (s) {
      return s.path + ' <span class="sha">sha256 ' + s.sha256.slice(0, 12) + "</span>";
    }).join("<br>");
  }

  function bar(label, mean, std, digits, max, colour, title) {
    var pct = mean === null || mean === undefined ? 0 :
      Math.max(0, Math.min(1, mean / max)) * 100;
    var line = el("div", "bars__row");
    if (mean === null || mean === undefined) {
      line.innerHTML = '<span class="bars__fillwrap"><span class="bars__nm">' +
        NOT_MEASURED + "</span></span>";
    } else {
      line.innerHTML =
        '<span class="bars__fillwrap"><span class="bars__fill" style="width:' +
        pct.toFixed(1) + "%;background:" + colour + '"></span></span>' +
        '<span class="bars__val">' + pm(mean, std, digits) + "</span>";
    }
    if (title) { line.title = title; }
    return line;
  }

  function tableView(summary, tableHtml) {
    var det = el("details", "table-view");
    det.appendChild(el("summary", null, summary));
    det.insertAdjacentHTML("beforeend", tableHtml);
    return det;
  }

  /* ======================================================================
   * KPI cards
   * ==================================================================== */

  function renderKPIs(host, d) {
    var totalRows = d.coverage.rows.length;
    var causeCounts = { provider_error: 0, genuine: 0, none: 0 };
    d.coverage.rows.forEach(function (r) {
      var c = r.zero_claim_cause;
      if (c === "provider_error") { causeCounts.provider_error++; }
      else if (c === "genuine") { causeCounts.genuine++; }
      else { causeCounts.none++; }
    });
    var zeroClaim = causeCounts.provider_error + causeCounts.genuine;

    var angleFullCoverage = ANGLE_ORDER.filter(function (a) {
      return d.coverage.rows.every(function (r) {
        var e = r.angles[a];
        return e && typeof e.records === "number" && e.records > 0;
      });
    });

    var cards = [
      ["V2 dataset rows", String(d.dataset.n_rows), "data/v2/evaluation_queries.json"],
      ["angles with retrieval on every row", angleFullCoverage.length + " of " + ANGLE_ORDER.length,
        angleFullCoverage.length ? angleFullCoverage.join(", ") : "none"],
      ["V2 rows with zero claims", zeroClaim + " of " + totalRows,
        causeCounts.provider_error + " provider_error, " + causeCounts.genuine + " genuine"],
      ["V1 non-clean conditions PrismFlow lost", d.failures.n_included + " of " + (d.failures.n_conditions_total - 1),
        "clean excluded: tied at 0.000, not lost — 5 seeds each — results/chorus/attack_metrics.json"]
    ];
    host.innerHTML = "";
    cards.forEach(function (c) {
      host.appendChild(el("div", "panel stat",
        '<span class="stat__value">' + esc(c[1]) + "</span>" +
        '<span class="stat__label">' + esc(c[0]) + "</span>" +
        '<span class="stat__src">' + esc(c[2]) + "</span>"));
    });
  }

  /* ======================================================================
   * V2 -- dataset composition by domain
   * ==================================================================== */

  function renderDomains(host, data, domainFilter) {
    host.className = "";
    host.innerHTML = "";
    var domains = Object.keys(data.domains).sort();
    var max = Math.max.apply(null, domains.map(function (k) { return data.domains[k]; }));

    var wrap = el("div", "panel bars");
    var group = el("div", "bars__group");
    group.appendChild(el("div", "bars__label", "rows by domain (" + data.n_rows + " total)"));
    var track = el("div", "bars__track");
    domains.forEach(function (dom) {
      var n = data.domains[dom];
      var dimmed = domainFilter && domainFilter !== dom;
      var colour = dimmed ? "var(--status-not-measured)" : "var(--accent)";
      var line = bar(dom, n, null, 0, max, colour,
        dom + " — " + n + " rows — data/v2/evaluation_queries.json");
      line.insertAdjacentHTML("afterbegin",
        '<span class="bars__label" style="min-width:8rem;flex:0 0 auto">' + esc(dom) + "</span>");
      track.appendChild(line);
    });
    group.appendChild(track);
    wrap.appendChild(group);
    host.appendChild(wrap);

    var rows = domainFilter
      ? data.rows.filter(function (r) { return r.domain === domainFilter; })
      : data.rows;
    var t = ['<div class="table-wrap"><table><caption class="visually-hidden">',
      'V2 dataset rows, id, domain, recorded outcome and ground-truth source.',
      '</caption><thead><tr><th scope="col">id</th><th scope="col">domain</th>',
      '<th scope="col">outcome</th><th scope="col">outcome date</th>',
      '<th scope="col">conflict expected</th></tr></thead><tbody>'];
    rows.forEach(function (r) {
      t.push("<tr><th scope=\"row\" class=\"mono\">" + esc(r.id) + "</th><td>" +
        esc(r.domain) + "</td><td>" + esc(r.outcome) + "</td><td class=\"mono\">" +
        esc(r.outcome_date) + "</td><td>" + esc(r.conflict_expected) + "</td></tr>");
    });
    t.push("</tbody></table></div>");
    host.appendChild(tableView(
      "Table view — " + rows.length + " of " + data.n_rows + " rows shown", t.join("")));
    host.insertAdjacentHTML("beforeend", '<p class="src">' + provenanceLine(data.provenance) + "</p>");
  }

  /* ======================================================================
   * V2 -- per-angle retrieval coverage
   * ==================================================================== */

  function renderCoverage(host, data, domainFilter, angleFilter) {
    host.className = "";
    host.innerHTML = "";
    var rows = domainFilter
      ? data.rows.filter(function (r) { return r.domain === domainFilter; })
      : data.rows;
    var n = rows.length;

    var legend = el("div", "legend");
    ANGLE_ORDER.forEach(function (a) {
      legend.insertAdjacentHTML("beforeend",
        '<span class="legend__item"><i style="background:' + ANGLE_COLOUR[a] + '"></i>' +
        ANGLE_LABEL[a] + "</span>");
    });
    host.appendChild(legend);

    var wrap = el("div", "panel bars");
    ["records", "claims"].forEach(function (field) {
      var group = el("div", "bars__group");
      group.appendChild(el("div", "bars__label",
        "rows with >=1 " + field + (domainFilter ? " (" + domainFilter + " only)" : "") +
        ", of " + n));
      var track = el("div", "bars__track");
      ANGLE_ORDER.forEach(function (a) {
        var count = rows.filter(function (r) {
          var e = r.angles[a];
          return e && typeof e[field] === "number" && e[field] > 0;
        }).length;
        var dimmed = angleFilter && angleFilter !== a;
        var colour = dimmed ? "var(--status-not-measured)" : ANGLE_COLOUR[a];
        var line = bar(ANGLE_LABEL[a], count, null, 0, Math.max(1, n), colour,
          ANGLE_LABEL[a] + " — " + count + " of " + n + " rows — " +
          "results/v2/part24_pipeline_verification.json (single pass, NOT A PART 24 FINDING)");
        line.insertAdjacentHTML("afterbegin",
          '<span class="bars__label" style="min-width:7rem;flex:0 0 auto">' +
          esc(ANGLE_LABEL[a]) + "</span>");
        track.appendChild(line);
      });
      group.appendChild(track);
      wrap.appendChild(group);
    });
    host.appendChild(wrap);

    var visibleAngles = angleFilter ? [angleFilter] : ANGLE_ORDER;
    var t = ['<div class="table-wrap"><table><caption class="visually-hidden">',
      'Per-row angle coverage: records retrieved and claims produced.</caption>',
      "<thead><tr><th scope=\"col\">id</th><th scope=\"col\">domain</th>"];
    visibleAngles.forEach(function (a) {
      t.push('<th scope="col" class="num">' + esc(ANGLE_LABEL[a]) + " rec/claims</th>");
    });
    t.push('<th scope="col">zero-claim cause</th></tr></thead><tbody>');
    rows.forEach(function (r) {
      t.push("<tr><th scope=\"row\" class=\"mono\">" + esc(r.id) + "</th><td>" + esc(r.domain) + "</td>");
      visibleAngles.forEach(function (a) {
        var e = r.angles[a] || {};
        var recs = typeof e.records === "number" ? e.records : "not measured";
        var claims = typeof e.claims === "number" ? e.claims : "not measured";
        t.push('<td class="num mono">' + esc(recs) + " / " + esc(claims) + "</td>");
      });
      t.push("<td>" + (r.zero_claim_cause ? esc(r.zero_claim_cause) : "—") + "</td></tr>");
    });
    t.push("</tbody></table></div>");
    host.appendChild(tableView("Table view — " + n + " rows shown", t.join("")));
    host.insertAdjacentHTML("beforeend", '<p class="src">' + provenanceLine(data.provenance) + "</p>");
  }

  /* ======================================================================
   * Real failures -- V1 conditions PrismFlow lost + V2 zero-claim rows
   * ==================================================================== */

  function renderFailures(host, failures, coverage) {
    host.className = "";
    host.innerHTML = "";

    host.appendChild(el("h4", "chart-panel__title",
      "V1 — attack conditions where PrismFlow's success rate exceeded naive's"));
    var t1 = ['<div class="table-wrap"><table><caption class="visually-hidden">',
      'V1 attack conditions where PrismFlow lost to naive, mean and standard ',
      'deviation over 5 seeds.</caption><thead><tr><th scope="col">condition</th>',
      '<th scope="col" class="num">naive success</th>',
      '<th scope="col" class="num">prismflow success</th>',
      '<th scope="col" class="num">delta</th>',
      '<th scope="col" class="num">seeds worse / total</th>',
      '<th scope="col">fooled example</th></tr></thead><tbody>'];
    failures.conditions.forEach(function (c) {
      t1.push("<tr><th scope=\"row\" class=\"mono\">" + esc(c.condition) + "</th><td class=\"num\">" +
        pm(c.naive.mean, c.naive.std, 4) + '</td><td class="num">' +
        pm(c.prismflow.mean, c.prismflow.std, 4) + '</td><td class="num">' +
        num(c.delta_mean, 4) + '</td><td class="num mono">' +
        esc(c.seeds_prismflow_worse) + " / " + esc(c.n_seeds) + "</td><td>" +
        esc(c.example_note) + "</td></tr>");
    });
    t1.push("</tbody></table></div>");
    host.insertAdjacentHTML("beforeend", t1.join(""));
    host.insertAdjacentHTML("beforeend",
      '<p class="axis-note">excluded: ' + failures.excluded.map(function (e) {
        return esc(e.condition) + " (" + esc(e.why) + ")";
      }).join("; ") + "</p>");
    host.insertAdjacentHTML("beforeend", '<p class="src">' + provenanceLine(failures.provenance) + "</p>");

    host.appendChild(el("h4", "chart-panel__title", "V2 — rows that produced zero claims, by cause"));
    var byCause = { provider_error: [], genuine: [] };
    coverage.rows.forEach(function (r) {
      if (r.zero_claim_cause === "provider_error" || r.zero_claim_cause === "genuine") {
        var erroredAngles = ANGLE_ORDER.filter(function (a) {
          return r.angles[a] && r.angles[a].reasoner_error;
        });
        byCause[r.zero_claim_cause].push({ id: r.id, domain: r.domain, angles: erroredAngles });
      }
    });
    var t2 = ['<div class="table-wrap"><table><caption class="visually-hidden">',
      'V2 rows with zero claims, grouped by cause.</caption>',
      '<thead><tr><th scope="col">cause</th><th scope="col" class="num">rows</th>',
      '<th scope="col">row ids</th></tr></thead><tbody>'];
    ["provider_error", "genuine"].forEach(function (cause) {
      var list = byCause[cause];
      t2.push("<tr><th scope=\"row\">" + esc(cause) + "</th><td class=\"num mono\">" +
        list.length + "</td><td class=\"mono\">" +
        list.map(function (r) { return r.id; }).join(", ") + "</td></tr>");
    });
    t2.push("</tbody></table></div>");
    host.insertAdjacentHTML("beforeend", t2.join(""));
    host.insertAdjacentHTML("beforeend",
      '<p class="axis-note">"provider_error" rows carry a reasoner_error on at ' +
      "least one angle (Groq rate limits and malformed-JSON responses, per-row " +
      "in results/v2/part24_pipeline_verification.json); \"genuine\" rows " +
      "retrieved evidence and the reasoner ran, but produced no claims. Single " +
      "pass — not a Part 24 finding.</p>");
  }

  /* ======================================================================
   * V1 -- attack success rate, grouped by condition and system
   * ==================================================================== */

  function renderAttack(host, data, systemFilter) {
    host.className = "";
    host.innerHTML = "";
    var conditions = [];
    data.rows.forEach(function (r) {
      if (conditions.indexOf(r.condition) === -1) { conditions.push(r.condition); }
    });
    var systems = ["naive", "prismflow_nodiscount", "prismflow"];
    var shown = systemFilter && systems.indexOf(systemFilter) !== -1 ? [systemFilter] : systems;

    var legend = el("div", "legend");
    shown.forEach(function (s) {
      legend.insertAdjacentHTML("beforeend",
        '<span class="legend__item"><i style="background:' + SYSTEM_COLOUR[s] + '"></i>' + s + "</span>");
    });
    legend.insertAdjacentHTML("beforeend",
      '<span class="legend__note">rho fixed at ' + num(data.rho_fixed, 1) +
      " — not an axis in this experiment</span>");
    host.appendChild(legend);

    var wrap = el("div", "panel bars");
    conditions.forEach(function (cond) {
      var group = el("div", "bars__group");
      group.appendChild(el("div", "bars__label", cond));
      var track = el("div", "bars__track");
      shown.forEach(function (sys) {
        var row = data.rows.filter(function (r) { return r.condition === cond && r.system === sys; })[0];
        var line = bar(sys, row ? row.success_mean : null, row ? row.success_std : null, 3, 1,
          SYSTEM_COLOUR[sys],
          row && row.success_mean !== null
            ? cond + " / " + sys + " — success " + row.success_mean.toFixed(4) +
              " ± " + (row.success_std === null ? "n/a" : row.success_std.toFixed(4)) +
              " (n=" + row.n_seeds + ") — results/chorus/attack_metrics.json"
            : undefined);
        track.appendChild(line);
      });
      group.appendChild(track);
      wrap.appendChild(group);
    });
    host.appendChild(wrap);

    var t = ['<div class="table-wrap"><table><caption class="visually-hidden">',
      'Attack success rate by condition and system.</caption><thead><tr>',
      '<th scope="col">condition</th><th scope="col">system</th>',
      '<th scope="col" class="num">success mean</th><th scope="col" class="num">std</th>',
      '<th scope="col" class="num">seeds</th></tr></thead><tbody>'];
    data.rows.forEach(function (r) {
      if (systemFilter && r.system !== systemFilter) { return; }
      t.push("<tr><th scope=\"row\">" + esc(r.condition) + "</th><td>" + esc(r.system) +
        '</td><td class="num">' + num(r.success_mean, 4) + '</td><td class="num">' +
        num(r.success_std, 4) + '</td><td class="num">' + num(r.n_seeds, 0) + "</td></tr>");
    });
    t.push("</tbody></table></div>");
    host.appendChild(tableView("Table view — attack success, as committed", t.join("")));
    host.insertAdjacentHTML("beforeend", '<p class="src">' + provenanceLine(data.provenance) + "</p>");
  }

  /* ======================================================================
   * V1 -- calibration under view duplication
   * ==================================================================== */

  function renderCalibration(host, data, systemFilter) {
    host.className = "";
    host.innerHTML = "";
    var systems = data.axes.system.values;
    var shown = systemFilter && systems.indexOf(systemFilter) !== -1 ? [systemFilter] : systems;

    var legend = el("div", "legend");
    shown.forEach(function (s) {
      legend.insertAdjacentHTML("beforeend",
        '<span class="legend__item"><i style="background:' + SYSTEM_COLOUR[s] + '"></i>' + s + "</span>");
    });
    host.appendChild(legend);

    [["accuracy", 3, 1], ["prob_ece", 4, 0.5]].forEach(function (spec) {
      var metric = spec[0], digits = spec[1], max = spec[2];
      var wrap = el("div", "panel bars");
      data.axes.k.values.forEach(function (k) {
        var group = el("div", "bars__group");
        group.appendChild(el("div", "bars__label", metric + " — k=" + k + " duplicate view(s)"));
        var track = el("div", "bars__track");
        shown.forEach(function (sys) {
          var cell = data.cells.filter(function (c) { return c.k === k && c.system === sys; })[0];
          var mean = cell ? cell[metric + "_mean"] : null;
          var std = cell ? cell[metric + "_std"] : null;
          var line = bar(sys, mean, std, digits, max, SYSTEM_COLOUR[sys],
            sys + " k=" + k + " " + metric + " — " +
            (mean === null ? "not measured" : mean.toFixed(digits)) +
            " — results/calibration_duplicated/k" + k + "_" + sys + "/metrics.json");
          line.insertAdjacentHTML("afterbegin",
            '<span class="bars__label" style="min-width:9rem;flex:0 0 auto">' + esc(sys) + "</span>");
          track.appendChild(line);
        });
        group.appendChild(track);
        wrap.appendChild(group);
      });
      host.appendChild(wrap);
    });

    var t = ['<div class="table-wrap"><table><caption class="visually-hidden">',
      'Calibration under view duplication, all committed metrics.</caption>',
      "<thead><tr><th scope=\"col\">k</th><th scope=\"col\">system</th>"];
    data.metrics.forEach(function (m) { t.push('<th scope="col" class="num">' + esc(m) + "</th>"); });
    t.push("</tr></thead><tbody>");
    data.cells.forEach(function (c) {
      if (systemFilter && c.system !== systemFilter) { return; }
      t.push("<tr><th scope=\"row\" class=\"mono\">" + esc(c.k) + "</th><td>" + esc(c.system) + "</td>");
      data.metrics.forEach(function (m) {
        t.push('<td class="num">' + pm(c[m + "_mean"], c[m + "_std"], 4) + "</td>");
      });
      t.push("</tr>");
    });
    t.push("</tbody></table></div>");
    host.appendChild(tableView("Table view — every committed calibration metric", t.join("")));
    host.insertAdjacentHTML("beforeend", '<p class="src">' + provenanceLine(data.provenance) + "</p>");
  }

  /* ======================================================================
   * V1 -- ENIV under view duplication
   * ==================================================================== */

  function renderENIV(host, data, systemFilter) {
    host.className = "";
    host.innerHTML = "";
    var systems = ["naive", "prismflow"];
    var shown = systemFilter && systems.indexOf(systemFilter) !== -1 ? [systemFilter] : systems;
    var kValues = data.axes.k.values;

    var legend = el("div", "legend");
    shown.forEach(function (s) {
      legend.insertAdjacentHTML("beforeend",
        '<span class="legend__item"><i style="background:' + SYSTEM_COLOUR[s] + '"></i>' + s + "</span>");
    });
    legend.insertAdjacentHTML("beforeend", '<span class="legend__note">rho = 0.0</span>');
    host.appendChild(legend);

    var wrap = el("div", "panel bars");
    kValues.forEach(function (k) {
      var group = el("div", "bars__group");
      group.appendChild(el("div", "bars__label", "ENIV — k=" + k + " duplicate copies (rho=0.0)"));
      var track = el("div", "bars__track");
      shown.forEach(function (sys) {
        var cell = data.cells.filter(function (c) {
          return c.k === k && c.rho === 0 && c.system === sys;
        })[0];
        var mean = cell ? cell.eniv_mean : null;
        var std = cell ? cell.eniv_std : null;
        var line = bar(sys, mean, std, 3, 5, SYSTEM_COLOUR[sys],
          sys + " k=" + k + " ENIV — " + (mean === null ? "not measured" : mean.toFixed(3)) +
          " — results/clone_eigen/summary.json");
        line.insertAdjacentHTML("afterbegin",
          '<span class="bars__label" style="min-width:7rem;flex:0 0 auto">' + esc(sys) + "</span>");
        track.appendChild(line);
      });
      group.appendChild(track);
      wrap.appendChild(group);
    });
    host.appendChild(wrap);
    host.insertAdjacentHTML("beforeend", '<p class="axis-note">' + esc(data.attack_success_note) + "</p>");

    var t = ['<div class="table-wrap"><table><caption class="visually-hidden">',
      'ENIV under view duplication, both rho values.</caption><thead><tr>',
      '<th scope="col">rho</th><th scope="col">k</th><th scope="col">system</th>',
      '<th scope="col" class="num">ENIV</th><th scope="col" class="num">confidence</th>',
      '<th scope="col" class="num">accuracy</th></tr></thead><tbody>'];
    data.cells.forEach(function (c) {
      if (systemFilter && c.system !== systemFilter) { return; }
      t.push("<tr><td class=\"mono\">" + esc(c.rho) + "</td><td class=\"mono\">" + esc(c.k) +
        "</td><td>" + esc(c.system) + '</td><td class="num">' +
        pm(c.eniv_mean, c.eniv_std, 4) + '</td><td class="num">' +
        pm(c.confidence_mean, c.confidence_std, 4) + '</td><td class="num">' +
        num(c.accuracy_mean, 4) + "</td></tr>");
    });
    t.push("</tbody></table></div>");
    host.appendChild(tableView("Table view — ENIV, confidence, accuracy, both rho values", t.join("")));
    host.insertAdjacentHTML("beforeend", '<p class="src">' + provenanceLine(data.provenance) + "</p>");
  }

  /* ======================================================================
   * boot
   * ==================================================================== */

  var kpiHost = document.getElementById("sc-kpi-cards");
  if (!kpiHost) { return; }

  Promise.all([
    load("data/v2_dataset.json"),
    load("data/v2_coverage.json"),
    load("data/v1_attack.json"),
    load("data/v1_failures.json"),
    load("data/calibration_duplicated.json"),
    load("data/v1_surface.json")
  ]).then(function (all) {
    var d = {
      dataset: all[0], coverage: all[1], attack: all[2],
      failures: all[3], calibration: all[4], surface: all[5]
    };

    renderKPIs(kpiHost, d);

    var state = { domain: "", angle: "", system: "" };

    var fDomain = document.getElementById("sc-f-domain");
    Object.keys(d.dataset.domains).sort().forEach(function (dom) {
      fDomain.insertAdjacentHTML("beforeend",
        '<option value="' + esc(dom) + '">' + esc(dom) + "</option>");
    });
    var fAngle = document.getElementById("sc-f-angle");
    ANGLE_ORDER.forEach(function (a) {
      fAngle.insertAdjacentHTML("beforeend",
        '<option value="' + esc(a) + '">' + esc(ANGLE_LABEL[a]) + "</option>");
    });
    var fSystem = document.getElementById("sc-f-system");
    ["naive", "prismflow", "prismflow_nodiscount", "naive_weights_discounted"].forEach(function (s) {
      fSystem.insertAdjacentHTML("beforeend",
        '<option value="' + esc(s) + '">' + esc(s) + "</option>");
    });

    function draw() {
      renderDomains(document.getElementById("sc-panel-domains"), d.dataset, state.domain);
      renderCoverage(document.getElementById("sc-panel-coverage"), d.coverage, state.domain, state.angle);
      renderFailures(document.getElementById("sc-panel-failures"), d.failures, d.coverage);
      renderAttack(document.getElementById("sc-panel-attack"), d.attack, state.system);
      renderCalibration(document.getElementById("sc-panel-calibration"), d.calibration, state.system);
      renderENIV(document.getElementById("sc-panel-eniv"), d.surface, state.system);
    }

    fDomain.addEventListener("change", function () { state.domain = fDomain.value; draw(); });
    fAngle.addEventListener("change", function () { state.angle = fAngle.value; draw(); });
    fSystem.addEventListener("change", function () { state.system = fSystem.value; draw(); });
    document.getElementById("sc-f-reset").addEventListener("click", function () {
      state.domain = ""; state.angle = ""; state.system = "";
      fDomain.value = ""; fAngle.value = ""; fSystem.value = "";
      draw();
    });

    draw();
  }).catch(function (e) {
    kpiHost.className = "notmeasured";
    kpiHost.innerHTML = "could not load committed data — " + esc(e && e.message ? e.message : e) +
      "<br>run <code>python web/build_data.py</code>";
  });
})();
