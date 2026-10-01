/* PrismFlow Part 25 -- phase (b). Charts wired to committed data only.
 *
 * Every value rendered here comes from a file in web/data/, which build_data.py
 * copied from a committed result file and stamped with a sha256. Nothing is
 * computed from a formula, interpolated, or carried across experiments. Where a
 * value is null the cell renders "not measured" in the dimmest token -- never
 * blank, never zero, never a guess.
 *
 * A NOTE ON THE ANGLE COLOURS, because a validator disagrees with them.
 * The five angle hues fail an adjacent-pair CVD check as a categorical SET:
 * Market and Financial sit at protan deltaE 5.8. They are not used as a
 * discriminating scale anywhere on this page -- every angle is a labelled
 * COLUMN in a table, so identity comes from position and a text header, and
 * colour is redundant reinforcement. Fixing the pair by darkening Market would
 * collapse its greyscale separation from Tech (1.39 -> 1.10) and break the
 * luminance staircase that makes the set work in print. Position encoding was
 * the cheaper correct answer.
 */
(function () {
  "use strict";

  var NOT_MEASURED = '<span class="nm">not measured</span>';
  var ANGLE_LABEL = {
    tech: "Tech", market: "Market", financial: "Financial",
    regulatory: "Regulatory", sentiment: "Sentiment"
  };

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) { n.className = cls; }
    if (html !== undefined) { n.innerHTML = html; }
    return n;
  }

  /* A number is printed only if it really is a number. null, undefined and NaN
   * all become "not measured" -- the three ways a value can be absent must not
   * render as three different things, and none of them may render as 0. */
  function num(value, digits) {
    if (value === null || value === undefined || (typeof value === "number" && !isFinite(value))) {
      return NOT_MEASURED;
    }
    return Number(value).toFixed(digits === undefined ? 2 : digits);
  }

  function pm(mean, std, digits) {
    if (mean === null || mean === undefined) { return NOT_MEASURED; }
    var out = num(mean, digits);
    if (std !== null && std !== undefined && isFinite(std)) {
      out += ' <span class="pm">± ' + num(std, digits) + "</span>";
    }
    return out;
  }

  /* Everything below builds HTML strings, and the row-detail view is the first
   * thing on this page to print FREE TEXT from a data file -- pitches, dataset
   * notes, provider error bodies. Those contain quotes, ampersands and angle
   * brackets, so they are escaped rather than trusted. Numbers and known enum
   * strings elsewhere in this file were never a concern; free prose is. */
  function esc(value) {
    if (value === null || value === undefined) { return ""; }
    return String(value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function fail(host, message) {
    host.className = "notmeasured";
    host.innerHTML = "could not load committed data — " + message +
      "<br>run <code>python web/build_data.py</code>";
  }

  function load(path) {
    return fetch(path, { cache: "no-store" }).then(function (r) {
      if (!r.ok) { throw new Error(path + " → HTTP " + r.status); }
      return r.json();
    });
  }

  function provenanceLine(prov) {
    if (!prov || !prov.sources) { return ""; }
    return prov.sources.map(function (s) {
      return s.path + " <span class=\"sha\">sha256 " + s.sha256.slice(0, 12) + "</span>";
    }).join("<br>");
  }

  /* ==================================================================
   * 1. V1 -- ENIV over (k copies, rho), as small multiples
   * ================================================================== */

  var RHO_COLOUR = { "0": "var(--angle-tech)", "0.5": "var(--angle-sentiment)" };

  function sparkPanel(system, cells, kValues, rhoValues, domain) {
    var panel = el("div", "panel chart-panel");
    panel.appendChild(el("h4", "chart-panel__title", system));

    var W = 260, H = 150, PAD_L = 38, PAD_B = 26, PAD_T = 10, PAD_R = 8;
    var lo = domain[0], hi = domain[1];
    function x(k) { return PAD_L + (kValues.indexOf(k) / (kValues.length - 1)) * (W - PAD_L - PAD_R); }
    function y(v) { return PAD_T + (1 - (v - lo) / (hi - lo)) * (H - PAD_T - PAD_B); }

    // aria-describedby points at the table holding the same numbers, so the
    // chart is not a dead end for anyone who cannot read the shape.
    var svg = ['<svg viewBox="0 0 ' + W + " " + H + '" class="spark" role="img" ' +
      'aria-describedby="v1-surface-table" ' +
      'aria-label="ENIV versus duplicate copies for ' + system +
      '. The numbers behind this chart are in the table view below it.">'];

    // recessive gridlines + axis ticks
    var ticks = 4, i;
    for (i = 0; i <= ticks; i++) {
      var v = lo + (hi - lo) * (i / ticks);
      svg.push('<line x1="' + PAD_L + '" y1="' + y(v).toFixed(1) + '" x2="' + (W - PAD_R) +
               '" y2="' + y(v).toFixed(1) + '" class="grid"/>');
      svg.push('<text x="' + (PAD_L - 5) + '" y="' + (y(v) + 3).toFixed(1) +
               '" class="tick tick--y">' + v.toFixed(1) + "</text>");
    }
    kValues.forEach(function (k) {
      svg.push('<text x="' + x(k).toFixed(1) + '" y="' + (H - 8) + '" class="tick tick--x">' + k + "</text>");
    });

    rhoValues.forEach(function (rho) {
      var series = cells.filter(function (c) {
        return c.system === system && c.rho === rho && c.eniv_mean !== null;
      }).sort(function (a, b) { return a.k - b.k; });
      if (!series.length) { return; }
      var colour = RHO_COLOUR[String(rho)] || "var(--text-secondary)";

      // std as a band -- the spread is part of the measurement, not decoration
      var up = series.map(function (c) { return x(c.k).toFixed(1) + "," + y(c.eniv_mean + (c.eniv_std || 0)).toFixed(1); });
      var dn = series.map(function (c) { return x(c.k).toFixed(1) + "," + y(c.eniv_mean - (c.eniv_std || 0)).toFixed(1); }).reverse();
      svg.push('<polygon points="' + up.concat(dn).join(" ") + '" fill="' + colour + '" opacity="0.16"/>');

      var pts = series.map(function (c) { return x(c.k).toFixed(1) + "," + y(c.eniv_mean).toFixed(1); });
      svg.push('<polyline points="' + pts.join(" ") + '" fill="none" stroke="' + colour + '" stroke-width="2"/>');
      series.forEach(function (c) {
        svg.push('<circle cx="' + x(c.k).toFixed(1) + '" cy="' + y(c.eniv_mean).toFixed(1) +
                 '" r="3.2" fill="' + colour + '"><title>' + system + "  rho=" + c.rho +
                 "  k=" + c.k + "  ENIV " + c.eniv_mean.toFixed(4) +
                 " ± " + (c.eniv_std === null ? "n/a" : c.eniv_std.toFixed(4)) +
                 "  (" + c.n_seeds + " seeds)</title></circle>");
      });
    });

    svg.push("</svg>");
    panel.insertAdjacentHTML("beforeend", svg.join(""));
    panel.appendChild(el("p", "axis-note", "k = duplicate copies of view 0 →"));
    return panel;
  }

  function renderV1Surface(host, data) {
    host.className = "";
    host.innerHTML = "";

    var kValues = data.axes.k.values;
    var rhoValues = data.axes.rho.values;
    var systems = [];
    data.cells.forEach(function (c) {
      if (c.system && systems.indexOf(c.system) === -1) { systems.push(c.system); }
    });

    // one shared y-domain across the small multiples, so the panels are
    // comparable -- per-panel scaling would make three different pictures
    var vals = data.cells.filter(function (c) { return c.eniv_mean !== null; })
      .map(function (c) { return c.eniv_mean; });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var padding = (hi - lo) * 0.12;
    var domain = [lo - padding, hi + padding];

    var legend = el("div", "legend");
    rhoValues.forEach(function (rho) {
      legend.insertAdjacentHTML("beforeend",
        '<span class="legend__item"><i style="background:' + (RHO_COLOUR[String(rho)] || "#888") +
        '"></i>rho = ' + rho + "</span>");
    });
    legend.insertAdjacentHTML("beforeend",
      '<span class="legend__note">shaded band = ± 1 sample std over ' +
      (data.seeds ? data.seeds.length : "?") + " seeds</span>");
    host.appendChild(legend);

    var grid = el("div", "grid grid--3");
    systems.forEach(function (s) {
      grid.appendChild(sparkPanel(s, data.cells, kValues, rhoValues, domain));
    });
    host.appendChild(grid);

    // The colour channel the brief asked for, and why it is absent.
    var banner = el("div", "notmeasured notmeasured--wide");
    banner.innerHTML = '<strong class="nm">attack success — not measured on this grid</strong><br>' +
      data.attack_success_note;
    host.appendChild(banner);

    // Table view: the numbers themselves, for screen readers and for anyone
    // who wants the value rather than the shape.
    var details = el("details", "table-view");
    details.id = "v1-surface-table";
    details.appendChild(el("summary", null, "Table view — every cell, as committed"));
    var rows = ['<table><caption class="visually-hidden">ENIV by system, rho and ' +
                'duplicate-copy count k, with sample standard deviation and seed ' +
                'count. The attack success column is empty because no committed ' +
                'file measures it on this grid.</caption>' +
                '<thead><tr><th scope="col">system</th><th scope="col" class="num">rho</th>' +
                '<th scope="col" class="num">k</th>' +
                '<th scope="col" class="num">ENIV mean</th><th scope="col" class="num">std</th>' +
                '<th scope="col" class="num">seeds</th>' +
                '<th scope="col" class="num">attack success</th></tr></thead><tbody>'];
    data.cells.slice().sort(function (a, b) {
      return a.system.localeCompare(b.system) || a.rho - b.rho || a.k - b.k;
    }).forEach(function (c) {
      rows.push('<tr><th scope="row">' + c.system + '</th><td class="num">' + c.rho + '</td><td class="num">' + c.k +
        '</td><td class="num">' + num(c.eniv_mean, 4) + '</td><td class="num">' + num(c.eniv_std, 4) +
        '</td><td class="num">' + num(c.n_seeds, 0) + '</td><td class="num">' + num(c.attack_success, 3) +
        "</td></tr>");
    });
    rows.push("</tbody></table>");
    details.insertAdjacentHTML("beforeend", '<div class="table-wrap">' + rows.join("") + "</div>");
    host.appendChild(details);

    host.insertAdjacentHTML("beforeend", '<p class="src">' + provenanceLine(data.provenance) + "</p>");
  }

  /* ==================================================================
   * 2. V1 -- attack success, on its own axes
   * ================================================================== */

  var SYSTEM_COLOUR = {
    naive: "var(--text-secondary)",
    prismflow: "var(--angle-tech)",
    prismflow_nodiscount: "var(--angle-regulatory)"
  };

  function renderV1Attack(host, data) {
    host.className = "";
    host.innerHTML = "";

    var conditions = [];
    data.rows.forEach(function (r) {
      if (conditions.indexOf(r.condition) === -1) { conditions.push(r.condition); }
    });
    var systems = ["naive", "prismflow_nodiscount", "prismflow"];

    var legend = el("div", "legend");
    systems.forEach(function (s) {
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
      systems.forEach(function (sys) {
        var row = data.rows.filter(function (r) { return r.condition === cond && r.system === sys; })[0];
        var line = el("div", "bars__row");
        if (!row || row.success_mean === null) {
          line.innerHTML = '<span class="bars__fillwrap"><span class="bars__nm">' +
            NOT_MEASURED + "</span></span>";
        } else {
          var pct = Math.max(0, Math.min(1, row.success_mean)) * 100;
          line.innerHTML =
            '<span class="bars__fillwrap"><span class="bars__fill" style="width:' + pct.toFixed(1) +
            "%;background:" + SYSTEM_COLOUR[sys] + '"></span></span>' +
            '<span class="bars__val">' + pm(row.success_mean, row.success_std, 3) +
            ' <span class="pm">n=' + num(row.n_seeds, 0) + "</span></span>";
          line.title = cond + " / " + sys + " — success " + row.success_mean.toFixed(4) +
            " ± " + (row.success_std === null ? "n/a" : row.success_std.toFixed(4));
        }
        track.appendChild(line);
      });
      group.appendChild(track);
      wrap.appendChild(group);
    });
    host.appendChild(wrap);

    // The bars are divs, so their values exist only as pixel widths. This is
    // real content, not decoration, so it gets a real table.
    var det = el("details", "table-view");
    det.appendChild(el("summary", null, "Table view — attack success, as committed"));
    var t = ['<div class="table-wrap"><table><caption class="visually-hidden">' +
      'Attack success rate by condition and system, mean and sample standard ' +
      'deviation over seeds.</caption><thead><tr><th scope="col">condition</th>' +
      '<th scope="col">system</th><th scope="col" class="num">success mean</th>' +
      '<th scope="col" class="num">std</th><th scope="col" class="num">seeds</th>' +
      '<th scope="col" class="num">ENIV measured</th></tr></thead><tbody>'];
    data.rows.forEach(function (r) {
      t.push('<tr><th scope="row">' + r.condition + "</th><td>" + r.system +
        '</td><td class="num">' + num(r.success_mean, 4) +
        '</td><td class="num">' + num(r.success_std, 4) +
        '</td><td class="num">' + num(r.n_seeds, 0) +
        '</td><td class="num">' + num(r.eniv_measured, 4) + "</td></tr>");
    });
    t.push("</tbody></table></div>");
    det.insertAdjacentHTML("beforeend", t.join(""));
    host.appendChild(det);

    host.insertAdjacentHTML("beforeend", '<p class="src">' + provenanceLine(data.provenance) + "</p>");
  }

  /* ==================================================================
   * 3. V2 -- per-row angle coverage, all 48 rows
   * ================================================================== */

  function coverageCell(name, a) {
    if (a.retrieval_error) {
      return '<td class="cov cov--err" title="retrieval error: ' +
        String(a.retrieval_error).replace(/"/g, "&quot;") + '"><span class="cov__mark">err</span></td>';
    }
    var records = a.records;
    if (records === null) {
      return '<td class="cov cov--nm" title="not measured"><span class="cov__mark">·</span></td>';
    }
    var cls = "cov cov--" + name + (records > 0 ? " cov--on" : " cov--off");
    // Opacity carries magnitude within a column; the column header carries
    // identity. Colour alone never has to be read.
    var strength = records > 0 ? Math.min(1, 0.32 + (records / 10) * 0.68) : 0;
    var style = records > 0 ? ' style="--strength:' + strength.toFixed(2) + '"' : "";
    var claims = a.claims === null ? "not measured" : a.claims + " claims";
    var errNote = a.reasoner_error ? " · reasoner error" : "";
    // The visible glyph is the record count; the hidden span carries the units
    // and the claim count, because a bare "10" announced on its own says
    // nothing, and a title attribute is not reliably announced at all.
    return "<td class=\"" + cls + "\"" + style + ' title="' + ANGLE_LABEL[name] + ": " +
      records + " records, " + claims + errNote + '"><span class="cov__mark">' +
      (records > 0 ? records : "0") + '</span><span class="visually-hidden"> records, ' +
      claims + errNote + "</span></td>";
  }

  /* ---- row detail ------------------------------------------------------
   * The grid answers "which angles retrieved"; it has no room for the row
   * itself. This opens the whole row: the pitch the pipeline was actually
   * given, the label and the URL that label came from, and every per-angle
   * field the committed run wrote.
   *
   * WHAT IS NOT HERE, AND WHY IT IS STATED RATHER THAN LEFT BLANK.
   * There is no claim text. results/v2/part24_pipeline_verification.json
   * records, per angle, a record count, a claim count, a token count and the
   * mean of the reasoner's self-reported per-claim confidence -- it never
   * wrote the claims. So the detail view prints that sentence instead of an
   * empty panel, because an empty panel reads as a load failure and a dash
   * reads as "no claims". Neither is true: the claims existed and were not
   * retained.
   *
   * mean_claim_conf is labelled as what it is. It is the reasoner's own
   * confidence in its own claims, averaged -- not a calibrated probability,
   * and not the fused confidence, which does not exist on any row because no
   * row fused.
   */

  function sourceLink(url) {
    if (!url) { return NOT_MEASURED; }
    // Only http(s) becomes a link. Anything else is printed as text, so a
    // javascript: or data: value in a data file cannot become clickable.
    if (!/^https?:\/\//i.test(url)) {
      return '<span class="mono">' + esc(url) + "</span>";
    }
    return '<a class="detail__link" href="' + esc(url) +
      '" target="_blank" rel="noopener noreferrer">' + esc(url) +
      '<span class="visually-hidden"> (opens in a new tab)</span></a>';
  }

  function detailAngleRow(name, a, angleNote) {
    var cells = [];
    cells.push('<th scope="row" class="detail__angle detail__angle--' + name + '">' +
      ANGLE_LABEL[name] +
      (angleNote ? '<span class="detail__anglenote">' + esc(angleNote) + "</span>" : "") +
      "</th>");

    // A retrieval exception is not a count of zero, so it has no records cell.
    if (a.retrieval_error) {
      cells.push('<td class="num">' + NOT_MEASURED + "</td>");
    } else {
      cells.push('<td class="num">' + num(a.records, 0) + "</td>");
    }
    cells.push('<td class="num">' + num(a.claims, 0) + "</td>");
    cells.push('<td class="num">' + num(a.mean_claim_conf, 3) + "</td>");
    cells.push('<td class="num">' + num(a.tokens, 0) + "</td>");

    // Which connectors ran, and what each returned. An empty provenance map
    // means no connector was called at all -- printed as such, not as 0.
    var keys = Object.keys(a.provenance || {});
    if (!keys.length) {
      cells.push("<td>" + NOT_MEASURED + "</td>");
    } else {
      cells.push("<td>" + keys.sort().map(function (k) {
        return '<span class="detail__conn' + (a.provenance[k] > 0 ? " detail__conn--on" : "") +
          '">' + esc(k) + " " + num(a.provenance[k], 0) + "</span>";
      }).join(" ") + "</td>");
    }

    // One state column, in the run's own words wherever it wrote any.
    var state = [];
    if (a.retrieval_error) {
      state.push('<span class="pill pill--error">retrieval error</span>' +
        '<span class="detail__err">' + esc(a.retrieval_error) + "</span>");
    }
    if (a.reasoner_error) {
      state.push('<span class="pill pill--error">reasoner error</span>' +
        '<span class="detail__err">' + esc(a.reasoner_error) + "</span>");
    }
    if (a.note) { state.push('<span class="detail__note">' + esc(a.note) + "</span>"); }
    if (!state.length && a.claims > 0) {
      state.push('<span class="pill pill--live">claims produced</span>');
    }
    cells.push("<td>" + (state.length ? state.join("") : NOT_MEASURED) + "</td>");
    return "<tr>" + cells.join("") + "</tr>";
  }

  function detailHTML(data, row, index) {
    var angles = data.angle_order;
    var out = [];

    out.push('<div class="detail__head">' +
      '<p class="detail__id mono">row ' + esc(row.id) + '<span class="detail__pos">' +
      (index + 1) + " of " + data.rows.length + "</span></p>" +
      '<h4 class="detail__title" id="rowdetail-title">' +
      esc(row.domain || "domain not measured") + " — " +
      esc(row.outcome || "outcome not measured") + "</h4></div>");

    out.push('<dl class="detail__meta">' +
      '<dt>idea pitch</dt><dd class="detail__pitch">' +
      (row.idea_pitch ? esc(row.idea_pitch) : NOT_MEASURED) + "</dd>" +
      "<dt>domain</dt><dd>" + (row.domain ? esc(row.domain) : NOT_MEASURED) + "</dd>" +
      "<dt>actual outcome</dt><dd>" + (row.outcome ? esc(row.outcome) : NOT_MEASURED) +
      (row.outcome_date ? ' <span class="pm">' + esc(row.outcome_date) + "</span>" : "") + "</dd>" +
      "<dt>ground truth source</dt><dd>" + sourceLink(row.ground_truth_source) + "</dd>" +
      "<dt>conflict expected</dt><dd>" +
      (row.conflict_expected ? esc(row.conflict_expected) : NOT_MEASURED) + "</dd>" +
      "<dt>dataset notes</dt><dd>" +
      (row.dataset_notes ? esc(row.dataset_notes) : NOT_MEASURED) + "</dd>" +
      "</dl>");

    out.push('<h5 class="detail__h">Per-angle detail</h5>');
    out.push('<div class="table-wrap"><table class="detail__table">' +
      '<caption class="visually-hidden">For row ' + esc(row.id) +
      ": records retrieved, claims produced, mean self-reported claim " +
      "confidence, evidence tokens, which connectors ran, and the run state " +
      "for each of the five angles.</caption><thead><tr>" +
      '<th scope="col">angle</th><th scope="col" class="num">records</th>' +
      '<th scope="col" class="num">claims</th>' +
      '<th scope="col" class="num">mean claim conf</th>' +
      '<th scope="col" class="num">tokens</th>' +
      '<th scope="col">connectors</th><th scope="col">state</th>' +
      "</tr></thead><tbody>");
    angles.forEach(function (name) {
      out.push(detailAngleRow(name, row.angles[name] || {},
        (data.angle_notes || {})[name]));
    });
    out.push("</tbody></table></div>");

    // The honest sentence about claim text. Stated, not implied by a blank.
    out.push('<p class="detail__absent"><span class="nm">claim text — not ' +
      "measured</span> The committed run wrote per-angle counts, a token count " +
      "and the mean of the reasoner's self-reported per-claim confidence. It " +
      "never wrote the claims themselves, so no claim text exists in any file " +
      "this site can read, and none is shown. <em>mean claim conf</em> is the " +
      "reasoner grading its own claims, averaged — not a calibrated " +
      "probability.</p>");

    out.push('<h5 class="detail__h">Fusion</h5>');
    out.push('<dl class="detail__meta detail__meta--num">' +
      "<dt>angles with claims</dt><dd>" + num(row.angles_with_claims, 0) + "</dd>" +
      "<dt>angles with reasoner error</dt><dd>" + num(row.angles_with_reasoner_error, 0) + "</dd>" +
      "<dt>ENIV</dt><dd>" + num(row.eniv, 3) + "</dd>" +
      "<dt>discount</dt><dd>" + num(row.discount, 3) + "</dd>" +
      "<dt>fused confidence</dt><dd>" + num(row.confidence, 3) + "</dd>" +
      "<dt>contradictions</dt><dd>" + num(row.contradictions, 0) + "</dd>" +
      "<dt>wall clock</dt><dd>" + num(row.seconds, 2) +
      (row.seconds === null || row.seconds === undefined ? "" : " s") + "</dd>" +
      "</dl>");
    if (row.run_note) {
      out.push('<p class="pass-note">' + esc(row.run_note) + "</p>");
    }

    // Retrieval warnings are the only place the run explains WHY an angle got
    // nothing -- query truncation, and the missing regulatory connector. They
    // are verbose, so they collapse.
    var warnings = [];
    angles.forEach(function (name) {
      ((row.angles[name] || {}).warnings || []).forEach(function (w) {
        warnings.push('<li><span class="mono">' + ANGLE_LABEL[name] + "</span> " + esc(w) + "</li>");
      });
    });
    if (warnings.length) {
      out.push('<details class="table-view"><summary>Retrieval warnings (' +
        warnings.length + ") — verbatim from the run</summary>" +
        '<ul class="detail__warn">' + warnings.join("") + "</ul></details>");
    }

    return out.join("");
  }

  /* One dialog, reused. Building 48 of them up front would put 48 copies of
   * every pitch in the DOM for a view that shows one row at a time. */
  function mountRowDetail(host, data) {
    var dlg = el("dialog", "rowdlg");
    dlg.setAttribute("aria-labelledby", "rowdetail-title");
    var nav = el("div", "rowdlg__nav");
    nav.innerHTML =
      '<button type="button" class="rowdlg__btn" data-step="-1">‹ previous row</button>' +
      '<button type="button" class="rowdlg__btn" data-step="1">next row ›</button>' +
      '<button type="button" class="rowdlg__btn rowdlg__btn--close" data-close="1">Close</button>';
    var body = el("div", "rowdlg__body");
    dlg.appendChild(nav);
    dlg.appendChild(body);
    host.appendChild(dlg);

    var current = -1;

    function show(index) {
      if (index < 0 || index >= data.rows.length) { return; }
      current = index;
      body.innerHTML = detailHTML(data, data.rows[index], index);
      body.scrollTop = 0;
      Array.prototype.forEach.call(nav.querySelectorAll("[data-step]"), function (b) {
        var next = index + Number(b.getAttribute("data-step"));
        b.disabled = (next < 0 || next >= data.rows.length);
      });
      if (!dlg.open) {
        // <dialog> is baseline in every browser this site targets, but a
        // missing showModal must still leave the content reachable rather
        // than swallow the click.
        if (dlg.showModal) { dlg.showModal(); } else { dlg.setAttribute("open", ""); }
      }
    }

    function close() {
      if (dlg.close) { dlg.close(); } else { dlg.removeAttribute("open"); }
    }

    nav.addEventListener("click", function (e) {
      var btn = e.target.closest ? e.target.closest("button") : null;
      if (!btn) { return; }
      if (btn.hasAttribute("data-close")) { close(); return; }
      show(current + Number(btn.getAttribute("data-step")));
    });

    // A click on the backdrop lands on the dialog element itself, never on a
    // child. Esc is handled by the element.
    dlg.addEventListener("click", function (e) {
      if (e.target === dlg) { close(); }
    });

    // Arrow keys step rows while the dialog is open, which is how anyone
    // actually reads 48 of these.
    //
    // Tab is trapped by hand as well. showModal() is supposed to do this on
    // its own, but measured directly (real CDP-dispatched Tab keypresses,
    // not scripted .click()/.focus()): tabbing off the last focusable node
    // in the body (the <details><summary> warnings list) lands on
    // document.body for one step before the browser's own cycle recovers --
    // a real, reproducible gap, not a hypothetical one. This listener makes
    // the wrap happen on the same keypress instead of one late.
    var FOCUSABLE = 'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"]), summary';
    dlg.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight") { show(current + 1); return; }
      if (e.key === "ArrowLeft") { show(current - 1); return; }
      if (e.key !== "Tab") { return; }
      var nodes = Array.prototype.filter.call(
        dlg.querySelectorAll(FOCUSABLE),
        function (n) { return n.offsetParent !== null; } // skip hidden (e.g. inside a closed <details>)
      );
      if (!nodes.length) { return; }
      var first = nodes[0], last = nodes[nodes.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus();
      }
    });

    return show;
  }

  function renderV2Coverage(host, data) {
    host.className = "";
    host.innerHTML = "";

    var angles = data.angle_order;

    var legend = el("div", "legend");
    angles.forEach(function (a) {
      var extra = (a === "regulatory") ? " legend__item--unbuilt" : "";
      legend.insertAdjacentHTML("beforeend",
        '<span class="legend__item' + extra + '"><i class="cov--' + a +
        '" style="--strength:1"></i>' + ANGLE_LABEL[a] + "</span>");
    });
    legend.insertAdjacentHTML("beforeend",
      '<span class="legend__note">cell shows record count; shade = how many. ' +
      "Regulatory is dashed throughout: no connector exists.</span>");
    host.appendChild(legend);

    var head = ['<div class="table-wrap"><table class="cov-table">' +
      '<caption class="visually-hidden">Per-row angle coverage over all ' +
      (data.n_rows || data.rows.length) + ' evaluation rows. For each row: the ' +
      'number of records each angle retrieved, how many angles produced claims, ' +
      'the outcome of the run, and ENIV where the row fused.</caption>' +
      '<thead><tr>' +
      '<th scope="col">row</th><th scope="col">domain</th><th scope="col">outcome</th>'];
    angles.forEach(function (a) {
      var cls = (a === "regulatory") ? ' class="th--unbuilt"' : "";
      var note = (a === "regulatory") ? " (no connector exists)" : "";
      head.push('<th scope="col"' + cls + ">" + ANGLE_LABEL[a] +
        '<span class="visually-hidden"> records' + note + "</span></th>");
    });
    head.push('<th scope="col" class="num">angles with claims</th>' +
      '<th scope="col">outcome of run</th>' +
      '<th scope="col" class="num">ENIV</th></tr></thead><tbody>');

    data.rows.forEach(function (r, i) {
      // The whole row is clickable, but the CONTROL is the button in the row
      // header. A click handler on a <tr> is invisible to a keyboard and to
      // assistive tech; a real button is focusable, announced, and works with
      // Enter and Space for free.
      var cells = ['<tr class="cov-row" data-index="' + i + '">'];
      cells.push('<th scope="row" class="mono"><button type="button" ' +
        'class="cov-row__btn" data-index="' + i + '" aria-haspopup="dialog">' +
        r.id + '<span class="visually-hidden">: open full detail for this row' +
        "</span></button></th>");
      cells.push("<td>" + (r.domain || NOT_MEASURED) + "</td>");
      cells.push("<td>" + (r.outcome || NOT_MEASURED) + "</td>");
      angles.forEach(function (a) { cells.push(coverageCell(a, r.angles[a] || {})); });
      cells.push('<td class="num">' + num(r.angles_with_claims, 0) + "</td>");

      var status;
      if (r.zero_claim_cause === "provider_error") {
        status = '<span class="pill pill--error">provider error</span>';
      } else if (r.zero_claim_cause === "genuine") {
        status = '<span class="pill pill--zero">genuine zero</span>';
      } else if (r.angles_with_reasoner_error) {
        status = '<span class="pill pill--error">claims + lost angle</span>';
      } else {
        status = '<span class="pill pill--live">produced claims</span>';
      }
      cells.push("<td>" + status + "</td>");
      // A row that never fused has no ENIV. That is not zero.
      cells.push('<td class="num">' + num(r.eniv, 3) + "</td>");
      cells.push("</tr>");
      head.push(cells.join(""));
    });
    head.push("</tbody></table></div>");
    host.insertAdjacentHTML("beforeend", head.join(""));

    host.insertAdjacentHTML("beforeend",
      '<p class="table-wrap__hint">Every row opens: click it, or tab to a row ' +
      "number and press Enter, for the pitch, the ground-truth link and the " +
      "per-angle detail.</p>");

    // One delegated handler for all 48 rows, and one dialog for all of them.
    var showRow = mountRowDetail(host, data);
    var tbody = host.querySelector(".cov-table tbody");
    if (tbody) {
      tbody.addEventListener("click", function (e) {
        var tr = e.target.closest ? e.target.closest("tr[data-index]") : null;
        // A click on the ground-truth link inside a row must follow the link,
        // not open the dialog. There is no such link in the grid today, but
        // the guard costs nothing and is the usual trap here.
        if (!tr || (e.target.closest && e.target.closest("a"))) { return; }
        showRow(Number(tr.getAttribute("data-index")));
      });
    }

    if (data.pass1) {
      host.insertAdjacentHTML("beforeend",
        '<p class="pass-note">' + data.pass1.note + "</p>");
    }
    host.insertAdjacentHTML("beforeend", '<p class="src">' + provenanceLine(data.provenance) + "</p>");
  }

  /* ================================================================== */

  function mount(id, path, renderer) {
    var host = document.getElementById(id);
    if (!host) { return; }
    load(path).then(function (data) {
      try { renderer(host, data); }
      catch (e) { fail(host, String(e && e.message ? e.message : e)); }
    }).catch(function (e) {
      fail(host, String(e && e.message ? e.message : e));
    });
  }

  mount("v1-surface", "data/v1_surface.json", renderV1Surface);
  mount("v1-attack", "data/v1_attack.json", renderV1Attack);
  mount("v2-coverage", "data/v2_coverage.json", renderV2Coverage);
})();
