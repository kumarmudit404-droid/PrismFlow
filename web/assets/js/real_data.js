/* real_data.js -- the Part 31 real-data section.
 *
 * Four mount points, all fed from web/data/real_*.json, which build_data.py
 * wrote from committed results/ files. Nothing here computes a figure; the
 * only arithmetic is formatting.
 *
 * TWO RULES THIS FILE ADDS TO THE SITE'S USUAL ONES
 * -------------------------------------------------
 * AIT'S TWO DIRECTIONS ARE NEVER MERGED. Part 29 forbids pooling the two
 * scenarios, so every AIT table here is keyed by direction and there is no
 * "both" row anywhere. The same goes for the dns-compromised split: L17 says
 * a cell that does not name the compromised set is uninterpretable, so the
 * split is a column, not a filter that could be left off.
 *
 * A VERDICT IS NEVER SHOWN WITHOUT ITS RULE. The claim table's words are the
 * strongest thing on this page. Each cell therefore renders the verdict, the
 * written rule the cited file had to satisfy, what the file actually said,
 * and a link to that file's path with its sha256. A reader who disagrees with
 * a verdict can see exactly which test produced it.
 */
(function (global) {
  "use strict";

  var doc = global.document;
  var NOT_MEASURED = '<span class="pill pill--notmeasured">not measured</span>';

  function esc(value) {
    if (value === null || value === undefined) { return ""; }
    return String(value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function num(value, digits) {
    if (value === null || value === undefined || isNaN(value)) { return NOT_MEASURED; }
    return Number(value).toFixed(digits === undefined ? 4 : digits);
  }

  /* mean +/- std over n seeds. A mean with no std is labelled as one value
   * rather than printed as if it had a spread -- the 5-SEED rule is the whole
   * reason these tables exist. */
  function pm(block, digits) {
    if (!block || block.mean === null || block.mean === undefined) { return NOT_MEASURED; }
    var d = (digits === undefined) ? 4 : digits;
    var body = Number(block.mean).toFixed(d);
    if (block.std === null || block.std === undefined) {
      return body + ' <span class="unit">(single value, no std)</span>';
    }
    return body + " &plusmn; " + Number(block.std).toFixed(d) +
      (block.n_seeds ? ' <span class="unit">(' + block.n_seeds + " seeds)</span>" : "");
  }

  function provenanceLine(prov) {
    if (!prov || !prov.sources) { return ""; }
    return prov.sources.map(function (s) {
      return esc(s.path) + " <span class='mono'>" + esc(s.sha256.slice(0, 12)) + "</span>";
    }).join(" &middot; ");
  }

  function fail(host, message) {
    host.className = "placeholder placeholder--error";
    host.textContent = "could not load: " + message;
  }

  function load(path) {
    return fetch(path, { cache: "no-store" }).then(function (r) {
      if (!r.ok) { throw new Error(path + " -> HTTP " + r.status); }
      return r.json();
    });
  }

  /* ---- dataset cards -------------------------------------------------- */

  function datasetCard(d, extra) {
    var rows = [];
    function row(k, v) { rows.push("<dt>" + esc(k) + "</dt><dd>" + v + "</dd>"); }

    row("views", d.views ? d.views.map(function (v) {
      return '<code>' + esc(v) + "</code>";
    }).join(", ") : NOT_MEASURED);
    row("label method", d.label_method ? esc(d.label_method) : NOT_MEASURED);
    if (d.scenarios) { row("scenarios used", d.scenarios.map(esc).join(", ")); }
    if (d.days_train) {
      row("days used", "train " + d.days_train.join("-") +
        ", test " + d.days_test.join("-") +
        (d.window_seconds ? ", " + d.window_seconds + " s windows" : ""));
    }

    // Licence and attribution are an obligation, not decoration. Where the
    // repository holds the text it is quoted; where it does not, the card
    // says so instead of carrying an invented citation or a download link.
    if (d.licence) {
      row("licence", '<a href="' + esc(d.licence_url) + '" rel="noreferrer">' +
        esc(d.licence) + "</a>");
      row("source", '<a href="' + esc(d.source_url) + '" rel="noreferrer">' +
        esc(d.source_url) + "</a>");
      row("attribution", '<q class="attrib">' + esc(d.attribution) + "</q>" +
        ' <span class="unit">quoted from ' + esc(d.attribution_source) + "</span>");
    } else {
      row("licence", '<span class="pill pill--notmeasured">not in this repository</span>');
      row("attribution", '<p class="nm nm--why">' + esc(d.attribution_absent_note) +
        "</p>");
    }
    if (extra) { rows.push(extra); }

    return '<div class="panel"><h4>' + esc(d.name) + "</h4>" +
      '<dl class="kv">' + rows.join("") + "</dl></div>";
  }

  /* ---- dependence heatmap --------------------------------------------- */

  /* Reuses the coverage-grid shading idiom: one cell per pair, shade
   * proportional to the measured dependence, and the number printed in the
   * cell as well. Shade is never the only encoder -- the value is always
   * readable as text, which is the site's standing rule for colour. */
  function heatmap(cells, caption) {
    var systems = [];
    var pairs = [];
    cells.forEach(function (c) {
      if (systems.indexOf(c.system) < 0) { systems.push(c.system); }
      if (pairs.indexOf(c.pair) < 0) { pairs.push(c.pair); }
    });
    var values = cells.filter(function (c) { return c.mean !== null; })
      .map(function (c) { return c.mean; });
    var hi = values.length ? Math.max.apply(null, values) : 1;

    var out = ['<div class="table-wrap"><table class="heat">' +
      '<caption class="visually-hidden">' + esc(caption) + "</caption>" +
      '<thead><tr><th scope="col" class="heat__hd">system</th>'];
    pairs.forEach(function (p) {
      out.push('<th scope="col" class="heat__hd">' + esc(p) + "</th>");
    });
    out.push("</tr></thead><tbody>");
    systems.forEach(function (sys) {
      out.push('<tr><th scope="row" class="heat__hd">' + esc(sys) + "</th>");
      pairs.forEach(function (p) {
        var c = cells.filter(function (x) {
          return x.system === sys && x.pair === p;
        })[0];
        if (!c || c.mean === null) {
          out.push('<td class="heat__cell heat__cell--nm" ' +
            'title="not measured">&middot;</td>');
          return;
        }
        var strength = hi > 0 ? (c.mean / hi) : 0;
        out.push('<td class="heat__cell" style="--s:' + strength.toFixed(3) +
          '" title="' + esc(p + " " + sys + ": " + num(c.mean, 4) +
          " +/- " + num(c.std, 4)) + '">' + num(c.mean, 3) + "</td>");
      });
      out.push("</tr>");
    });
    out.push("</tbody></table></div>");
    return out.join("");
  }

  /* ---- AIT ------------------------------------------------------------ */

  function renderAit(host, data) {
    host.className = "";
    var out = [datasetCard(data.dataset)];

    out.push('<p class="axis-note">' + esc(data.gz_note) + "</p>");

    data.directions.forEach(function (d) {
      out.push('<div class="panel"><h4>' + esc(d.label || d.id) + "</h4>");
      out.push('<p class="axis-note">train <code>' + esc(d.train_scenario) +
        "</code> (n = " + num(d.n_train, 0) + "), test <code>" +
        esc(d.test_scenario) + "</code> (n = " + num(d.n_test, 0) +
        "). Clean ENIV " + pm(d.clean.eniv_eigen, 4) +
        ", clean detector AUC " + pm(d.clean.detector_auc, 4) + ".</p>");

      out.push("<h5>Dependence between views, clean data</h5>");
      out.push('<p class="axis-note">' + esc(d.dependence.label) +
        ". Most dependent: " + (d.dependence.most_dependent || []).map(function (p) {
          return "<code>" + esc(p) + "</code>";
        }).join(", ") + ". Both systems agree on the ordering: <strong>" +
        (d.dependence.systems_agree_on_ordering ? "yes" : "no") +
        "</strong>. Read from <code>" + esc(d.dependence.source) +
        "</code>, not re-derived.</p>");
      out.push(heatmap(d.dependence.cells,
        "Mean pairwise dependence per view pair and system, " + d.id));
      if (d.dependence.caveat) {
        out.push('<p class="pass-note">' + esc(d.dependence.caveat) + "</p>");
      }

      out.push("<h5>Under attack: ENIV and detector AUC by k</h5>");
      out.push('<p class="axis-note">Split by whether <code>dns</code> was in ' +
        "the compromised set. That split is not optional: <code>dns</code> " +
        "alone solves this label, so a row that does not name the " +
        "compromised set cannot be interpreted (L17).</p>");
      var rows = ['<div class="table-wrap"><table>' +
        '<caption class="visually-hidden">Attack grid for ' + esc(d.id) +
        ": ENIV, detector AUC and attack success per cell, by whether dns " +
        "was compromised.</caption><thead><tr>" +
        '<th scope="col">attack</th><th scope="col" class="num">k</th>' +
        '<th scope="col" class="num">eps</th>' +
        '<th scope="col">dns compromised</th>' +
        '<th scope="col" class="num">ENIV (eigen)</th>' +
        '<th scope="col" class="num">detector AUC</th>' +
        '<th scope="col" class="num">attack success</th></tr></thead><tbody>'];
      d.attack_by_k.forEach(function (c) {
        rows.push("<tr><td>" + esc(c.kind) + '</td><td class="num">' +
          num(c.k, 0) + '</td><td class="num">' + num(c.epsilon, 1) +
          "</td><td>" + (c.dns_compromised
            ? '<span class="pill pill--error">yes</span>'
            : '<span class="pill pill--live">no</span>') +
          '</td><td class="num">' + pm(c.eniv_eigen, 4) +
          '</td><td class="num">' + pm(c.detector_auc, 4) +
          '</td><td class="num">' + pm(c.attack_success, 4) + "</td></tr>");
      });
      rows.push("</tbody></table></div>");
      out.push(rows.join(""));
      out.push("</div>");
    });

    out.push('<div class="panel"><h4>Honest duplication (Part 29 test C)</h4>' +
      '<p><span class="pill pill--notmeasured">' +
      esc(data.honest_duplication.verdict) + "</span></p>" +
      '<p class="nm nm--why">' + esc(data.honest_duplication.reason) + "</p>" +
      "</div>");

    out.push('<p class="src">' + provenanceLine(data.provenance) + "</p>");
    host.innerHTML = out.join("");
  }

  /* ---- LANL ----------------------------------------------------------- */

  function renderLanl(host, data) {
    host.className = "";
    var s = data.sample;
    var extra = "<dt>sample design</dt><dd>" + esc(s.design) +
      ", test prevalence <strong>" + num(s.test_prevalence, 4) + "</strong> (" +
      num(s.n_test_attack, 0) + " attack of " + num(s.n_test, 0) +
      ")</dd>";
    var out = [datasetCard(data.dataset, extra)];

    // The two limits that bound every AUC below. They are stated above the
    // table, not under it, because an AUC read without them is read wrong.
    out.push('<div class="panel panel--caution"><h4>Read the AUCs against these</h4>' +
      "<p><strong>" + esc(s.limitation) + ".</strong> " + esc(s.limitation_note) +
      "</p><p><strong>" + esc(data.seen_unseen.limitation) + ".</strong> " +
      esc(data.seen_unseen.limitation_note) + "</p>" +
      "<p>The untrained scalar <code>" + esc(data.trivial_scalar.name) +
      "</code> reaches test ROC AUC <strong>" +
      num(data.trivial_scalar.test_auc, 4) +
      "</strong> on the same test block, with no training at all. Every arm " +
      "below should be read as its distance from that number, not from " +
      "zero.</p></div>");

    var rows = ['<div class="table-wrap"><table>' +
      '<caption class="visually-hidden">Per view configuration: naive and ' +
      "PrismFlow test ROC AUC over 5 seeds, their paired difference, and a " +
      "2000-replicate cluster bootstrap CI over test computers.</caption>" +
      '<thead><tr><th scope="col">arm</th>' +
      '<th scope="col" class="num">naive ROC AUC</th>' +
      '<th scope="col" class="num">PrismFlow ROC AUC</th>' +
      '<th scope="col" class="num">paired difference</th>' +
      '<th scope="col" class="num">bootstrap 95% CI</th>' +
      '<th scope="col" class="num">untrained scalar</th></tr></thead><tbody>'];
    data.arms.forEach(function (a) {
      var ci = a.bootstrap_ci95;
      var ciText = (ci.low === null || ci.high === null) ? NOT_MEASURED
        : "[" + num(ci.low, 4) + ", " + num(ci.high, 4) + "]" +
          '<span class="unit"> (' + num(ci.replicates, 0) + " replicates, " +
          num(ci.n_test_computers, 0) + " computers)</span>";
      rows.push('<tr><th scope="row"><code>' + esc(a.arm) + "</code></th>" +
        '<td class="num">' + pm(a.naive_roc_auc, 4) + "</td>" +
        '<td class="num">' + pm(a.prismflow_roc_auc, 4) + "</td>" +
        '<td class="num">' + pm(a.paired_difference, 4) +
        (a.paired_difference.n_seeds_favouring_prismflow !== null
          ? '<span class="unit"> ' +
            a.paired_difference.n_seeds_favouring_prismflow + "/" +
            a.paired_difference.n_seeds + " seeds favour PrismFlow</span>"
          : "") + "</td>" +
        '<td class="num">' + ciText + "</td>" +
        '<td class="num">' + num(data.trivial_scalar.test_auc, 4) + "</td></tr>");
    });
    rows.push("</tbody></table></div>");
    out.push(rows.join(""));

    if (data.bootstrap_note) {
      out.push('<p class="pass-note">' + esc(data.bootstrap_note) + "</p>");
    }
    out.push('<p class="src">' + provenanceLine(data.provenance) + "</p>");
    host.innerHTML = out.join("");
  }

  /* ---- the claim table ------------------------------------------------ */

  function verdictClass(verdict) {
    if (verdict === "CONFIRMS") { return "pill pill--live"; }
    if (verdict === "CONTRADICTS") { return "pill pill--error"; }
    return "pill pill--notmeasured";
  }

  function renderSummary(host, data) {
    host.className = "";
    var out = ['<p class="axis-note">' + esc(data.note) + "</p>"];

    var t = ['<div class="table-wrap"><table class="claims">' +
      '<caption class="visually-hidden">' + esc(data.title) +
      ". Each cell gives a verdict, the written rule that verdict had to " +
      "satisfy, what the cited file said, and that file's sha256.</caption>" +
      '<thead><tr><th scope="col">claim</th>'];
    data.columns.forEach(function (c) {
      t.push('<th scope="col">' + esc(c.label) +
        '<span class="unit"> ' + esc(c.note) + "</span></th>");
    });
    t.push("</tr></thead><tbody>");

    data.rows.forEach(function (row) {
      t.push('<tr><th scope="row">' + esc(row.claim) + "</th>");
      data.columns.forEach(function (col) {
        var c = row.cells[col.id];
        if (!c) { t.push("<td>" + NOT_MEASURED + "</td>"); return; }
        t.push("<td>" +
          '<p><span class="' + verdictClass(c.verdict) + '">' +
          esc(c.verdict) + "</span></p>" +
          '<p class="claims__rule"><strong>Rule.</strong> ' + esc(c.rule) + "</p>" +
          (c.detail ? '<p class="claims__detail"><strong>What the file says.</strong> ' +
            esc(c.detail) + "</p>" : "") +
          '<p class="src"><code>' + esc(c.source_file) + "</code> " +
          '<span class="mono">' + esc((c.sha256 || "").slice(0, 12)) + "</span></p>" +
          "</td>");
      });
      t.push("</tr>");
    });
    t.push("</tbody></table></div>");
    out.push(t.join(""));
    out.push('<p class="src">' + provenanceLine(data.provenance) + "</p>");
    host.innerHTML = out.join("");
  }

  /* ---- the real-data limitations group -------------------------------- *
   * L16-L30 came out of Parts 28-30. limits.js renders the two groups it
   * already knew about; this renders the third, from the same file, so the
   * hub keeps showing every entry the document holds. */
  function renderRealLimits(host, data) {
    host.className = "";
    var entries = data.v1.entries.filter(function (e) {
      return e.group === "real_data";
    });
    if (!entries.length) {
      host.innerHTML = '<p class="nm nm--why">No entry in ' +
        esc(data.v1.source) + " is grouped as a real-data finding.</p>";
      return;
    }
    host.innerHTML = '<div class="panel">' + entries.map(function (e) {
      return '<p class="limit"><strong>' + esc(e.id) + ".</strong> " +
        esc(e.title) + "</p>";
    }).join("") + '<p class="src">' + esc(data.v1.source) +
      " — headings verbatim, group “" +
      esc(data.v1.group_labels.real_data) + "”</p></div>";
  }

  /* ---- mount ---------------------------------------------------------- */

  function mount(id, path, renderer) {
    var host = doc.getElementById(id);
    if (!host) { return; }
    load(path).then(function (data) {
      try { renderer(host, data); }
      catch (e) { fail(host, String(e && e.message ? e.message : e)); }
    }).catch(function (e) {
      fail(host, String(e && e.message ? e.message : e));
    });
  }

  mount("real-ait", "data/real_ait.json", renderAit);
  mount("real-lanl", "data/real_lanl.json", renderLanl);
  mount("real-claims", "data/real_summary.json", renderSummary);
  mount("limits-v1-realdata", "data/limitations.json", renderRealLimits);
})(this);
