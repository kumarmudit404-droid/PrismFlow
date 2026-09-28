/* PrismFlow Part 25 -- phase (e), step 5. The limitations hub, and the fix
 * that closed two of its entries.
 *
 * WHY THIS FILE EXISTS. index.html used to carry the limitation lists as hand
 * written HTML, in three places: Chapter 2, Chapter 4 and the hub. When V2-L2
 * and V2-L3 were fixed and V2-L7 was opened, the document recorded it and the
 * page did not, because nothing connected the two. A list with no link to its
 * source drifts silently. These lists are now read from
 * docs/KNOWN_LIMITATIONS.md and docs/v2-known-limitations.md at build time, so
 * an entry that is resolved there cannot stay open here.
 *
 * ONE COPY, NOT THREE. The full list lives in the hub. The chapters carry a
 * short status line and a link to it. That is the "keep the Known Limitations
 * hub" rule: the hub is the only place the entries are printed in full.
 */
(function () {
  "use strict";

  function esc(v) {
    if (v === null || v === undefined) { return ""; }
    return String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function load(path) {
    return fetch(path, { cache: "no-store" }).then(function (r) {
      if (!r.ok) { throw new Error(path + " → HTTP " + r.status); }
      return r.json();
    });
  }

  function statusPill(status) {
    if (!status) { return ""; }
    if (status.indexOf("RESOLVED") === 0) {
      return ' <span class="pill pill--live">' + esc(status) + "</span>";
    }
    return ' <span class="pill pill--notmeasured">' + esc(status) + "</span>";
  }

  function entryHTML(e) {
    // Where the heading itself opens with the marker -- V2-L7 reads "NEW,
    // created by ..." -- the pill would print the word twice, so it is
    // dropped. The heading is never edited to make room for it.
    var inTitle = e.status &&
      String(e.title).toUpperCase().indexOf(e.status.toUpperCase()) === 0;
    return '<div class="limit' + (e.status && e.status.indexOf("RESOLVED") === 0
      ? " limit--resolved" : "") + '">' +
      '<span class="limit__id">' + esc(e.id) + ".</span> " +
      '<span class="limit__title">' + esc(e.title) + "</span>" +
      (inTitle ? "" : statusPill(e.status)) + "</div>";
  }

  function listPanel(entries, source) {
    return '<div class="panel">' +
      entries.map(entryHTML).join("") +
      '<p class="src">' + esc(source) + " — headings verbatim</p></div>";
  }

  /* ---- the hub ---- */
  function renderHub(data) {
    var v1 = data.v1.entries;
    function group(g) { return v1.filter(function (e) { return e.group === g; }); }

    var slots = [
      ["limits-v1-parts", group("per_part"), data.v1.source],
      ["limits-v1-standing", group("standing"), data.v1.source],
      ["limits-v2", data.v2.entries, data.v2.source]
    ];
    slots.forEach(function (s) {
      var node = document.getElementById(s[0]);
      if (!node) { return; }
      node.className = "";
      node.innerHTML = listPanel(s[1], s[2]);
    });

    var note = document.getElementById("limits-note");
    if (note) {
      note.innerHTML = esc(data.note) + " V1: " + data.v1.counts.total +
        " entries, all open. V2: " + data.v2.counts.total + " entries, " +
        data.v2.counts.resolved + " resolved, " + data.v2.counts.open + " open.";
    }
  }

  /* ---- the short status line a chapter carries instead of the list ---- */
  function renderSummary(id, block, heading) {
    var node = document.getElementById(id);
    if (!node) { return; }
    var c = block.counts;
    node.className = "limits-summary";
    node.innerHTML =
      '<p class="limits-summary__line"><strong>' + esc(heading) + "</strong> " +
      c.total + " entries" +
      (c.resolved ? ", " + c.resolved + " resolved" : "") +
      (c.new ? ", " + c.new + " opened by a fix" : "") +
      ", " + c.open + " still open.</p>" +
      '<p class="limits-summary__ids">' +
      block.entries.map(function (e) {
        return '<a class="limits-summary__id' +
          (e.status && e.status.indexOf("RESOLVED") === 0
            ? " limits-summary__id--resolved" : "") +
          '" href="#limits" title="' + esc(e.title) + '">' + esc(e.id) + "</a>";
      }).join("") + "</p>" +
      '<p class="src">' + esc(block.source) +
      ' — <a href="#limits">every entry, in full, in the hub</a></p>';
  }

  /* ==================================================================
   * V2-L2 / V2-L3: retrieval before and after the connector fix
   * ================================================================== */

  function cell(before, after) {
    if (before === null && after === null) {
      return '<td class="num"><span class="nm">not measured</span></td>';
    }
    var moved = before !== after;
    return '<td class="num">' +
      '<span class="rv-before">' + (before === null ? "—" : before) + "</span>" +
      '<span class="rv-arrow" aria-hidden="true">→</span>' +
      '<span class="' + (moved ? "rv-after rv-after--moved" : "rv-after") + '">' +
      (after === null ? "—" : after) + "</span></td>";
  }

  function renderReverify(node, data) {
    node.className = "";
    var s = data.summary;

    var head = '<div class="grid grid--4 rv-stats">' +
      [["fusible rows", s.fusible_before + " → " + s.fusible_after,
        "of " + s.n_rows + " re-run rows"],
       ["rows with GitHub records", s.github_rows_with_records_before + " → " +
        s.github_rows_with_records_after, "GitHub returned 0 on every row before"],
       ["distinct tech token totals", s.distinct_tech_token_totals_before + " → " +
        s.distinct_tech_token_totals_after,
        "identical totals meant identical records"],
       ["rows on the collapsed listing", s.rows_on_collapsed_1076_before + " → " +
        s.rows_on_collapsed_1076_after, "arXiv's generic physics result"]
      ].map(function (t) {
        return '<div class="panel stat"><span class="stat__value">' + esc(t[1]) +
          '</span><span class="stat__label">' + esc(t[0]) +
          '</span><span class="stat__src">' + esc(t[2]) + "</span></div>";
      }).join("") + "</div>";

    var rows = data.rows.map(function (r) {
      return '<tr><th scope="row" class="mono">' + esc(r.id) + "</th>" +
        "<td>" + esc(r.domain) + "</td>" +
        cell(r.github_before, r.github_after) +
        cell(r.arxiv_before, r.arxiv_after) +
        cell(r.tech_tokens_before, r.tech_tokens_after) +
        cell(r.angles_with_claims_before, r.angles_with_claims_after) +
        '<td class="num">' +
          (r.eniv_after === null
            ? '<span class="nm">not defined</span>'
            : Number(r.eniv_after).toFixed(4)) + "</td>" +
        "<td>" + (r.note_after ? esc(r.note_after) : "") + "</td></tr>";
    }).join("");

    node.innerHTML = head +
      '<div class="table-wrap"><table class="rv-table">' +
      '<caption class="visually-hidden">Per-row retrieval before and after the ' +
      "connector fix, on the same six rows. Each cell shows the value before " +
      "the arrow and after it. ENIV is only defined where at least two angles " +
      "produced claims.</caption><thead><tr>" +
      '<th scope="col">row</th><th scope="col">domain</th>' +
      '<th scope="col" class="num">GitHub records</th>' +
      '<th scope="col" class="num">arXiv records</th>' +
      '<th scope="col" class="num">tech tokens</th>' +
      '<th scope="col" class="num">angles with claims</th>' +
      '<th scope="col" class="num">ENIV after</th>' +
      '<th scope="col">outcome</th></tr></thead><tbody>' + rows +
      "</tbody></table></div>" +
      '<div class="notmeasured notmeasured--wide"><strong class="nm">' +
      "read this before quoting any number above</strong> " + esc(data.label) +
      "</div>" +
      "<ul class=\"rv-caveats\">" +
      (data.caveats || []).map(function (c) {
        return "<li>" + esc(c) + "</li>";
      }).join("") + "</ul>" +
      '<p class="src">' + (data.provenance.sources || []).map(function (x) {
        return esc(x.path) + ' <span class="sha">sha256 ' +
          esc(x.sha256.slice(0, 12)) + "</span>";
      }).join(" · ") + "</p>";
  }

  function fail(node, e) {
    node.className = "notmeasured";
    node.innerHTML = "could not load committed data — " +
      esc(e && e.message ? e.message : e) +
      "<br>run <code>python web/build_data.py</code>";
  }

  load("data/limitations.json").then(function (data) {
    renderHub(data);
    renderSummary("limits-ch2", data.v1, "V1 —");
    renderSummary("limits-ch4", data.v2, "V2 —");
  }).catch(function (e) {
    ["limits-v1-parts", "limits-v1-standing", "limits-v2",
     "limits-ch2", "limits-ch4"].forEach(function (id) {
      var n = document.getElementById(id);
      if (n) { fail(n, e); }
    });
  });

  var rv = document.getElementById("v2-reverify");
  if (rv) {
    load("data/v2_reverify.json")
      .then(function (data) { renderReverify(rv, data); })
      .catch(function (e) { fail(rv, e); });
  }
})();
