/* PrismFlow Part 25 -- phase (e), step 3. The V2 evaluation dataset, all 48 rows.
 *
 * SOURCE. web/data/v2_dataset.json, copied by web/build_data.py from
 * data/v2/evaluation_queries.json -- the derived file the Part 24 harness
 * actually loads. Not the workbook: the workbook holds a blank 040 and an EX
 * example row, and this table has to show the rows an evaluation would run on.
 * The workbook is offered separately as a download so the two can be compared,
 * and both carry a sha256.
 *
 * THIS TABLE HOLDS NO RESULT. There is no prediction column, no score and no
 * calibration metric, because none has been computed on these rows -- that is
 * V2-L6, and inventing a column here would be the exact failure the limitation
 * exists to record.
 *
 * Filtering, sorting and search are all over the loaded rows and nothing else.
 * No row is ever synthesised to fill a gap: 040 is absent from the table and
 * explained in the footer, in the words of the commit that left it out.
 */
(function () {
  "use strict";

  var NOT_RECORDED = '<span class="nm">not recorded</span>';

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

  /* Only http(s) becomes clickable. A javascript: or data: value arriving in a
   * data file must render as inert text, not as something a reader can click.
   * Same rule as charts.js, deliberately repeated rather than shared, because a
   * link-safety check is not worth a cross-file dependency to get wrong. */
  function sourceLink(url, label) {
    if (!url) { return NOT_RECORDED; }
    if (!/^https?:\/\//i.test(url)) {
      return '<span class="mono">' + esc(url) + "</span>";
    }
    var text = label || url.replace(/^https?:\/\//i, "").replace(/\/$/, "");
    return '<a class="detail__link" href="' + esc(url) +
      '" target="_blank" rel="noopener noreferrer">' + esc(text) +
      '<span class="visually-hidden"> (opens in a new tab)</span></a>';
  }

  function hostOf(url) {
    try { return new URL(url).hostname.replace(/^www\./, ""); }
    catch (e) { return url; }
  }

  function load(path) {
    return fetch(path, { cache: "no-store" }).then(function (r) {
      if (!r.ok) { throw new Error(path + " → HTTP " + r.status); }
      return r.json();
    });
  }

  /* ---- facets ------------------------------------------------------------
   * Every option is built from the rows themselves, so a value that appears in
   * the data can never be missing from the filter, and a filter can never offer
   * a value no row has. */
  function facet(rows, field) {
    var seen = {};
    rows.forEach(function (r) { seen[r[field] || "not recorded"] = 1; });
    return Object.keys(seen).sort();
  }

  var COLUMNS = [
    { key: "id", label: "row", sortable: true },
    { key: "domain", label: "domain", sortable: true },
    { key: "outcome", label: "outcome", sortable: true },
    { key: "outcome_date", label: "outcome date", sortable: true },
    { key: "conflict_expected", label: "conflict expected", sortable: true },
    { key: "ground_truth_source", label: "ground truth source", sortable: false }
  ];

  function cmp(a, b, key) {
    var x = a[key], y = b[key];
    // A missing value sorts last in BOTH directions. It is not an empty string
    // that happens to come first alphabetically; it is an absence, and it should
    // not migrate to the top of the table when the sort is reversed.
    if (!x && !y) { return 0; }
    if (!x) { return 1; }
    if (!y) { return -1; }
    return String(x).localeCompare(String(y), undefined, { numeric: true });
  }

  function detailRow(r, colspan) {
    return '<tr class="ds-detail" data-detail-for="' + esc(r.id) + '" hidden>' +
      '<td colspan="' + colspan + '">' +
      '<div class="ds-detail__body">' +
      "<h4>Row " + esc(r.id) + " — the pitch, as the model receives it</h4>" +
      '<p class="ds-detail__pitch">' + (r.idea_pitch ? esc(r.idea_pitch) : NOT_RECORDED) + "</p>" +
      '<dl class="detail__meta">' +
      "<dt>domain</dt><dd>" + (r.domain ? esc(r.domain) : NOT_RECORDED) + "</dd>" +
      "<dt>recorded outcome</dt><dd>" + (r.outcome ? esc(r.outcome) : NOT_RECORDED) + "</dd>" +
      "<dt>outcome date</dt><dd>" + (r.outcome_date ? esc(r.outcome_date) : NOT_RECORDED) + "</dd>" +
      "<dt>conflict expected</dt><dd>" +
        (r.conflict_expected ? esc(r.conflict_expected) : NOT_RECORDED) + "</dd>" +
      "<dt>ground truth source</dt><dd>" + sourceLink(r.ground_truth_source) + "</dd>" +
      "</dl>" +
      "<h4>Why this outcome is the recorded one</h4>" +
      '<p class="ds-detail__notes">' + (r.notes ? esc(r.notes) : NOT_RECORDED) + "</p>" +
      "</div></td></tr>";
  }

  function bodyRow(r, colspan) {
    var out = ['<tr class="ds-row" data-id="' + esc(r.id) + '">'];
    // The row number is a real <button>, not a click handler on the <tr>: a
    // handler on a row is invisible to a keyboard and to assistive tech, while a
    // button is focusable, announced, and works with Enter and Space for free.
    out.push('<th scope="row" class="mono"><button type="button" class="ds-row__btn" ' +
      'data-id="' + esc(r.id) + '" aria-expanded="false">' + esc(r.id) +
      '<span class="visually-hidden">: show the pitch and the ground-truth note' +
      " for this row</span></button></th>");
    out.push("<td>" + (r.domain ? esc(r.domain) : NOT_RECORDED) + "</td>");
    out.push("<td>" + (r.outcome ? esc(r.outcome) : NOT_RECORDED) + "</td>");
    out.push('<td class="mono">' + (r.outcome_date ? esc(r.outcome_date) : NOT_RECORDED) + "</td>");
    out.push("<td>" + (r.conflict_expected ? esc(r.conflict_expected) : NOT_RECORDED) + "</td>");
    out.push('<td class="ds-src">' +
      (r.ground_truth_source
        ? sourceLink(r.ground_truth_source, hostOf(r.ground_truth_source))
        : NOT_RECORDED) + "</td>");
    out.push("</tr>");
    out.push(detailRow(r, colspan));
    return out.join("");
  }

  function renderDataset(mountHost, data) {
    mountHost.className = "";
    mountHost.innerHTML = "";
    var rows = data.rows || [];
    var colspan = COLUMNS.length;

    var state = { q: "", domain: "", outcome: "", conflict: "", sort: "id", dir: 1 };

    /* ---- the summary, counted from the rows and never quoted ---- */
    var counts = el("div", "grid grid--4 ds-stats");
    [
      ["rows", String(data.n_rows),
        (data.id_gaps && data.id_gaps.length
          ? "id " + data.id_gaps.join(", ") + " absent"
          : "ids contiguous")],
      ["calibration n", String(data.denominators.calibration_n), "every row"],
      ["conflict n", String(data.denominators.conflict_n),
        "excludes the rows labelled Unsure"],
      ["domains", String(Object.keys(data.domains).length),
        Object.keys(data.domains).map(function (k) {
          return k + " " + data.domains[k];
        }).join(" · ")]
    ].forEach(function (s) {
      counts.appendChild(el("div", "panel stat",
        '<span class="stat__value">' + esc(s[1]) + "</span>" +
        '<span class="stat__label">' + esc(s[0]) + "</span>" +
        '<span class="stat__src">' + esc(s[2]) + "</span>"));
    });
    mountHost.appendChild(counts);
    mountHost.insertAdjacentHTML("beforeend",
      '<p class="axis-note">' + esc(data.denominators.note) + "</p>");

    /* ---- controls ---- */
    function select(name, label, values) {
      return '<label class="ds-ctl"><span class="ds-ctl__label">' + esc(label) + "</span>" +
        '<select class="ds-ctl__input" data-filter="' + name + '">' +
        '<option value="">all</option>' +
        values.map(function (v) {
          return '<option value="' + esc(v) + '">' + esc(v) + "</option>";
        }).join("") + "</select></label>";
    }

    var controls = el("div", "ds-controls");
    controls.innerHTML =
      '<label class="ds-ctl ds-ctl--search"><span class="ds-ctl__label">search</span>' +
      '<input class="ds-ctl__input" type="search" data-search ' +
      'placeholder="id, pitch, outcome, notes, source…" ' +
      'aria-describedby="ds-searchnote"></label>' +
      select("domain", "domain", facet(rows, "domain")) +
      select("outcome", "outcome", facet(rows, "outcome")) +
      select("conflict_expected", "conflict expected", facet(rows, "conflict_expected")) +
      '<button type="button" class="ds-ctl__reset" data-reset>reset</button>';
    mountHost.appendChild(controls);
    mountHost.insertAdjacentHTML("beforeend",
      '<p class="axis-note" id="ds-searchnote">Search covers every field of a ' +
      "row, including the pitch and the ground-truth note, not only what the " +
      "table columns show.</p>");

    var countLine = el("p", "ds-count");
    countLine.setAttribute("aria-live", "polite");
    mountHost.appendChild(countLine);

    /* ---- the table ---- */
    var head = COLUMNS.map(function (c) {
      if (!c.sortable) { return '<th scope="col">' + esc(c.label) + "</th>"; }
      return '<th scope="col" aria-sort="none" data-col="' + c.key + '">' +
        '<button type="button" class="ds-sort" data-col="' + c.key + '">' +
        esc(c.label) + '<span class="ds-sort__mark" aria-hidden="true"></span>' +
        '<span class="visually-hidden">: sort by this column</span></button></th>';
    }).join("");

    var wrap = el("div", "table-wrap");
    wrap.innerHTML = '<table class="ds-table">' +
      '<caption class="visually-hidden">The ' + esc(String(data.n_rows)) +
      " evaluation rows: id, domain, recorded outcome, outcome date, whether a " +
      "conflict between angles is expected, and the ground-truth source. Each " +
      "row number is a button that reveals the pitch and the ground-truth " +
      "note. This table contains no prediction and no score.</caption>" +
      "<thead><tr>" + head + "</tr></thead><tbody></tbody></table>";
    mountHost.appendChild(wrap);
    var tbody = wrap.querySelector("tbody");

    function matches(r) {
      if (state.domain && (r.domain || "not recorded") !== state.domain) { return false; }
      if (state.outcome && (r.outcome || "not recorded") !== state.outcome) { return false; }
      if (state.conflict &&
          (r.conflict_expected || "not recorded") !== state.conflict) { return false; }
      if (!state.q) { return true; }
      var hay = [r.id, r.domain, r.outcome, r.outcome_date, r.conflict_expected,
                 r.ground_truth_source, r.idea_pitch, r.notes]
        .filter(Boolean).join(" ").toLowerCase();
      return state.q.split(/\s+/).every(function (t) { return hay.indexOf(t) >= 0; });
    }

    function draw() {
      var shown = rows.filter(matches).slice().sort(function (a, b) {
        return state.dir * cmp(a, b, state.sort);
      });
      tbody.innerHTML = shown.map(function (r) { return bodyRow(r, colspan); }).join("");

      countLine.innerHTML = shown.length === rows.length
        ? "showing all " + rows.length + " rows"
        : "showing " + shown.length + " of " + rows.length + " rows" +
          (shown.length ? "" : " — no row matches; nothing is substituted");

      Array.prototype.forEach.call(wrap.querySelectorAll("th[data-col]"), function (th) {
        var on = th.getAttribute("data-col") === state.sort;
        th.setAttribute("aria-sort", on ? (state.dir > 0 ? "ascending" : "descending") : "none");
        var mark = th.querySelector(".ds-sort__mark");
        if (mark) { mark.textContent = on ? (state.dir > 0 ? "▲" : "▼") : ""; }
      });
    }

    controls.addEventListener("input", function (e) {
      var t = e.target;
      if (t.hasAttribute("data-search")) { state.q = t.value.trim().toLowerCase(); draw(); }
      else if (t.hasAttribute("data-filter")) {
        var f = t.getAttribute("data-filter");
        if (f === "domain") { state.domain = t.value; }
        if (f === "outcome") { state.outcome = t.value; }
        if (f === "conflict_expected") { state.conflict = t.value; }
        draw();
      }
    });
    controls.addEventListener("click", function (e) {
      if (!e.target.hasAttribute || !e.target.hasAttribute("data-reset")) { return; }
      state.q = ""; state.domain = ""; state.outcome = ""; state.conflict = "";
      state.sort = "id"; state.dir = 1;
      var s = controls.querySelector("[data-search]");
      if (s) { s.value = ""; }
      Array.prototype.forEach.call(controls.querySelectorAll("[data-filter]"),
        function (n) { n.value = ""; });
      draw();
    });

    wrap.addEventListener("click", function (e) {
      var sort = e.target.closest ? e.target.closest(".ds-sort") : null;
      if (sort) {
        var col = sort.getAttribute("data-col");
        if (state.sort === col) { state.dir = -state.dir; }
        else { state.sort = col; state.dir = 1; }
        draw();
        // draw() replaces the header, so focus has to be put back on the
        // equivalent button or a keyboard user is dropped out of the table.
        var again = wrap.querySelector('.ds-sort[data-col="' + col + '"]');
        if (again) { again.focus(); }
        return;
      }
      // A click on the ground-truth link must follow the link, never toggle
      // the row open underneath it.
      if (e.target.closest && e.target.closest("a")) { return; }
      var btn = e.target.closest ? e.target.closest(".ds-row__btn") : null;
      if (!btn) { return; }
      var id = btn.getAttribute("data-id");
      var det = wrap.querySelector('tr[data-detail-for="' + id + '"]');
      if (!det) { return; }
      var open = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", open ? "false" : "true");
      det.hidden = open;
    });

    draw();

    /* ---- the workbook, and the one row that is not here ---- */
    var foot = el("div", "panel ds-foot");
    foot.innerHTML =
      "<h4>The dataset as a file</h4>" +
      (data.download && data.download.file
        ? '<p><a class="ds-download" href="data/' + esc(data.download.file) + '" download>' +
          "Download the labelled workbook (" +
          Math.round(data.download.bytes / 1024) + " KB)</a></p>" +
          '<p class="src">' + esc(data.download.source) +
          ' <span class="sha">sha256 ' + esc((data.download.sha256 || "").slice(0, 16)) +
          "</span></p>" +
          '<p class="axis-note">' + esc(data.download.note) + "</p>"
        : '<p class="notmeasured">the workbook was not present at build time, ' +
          "so no download is offered</p>");
    mountHost.appendChild(foot);

    var gap = el("div", "panel ds-foot ds-foot--gap");
    gap.innerHTML =
      "<h4>Why this table has " + esc(String(data.n_rows)) + " rows and not 49</h4>" +
      '<p class="axis-note">' + esc(data.gap_note) + "</p>" +
      '<figure class="quote"><blockquote>' +
      "<p><strong>" + esc(data.row_040.heading) + "</strong></p>" +
      '<pre class="quote__pre">' + esc(data.row_040.text) + "</pre>" +
      "</blockquote><figcaption>commit <span class=\"mono\">" +
      esc(data.row_040.commit) + "</span> — " + esc(data.row_040.subject) +
      '<span class="src mono">' + esc(data.row_040.commit_full || "") + "</span>" +
      "</figcaption></figure>";
    mountHost.appendChild(gap);

    mountHost.insertAdjacentHTML("beforeend",
      '<p class="pass-note">' + esc(data.label) + "</p>");

    var src = (data.provenance.sources || [])[0];
    mountHost.insertAdjacentHTML("beforeend",
      '<p class="src">' + esc(src ? src.path : "source not recorded") +
      ' <span class="sha">sha256 ' + esc(src ? src.sha256.slice(0, 16) : "") +
      "</span> — copied by web/build_data.py, generated " +
      esc(data.provenance.generated_utc) + "</p>");
  }

  var node = document.getElementById("v2-dataset");
  if (!node) { return; }
  Promise.all([
    load("data/v2_dataset.json"),
    // The download descriptor is optional: if it is missing the table still
    // renders and the footer says there is no file, rather than the whole
    // section failing over a missing offer.
    load("data/v2_dataset_download.json").catch(function () { return null; })
  ]).then(function (both) {
    var data = both[0];
    data.download = both[1];
    renderDataset(node, data);
  }).catch(function (e) {
    node.className = "notmeasured";
    node.innerHTML = "could not load committed data — " +
      esc(e && e.message ? e.message : e) +
      "<br>run <code>python web/build_data.py</code>";
  });
})();
