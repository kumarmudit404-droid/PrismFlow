/* PrismFlow Part 25 -- phase (e), step 4. Real failures, and only real ones.
 *
 * WHAT THIS SECTION IS. Every non-clean attack condition in the committed Part
 * 09 run where PrismFlow's mean success rate is HIGHER than the naive
 * baseline's -- that is, where the discounted pipeline was fooled more often
 * than the thing it is supposed to improve on. Selection is mechanical, done in
 * web/build_data.py by comparing the two means, so no cell is here for being
 * dramatic and none is missing for being small.
 *
 * WHY NOT JUST eps 1.0. Showing one epsilon would make a 12-cell result look
 * like a single finding. The two near-zero cells (eps 0.2, both attacks) are
 * the ones that show the effect is not uniform, and they are exactly the cells
 * a flattering selection would drop.
 *
 * THE PER-SEED SIGN IS SHOWN NEXT TO EVERY MEAN. A positive mean over 5 seeds
 * can hide two seeds that went the other way, and a reader given only the mean
 * cannot tell a consistent effect from a coin flip. Every row therefore carries
 * "n of 5 seeds", and opening a row lists the per-seed differences themselves.
 *
 * NO FOOLED EXAMPLE IS INVENTED. The committed run records aggregates, per-seed
 * rows and a perturbation tensor. Nothing records which sample was fooled, what
 * it was predicted as, or what its true label was, so every example cell reads
 * "not recorded". Building one from the perturbation tensor would be a new
 * experiment wearing a page's clothes.
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

  function num(v, digits) {
    if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) {
      return NOT_RECORDED;
    }
    return Number(v).toFixed(digits === undefined ? 3 : digits);
  }

  /* A mean without its spread is not a result under the 5-SEED RULE, so the
   * two are never printed apart. A null std is printed as such, not as 0. */
  function pm(s, digits) {
    if (!s || s.mean === null || s.mean === undefined) { return NOT_RECORDED; }
    var d = digits === undefined ? 3 : digits;
    return Number(s.mean).toFixed(d) +
      (s.std === null || s.std === undefined
        ? ' <span class="nm">± not recorded</span>'
        : ' <span class="pm">± ' + Number(s.std).toFixed(d) + "</span>");
  }

  function load(path) {
    return fetch(path, { cache: "no-store" }).then(function (r) {
      if (!r.ok) { throw new Error(path + " → HTTP " + r.status); }
      return r.json();
    });
  }

  function knobs(c) {
    var bits = [];
    if (c.attack) { bits.push(esc(c.attack)); }
    if (c.k !== null && c.k !== undefined) { bits.push("k " + c.k); }
    // One decimal, so 1.0 does not print as "1" beside a condition named
    // eps1.0 and read as a different cell.
    if (c.epsilon !== null && c.epsilon !== undefined) {
      bits.push("ε " + Number(c.epsilon).toFixed(1));
    }
    if (c.beta !== null && c.beta !== undefined) {
      bits.push("β " + Number(c.beta).toFixed(1));
    }
    return bits.length ? bits.join(" · ") : NOT_RECORDED;
  }

  function detailBody(c) {
    var seeds = (c.per_seed || []).map(function (d) {
      return '<tr><th scope="row" class="num mono">' + d.seed + "</th>" +
        '<td class="num">' + num(d.naive) + "</td>" +
        '<td class="num">' + num(d.prismflow) + "</td>" +
        '<td class="num ' + (d.delta > 0 ? "fail-d--worse" : "fail-d--better") + '">' +
        (d.delta > 0 ? "+" : "") + num(d.delta) + "</td></tr>";
    }).join("");

    return '<div class="fail-detail__body">' +
      "<h4>Every seed, not the average of them</h4>" +
      '<div class="table-wrap"><table>' +
      '<caption class="visually-hidden">Attack success rate per seed for the ' +
      "naive baseline and for PrismFlow, and the difference. A positive " +
      "difference means PrismFlow was fooled more often on that seed.</caption>" +
      '<thead><tr><th scope="col" class="num">seed</th>' +
      '<th scope="col" class="num">naive</th>' +
      '<th scope="col" class="num">PrismFlow</th>' +
      '<th scope="col" class="num">difference</th></tr></thead><tbody>' +
      (seeds || '<tr><td colspan="4">' + NOT_RECORDED + "</td></tr>") +
      "</tbody></table></div>" +

      "<h4>What the attack achieved where it succeeded</h4>" +
      '<dl class="detail__meta detail__meta--num">' +
      "<dt>probability on the target class</dt><dd>" +
        pm(c.target_prob_on_success) + "</dd>" +
      "<dt>belief on the target class</dt><dd>" +
        pm(c.target_belief_on_success) + "</dd>" +
      "<dt>vacuity</dt><dd>" + pm(c.vacuity) + "</dd>" +
      "<dt>ENIV measured</dt><dd>" + pm(c.eniv_measured) + "</dd>" +
      "<dt>dependence, compromised pair</dt><dd>" +
        pm(c.dependence_compromised) + "</dd>" +
      "<dt>dependence, all pairs</dt><dd>" + pm(c.dependence_all_pairs) + "</dd>" +
      "</dl>" +

      "<h4>The sample that was fooled</h4>" +
      '<p class="fail-detail__ex">' + NOT_RECORDED +
      " — the committed run holds aggregates, per-seed rows and the " +
      "perturbation tensor. No file records which sample was fooled, what it " +
      "was predicted as, or what its true label was. Nothing is reconstructed " +
      "here to fill the gap.</p>" +
      "</div>";
  }

  function render(mountHost, data) {
    mountHost.className = "";
    mountHost.innerHTML = "";

    var conds = data.conditions || [];
    var consistent = conds.filter(function (c) {
      return c.n_seeds && c.seeds_prismflow_worse === c.n_seeds;
    }).length;

    var stats = el("div", "grid grid--4 fail-stats");
    [
      [String(data.n_included) + " of " + (data.n_conditions_total - 1),
        "non-clean conditions", "PrismFlow fooled more than naive"],
      [String(consistent) + " of " + data.n_included,
        "consistent across seeds", "every seed went the same way"],
      [(data.seeds ? String(data.seeds.length) : "not recorded"),
        "seeds per cell", "mean and standard deviation"],
      [num(data.rho_fixed, 1), "rho", "one fixed value, not an axis"]
    ].forEach(function (s) {
      stats.appendChild(el("div", "panel stat",
        '<span class="stat__value">' + esc(s[0]) + "</span>" +
        '<span class="stat__label">' + esc(s[1]) + "</span>" +
        '<span class="stat__src">' + esc(s[2]) + "</span>"));
    });
    mountHost.appendChild(stats);

    mountHost.insertAdjacentHTML("beforeend",
      '<p class="axis-note">Selection: ' + esc(data.criterion) + "</p>");

    (data.excluded || []).forEach(function (x) {
      mountHost.insertAdjacentHTML("beforeend",
        '<p class="axis-note">Excluded — <span class="mono">' +
        esc(x.condition) + "</span>: " + esc(x.why) + "</p>");
    });

    var rows = conds.map(function (c, i) {
      var mixed = c.seeds_prismflow_worse < c.n_seeds;
      return '<tr class="fail-row">' +
        '<th scope="row" class="mono"><button type="button" class="fail-row__btn" ' +
        'data-i="' + i + '" aria-expanded="false">' + esc(c.condition) +
        '<span class="visually-hidden">: show every seed and what the attack ' +
        "achieved</span></button></th>" +
        "<td>" + knobs(c) + "</td>" +
        '<td class="num">' + pm(c.naive) + "</td>" +
        '<td class="num">' + pm(c.prismflow) + "</td>" +
        '<td class="num">' + pm(c.prismflow_nodiscount) + "</td>" +
        '<td class="num fail-d--worse">+' + num(c.delta_mean) + "</td>" +
        '<td class="num">' +
          '<span class="pill ' + (mixed ? "pill--notmeasured" : "pill--error") + '">' +
          c.seeds_prismflow_worse + " of " + c.n_seeds + "</span></td>" +
        "<td>" + NOT_RECORDED + "</td>" +
        "</tr>" +
        '<tr class="fail-detail" data-detail-for="' + i + '" hidden>' +
        '<td colspan="8">' + detailBody(c) + "</td></tr>";
    }).join("");

    var wrap = el("div", "table-wrap");
    wrap.innerHTML = '<table class="fail-table">' +
      '<caption class="visually-hidden">Attack conditions where PrismFlow was ' +
      "fooled more often than the naive baseline. For each: the attack and its " +
      "settings, the mean success rate with standard deviation for the naive " +
      "baseline, for PrismFlow and for PrismFlow without the discount, the " +
      "difference, how many of the seeds individually went the wrong way, and " +
      "whether an individual fooled sample was recorded. Higher success rate " +
      "is worse.</caption>" +
      '<thead><tr><th scope="col">condition</th><th scope="col">attack</th>' +
      '<th scope="col" class="num">naive</th>' +
      '<th scope="col" class="num">PrismFlow</th>' +
      '<th scope="col" class="num">PrismFlow, no discount</th>' +
      '<th scope="col" class="num">difference</th>' +
      '<th scope="col" class="num">seeds worse</th>' +
      '<th scope="col">fooled example</th></tr></thead><tbody>' +
      rows + "</tbody></table></div>";
    mountHost.appendChild(wrap);

    wrap.addEventListener("click", function (e) {
      var btn = e.target.closest ? e.target.closest(".fail-row__btn") : null;
      if (!btn) { return; }
      var i = btn.getAttribute("data-i");
      var det = wrap.querySelector('tr[data-detail-for="' + i + '"]');
      if (!det) { return; }
      var open = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", open ? "false" : "true");
      det.hidden = open;
    });

    mountHost.insertAdjacentHTML("beforeend",
      '<p class="table-wrap__hint">Open any condition for its per-seed ' +
      "differences and what the attack achieved where it succeeded.</p>");

    mountHost.insertAdjacentHTML("beforeend",
      '<div class="notmeasured notmeasured--wide"><strong class="nm">no fooled ' +
      "example is available</strong> " + esc(data.example_availability) + "</div>");

    mountHost.insertAdjacentHTML("beforeend",
      '<p class="pass-note">' + esc(data.label) + "</p>");

    var src = (data.provenance.sources || []).map(function (s) {
      return esc(s.path) + ' <span class="sha">sha256 ' + esc(s.sha256.slice(0, 12)) + "</span>";
    }).join(" · ");
    mountHost.insertAdjacentHTML("beforeend", '<p class="src">' + src + "</p>");
  }

  var node = document.getElementById("v1-failures");
  if (!node) { return; }
  load("data/v1_failures.json").then(function (data) {
    render(node, data);
  }).catch(function (e) {
    node.className = "notmeasured";
    node.innerHTML = "could not load committed data — " +
      esc(e && e.message ? e.message : e) +
      "<br>run <code>python web/build_data.py</code>";
  });
})();
