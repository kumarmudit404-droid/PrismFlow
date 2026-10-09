/* absence.js -- the one place that decides how a missing value is worded.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * Every page on this site has cells with no value in them, and the reasons are
 * NOT the same reason. A row with one retrieving angle has no ENIV because
 * dependence is undefined below two views. The regulatory angle has no records
 * because nobody built the connector. The financial angle has no records
 * because yfinance resolves tickers and the pitches are anonymised on purpose.
 * ECE has no value because the question was closed as not measured. gpt-4o has
 * no verification because the account is unfunded.
 *
 * Those are five different findings. A blank cell states none of them, a dash
 * states none of them, and a single blanket "not measured" on all five states
 * the wrong one four times. So each cause is registered here once, with the
 * document entry that evidences it, and every page renders the label from this
 * registry rather than from a string typed at the call site. If a cause ever
 * changes, it changes in one place and the wording cannot drift between the
 * coverage grid, the scores page and the limitations cross-references.
 *
 * NO CELL HERE EVER PRODUCES A NUMBER. Not an estimate, not a zero, not a
 * dash that could be read as one. The honest rendering of an absent value is a
 * sentence about why it is absent, plus a citation a reader can go and check.
 *
 * "needs more data" IS NOT UNIVERSAL, and that is deliberate. Three of the
 * causes below are not waiting on data at all:
 *   - financial    a design conflict; more rows of the same kind change nothing
 *   - reasoners    waiting on account credit, not on data
 *   - n/a cells    there is no absence to explain, the question does not apply
 * Labelling those "needs more data" would promise a fix that collecting more
 * data cannot deliver, so they get their own lead-ins. Causes that genuinely
 * are coverage-limited say "Needs more data" and mean it.
 */
(function (global) {
  "use strict";

  /* The limitations hub lives on index.html. A page that has the section links
   * within itself; a page that does not (scores.html) links across. Resolved at
   * call time, not at load time, because scripts may run before the section
   * they are testing for has parsed. */
  function hubHref() {
    return (global.document && global.document.getElementById("limits"))
      ? "#limits"
      : "index.html#limits";
  }

  function esc(value) {
    if (value === null || value === undefined) { return ""; }
    return String(value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  /* ------------------------------------------------------------------ *
   * The registry.
   *
   *   lead   the first clause. "Needs more data" ONLY where more data is
   *          actually the blocker -- see the header note.
   *   why    the specific cause, in the words the cited entry uses. A
   *          function where the cause depends on the row.
   *   entry  the limitations entry that evidences it. This is the whole point
   *          of the label: a reader can open that entry and check the claim.
   * ------------------------------------------------------------------ */
  var CAUSES = {

    /* ENIV, the discount, fused confidence and contradiction counts are
     * undefined below two claim-producing angles -- aggregate_dependence drops
     * angles with no claims, so a one-angle row yields a 1x1 matrix. 46 of the
     * 48 rows are in this state. */
    "eniv-below-threshold": {
      lead: "Needs more data",
      why: function (o) {
        var n = (o && typeof o.angles === "number") ? o.angles : null;
        if (n === null) { return "fewer than 2 angles produced claims, ENIV requires 2+"; }
        return "only " + n + " angle" + (n === 1 ? "" : "s") +
          " produced claims, ENIV requires 2+";
      },
      entry: "V2-L1"
    },

    /* The regulatory angle. Not a zero from a connector -- there is no
     * connector, so nothing was ever called. Its provenance map is empty on
     * all 48 rows, which is a different fact from a connector returning 0. */
    "no-connector": {
      lead: "Needs more data",
      why: function () { return "no connector exists, nothing was called"; },
      entry: "V2-L1"
    },

    /* The financial angle. NOT a coverage problem and NOT waiting on data:
     * YFinanceConnector resolves a query to a ticker, and the dataset's
     * pitches are anonymised by design, so the company is exactly what has
     * been removed. V2-L4 calls this a design conflict requiring a decision,
     * not a patch -- so this label does not say "needs more data", because
     * more data of the same shape would not move it. */
    "ticker-mismatch": {
      lead: "Not obtainable as the dataset stands",
      why: function () {
        return "yfinance needs a ticker and the pitches are anonymised by design";
      },
      entry: "V2-L4"
    },

    /* A connector that WAS called and returned nothing: market on 42 rows,
     * sentiment on 47, tech on 4. An honest zero for that query against that
     * source, and the thing that would change it is a connector whose queries
     * match the dataset's language. */
    "connector-empty": {
      lead: "Needs more data",
      why: function (o) {
        var src = o && o.source;
        return (src ? src + " was called and " : "the connector was called and ") +
          "returned no records for this row";
      },
      entry: "V2-L1"
    },

    /* The fallback chain stops at the first slot that returns. Where GitHub
     * now fills the tech primary slot, arXiv is never reached -- so its cell
     * is not a zero, it is a call that did not happen. */
    "fallback-not-reached": {
      lead: "Not called",
      why: function () {
        return "the primary slot returned, so the fallback chain stopped before it";
      },
      entry: "V2-L3"
    },

    /* mean_claim_conf is the mean of the reasoner's self-reported per-claim
     * confidence. With no claims there is nothing to take a mean of. Null on
     * 196 of the 240 angle cells. */
    "no-claims-to-average": {
      lead: "Needs more data",
      why: function () {
        return "the reasoner produced no claims here, so there is nothing to average";
      },
      entry: "V2-L1"
    },

    /* ECE and Brier for the V2 dataset. Closed, not pending: no figure has
     * ever been computed or committed, and the postfix run's n=2 would not
     * satisfy the 5-SEED RULE even though it clears the harness's own gate. */
    "calibration-closed": {
      lead: "Needs more data",
      why: function () {
        return "closed as not measured; n=2 fused rows is not a distribution, " +
          "and one unseeded pass is not a finding";
      },
      entry: "V2-L9"
    },

    /* claude_reasoner.py and the gpt-4o path. Both keys authenticate; both
     * accounts are unfunded. Not a code defect and not a data shortage. */
    "reasoner-unfunded": {
      lead: "Needs more data",
      why: function (o) {
        var who = o && o.who;
        return (who ? who + ": " : "") +
          "blocked on funding, no live response has ever been logged";
      },
      entry: "V2-L8"
    },

    /* Not an absence at all. 42 of 48 rows have no zero-claim cause because
     * they produced claims -- there is no zero to explain. Calling that
     * "needs more data" would invent a gap that is not there. */
    "not-applicable": {
      lead: "Not applicable",
      why: function (o) {
        return (o && o.because) || "the row produced claims, so there is no zero to explain";
      },
      entry: null
    },

    /* A field the run simply did not write for this row. The file is the
     * authority on its own contents, and inventing a note would be writing
     * prose and attributing it to the run. */
    "run-wrote-none": {
      lead: "Not recorded",
      why: function (o) {
        return "the run wrote no " + ((o && o.field) || "value") + " for this row";
      },
      entry: null
    }
  };

  function resolve(causeId, opts) {
    var c = CAUSES[causeId];
    if (!c) {
      // An unknown cause id is a bug in the caller, and it must not render as
      // a blank or as a confident label. It says so instead.
      return { lead: "Not labelled", why: "unknown cause id " + causeId, entry: null };
    }
    return {
      lead: c.lead,
      why: (typeof c.why === "function") ? c.why(opts || {}) : c.why,
      entry: c.entry
    };
  }

  /* Plain text, for a title attribute or an aria-label. */
  function text(causeId, opts) {
    var r = resolve(causeId, opts);
    return r.lead + " — " + r.why + (r.entry ? ", see " + r.entry : "");
  }

  /* The full label: lead, cause, and the entry as a real link into the
   * limitations hub. Used where there is room to read a sentence -- row
   * detail panels, callout pills, prose. */
  function label(causeId, opts) {
    var r = resolve(causeId, opts);
    var cite = r.entry
      ? ' <a class="nm__cite" href="' + hubHref() + '">' + esc(r.entry) + "</a>"
      : "";
    return '<span class="nm nm--why">' + esc(r.lead) +
      '<span class="nm__why"> — ' + esc(r.why) + "</span>" + cite + "</span>";
  }

  /* The compact label, for a narrow table column. The lead is visible, the
   * full cause reaches a screen reader and a mouse, and the citation stays a
   * link -- a hover-only explanation would not be checkable. */
  function labelCompact(causeId, opts) {
    var r = resolve(causeId, opts);
    var cite = r.entry
      ? ' <a class="nm__cite" href="' + hubHref() + '">' + esc(r.entry) + "</a>"
      : "";
    return '<span class="nm nm--why" title="' + esc(text(causeId, opts)) + '">' +
      esc(r.lead) +
      '<span class="visually-hidden"> — ' + esc(r.why) + "</span>" + cite +
      "</span>";
  }

  /* ------------------------------------------------------------------ *
   * Static cells in the markup.
   *
   * index.html has hand-written callout pills -- ECE, Brier, gpt-4o, Claude --
   * that were reading "not measured" and "unverified" with no cause and no
   * citation. They could have been rewritten in place, but then the wording
   * would live in two files and drift the first time a cause changed. Instead
   * each pill declares WHICH cause it is an instance of, and takes its text
   * from the registry above:
   *
   *   <span class="pill pill--notmeasured"
   *         data-absent="calibration-closed" data-absent-label="ECE"></span>
   *
   * data-absent-label prefixes the metric's own name. data-absent-who and
   * data-absent-field pass the per-cell options some causes take.
   * ------------------------------------------------------------------ */
  function hydrate(root) {
    var doc = root || global.document;
    if (!doc || !doc.querySelectorAll) { return 0; }
    var nodes = doc.querySelectorAll("[data-absent]");
    var n = 0;
    Array.prototype.forEach.call(nodes, function (node) {
      var causeId = node.getAttribute("data-absent");
      var opts = {
        who: node.getAttribute("data-absent-who") || undefined,
        field: node.getAttribute("data-absent-field") || undefined,
        source: node.getAttribute("data-absent-source") || undefined,
        because: node.getAttribute("data-absent-because") || undefined
      };
      var prefix = node.getAttribute("data-absent-label");
      node.innerHTML = (prefix ? esc(prefix) + " — " : "") + label(causeId, opts);
      n += 1;
    });
    return n;
  }

  global.PrismFlowAbsence = {
    CAUSES: CAUSES,
    text: text,
    label: label,
    labelCompact: labelCompact,
    hydrate: hydrate
  };

  /* Hydrate as soon as the markup exists. These pills carry a limitation's
   * status, so they must never be left showing nothing if a later script on
   * the page throws. */
  if (global.document) {
    if (global.document.readyState === "loading") {
      global.document.addEventListener("DOMContentLoaded", function () { hydrate(); });
    } else {
      hydrate();
    }
  }
})(this);
