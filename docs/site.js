/* causalab-mini docs — the only script on the site.
 *
 * Two jobs, both of which the pages work without: the theme override and the
 * narrow-screen nav toggle. Every read of localStorage is wrapped, because a
 * private window throws on the accessor rather than returning null.
 *
 * It does NOT set aria-current. Each page marks its own nav entry in the
 * markup, so the highlight is right before a byte of JavaScript has run.
 */
(function () {
  "use strict";

  var root = document.documentElement;

  function stored(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }
  function store(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* no-op */ }
  }

  // The override, if this viewer set one. Absent, the page follows
  // prefers-color-scheme, which the stylesheet handles on its own.
  var saved = stored("theme");
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);

  function current() {
    var explicit = root.getAttribute("data-theme");
    if (explicit) return explicit;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function wire() {
    document.body.setAttribute("data-js", "on");

    var toggles = document.querySelectorAll("[data-theme-toggle]");
    for (var i = 0; i < toggles.length; i++) {
      toggles[i].addEventListener("click", function () {
        var next = current() === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        store("theme", next);
        label();
      });
    }

    var nav = document.querySelectorAll("[data-nav-toggle]");
    for (var j = 0; j < nav.length; j++) {
      nav[j].addEventListener("click", function () {
        var open = document.body.getAttribute("data-nav") === "open";
        document.body.setAttribute("data-nav", open ? "closed" : "open");
        this.setAttribute("aria-expanded", open ? "false" : "true");
      });
    }

    label();
  }

  function label() {
    var next = current() === "dark" ? "light" : "dark";
    var toggles = document.querySelectorAll("[data-theme-toggle]");
    for (var i = 0; i < toggles.length; i++) {
      toggles[i].textContent = current() === "dark" ? "light" : "dark";
      toggles[i].setAttribute("aria-label", "Switch to " + next + " theme");
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})();
