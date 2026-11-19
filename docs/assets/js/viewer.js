/* TopoPT project page: config-driven links, video, slide viewer, poster,
   image lightbox, scroll-spy and heading anchors. No dependencies except the
   self-hosted PDF.js, which is loaded only when the slides come into view. */

(function () {
  "use strict";

  var CFG = window.TOPOPT || {};

  /* ------------------------------------------------------- config links */
  function activate(key, url, label) {
    document.querySelectorAll('[data-link="' + key + '"]').forEach(function (a) {
      a.href = url;
      a.hidden = false;
      a.classList.remove("is-soon");
      a.removeAttribute("aria-disabled");
      if (label) a.textContent = label;
    });
  }
  if (CFG.paper) activate("paper", CFG.paper, "Paper");
  if (CFG.openreview) activate("openreview", CFG.openreview, "OpenReview");

  /* ---------------------------------------------------------- video URL */
  function toEmbed(raw) {
    if (!raw) return null;
    var u;
    try { u = new URL(raw, location.href); } catch (e) { return null; }
    var h = u.hostname.replace(/^www\./, "");
    var m;

    if (/^recorder/.test(h)) {
      console.warn("[TopoPT] recorder links are speaker tools, not public players. Ignored.");
      return null;
    }
    if (/\.(mp4|webm|ogv)$/i.test(u.pathname)) return { kind: "file", src: u.href };

    if (h === "youtu.be" && u.pathname.length > 1) {
      return { kind: "iframe", src: "https://www.youtube-nocookie.com/embed/" + u.pathname.slice(1) };
    }
    if (/(^|\.)youtube(-nocookie)?\.com$/.test(h)) {
      var id = u.searchParams.get("v");
      m = u.pathname.match(/^\/(?:embed|shorts)\/([\w-]+)/);
      if (!id && m) id = m[1];
      return id ? { kind: "iframe", src: "https://www.youtube-nocookie.com/embed/" + id } : null;
    }
    if (h === "vimeo.com" || h === "player.vimeo.com") {
      m = u.pathname.match(/(\d{5,})/);
      return m ? { kind: "iframe", src: "https://player.vimeo.com/video/" + m[1] } : null;
    }
    if (h === "slideslive.com") {
      if (/^\/embed\//.test(u.pathname)) return { kind: "iframe", src: u.href };
      m = u.pathname.match(/^\/(\d+)/);
      return m ? { kind: "iframe", src: "https://slideslive.com/embed/presentation/" + m[1] } : null;
    }
    return u.protocol === "https:" ? { kind: "iframe", src: u.href } : null;
  }
  window.__topoptToEmbed = toEmbed; // exposed for testing

  /* -------------------------------------------------------------- video */
  (function video() {
    var root = document.getElementById("video");
    var hero = document.querySelector('[data-link="video"]');
    var embed = toEmbed(CFG.video);

    if (!embed) {
      if (hero) { hero.classList.add("is-soon"); hero.setAttribute("aria-disabled", "true"); hero.textContent = "Video (soon)"; }
    }
    if (!root) return;

    var poster = root.querySelector(".video-poster");
    if (poster) {
      poster.addEventListener("error", function () { poster.hidden = true; });
      if (poster.complete && poster.naturalWidth === 0) poster.hidden = true; // failed before this script ran
    }
    if (!embed) return;

    var play = root.querySelector(".video-play");
    root.setAttribute("data-state", "ready");
    play.hidden = false;
    play.addEventListener("click", function () {
      var stage = root.querySelector(".video-stage"), el;
      if (embed.kind === "file") {
        el = document.createElement("video");
        el.controls = true; el.autoplay = true; el.playsInline = true;
        el.src = embed.src;
        if (poster && !poster.hidden) el.poster = poster.currentSrc || poster.src;
      } else {
        el = document.createElement("iframe");
        el.src = embed.src;
        el.title = "TopoPT presentation video";
        el.allow = "autoplay; fullscreen; picture-in-picture; encrypted-media";
        el.setAttribute("allowfullscreen", "");
        el.referrerPolicy = "strict-origin-when-cross-origin";
      }
      stage.innerHTML = "";
      stage.appendChild(el);
      root.setAttribute("data-state", "playing");
    });
  })();

  /* ---------------------------------------------------------- lightbox */
  var lastFocus = null;
  function openLightbox(src, alt, trigger) {
    lastFocus = trigger || document.activeElement;
    var lb = document.createElement("div");
    lb.className = "lb";
    lb.setAttribute("role", "dialog");
    lb.setAttribute("aria-modal", "true");
    lb.setAttribute("aria-label", "Enlarged image");
    lb.innerHTML =
      '<div class="lb-bar">' +
      '<button type="button" class="icon-btn lb-size">Actual size</button>' +
      '<button type="button" class="icon-btn lb-close">Close</button></div>' +
      '<div class="lb-body"><img alt=""></div>';
    var img = lb.querySelector("img");
    img.alt = alt || "";
    img.src = src;
    document.body.appendChild(lb);
    document.body.style.overflow = "hidden";

    var size = lb.querySelector(".lb-size"), close = lb.querySelector(".lb-close");
    function shut() {
      document.removeEventListener("keydown", onKey, true);
      document.body.style.overflow = "";
      lb.remove();
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }
    function onKey(e) {
      if (e.key === "Escape") { e.preventDefault(); shut(); }
      if (e.key === "Tab") {
        e.preventDefault();
        (document.activeElement === size ? close : size).focus();
      }
    }
    size.addEventListener("click", function () {
      var full = lb.classList.toggle("is-full");
      size.textContent = full ? "Fit to screen" : "Actual size";
    });
    close.addEventListener("click", shut);
    lb.querySelector(".lb-body").addEventListener("click", function (e) { if (e.target !== img) shut(); });
    document.addEventListener("keydown", onKey, true);
    close.focus();
  }

  document.querySelectorAll("img[data-zoom]").forEach(function (img) {
    img.tabIndex = 0;
    img.setAttribute("role", "button");
    img.setAttribute("aria-label", "Enlarge: " + (img.alt || "image"));
    function open() { openLightbox(img.getAttribute("data-full") || img.currentSrc || img.src, img.alt, img); }
    img.addEventListener("click", open);
    img.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
    });
  });

  /* ------------------------------------------------------------- poster */
  (function poster() {
    var img = document.getElementById("poster-img");
    if (!img) return;
    var full = img.getAttribute("data-full"), triedFull = false;
    img.addEventListener("error", function () {
      if (!triedFull && full) { triedFull = true; img.src = full; return; }
      var fig = img.closest(".poster");
      fig.setAttribute("data-state", "missing");
    });
  })();

  /* ------------------------------------------------------ slide viewer */
  (function slides() {
    var root = document.getElementById("slides-viewer");
    if (!root) return;
    var src = root.getAttribute("data-src");
    var stage = root.querySelector(".slides-stage");
    var canvas = root.querySelector("canvas");
    var msg = root.querySelector(".slides-msg");
    var prev = root.querySelector(".s-prev"), next = root.querySelector(".s-next");
    var count = root.querySelector(".s-count"), fs = root.querySelector(".s-full");
    var pdf = null, page = 1, busy = false, pending = 0, started = false;

    function say(t) { msg.textContent = t; msg.hidden = !t; }
    function controls() {
      var n = pdf ? pdf.numPages : 0;
      count.textContent = pdf ? page + " / " + n : "";
      prev.disabled = !pdf || page <= 1;
      next.disabled = !pdf || page >= n;
    }
    function fail(err) {
      var missing = err && (err.name === "MissingPDFException" || err.status === 404);
      say(missing ? "The slides will be added soon." : "The slides could not be loaded here. Use the download link instead.");
      canvas.hidden = true;
      controls();
    }

    function show(n) {
      if (!pdf) return;
      page = Math.max(1, Math.min(pdf.numPages, n));
      controls();
      if (busy) { pending = page; return; }
      busy = true;
      pdf.getPage(page).then(function (p) {
        var base = p.getViewport({ scale: 1 });
        var w = stage.clientWidth, h = stage.clientHeight;
        var scale = Math.min(w / base.width, h / base.height);
        var dpr = Math.min(window.devicePixelRatio || 1, 2);
        var vp = p.getViewport({ scale: scale * dpr });
        canvas.width = Math.floor(vp.width);
        canvas.height = Math.floor(vp.height);
        canvas.style.width = Math.floor(vp.width / dpr) + "px";
        canvas.style.height = Math.floor(vp.height / dpr) + "px";
        return p.render({ canvasContext: canvas.getContext("2d"), viewport: vp }).promise;
      }).catch(fail).then(function () {
        busy = false;
        if (pending) { var n2 = pending; pending = 0; show(n2); }
      });
    }

    function start() {
      if (started) return;
      started = true;
      say("Loading slides…");
      var s = document.createElement("script");
      s.src = "assets/vendor/pdfjs/pdf.min.js";
      s.onerror = fail;
      s.onload = function () {
        window.pdfjsLib.GlobalWorkerOptions.workerSrc = "assets/vendor/pdfjs/pdf.worker.min.js";
        window.pdfjsLib.getDocument(src).promise.then(function (d) {
          pdf = d;
          return d.getPage(1);
        }).then(function (p) {
          var v = p.getViewport({ scale: 1 });
          stage.style.setProperty("--ar", v.width + " / " + v.height);
          say("");
          canvas.hidden = false;
          show(1);
        }).catch(fail);
      };
      document.head.appendChild(s);
    }

    prev.addEventListener("click", function () { show(page - 1); });
    next.addEventListener("click", function () { show(page + 1); });
    root.addEventListener("keydown", function (e) {
      if (e.target.closest("a, button") && (e.key === " " || e.key === "Enter")) return;
      var k = e.key;
      if (k === "ArrowRight" || k === "PageDown") { e.preventDefault(); show(page + 1); }
      else if (k === "ArrowLeft" || k === "PageUp") { e.preventDefault(); show(page - 1); }
      else if (k === "Home") { e.preventDefault(); show(1); }
      else if (k === "End" && pdf) { e.preventDefault(); show(pdf.numPages); }
    });

    var x0 = null;
    stage.addEventListener("pointerdown", function (e) { x0 = e.clientX; });
    stage.addEventListener("pointerup", function (e) {
      if (x0 === null) return;
      var dx = e.clientX - x0; x0 = null;
      if (Math.abs(dx) > 50) show(page + (dx < 0 ? 1 : -1));
    });

    if (document.fullscreenEnabled && root.requestFullscreen) {
      fs.addEventListener("click", function () {
        if (document.fullscreenElement) document.exitFullscreen(); else root.requestFullscreen();
      });
      document.addEventListener("fullscreenchange", function () { setTimeout(function () { show(page); }, 80); });
    } else {
      fs.hidden = true;
    }

    var t = 0;
    window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(function () { show(page); }, 150); });

    controls();
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (en) {
        if (en[0].isIntersecting) { io.disconnect(); start(); }
      }, { rootMargin: "400px 0px" });
      io.observe(root);
    } else {
      start();
    }
  })();

  /* ---------------------------------------------- scroll-spy + anchors */
  (function nav() {
    var links = [].slice.call(document.querySelectorAll(".top nav a"));
    var map = {};
    links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var bar = document.querySelector(".top nav");

    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          var a = map[e.target.id];
          if (!a) return;
          links.forEach(function (l) { l.removeAttribute("aria-current"); });
          a.setAttribute("aria-current", "true");
          if (bar && bar.scrollWidth > bar.clientWidth) {
            bar.scrollTo({ left: a.offsetLeft - bar.clientWidth / 2 + a.clientWidth / 2, behavior: "auto" });
          }
        });
      }, { rootMargin: "-35% 0px -60% 0px" });
      Object.keys(map).forEach(function (id) {
        var el = document.getElementById(id);
        if (el) io.observe(el);
      });
    }

    document.querySelectorAll("section[id] .sec-head h2").forEach(function (h) {
      var a = document.createElement("a");
      a.className = "anchor";
      a.href = "#" + h.closest("section").id;
      a.setAttribute("aria-label", "Link to this section");
      a.textContent = "#";
      h.appendChild(a);
    });
  })();
})();
