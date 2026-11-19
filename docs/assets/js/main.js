/* TopoPT project page: tabs, copy buttons, and the relational-matching demo. */

document.documentElement.classList.add("js");

/* ------------------------------------------------------------------ tabs */
(function tabs() {
  document.querySelectorAll("[data-tabs]").forEach(function (root) {
    var buttons = root.querySelectorAll('[role="tab"]');
    var panels = root.querySelectorAll('[role="tabpanel"]');

    function select(btn) {
      buttons.forEach(function (b) {
        var on = b === btn;
        b.setAttribute("aria-selected", on ? "true" : "false");
        b.tabIndex = on ? 0 : -1;
      });
      panels.forEach(function (p) {
        p.hidden = p.id !== btn.getAttribute("aria-controls");
      });
    }

    buttons.forEach(function (btn, i) {
      btn.addEventListener("click", function () { select(btn); });
      btn.addEventListener("keydown", function (e) {
        var next = null;
        if (e.key === "ArrowRight") next = buttons[(i + 1) % buttons.length];
        if (e.key === "ArrowLeft") next = buttons[(i - 1 + buttons.length) % buttons.length];
        if (next) { e.preventDefault(); next.focus(); select(next); }
      });
    });
    select(buttons[0]);
  });
})();

/* ----------------------------------------------------------- copy buttons */
(function copyButtons() {
  document.querySelectorAll(".code .copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.parentElement.querySelector("pre").innerText;
      var done = function () {
        var old = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = old; }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else {
        fallback();
      }
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); done(); } catch (e) { /* ignore */ }
        document.body.removeChild(ta);
      }
    });
  });
})();

/* ------------------------------------------------------------------ demo
   A tiny, real optimisation. A "teacher" has 24 token features (16-D) that
   fall into four groups. A "student" has 6-D features that start random.
   We update the student ONLY with the relational objective
       || S_teacher - S_student ||_F^2 ,  S = pairwise cosine similarity,
   and record every step, so the slider replays an actual descent. The student
   never sees the teacher's features, only how its tokens relate to each other.
   (Illustration on synthetic tokens; the paper uses up to 512 tokens per stage
   plus a pointwise term and a learned projector.)
   ----------------------------------------------------------------------- */
(function demo() {
  var root = document.getElementById("demo");
  if (!root) return;

  var N = 24, DT = 16, DS = 6, STEPS = 160, LR = 0.02, NOISE = 0.55;
  var SIZES = [8, 6, 6, 4];

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var rnd = mulberry32(7);
  function gauss() {
    var u = 0; while (u === 0) u = rnd();
    var v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function norm(v) {
    var n = 0, i; for (i = 0; i < v.length; i++) n += v[i] * v[i];
    n = Math.sqrt(n) || 1;
    return v.map(function (x) { return x / n; });
  }
  function simMatrix(U, d) {
    var S = new Float32Array(N * N), i, j, k, s;
    for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
      s = 0; for (k = 0; k < d; k++) s += U[i][k] * U[j][k];
      S[i * N + j] = s;
    }
    return S;
  }

  // teacher features: four groups of tokens
  var group = [];
  SIZES.forEach(function (n, g) { for (var i = 0; i < n; i++) group.push(g); });
  // group centres: made orthogonal so groups read as clean blocks (cross-group similarity near 0)
  var centers = [];
  SIZES.forEach(function () {
    var v = Array.from({ length: DT }, gauss);
    centers.forEach(function (c) {
      var d = 0, i; for (i = 0; i < DT; i++) d += v[i] * c[i];
      for (i = 0; i < DT; i++) v[i] -= d * c[i];
    });
    centers.push(norm(v).map(function (x) { return x * 4; }));
  });
  var teacherU = group.map(function (g) {
    var c = centers[g], scale = Math.sqrt(c.reduce(function (a, x) { return a + x * x; }, 0) / DT);
    return norm(c.map(function (x) { return x + NOISE * scale * gauss(); }));
  });
  var T = simMatrix(teacherU, DT);

  // student: random start, optimise the relational loss only
  var X = Array.from({ length: N }, function () { return Array.from({ length: DS }, gauss); });
  var frames = [], dist = [];
  for (var t = 0; t <= STEPS; t++) {
    var U = X.map(norm), S = simMatrix(U, DS), f = 0, i, j, d;
    for (i = 0; i < N * N; i++) f += (S[i] - T[i]) * (S[i] - T[i]);
    frames.push(S); dist.push(Math.sqrt(f));

    var G = U.map(function () { return new Array(DS).fill(0); });
    for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
      var e = S[i * N + j] - T[i * N + j];
      for (d = 0; d < DS; d++) G[i][d] += 4 * e * U[j][d];
    }
    for (i = 0; i < N; i++) {
      var n = Math.sqrt(X[i].reduce(function (a, x) { return a + x * x; }, 0)) || 1, dot = 0;
      for (d = 0; d < DS; d++) dot += G[i][d] * U[i][d];
      for (d = 0; d < DS; d++) X[i][d] -= LR * (G[i][d] - dot * U[i][d]) / n;
    }
  }

  // drawing
  var cT = root.querySelector("#cv-teacher");
  var cS = root.querySelector("#cv-student");
  var cD = root.querySelector("#cv-diff");
  var slider = root.querySelector("#step");
  var readStep = root.querySelector("#read-step");
  var readDist = root.querySelector("#read-dist");
  var playBtn = root.querySelector("#play");

  var BLUE = [47, 111, 222], WHITE = [255, 255, 255], ORANGE = [232, 137, 43];
  function ramp(v) { // -1 blue, 0 white, +1 orange (same scale as the paper's figure)
    var a = v < 0 ? BLUE : ORANGE, k = Math.min(1, Math.abs(v));
    return "rgb(" + [0, 1, 2].map(function (c) { return Math.round(WHITE[c] + (a[c] - WHITE[c]) * k); }).join(",") + ")";
  }
  function inkRamp(v) { // 0 white -> 1 ink
    var k = Math.min(1, v / 0.9), a = [20, 32, 46];
    return "rgb(" + [0, 1, 2].map(function (c) { return Math.round(255 + (a[c] - 255) * k); }).join(",") + ")";
  }

  function size(canvas) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2), w = canvas.clientWidth || 240;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(w * dpr);
    return canvas.getContext("2d");
  }
  function paint(canvas, fn) {
    var ctx = size(canvas), cell = canvas.width / N, i, j;
    for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
      ctx.fillStyle = fn(i * N + j);
      ctx.fillRect(Math.floor(j * cell), Math.floor(i * cell), Math.ceil(cell), Math.ceil(cell));
    }
  }

  var step = 0;
  function render(s) {
    step = Math.max(0, Math.min(STEPS, Math.round(s)));
    var S = frames[step];
    paint(cT, function (k) { return ramp(T[k]); });
    paint(cS, function (k) { return ramp(S[k]); });
    paint(cD, function (k) { return inkRamp(Math.abs(T[k] - S[k])); });
    slider.value = step;
    readStep.textContent = step + " / " + STEPS;
    readDist.textContent = dist[step].toFixed(2);
    var msg = "Student similarity matrix at training step " + step + " of " + STEPS +
      "; Frobenius distance to the teacher matrix " + dist[step].toFixed(2) + ".";
    cS.setAttribute("aria-label", msg);
  }

  // playback
  var raf = 0, playing = false, last = 0, pos = 0;
  function frame(ts) {
    if (!playing) return;
    var dt = Math.min(0.05, (ts - last) / 1000); last = ts;
    pos += dt * 38; // steps per second
    if (pos >= STEPS) { pos = STEPS; render(pos); stop(); return; }
    render(pos);
    raf = requestAnimationFrame(frame);
  }
  function play() {
    if (pos >= STEPS) pos = 0;
    playing = true; last = performance.now();
    playBtn.textContent = "Pause";
    raf = requestAnimationFrame(frame);
  }
  function stop() {
    playing = false; cancelAnimationFrame(raf);
    playBtn.textContent = pos >= STEPS ? "Replay" : "Play";
  }

  playBtn.addEventListener("click", function () { playing ? stop() : play(); });
  slider.addEventListener("input", function () { pos = +slider.value; if (playing) stop(); render(pos); playBtn.textContent = pos >= STEPS ? "Replay" : "Play"; });
  window.addEventListener("resize", function () { render(step); });

  render(0);

  // one orchestrated moment: play once when the demo scrolls into view
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce && "IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      if (entries[0].isIntersecting) { io.disconnect(); setTimeout(play, 500); }
    }, { threshold: 0.55 });
    io.observe(root);
  } else {
    pos = STEPS; render(STEPS); playBtn.textContent = "Replay";
  }
})();
