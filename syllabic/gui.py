"""The live window (pyqtgraph, like main11).

Left:  timeline of the last few seconds, laid out like a Praat TextGrid: Whisper's words in a tier on
       top, the detected syllables (labelled with their dictionary IPA once Whisper has the words) in a
       tier at the bottom, the vowel-band level in between. Below it: the live syllabic transcription
       and the live accuracy scores.
Right: your vowel space vs a target (F1/F2, or F2/F3, F3/F4), a table of the same numbers, and the
       apparent vocal-tract length from F1-F4 over the session.

Colour use: series colours only say WHO a point belongs to (you / target / right now); vowels are told
apart by their IPA symbol, the way phoneticians draw vowel charts. A vowel of yours with fewer than
vowelspace.SETTLED_N tokens is a hollow ring with a grey label and stays out of the outline, the spread
and the links to the target: one odd token can put such an average anywhere. Correct / wrong uses the status
colours, always together with a symbol (check / cross) so it never depends on colour alone.
Legends sit in a text row above each chart so they never cover data.
"""
import html
import queue
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import formants as fmt
from . import pronounce, vowelspace
from .frontend import FRAME_S

THEMES = {
    "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "you": "#3987e5", "target": "#d95926", "now": "#199e70",
             "good": "#0ca30c", "serious": "#ec835a", "critical": "#d03b3b", "span_a": "#3b3b38",
             "span_b": "#55544f", "good_text": "#0ca30c"},
    "light": {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "you": "#2a78d6", "target": "#eb6834", "now": "#1baf7a",
              "good": "#0ca30c", "serious": "#ec835a", "critical": "#d03b3b", "span_a": "#e1e0d9",
              "span_b": "#c3c2b7", "good_text": "#006300"},
}
AXES = {"F1 / F2 (vowel quality)": ("F2", "F1", True, True),
        "F2 / F3": ("F2", "F3", True, False),
        "F3 / F4 (size / resonance)": ("F4", "F3", True, False)}
CORNERS = ("IY", "AE", "AA", "UW")   # the usual vowel-space-area quadrilateral /i æ ɑ u/
TIMELINE_S = 8.0


def _short(word, n=18):
    """Whisper writes held vowels as one endless word ("eeeeeeee..."): keep the display tidy."""
    return word if len(word) <= n else word[: n - 1] + "…"


def _color(hex_, alpha=255):
    c = QtGui.QColor(hex_)
    c.setAlpha(int(alpha))
    return c


class TextPool:
    """Re-usable TextItems (creating them every frame is slow)."""

    def __init__(self, plot, color, anchor=(0.5, 1.0), size=9, n=40, bold=False):
        self.items, self.used = [], 0
        self.font = QtGui.QFont()
        self.font.setPointSize(size)
        self.font.setBold(bold)
        self.metrics = QtGui.QFontMetricsF(self.font)
        for _ in range(n):
            it = pg.TextItem("", color=color, anchor=anchor)
            it.setFont(self.font)
            it.setVisible(False)
            plot.addItem(it)
            self.items.append(it)

    def width_px(self, text):
        return self.metrics.horizontalAdvance(text) + 6

    def begin(self):
        self.used = 0

    def put(self, x, y, text, color=None):
        if self.used >= len(self.items):
            return
        it = self.items[self.used]
        self.used += 1
        if it.textItem.toPlainText() != text:
            it.setText(text)
        if color is not None:
            it.setColor(color)
        it.setPos(x, y)
        it.setVisible(True)

    def end(self):
        for it in self.items[self.used:]:
            it.setVisible(False)


class LiveWindow(QtWidgets.QMainWindow):
    def __init__(self, pipeline, source, theme="dark", target_key=None, history=None, title="",
                 on_close=None, whisper=None):
        super().__init__()
        self.p, self.source, self.whisper = pipeline, source, whisper
        self.T = THEMES[theme]
        self.on_close = on_close
        self.space = vowelspace.VowelSpace()
        self.history = [t for t in history or [] if t.get("chart")]
        self.history_n = sum(self.space.add(tok) for tok in self.history)
        self.target = None
        self.paused = False
        self.dirty_words = True
        self.vc_fit = True
        self.chart_tokens = []
        self.tl_yr = None
        self.p.listeners.append(self._on_event)
        pg.setConfigOptions(antialias=True, background=self.T["surface"], foreground=self.T["muted"])
        self.setWindowTitle(title or "Live syllables & vowel space")
        self.resize(1560, 960)
        self._build()
        self.target_box.addItem("(no target)", None)
        for key, name in vowelspace.list_targets():
            self.target_box.addItem(name, key)
        if target_key:
            idx = self.target_box.findData(target_key)
            if idx < 0:
                self.target_box.addItem(str(target_key), target_key)
                idx = self.target_box.count() - 1
            self.target_box.setCurrentIndex(idx)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    # ------------------------------------------------------------------------------------------------
    def _plot(self, title, xlabel, ylabel):
        w = pg.PlotWidget()
        w.setTitle(title, color=self.T["ink"], size="10pt")
        pi = w.getPlotItem()
        for name in ("left", "bottom"):
            ax = pi.getAxis(name)
            ax.setPen(pg.mkPen(self.T["axis"]))
            ax.setTextPen(pg.mkPen(self.T["muted"]))
        pi.setLabel("bottom", xlabel, color=self.T["ink2"])
        pi.setLabel("left", ylabel, color=self.T["ink2"])
        pi.showGrid(x=True, y=True, alpha=0.35)
        pi.setMenuEnabled(False)
        return w

    def _legend(self, entries):
        """A legend as a row of text above the chart (never covers data). kind: dot / ring / cross / line."""
        glyph = {"dot": "●", "ring": "○", "cross": "✕", "line": "━"}
        parts = [f"<span style='color:{c}; font-size:12pt'>{glyph[k]}</span>&nbsp;"
                 f"<span style='color:{self.T['ink2']}'>{html.escape(t)}</span>" for k, c, t in entries]
        lab = QtWidgets.QLabel("&nbsp;&nbsp;&nbsp; ".join(parts))
        lab.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lab.setWordWrap(True)
        return lab

    def _build(self):
        T = self.T
        root = QtWidgets.QWidget()
        root.setStyleSheet(
            f"QWidget {{ background:{T['page']}; color:{T['ink']}; }}"
            f"QPushButton, QComboBox {{ background:{T['surface']}; color:{T['ink']}; border:1px solid {T['axis']};"
            f" padding:4px 10px; }}"
            f"QComboBox QAbstractItemView {{ background:{T['surface']}; color:{T['ink']}; }}"
            f"QTableWidget {{ background:{T['surface']}; color:{T['ink']}; gridline-color:{T['grid']};"
            f" border:1px solid {T['axis']}; }}"
            f"QHeaderView::section {{ background:{T['page']}; color:{T['ink2']}; border:0px; padding:3px; }}"
            f"QTextBrowser {{ background:{T['surface']}; color:{T['ink']}; border:1px solid {T['axis']}; }}")
        self.setCentralWidget(root)
        lay = QtWidgets.QHBoxLayout(root)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lay.addWidget(split)

        # ---------------- left: timeline, transcript, scores ----------------
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        top = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("")
        self.status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{T['ink2']};")
        top.addWidget(self.status, 1)
        self.pause_btn = QtWidgets.QPushButton("Pause (space)")
        self.pause_btn.clicked.connect(self.toggle_pause)
        top.addWidget(self.pause_btn)
        lv.addLayout(top)

        lv.addWidget(self._legend([("line", T["ink2"], "vowel-band level"), ("line", T["muted"], "nucleus threshold"),
                                   ("ring", T["muted"], "syllable, waiting for Whisper"),
                                   ("dot", T["good"], "syllable matched to its word"),
                                   ("cross", T["serious"], "extra syllable"),
                                   ("line", T["good"], "✓ word start found (±100 ms)"),
                                   ("line", T["critical"], "✗ word start missed")]))
        self.tl = self._plot("Words (Whisper) on top · detected syllables (dictionary IPA) at the bottom",
                             "seconds", "vowel-band level (dB)")
        self.tl.setMinimumHeight(320)
        pi = self.tl.getPlotItem()
        self.env = pi.plot(pen=pg.mkPen(T["ink2"], width=2))
        self.thr = pi.plot(pen=pg.mkPen(T["muted"], width=1))
        self.span_a = pi.plot(pen=pg.mkPen(T["span_a"], width=14), connect="pairs")
        self.span_b = pi.plot(pen=pg.mkPen(T["span_b"], width=14), connect="pairs")
        self.tick_hit = pi.plot(pen=pg.mkPen(_color(T["good"], 150), width=2), connect="pairs")
        self.tick_miss = pi.plot(pen=pg.mkPen(_color(T["critical"], 150), width=2), connect="pairs")
        self.nuc_pending = pg.ScatterPlotItem(size=10, pen=pg.mkPen(T["muted"], width=2), brush=None, symbol="o",
                                              hoverable=True, tip=self._syl_tip)
        self.nuc_match = pg.ScatterPlotItem(size=10, pen=pg.mkPen(T["surface"], width=2), brush=pg.mkBrush(T["good"]),
                                            symbol="o", hoverable=True, tip=self._syl_tip)
        self.nuc_extra = pg.ScatterPlotItem(size=11, pen=pg.mkPen(T["serious"], width=2), brush=pg.mkBrush(T["serious"]),
                                            symbol="x", hoverable=True, tip=self._syl_tip)
        for it in (self.nuc_pending, self.nuc_match, self.nuc_extra):
            pi.addItem(it)
        self.word_text = TextPool(pi, T["ink"], anchor=(0.0, 0.5), size=10, n=40, bold=True)
        self.syl_text = TextPool(pi, T["ink"], anchor=(0.5, 0.5), size=10, n=80)
        lv.addWidget(self.tl, 3)

        lab = QtWidgets.QLabel("Live syllabic transcription  (word · dictionary syllables · "
                               "✓ or detected/expected syllables)")
        lab.setStyleSheet(f"color:{T['ink2']}; font-weight:600;")
        lv.addWidget(lab)
        self.transcript = QtWidgets.QTextBrowser()
        self.transcript.setStyleSheet("font-size:13pt;")
        lv.addWidget(self.transcript, 2)
        self.scores = QtWidgets.QLabel()
        self.scores.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.scores.setStyleSheet(f"background:{T['surface']}; color:{T['ink']}; border:1px solid {T['axis']};"
                                  " padding:6px;")
        lv.addWidget(self.scores)
        split.addWidget(left)

        # ---------------- right: vowel space ----------------
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Target:"))
        self.target_box = QtWidgets.QComboBox()
        self.target_box.setMinimumWidth(300)
        self.target_box.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.target_box.currentIndexChanged.connect(self._target_changed)
        ctl.addWidget(self.target_box, 3)
        self.axes_box = QtWidgets.QComboBox()
        self.axes_box.addItems(list(AXES))
        self.axes_box.setToolTip("Which formants the chart shows")
        self.axes_box.currentIndexChanged.connect(self._axes_changed)
        ctl.addWidget(self.axes_box, 2)
        self.reset_btn = QtWidgets.QPushButton("Reset my vowels")
        self.reset_btn.clicked.connect(self.reset_space)
        ctl.addWidget(self.reset_btn)
        rv.addLayout(ctl)

        rv.addWidget(self._legend([("dot", T["you"], "you: recent average per vowel, with its spread"),
                                   ("ring", T["you"], f"you, fewer than {vowelspace.SETTLED_N} tokens yet"),
                                   ("ring", T["target"], "target"), ("dot", T["now"], "what you are saying now"),
                                   ("ring", T["muted"], "new syllable, waiting for Whisper")]))
        self.vc = self._plot("Your vowel space", "F2 (Hz)", "F1 (Hz)")
        vpi = self.vc.getPlotItem()
        self.tgt_poly = vpi.plot(pen=pg.mkPen(T["target"], width=2))
        self.tgt_pts = pg.ScatterPlotItem(size=13, pen=pg.mkPen(T["target"], width=2), brush=None, symbol="o")
        self.you_poly = vpi.plot(pen=pg.mkPen(T["you"], width=2))
        self.links = vpi.plot(pen=pg.mkPen(T["muted"], width=1), connect="pairs")
        self.ellipses = vpi.plot(pen=pg.mkPen(_color(T["you"], 110), width=1), connect="finite")
        self.tokens_item = pg.ScatterPlotItem(size=8, pen=None, hoverable=True, tip=self._tok_tip)
        self.you_pts = pg.ScatterPlotItem(size=13, pen=pg.mkPen(T["surface"], width=2), brush=pg.mkBrush(T["you"]))
        self.you_early = pg.ScatterPlotItem(size=13, pen=pg.mkPen(T["you"], width=2), brush=None)
        self.pending_item = pg.ScatterPlotItem(size=9, pen=pg.mkPen(T["muted"], width=2), brush=None)
        self.trail = vpi.plot(pen=pg.mkPen(_color(T["now"], 160), width=2))
        self.cursor_item = pg.ScatterPlotItem(size=15, pen=pg.mkPen(T["surface"], width=2), brush=pg.mkBrush(T["now"]))
        for it in (self.tgt_pts, self.tokens_item, self.you_early, self.you_pts, self.pending_item, self.cursor_item):
            vpi.addItem(it)
        self.you_text = TextPool(vpi, T["ink"], anchor=(-0.25, 1.1), size=13, n=16, bold=True)
        self.tgt_text = TextPool(vpi, T["ink2"], anchor=(1.25, -0.1), size=12, n=16)
        rv.addWidget(self.vc, 5)

        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["vowel", "n", "your F1", "target F1", "your F2", "target F2",
                                              "your F3", "target F3"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        rv.addWidget(self.table, 2)

        rv.addWidget(self._legend([("dot", T["you"], "your vowels"), ("line", T["you"], "your running average"),
                                   ("line", T["target"], "target")]))
        self.vtl = self._plot("Apparent vocal-tract length from F1–F4 (lower = higher formants overall)",
                              "seconds", "cm")
        tpi = self.vtl.getPlotItem()
        self.vtl_pts = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(_color(T["you"], 110)))
        tpi.addItem(self.vtl_pts)
        self.vtl_avg = tpi.plot(pen=pg.mkPen(T["you"], width=2))
        self.vtl_tgt = tpi.plot(pen=pg.mkPen(T["target"], width=2))
        self.vtl.setMaximumHeight(220)
        rv.addWidget(self.vtl, 2)
        right.setMinimumWidth(560)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([900, 660])
        self._axes_changed()

    # ------------------------------------------------------------------------------------------------
    def _on_event(self, event, payload):
        if event == "words":
            self.dirty_words = True
            for tok in payload["tokens"]:
                if tok["chart"]:
                    ok = self.space.add(tok)
                    tok["outlier"] = not ok
                    self.chart_tokens.append(tok)
        elif event == "ceiling":
            # the speaker's formant ceiling was re-estimated: every point was re-expressed, so rebuild
            self.space = vowelspace.VowelSpace()
            for tok in self.history:
                self.space.add(tok)
            for tok in self.chart_tokens:
                tok["outlier"] = not self.space.add(tok)
            self.dirty_words = True
            self.vc_fit = True

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key.Key_Space:
            self.toggle_pause()
        else:
            super().keyPressEvent(ev)

    def toggle_pause(self):
        self.paused = not self.paused
        self.source.set_paused(self.paused)
        self.pause_btn.setText("Resume (space)" if self.paused else "Pause (space)")

    def reset_space(self):
        self.space.reset()
        self.chart_tokens = []
        self.history = []
        self.dirty_words = True
        self.vc_fit = True

    def _target_changed(self):
        key = self.target_box.currentData()
        try:
            self.target = vowelspace.load_target(key) if key else None
        except Exception as e:
            self.target = None
            self.status.setText(f"Could not load target: {e}")
        self.dirty_words = True
        self.vc_fit = True

    def _axes_changed(self):
        xk, yk, invx, invy = AXES[self.axes_box.currentText()]
        pi = self.vc.getPlotItem()
        pi.invertX(invx)
        pi.invertY(invy)
        pi.setLabel("bottom", f"{xk} (Hz)", color=self.T["ink2"])
        pi.setLabel("left", f"{yk} (Hz)", color=self.T["ink2"])
        self.vc_fit = True
        self.dirty_words = True

    def closeEvent(self, ev):
        self.timer.stop()
        if self.on_close:
            self.on_close()
        super().closeEvent(ev)

    # ------------------------------------------------------------------------------------------------
    def tick(self):
        try:
            self.source.pump(self.p)
            self.p.poll()
            self._draw_timeline()
            self._draw_cursor()
            if self.dirty_words:
                self.dirty_words = False
                self._draw_transcript()
                self._draw_scores()
                self._draw_vowels()
            self._draw_status()
        except Exception as e:  # never let one bad frame kill the window
            import traceback
            traceback.print_exc()
            self.status.setText(f"error: {type(e).__name__}: {e}")

    def _draw_status(self):
        T = self.T
        w = self.whisper
        warn = []
        if w is None:
            ws = "Whisper off"
        elif w.error:
            ws = "Whisper failed"
            warn.append(f"Whisper failed: {html.escape(w.error)}")
        elif not w.ready.is_set():
            ws = f"Whisper '{w.model_name}' loading..."
        else:
            lat = np.median(self.p.whisper_secs) if self.p.whisper_secs else None
            ws = f"Whisper '{w.model_name}' ready" + (f", {lat:.2f} s per phrase" if lat else "")
            if hasattr(w, "lag") and w.lag() > 5:
                warn.append(f"Whisper is {w.lag():.0f} s behind (older phrases get skipped so it stays live)")
            if getattr(w, "restarts", 0):
                ws += f", restarted {w.restarts}x"
        if self.p.clip_recent > 0.001:
            warn.append(f"input clipping ({100 * self.p.clip_recent:.1f}% of samples at full scale): turn the "
                        "mic / VoiceMeeter gain down, clipping distorts the formants")
        mode = self.p.cfg.ceiling
        ceil = f"formant ceiling {self.p.ceiling_now():.0f} Hz" + (" (fitted to your voice)" if mode == "speaker" else
                                                                     " (per vowel)" if mode == "auto" else "")
        hist = f" (+{self.history_n} from history)" if self.history_n else ""
        text = (f"{html.escape(self.source.describe())} · {self.p.now:6.1f} s · {ws} · {ceil} · "
                f"{len(self.p.syllables)} syllables, {len(self.chart_tokens)} vowel points{hist}"
                + (" · PAUSED" if self.paused else ""))
        if warn:
            text += "".join(f"<br><span style='color:{T['critical']}'>⚠ {m}</span>" for m in warn)
        self.status.setText(text)

    def _pack(self, pool, items, x0, sec_per_px, rows):
        """Places labels left to right, each in the first row where it does not overlap the previous
        label of that row; labels that fit nowhere are skipped (their info is in the transcript)."""
        pool.begin()
        last = [-1e9] * len(rows)
        for x, text, centred in items:
            w = pool.width_px(text) * sec_per_px
            left = x - w / 2 if centred else x
            for r, y in enumerate(rows):
                if left >= last[r]:
                    pool.put(x, y, text)
                    last[r] = left + w
                    break
        pool.end()

    def _draw_timeline(self):
        p, d = self.p, self.p.det
        now = p.now
        t0 = max(0.0, now - TIMELINE_S)
        k0, k1 = int(t0 / FRAME_S), d.n_smoothed
        if k1 <= k0 + 2:
            return
        t = d.h.view("t", k0, k1)
        s = d.h.view("sband", k0, k1)
        self.env.setData(t, s)
        if d.ref is not None:
            thr = d.ref - d.p.silence_rel_db
            self.thr.setData([t0, now], [thr, thr])
            lo, hi = max(d.floor - 6, thr - 12), d.ref + 6
            if self.tl_yr is None or abs(self.tl_yr[0] - lo) > 4 or abs(self.tl_yr[1] - hi) > 4:
                self.tl_yr = (lo, hi)
        if self.tl_yr is None:
            return
        lo, hi = self.tl_yr
        tier = 0.18 * (hi - lo)
        y_syl, y_word = lo - 0.5 * tier, (hi + 0.35 * tier, hi + 0.8 * tier)
        self.tl.setYRange(lo - tier, hi + tier * 1.1, padding=0)
        self.tl.setXRange(t0, max(now, t0 + 1), padding=0)
        sec_per_px = TIMELINE_S / max(self.tl.getPlotItem().getViewBox().width(), 100.0)

        sy = [x for x in p.syllables[-80:] if x.t_peak >= t0 and x.status != "quiet"]
        A, B, labels = [], [], []
        for x in sy:
            end = x.t_end if x.t_end is not None else min(now, x.t_peak + 0.08)
            a, b = x.t_onset + 0.004, max(end - 0.004, x.t_onset + 0.01)   # small gap between neighbours
            (A if x.idx % 2 == 0 else B).extend([(a, y_syl), (b, y_syl)])
            if x.label:
                labels.append(((a + b) / 2, x.label, True))
        for item, pts in ((self.span_a, A), (self.span_b, B)):
            a = np.array(pts) if pts else np.zeros((0, 2))
            item.setData(a[:, 0], a[:, 1])
        self._pack(self.syl_text, labels, t0, sec_per_px, [y_syl + 0.33 * tier, y_syl - 0.33 * tier])

        groups = {"pending": [], "matched": [], "extra": []}
        for x in sy:
            g = "matched" if x.status == "matched" else "extra" if x.status == "extra" else "pending"
            groups[g].append({"pos": (x.t_peak, x.peak_db), "data": x})
        self.nuc_pending.setData(groups["pending"])
        self.nuc_match.setData(groups["matched"])
        self.nuc_extra.setData(groups["extra"])

        hit, miss, wl = [], [], []
        for w in p.words[-60:]:
            if w.end < t0:
                continue
            ok = w.onset_err is not None and abs(w.onset_err) <= p.cfg.align.onset_tol_s
            judged = w.scored and not w.phrase_initial   # Whisper pins a phrase's first word to the clip start
            wl.append((w.start, _short(w.word) + ("" if not judged else (" ✓" if ok else " ✗")), False))
            if judged:
                (hit if ok else miss).extend([(w.start, hi + tier), (w.start, lo - tier)])
        self._pack(self.word_text, wl, t0, sec_per_px, list(y_word))
        for item, pts in ((self.tick_hit, hit), (self.tick_miss, miss)):
            a = np.array(pts) if pts else np.zeros((0, 2))
            item.setData(a[:, 0], a[:, 1])

    def _xy(self, d):
        xk, yk, _, _ = AXES[self.axes_box.currentText()]
        return d.get(xk, 0.0), d.get(yk, 0.0)

    def _draw_cursor(self):
        xk, yk, _, _ = AXES[self.axes_box.currentText()]
        idx = {"F1": 1, "F2": 2, "F3": 3, "F4": 4}
        now = self.p.now
        cur = [c for c in self.p.cursor if now - c[0] < 0.6 and c[idx[xk]] > 0 and c[idx[yk]] > 0]
        if cur:
            xs = [c[idx[xk]] for c in cur]
            ys = [c[idx[yk]] for c in cur]
            self.trail.setData(xs, ys)
            self.cursor_item.setData([xs[-1]], [ys[-1]])
        else:
            self.trail.setData([], [])
            self.cursor_item.setData([], [])
        pend = [s for s in self.p.syllables[-12:] if s.status == "pending" and s.formants
                and now - s.t_peak < 4.0 and s.formants[idx[xk] - 1] > 0 and s.formants[idx[yk] - 1] > 0]
        self.pending_item.setData([s.formants[idx[xk] - 1] for s in pend], [s.formants[idx[yk] - 1] for s in pend])

    def _outline(self, item, pts_by_vowel):
        xk, yk, _, _ = AXES[self.axes_box.currentText()]
        poly = [pts_by_vowel[v] for v in CORNERS if v in pts_by_vowel]
        if xk == "F2" and yk == "F1" and len(poly) >= 3:
            poly.append(poly[0])
            item.setData([a[0] for a in poly], [a[1] for a in poly])
        else:
            item.setData([], [])

    def _draw_vowels(self):
        T = self.T
        xk, yk, _, _ = AXES[self.axes_box.currentText()]
        means = self.space.means()
        tv = (self.target or {}).get("vowels", {})
        tgt = {v: self._xy(tv[v]) for v in pronounce.CHART_VOWELS if v in tv and min(self._xy(tv[v])) > 0}
        drawn = {v: self._xy(m) for v, m in means.items() if v in pronounce.CHART_VOWELS and min(self._xy(m)) > 0}
        # settled averages are solid; ones with only a few tokens are hollow and stay out of the outline,
        # the spread ellipses and the links to the target (a single odd token can put them anywhere)
        you = {v: xy for v, xy in drawn.items() if means[v]["n"] >= vowelspace.SETTLED_N}
        early = {v: xy for v, xy in drawn.items() if v not in you}
        self.tgt_pts.setData([a[0] for a in tgt.values()], [a[1] for a in tgt.values()])
        self._outline(self.tgt_poly, tgt)
        self.tgt_text.begin()
        for v, (x, y) in tgt.items():
            self.tgt_text.put(x, y, pronounce.IPA[v])
        self.tgt_text.end()
        self.you_pts.setData([a[0] for a in you.values()], [a[1] for a in you.values()])
        self.you_early.setData([a[0] for a in early.values()], [a[1] for a in early.values()])
        self._outline(self.you_poly, you)
        self.you_text.begin()
        for v, (x, y) in you.items():
            self.you_text.put(x, y, pronounce.IPA[v], T["ink"])
        for v, (x, y) in early.items():
            self.you_text.put(x, y, pronounce.IPA[v], T["muted"])
        self.you_text.end()
        links = []
        for v, xy in you.items():
            if v in tgt:
                links += [xy, tgt[v]]
        a = np.array(links) if links else np.zeros((0, 2))
        self.links.setData(a[:, 0], a[:, 1])
        ex, ey = [], []
        if xk == "F2" and yk == "F1":   # the running spread is kept over F1/F2
            th = np.linspace(0, 2 * np.pi, 40)
            for v in you:
                m = means[v]
                w, U = np.linalg.eigh(m["cov"])
                pts = U @ (np.sqrt(np.maximum(w, 1.0))[:, None] * np.vstack([np.cos(th), np.sin(th)]))
                ex += list(m["F2"] + pts[1]) + [np.nan]
                ey += list(m["F1"] + pts[0]) + [np.nan]
        self.ellipses.setData(ex, ey)
        toks = [t for t in self.chart_tokens[-200:] if not t.get("outlier")]
        spots, n = [], len(toks)
        for i, t in enumerate(toks):
            x, y = self._xy(t)
            if x > 0 and y > 0:
                spots.append({"pos": (x, y), "brush": pg.mkBrush(_color(T["you"], 40 + 170 * (i + 1) / max(n, 1))),
                              "data": t})
        self.tokens_item.setData(spots)
        xs = [s["pos"][0] for s in spots] + [a[0] for a in tgt.values()] + [a[0] for a in drawn.values()]
        ys = [s["pos"][1] for s in spots] + [a[1] for a in tgt.values()] + [a[1] for a in drawn.values()]
        dflt = {"F1": (250, 1000), "F2": (700, 2900), "F3": (2000, 3400), "F4": (3000, 4800)}
        (x0, x1), (y0, y1) = dflt[xk], dflt[yk]
        if len(xs) >= 2:
            px0, px1 = np.percentile(xs, [1, 99])
            py0, py1 = np.percentile(ys, [1, 99])
            x0, x1, y0, y1 = min(x0, px0), max(x1, px1), min(y0, py0), max(y1, py1)
        last = getattr(self, "vc_last_fit", None)
        grew = last is None or x0 < last[0] or x1 > last[1] or y0 < last[2] or y1 > last[3]
        if self.vc_fit or grew:   # re-fit only when the data outgrows the last fit, so a zoom you set stays
            self.vc.setXRange(x0, x1, padding=0.06)
            self.vc.setYRange(y0, y1, padding=0.08)
            self.vc_last_fit = (x0, x1, y0, y1)
            self.vc_fit = False
        self.vc.setTitle(f"Your vowel space  vs  {self.target['name'] if self.target else 'no target'}",
                         color=T["ink"], size="10pt")
        self._draw_table(means, tv)
        self._draw_vtl(tv)

    def _draw_table(self, means, tv):
        rows = [v for v in pronounce.CHART_VOWELS if v in means or v in tv]
        self.table.setRowCount(len(rows))

        def cell(r, c, text, color=None):
            it = QtWidgets.QTableWidgetItem(text)
            if color:
                it.setForeground(QtGui.QBrush(QtGui.QColor(color)))
            self.table.setItem(r, c, it)
        for r, v in enumerate(rows):
            m, t = means.get(v), tv.get(v)
            few = self.T["muted"] if m and m["n"] < vowelspace.SETTLED_N else None
            cell(r, 0, f"{pronounce.IPA[v]}  ({pronounce.KEYWORD[v]})")
            cell(r, 1, str(m["n"]) if m else "0", few)
            for c, f in ((2, "F1"), (4, "F2"), (6, "F3")):
                cell(r, c, f"{m[f]:.0f}" if m and m[f] > 0 else "–", few)
                cell(r, c + 1, f"{t[f]:.0f}" if t and t.get(f, 0) > 0 else "–", self.T["ink2"])

    def _draw_vtl(self, tv):
        toks = [t for t in self.chart_tokens if t.get("vtl_cm") and not t.get("outlier")]
        if toks:
            t = np.array([x["t"] for x in toks])
            v = np.array([x["vtl_cm"] for x in toks])
            self.vtl_pts.setData(t, v)
            avg, a = np.zeros_like(v), 0.0
            for i, x in enumerate(v):
                a = x if i == 0 else a + max(0.05, 1.0 / (i + 1)) * (x - a)
                avg[i] = a
            self.vtl_avg.setData(t, avg)
            x0, x1 = float(t[0]), float(max(t[-1], t[0] + 1))
        else:
            self.vtl_pts.setData([], [])
            self.vtl_avg.setData([], [])
            x0, x1 = 0.0, max(1.0, self.p.now)
        vt = [fmt.apparent_vtl_cm([m.get("F1", 0), m.get("F2", 0), m.get("F3", 0), m.get("F4", 0)])
              for m in tv.values()]
        vt = [x for x in vt if x > 0]
        if vt:
            tgt = float(np.median(vt))
            self.vtl_tgt.setData([x0, x1], [tgt, tgt])
        else:
            self.vtl_tgt.setData([], [])

    def _draw_transcript(self):
        T = self.T
        out = []
        notes = {"exercise": "held vowel — shown live on the vowel chart, no words to transcribe",
                 "behind": "skipped: Whisper was behind, so it jumped to your newest phrase",
                 "timeout": "skipped: Whisper got stuck on this one and was restarted"}
        shown = [u for u in self.p.utts if u["status"] in ("done", "sent", "error") or u["status"] in notes]
        for u in shown[-10:]:
            if u["status"] == "sent":
                out.append(f"<p style='color:{T['muted']}'>… transcribing {u['t0']:.1f}–{u['t1']:.1f} s</p>")
                continue
            if u["status"] in notes:
                out.append(f"<p style='color:{T['muted']}'>{u['t0']:.1f}s &nbsp;({notes[u['status']]})</p>")
                continue
            if u["status"] == "error":
                out.append(f"<p style='color:{T['critical']}'>Whisper error: {html.escape(u['text'])}</p>")
                continue
            parts = []
            for gi in u["words"]:
                w = self.p.words[gi]
                word = html.escape(_short(w.word))
                if not w.scored:
                    parts.append(f"<span style='color:{T['muted']}'>{word} <small>(not in dictionary)</small></span>")
                    continue
                ipa = "·".join(html.escape(x.ipa) for x in w.pron.syllables)
                tag = (f"<span style='color:{T['good_text']}'>✓</span>" if w.count_ok else
                       f"<span style='color:{T['critical']}'>✗ {w.n_det}/{w.n_exp}</span>")
                parts.append(f"<b>{word}</b> <span style='color:{T['ink2']}'>/{ipa}/</span> {tag}")
            out.append(f"<p><span style='color:{T['muted']}'>{u['t0']:.1f}s</span> &nbsp;" + " &nbsp; ".join(parts) + "</p>")
        self.transcript.setHtml("".join(reversed(out)) or
                                f"<p style='color:{T['muted']}'>Say something. Words appear here about half a second "
                                "after you pause.</p>")

    def _draw_scores(self):
        T = self.T
        r, a = self.p.score.summary("recent"), self.p.score.summary("all")

        def f(s, k, unit="%"):
            v = s.get(k)
            return "–" if v is None else (f"{v:+.0f} ms" if unit == "ms" else f"{v:.0f}%")
        rows = [("Words whose syllable count came out right", "count_exact", "%"),
                ("Word starts found (first syllable within ±100 ms)*", "onset_hit", "%"),
                ("  after a short pause", "onset_hit_after_pause", "%"),
                ("  in running speech", "onset_hit_connected", "%"),
                ("Dictionary syllables that were detected", "recall", "%"),
                ("Detected syllables that belong to a word syllable", "precision", "%"),
                ("Typical start offset (detected − Whisper)", "onset_median_ms", "ms")]
        h = (f"<table cellspacing='0' cellpadding='3'><tr><th align='left' style='color:{T['ink2']}'>"
             f"checked against Whisper + dictionary</th><th style='color:{T['ink2']}'>last {r.get('words', 0)} words</th>"
             f"<th style='color:{T['ink2']}'>session ({a.get('words', 0)})</th></tr>")
        for name, k, unit in rows:
            h += f"<tr><td>{name}</td><td align='right'>{f(r, k, unit)}</td><td align='right'>{f(a, k, unit)}</td></tr>"
        h += (f"</table><div style='color:{T['muted']}'>*Each phrase's first word is left out: Whisper pins its start "
              "to the start of the clip. Whisper's other word times are off by tens of ms, so these numbers mix "
              "its error with the detector's (on test speech they read 5-10 points below the truth).</div>")
        self.scores.setText(h)

    # ---- hover tips -------------------------------------------------------------------------------
    def _tok_tip(self, x, y, data):
        t = data
        return (f"{t['word']}  /{t['syllable']}/  vowel {t['ipa']}\nF1 {t['F1']:.0f}  F2 {t['F2']:.0f}  "
                f"F3 {t['F3']:.0f}  F4 {t['F4']:.0f} Hz\nat {t['t']:.2f} s, pitch {t['f0']:.0f} Hz")

    def _syl_tip(self, x, y, data):
        s = data
        F = s.formants or [0, 0, 0, 0]
        lab = f"/{s.label}/" if s.label else s.status
        return (f"syllable {s.idx}: {lab}\nstart {s.t_onset:.2f} s, nucleus {s.t_peak:.2f} s\n"
                f"F1 {F[0]:.0f}  F2 {F[1]:.0f}  F3 {F[2]:.0f} Hz, pitch {s.f0:.0f} Hz")


# --------------------------------------------------------------------------------------------------
# audio sources
# --------------------------------------------------------------------------------------------------
class MicSource:
    def __init__(self, device, samplerate, name=""):
        import sounddevice as sd
        self.sr = int(samplerate)
        self.name = name
        self.q = queue.Queue()
        self.paused = False
        self.stream = sd.InputStream(device=device, channels=1, samplerate=self.sr,
                                     blocksize=max(1, int(self.sr * 0.02)), callback=self._cb)

    def _cb(self, indata, frames, time_info, status):
        if not self.paused:
            self.q.put(indata[:, 0].copy())

    def start(self):
        self.stream.start()
        return self

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def set_paused(self, v):
        self.paused = v

    def pump(self, pipeline):
        blocks = []
        while True:
            try:
                blocks.append(self.q.get_nowait())
            except queue.Empty:
                break
        if blocks:
            pipeline.feed(np.concatenate(blocks))

    def describe(self):
        return f"mic: {self.name} @ {self.sr} Hz"


class FileSource:
    """Plays a recording through the live pipeline in real time (optionally out loud)."""

    def __init__(self, y, sr, name="", play=False, speed=1.0):
        self.y, self.sr, self.name, self.play, self.speed = y, int(sr), name, play, speed
        self.pos, self.t_start, self.paused, self.done = 0, None, False, False
        self.paused_at = None

    def start(self):
        self.t_start = time.perf_counter()
        if self.play:
            import sounddevice as sd
            sd.play(self.y, self.sr)
        return self

    def stop(self):
        if self.play:
            import sounddevice as sd
            sd.stop()

    def set_paused(self, v):
        if v and not self.paused:
            self.paused_at = time.perf_counter()
            if self.play:
                import sounddevice as sd
                sd.stop()
        elif not v and self.paused:
            self.t_start += time.perf_counter() - self.paused_at
            if self.play:
                import sounddevice as sd
                sd.play(self.y[self.pos:], self.sr)
        self.paused = v

    def pump(self, pipeline):
        if self.paused or self.done:
            return
        target = min(len(self.y), int((time.perf_counter() - self.t_start) * self.sr * self.speed))
        if target > self.pos:
            pipeline.feed(self.y[self.pos: target])
            self.pos = target
        if self.pos >= len(self.y):
            self.done = True
            pipeline.finish(wait=False)

    def describe(self):
        state = "finished" if self.done else f"{self.pos / self.sr:.1f}/{len(self.y) / self.sr:.1f} s"
        return f"file: {self.name} ({state})"
