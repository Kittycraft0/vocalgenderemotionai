"""The live window (pyqtgraph, like main11). The same window also reviews a recording or a saved session.

Left:  a timeline laid out like a Praat TextGrid: Whisper's words in a tier on top, the detected syllables
       (labelled with their dictionary IPA once Whisper has the words) in a tier at the bottom, the
       vowel-band level in between. It keeps the WHOLE session, not just the last seconds: drag it or turn
       the mouse wheel over it to go back, "Live" (or End) follows the newest audio again. Hovering puts a
       line on it and shows the word, the syllable and its formants at that moment (in a row under the
       timeline, so it covers nothing) and rings that syllable on the vowel chart. Clicking plays that
       word. Playback only works while paused or when reviewing, so the sound cannot go back into the
       microphone and end up in your own session.
       Below it: the syllabic transcription (click a phrase's time to jump there) and the accuracy scores.
Right: your vowel space vs a target, for the whole session or only for what the timeline shows, with
       optional men -> women lines; a table of the same numbers; the apparent vocal-tract length over the
       session (click it to jump the timeline to that moment).

Colour use: series colours only say WHO a point belongs to (you / target / right now); vowels are told
apart by their IPA symbol, the way phoneticians draw vowel charts. A vowel of yours with fewer than
vowelspace.SETTLED_N tokens is a hollow ring with a grey label and stays out of the outline, the spread
and the links to the target: one odd token can put such an average anywhere. Correct / wrong uses the status
colours, always together with a symbol (check / cross) so it never depends on colour alone.
Legends and read-outs sit in text rows next to the charts so they never cover data.

Axes: the F1/F2 chart uses the usual vowel-chart layout by default (F2 grows to the LEFT, F1 DOWNWARD,
so /i/ is top left and /ɑ/ at the bottom, like the IPA chart); untick "vowel-chart layout" for plain axes
where values grow right and up. F2/F3 and F3/F4 always use plain axes: with one axis flipped (as before
2026-11-10), a voice with higher F3 AND higher F4 moved up-left, which read as if F3 and F4 went in
opposite directions.
"""
import bisect
import html
import os
import queue
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import formants as fmt
from . import pronounce, vowelspace
from .frontend import FRAME_S, SR

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
# chart -> (formant on the x axis, formant on the y axis)
AXES = {"F1/F2 (vowels)": ("F2", "F1"),
        "F2/F3": ("F2", "F3"),
        "F3/F4 (size)": ("F3", "F4")}
CORNERS = ("IY", "AE", "AA", "UW")   # the usual vowel-space-area quadrilateral /i æ ɑ u/
TIMELINE_S = 8.0                     # seconds the timeline shows while it follows the newest audio
PLAY_MAX_S = 60.0                    # longest stretch "Play view" plays
# A live-cursor value outside these is more likely a tracking error than a vowel; only values inside
# them make the chart grow (an extreme "eeee" with F2 near 3 kHz used to run off the chart).
PLAUSIBLE = {"F1": (150, 1300), "F2": (500, 3600), "F3": (1500, 4600), "F4": (2400, 5800)}
# men -> women reference pairs for the "men → women" lines (both measured like your voice)
GENDER_PAIRS = {"running speech (ARCTIC)": ("arctic_men_measured", "arctic_women_measured"),
                "isolated words (Hillenbrand)": ("hillenbrand_men_measured", "hillenbrand_women_measured")}
STATUS_TEXT = {"pending": "waiting for Whisper", "extra": "extra syllable (matches no dictionary syllable)",
               "unscored": "word not in the dictionary", "exercise": "held vowel (exercise)",
               "quiet": "too quiet, left out"}


def _short(word, n=18):
    """Whisper writes held vowels as one endless word ("eeeeeeee..."): keep the display tidy."""
    return word if len(word) <= n else word[: n - 1] + "…"


def _color(hex_, alpha=255):
    c = QtGui.QColor(hex_)
    c.setAlpha(int(alpha))
    return c


def clock(t, decimals=1):
    """Seconds as 12.3 s, 4:05.2 or 1:02:03."""
    t = max(0.0, float(t))
    if t < 60:
        return f"{t:.{decimals}f} s"
    m, s = divmod(t, 60)
    if m < 60:
        return f"{int(m)}:{s:0{3 + decimals if decimals else 2}.{decimals}f}"
    h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02d}:{int(s):02d}"


class TimeAxis(pg.AxisItem):
    """Seconds, written as m:ss once the numbers pass a minute (sessions run for hours)."""

    def tickStrings(self, values, scale, spacing):
        if not values or max(abs(v) for v in values) < 60:
            return super().tickStrings(values, scale, spacing)
        out = []
        for v in values:
            m, s = divmod(max(float(v), 0.0), 60)
            h, m = divmod(int(m), 60)
            s = np.floor(s * 10) / 10 if spacing < 1 else int(s)
            body = f"{s:04.1f}" if spacing < 1 else f"{s:02d}"
            out.append(f"{h}:{m:02d}:{body}" if h else f"{m}:{body}")
        return out


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


class LevelTrace:
    """The whole session's vowel-band level for the timeline. The detector itself keeps only its last two
    minutes of frames; this copy (8 bytes per 10 ms frame, about 3 MB per hour) is what lets you scroll
    back through a session."""

    def __init__(self):
        self.k0 = None                       # absolute frame number of element 0
        self.n = 0
        self.level = np.zeros(0, np.float32)
        self.thr = np.zeros(0, np.float32)   # nucleus threshold as it was at that moment

    def pull(self, det):
        """Copies the frames the detector has finished smoothing since the last call."""
        end = det.n_smoothed
        start = max(det.h.base, det.h.base if self.k0 is None else self.k0 + self.n)
        if end <= start:
            return False
        if self.k0 is None:
            self.k0 = start
        lv = det.h.view("sband", start, end).astype(np.float32)
        thr = np.float32(det.ref - det.p.silence_rel_db) if det.ref is not None else np.float32(np.nan)
        gap = start - (self.k0 + self.n)    # frames the detector dropped before they were copied
        need = self.n + gap + len(lv)
        if need > len(self.level):
            cap = max(need, 2 * len(self.level), 1 << 15)
            grow = np.full(cap - self.n, np.nan, np.float32)
            self.level = np.concatenate([self.level[: self.n], grow])
            self.thr = np.concatenate([self.thr[: self.n], grow])
        self.level[self.n: self.n + gap] = np.nan
        self.thr[self.n: self.n + gap] = np.nan
        self.n += gap
        self.level[self.n: self.n + len(lv)] = lv
        self.thr[self.n: self.n + len(lv)] = thr
        self.n += len(lv)
        return True

    def window(self, t0, t1, max_points=3000):
        """(t, level, threshold) between t0 and t1 seconds. Long stretches keep the lowest and highest
        value of each screen-pixel's worth of frames, so every syllable peak still shows."""
        empty = np.zeros(0, np.float32)
        if self.k0 is None or self.n == 0:
            return empty, empty, empty
        a = max(0, int(np.floor(t0 / FRAME_S)) - self.k0)
        b = min(self.n, int(np.ceil(t1 / FRAME_S)) + 1 - self.k0)
        if b <= a:
            return empty, empty, empty
        lv, th = self.level[a:b], self.thr[a:b]
        t = (self.k0 + np.arange(a, b)) * FRAME_S
        step = int(np.ceil(2 * (b - a) / max(max_points, 2)))
        if step <= 1:
            return t, lv, th
        m = (b - a) // step * step
        if m == 0:
            return t, lv, th
        r = lv[:m].reshape(-1, step)
        lo, hi = np.fmin.reduce(r, axis=1), np.fmax.reduce(r, axis=1)
        tb = t[:m].reshape(-1, step)
        tt = np.empty(2 * len(lo))
        yy = np.empty(2 * len(lo), np.float32)
        hh = np.repeat(th[:m].reshape(-1, step)[:, 0], 2)
        tt[0::2], tt[1::2] = tb[:, 0], tb[:, -1]
        yy[0::2], yy[1::2] = lo, hi
        return tt, yy, hh


class LiveWindow(QtWidgets.QMainWindow):
    def __init__(self, pipeline, source, theme="dark", target_key=None, history=None, title="",
                 on_close=None, whisper=None, review=False, words_note=None, open_cmd=None):
        """review: the source is a recording or a saved session, not a microphone (playback always works).
        words_note: where the words come from when it is not a live Whisper (e.g. a saved session).
        open_cmd: command prefix that opens a recording/session in a new window ("Open…" button)."""
        super().__init__()
        self.p, self.source, self.whisper = pipeline, source, whisper
        self.review, self.words_note, self.open_cmd = review, words_note, open_cmd
        self.theme = theme
        self.T = THEMES[theme]
        self.on_close = on_close
        self.space = vowelspace.VowelSpace()
        self.history = [t for t in history or [] if t.get("chart")]
        self.history_n = sum(self.space.add(tok) for tok in self.history)
        self.target = None
        self.gender = None                   # (men target, women target) for the men -> women lines
        self.paused = False
        self.dirty_words = True
        self.vc_fit = True
        self.chart_tokens = []
        self.tok_at = {}                     # nucleus time in ms -> its vowel token (hover read-out)
        self.tl_yr = None
        self.trace = LevelTrace()
        self.follow = True                   # timeline follows the newest audio; off once you drag/zoom it
        self.span = TIMELINE_S
        self.view = (0.0, TIMELINE_S)        # what the timeline shows, seconds
        self._syl_t, self._word_t = [], []   # nucleus / word-start times, for finding what is in view
        self._tl_key = self._hover_key = self._scope_key = None
        self._scope_drawn = 0.0
        self.version = 0                     # bumped whenever words, labels or formants change
        self.hover_scene = None
        self.play = None                     # (t0, t1, wall-clock start) of what is playing
        self.note = ("", 0.0)                # a short message in the status row, and until when
        self.cursor_out = 0
        self.p.listeners.append(self._on_event)
        pg.setConfigOptions(antialias=True, background=self.T["surface"], foreground=self.T["muted"])
        self.setWindowTitle(title or "Live syllables & vowel space")
        self.resize(1560, 980)
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
    def _plot(self, title, xlabel, ylabel, time_axis=False):
        w = pg.PlotWidget(axisItems={"bottom": TimeAxis(orientation="bottom")} if time_axis else None)
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
        glyph = {"dot": "●", "ring": "○", "cross": "✕", "line": "━", "dash": "┅"}
        parts = [f"<span style='color:{c}; font-size:12pt'>{glyph[k]}</span>&nbsp;"
                 f"<span style='color:{self.T['ink2']}'>{html.escape(t)}</span>" for k, c, t in entries]
        lab = QtWidgets.QLabel("&nbsp;&nbsp;&nbsp; ".join(parts))
        lab.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lab.setWordWrap(True)
        return lab

    def _button(self, text, slot, tip=""):
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(slot)
        b.setToolTip(tip)
        return b

    def _build(self):
        T = self.T
        root = QtWidgets.QWidget()
        root.setStyleSheet(
            f"QWidget {{ background:{T['page']}; color:{T['ink']}; }}"
            f"QPushButton, QComboBox {{ background:{T['surface']}; color:{T['ink']}; border:1px solid {T['axis']};"
            f" padding:4px 10px; }}"
            f"QPushButton:disabled {{ color:{T['muted']}; }}"
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
        if self.open_cmd:
            top.addWidget(self._button("Open…", self.open_other,
                                       "Open a recording (any audio file) or a saved session (pick any file in its "
                                       "folder) in a new window, to look through it with the words"))
        self.live_btn = self._button("Live ▶▶", self.go_live, "Follow the newest audio again (End). Dragging or "
                                     "zooming the timeline stops following.")
        top.addWidget(self.live_btn)
        self.play_btn = self._button("▶ Play view", self.play_view,
                                     f"Play what the timeline shows (up to {PLAY_MAX_S:.0f} s; Esc stops). Clicking "
                                     "a word plays just that word. Live, pause first so the sound cannot go back "
                                     "into the microphone.")
        top.addWidget(self.play_btn)
        self.pause_btn = self._button("", self.toggle_pause, "Space")
        top.addWidget(self.pause_btn)
        self._pause_text()
        lv.addLayout(top)

        lv.addWidget(self._legend([("line", T["ink2"], "vowel-band level"), ("line", T["muted"], "nucleus threshold"),
                                   ("ring", T["muted"], "syllable, waiting for Whisper"),
                                   ("dot", T["good"], "syllable matched to its word"),
                                   ("cross", T["serious"], "extra syllable"),
                                   ("line", T["good"], "✓ word start found (±100 ms)"),
                                   ("line", T["critical"], "✗ word start missed"),
                                   ("dash", T["ink"], "mouse: what was said there")]))
        self.tl = self._plot("Words (Whisper) on top · detected syllables (dictionary IPA) at the bottom · "
                             "drag or scroll to go back", "time", "vowel-band level (dB)", time_axis=True)
        self.tl.setMinimumHeight(320)
        pi = self.tl.getPlotItem()
        vb = pi.getViewBox()
        vb.setMouseEnabled(x=True, y=False)
        vb.setLimits(xMin=-2.0, minXRange=0.5)
        vb.sigRangeChangedManually.connect(self._tl_dragged)
        self.env = pi.plot(pen=pg.mkPen(T["ink2"], width=2))
        self.thr = pi.plot(pen=pg.mkPen(T["muted"], width=1))
        self.span_a = pi.plot(pen=pg.mkPen(T["span_a"], width=14), connect="pairs")
        self.span_b = pi.plot(pen=pg.mkPen(T["span_b"], width=14), connect="pairs")
        self.tick_hit = pi.plot(pen=pg.mkPen(_color(T["good"], 150), width=2), connect="pairs")
        self.tick_miss = pi.plot(pen=pg.mkPen(_color(T["critical"], 150), width=2), connect="pairs")
        self.nuc_pending = pg.ScatterPlotItem(size=10, pen=pg.mkPen(T["muted"], width=2), brush=None, symbol="o")
        self.nuc_match = pg.ScatterPlotItem(size=10, pen=pg.mkPen(T["surface"], width=2), brush=pg.mkBrush(T["good"]),
                                            symbol="o")
        self.nuc_extra = pg.ScatterPlotItem(size=11, pen=pg.mkPen(T["serious"], width=2), brush=pg.mkBrush(T["serious"]),
                                            symbol="x")
        for it in (self.nuc_pending, self.nuc_match, self.nuc_extra):
            pi.addItem(it)
        self.word_text = TextPool(pi, T["ink"], anchor=(0.0, 0.5), size=10, n=40, bold=True)
        self.syl_text = TextPool(pi, T["ink"], anchor=(0.5, 0.5), size=10, n=80)
        self.hover_line = pg.InfiniteLine(angle=90, movable=False,
                                          pen=pg.mkPen(T["ink"], width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.play_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(T["now"], width=2))
        for it in (self.hover_line, self.play_line):
            it.setVisible(False)
            it.setZValue(50)
            pi.addItem(it, ignoreBounds=True)
        self.tl.scene().sigMouseMoved.connect(self._tl_mouse)
        self.tl.scene().sigMouseClicked.connect(self._tl_click)
        lv.addWidget(self.tl, 3)
        self.hover_info = QtWidgets.QLabel()
        self.hover_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.hover_info.setWordWrap(True)
        self.hover_info.setMinimumHeight(44)
        self.hover_info.setStyleSheet(f"background:{T['surface']}; color:{T['ink']}; border:1px solid {T['axis']};"
                                      " padding:4px;")
        lv.addWidget(self.hover_info)

        lab = QtWidgets.QLabel("Syllabic transcription  (word · dictionary syllables · ✓ or detected/expected "
                               "syllables; click a time to jump there)")
        lab.setStyleSheet(f"color:{T['ink2']}; font-weight:600;")
        lv.addWidget(lab)
        self.transcript = QtWidgets.QTextBrowser()
        self.transcript.setStyleSheet("font-size:13pt;")
        self.transcript.setOpenLinks(False)
        self.transcript.anchorClicked.connect(self._transcript_link)
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
        self.target_box.setMinimumWidth(260)
        self.target_box.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.target_box.currentIndexChanged.connect(self._target_changed)
        ctl.addWidget(self.target_box, 3)
        ctl.addWidget(self._button("Save mine as target", self.save_as_target,
                                   "Save the vowels the chart shows now (the whole session, or only what the "
                                   "timeline shows) as a target, to compare a later session with"))
        self.reset_btn = self._button("Reset my vowels", self.reset_space)
        ctl.addWidget(self.reset_btn)
        rv.addLayout(ctl)
        ctl2 = QtWidgets.QHBoxLayout()
        fit = QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        self.axes_box = QtWidgets.QComboBox()
        self.axes_box.addItems(list(AXES))
        self.axes_box.setSizeAdjustPolicy(fit)
        self.axes_box.setToolTip("Which formants the chart shows: F1/F2 tells the vowels apart; F3/F4 mostly "
                                 "reflect the size of the vocal tract")
        self.axes_box.currentIndexChanged.connect(self._axes_changed)
        ctl2.addWidget(self.axes_box)
        self.layout_box = QtWidgets.QCheckBox("vowel-chart layout")
        self.layout_box.setChecked(True)
        self.layout_box.setToolTip("Ticked: the usual vowel-chart layout, F2 grows to the LEFT and F1 grows "
                                   "DOWNWARD, so /i/ is top left, /u/ top right and /ɑ/ at the bottom, like the IPA "
                                   "chart. Higher formants overall (a shorter vocal tract) then sit down and to the "
                                   "left.\nUnticked: plain axes, values grow to the right and upward.")
        self.layout_box.toggled.connect(self._axes_changed)
        ctl2.addWidget(self.layout_box)
        self.scope_box = QtWidgets.QComboBox()
        self.scope_box.addItems(["whole session", "timeline view only"])
        self.scope_box.setSizeAdjustPolicy(fit)
        self.scope_box.setToolTip("Whole session: running averages (recent tokens count more).\nTimeline view "
                                  "only: averages of just the stretch the timeline shows, e.g. one reading passage.")
        self.scope_box.currentIndexChanged.connect(self._scope_changed)
        ctl2.addWidget(self.scope_box)
        self.gender_box = QtWidgets.QComboBox()
        self.gender_box.setSizeAdjustPolicy(fit)
        self.gender_box.addItem("men→women: off", None)
        for name, pair in GENDER_PAIRS.items():
            self.gender_box.addItem(f"men→women: {name}", pair)
        self.gender_box.setToolTip(
            "Draws, for each vowel, a dashed line from the men's average to the women's average of one reference "
            "corpus, and places your settled vowels along those lines (F1-F3, on a ratio scale): 0% = the men's "
            "average, 100% = the women's, over 100% = beyond it (children's voices).\n\n"
            "How real speakers spread (syllabic/README.md):\n"
            "  Hillenbrand's 139 speakers, each against averages without them: middle 80% of men -28% to 19%, of "
            "women 60% to 128%, boys 95-167%, girls 122-196%; 99% of adults on the right side of 50%.\n"
            "  Running speech (ARCTIC, measured like you): one woman at 45-50%, one man at -41% to -64%.\n"
            "  To the side of the lines: typical speakers 5-12%.\n"
            "So it describes formants; it is not a verdict on how a voice is heard. Pitch, voice quality and "
            "intonation weigh in too, and the numbers per vowel scatter about ±25 points even for typical speakers.")
        self.gender_box.currentIndexChanged.connect(self._gender_changed)
        ctl2.addWidget(self.gender_box)
        ctl2.addStretch(1)
        rv.addLayout(ctl2)

        rv.addWidget(self._legend([("dot", T["you"], "you (average ± spread)"),
                                   ("ring", T["you"], f"you, < {vowelspace.SETTLED_N} tokens"),
                                   ("ring", T["target"], "target"), ("dot", T["now"], "now"),
                                   ("ring", T["muted"], "new, no word yet"),
                                   ("ring", T["ink"], "under the mouse"),
                                   ("dash", T["muted"], "men → women")]))
        self.vc = self._plot("Your vowel space", "F2 (Hz)", "F1 (Hz)")
        vpi = self.vc.getPlotItem()
        self.mw_lines = vpi.plot(pen=pg.mkPen(T["muted"], width=1, style=QtCore.Qt.PenStyle.DashLine),
                                 connect="pairs")
        self.mw_ends = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(T["muted"]), symbol="s")
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
        self.hover_pt = pg.ScatterPlotItem(size=28, pen=pg.mkPen(T["ink"], width=2), brush=None)
        for it in (self.mw_ends, self.tgt_pts, self.tokens_item, self.you_early, self.you_pts, self.pending_item,
                   self.cursor_item, self.hover_pt):
            vpi.addItem(it)
        self.you_text = TextPool(vpi, T["ink"], anchor=(-0.25, 1.1), size=13, n=16, bold=True)
        self.tgt_text = TextPool(vpi, T["ink2"], anchor=(1.25, -0.1), size=12, n=16)
        rv.addWidget(self.vc, 5)
        self.mw_info = QtWidgets.QLabel()
        self.mw_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.mw_info.setWordWrap(True)
        self.mw_info.setStyleSheet(f"color:{T['ink2']};")
        self.mw_info.setVisible(False)
        rv.addWidget(self.mw_info)

        self.table = QtWidgets.QTableWidget(0, 0)   # columns are set by _draw_table (one more with men → women)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        rv.addWidget(self.table, 2)

        rv.addWidget(self._legend([("dot", T["you"], "your vowels"), ("line", T["you"], "your running average"),
                                   ("line", T["target"], "target"), ("dash", T["ink"], "mouse position")]))
        self.vtl = self._plot("Apparent vocal-tract length from F1–F4 (lower = higher formants overall) · "
                              "click to jump there", "time", "cm", time_axis=True)
        tpi = self.vtl.getPlotItem()
        self.vtl_pts = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(_color(T["you"], 110)))
        tpi.addItem(self.vtl_pts)
        self.vtl_avg = tpi.plot(pen=pg.mkPen(T["you"], width=2))
        self.vtl_tgt = tpi.plot(pen=pg.mkPen(T["target"], width=2))
        self.vtl_view = pg.LinearRegionItem(values=(0, 1), movable=False, brush=pg.mkBrush(_color(T["ink2"], 28)),
                                            pen=pg.mkPen(_color(T["ink2"], 60)))
        self.vtl_line = pg.InfiniteLine(angle=90, movable=False,
                                        pen=pg.mkPen(T["ink"], width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.vtl_line.setVisible(False)
        for it in (self.vtl_view, self.vtl_line):
            tpi.addItem(it, ignoreBounds=True)
        self.vtl.scene().sigMouseClicked.connect(self._vtl_click)
        self.vtl.setMaximumHeight(220)
        rv.addWidget(self.vtl, 2)
        right.setMinimumWidth(560)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([900, 660])
        self._axes_changed()
        self._show_hover_hint()

    # ------------------------------------------------------------------------------------------------
    def _on_event(self, event, payload):
        if event == "words":
            for tok in payload["tokens"]:
                self.tok_at[int(round(tok["t"] * 1000))] = tok
                if tok["chart"]:
                    tok["outlier"] = not self.space.add(tok)
                    self.chart_tokens.append(tok)
            self.dirty_words = True
            self.version += 1
        elif event == "ceiling":
            # the speaker's formant ceiling was re-estimated: every point was re-expressed, so rebuild
            self.space = vowelspace.VowelSpace()
            for tok in self.history:
                self.space.add(tok)
            for tok in self.chart_tokens:
                tok["outlier"] = not self.space.add(tok)
            self.dirty_words = True
            self.vc_fit = True
            self.version += 1

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == QtCore.Qt.Key.Key_Space:
            self.toggle_pause()
        elif k == QtCore.Qt.Key.Key_End:
            self.go_live()
        elif k == QtCore.Qt.Key.Key_Escape:
            self.stop_audio()
        else:
            super().keyPressEvent(ev)

    def _pause_text(self):
        what = "processing" if self.review else ""
        self.pause_btn.setText(f"{'Resume' if self.paused else 'Pause'} {what} (space)".replace("  ", " "))

    def toggle_pause(self):
        self.paused = not self.paused
        self.source.set_paused(self.paused)
        if not self.paused and not self.review:
            self.stop_audio()   # live again: anything still playing would go into the microphone
            self.go_live()
        self._pause_text()

    def go_live(self):
        self.follow = True
        self.span = float(np.clip(self.view[1] - self.view[0], 2.0, 120.0))
        self._tl_key = None
        self.dirty_words = True   # the transcript goes back to the newest phrases

    def reset_space(self):
        self.space.reset()
        self.chart_tokens = []
        self.history = []
        self.dirty_words = True
        self.vc_fit = True

    def _note(self, text, secs=6.0):
        self.note = (text, time.time() + secs)

    def _target_changed(self):
        key = self.target_box.currentData()
        try:
            self.target = vowelspace.load_target(key) if key else None
        except Exception as e:
            self.target = None
            self._note(f"Could not load target: {e}")
        self.dirty_words = True
        self.vc_fit = True

    def _gender_changed(self):
        pair = self.gender_box.currentData()
        try:
            self.gender = tuple(vowelspace.load_target(k) for k in pair) if pair else None
        except Exception as e:
            self.gender = None
            self._note(f"Could not load the men/women references: {e}")
        self.mw_info.setVisible(self.gender is not None)
        self.dirty_words = True
        self.vc_fit = True

    def _scope_changed(self):
        self._scope_key = None
        self.dirty_words = True
        self.vc_fit = True

    def _axes_changed(self):
        xk, yk = AXES[self.axes_box.currentText()]
        f12 = (xk, yk) == ("F2", "F1")
        self.layout_box.setEnabled(f12)
        flip = f12 and self.layout_box.isChecked()
        pi = self.vc.getPlotItem()
        pi.invertX(flip)
        pi.invertY(flip)
        pi.setLabel("bottom", f"{xk} (Hz)" + ("  ·  higher to the LEFT" if flip else ""), color=self.T["ink2"])
        pi.setLabel("left", f"{yk} (Hz)" + ("  ·  higher DOWNWARD" if flip else ""), color=self.T["ink2"])
        self.vc_fit = True
        self.dirty_words = True
        self._hover_key = None

    def closeEvent(self, ev):
        self.timer.stop()
        self.stop_audio()
        if self.on_close:
            self.on_close()
        super().closeEvent(ev)

    # ------------------------------------------------------------------------------------------------
    def tick(self):
        try:
            self.source.pump(self.p)
            self.p.poll()
            self._draw_timeline()
            self._draw_hover()
            self._draw_play()
            self._draw_cursor()
            if self.scope_box.currentIndex() == 1:
                key = (round(self.view[0], 1), round(self.view[1], 1))
                if key != self._scope_key and time.time() - self._scope_drawn > 0.4:
                    self._scope_key = key
                    self.dirty_words = True
            if not self.follow:
                key = ("transcript", round(self.view[0]), round(self.view[1]))
                if key != getattr(self, "_tr_key", None):
                    self._tr_key = key
                    self._draw_transcript()
            # reviewing a file, words arrive every frame: redraw the heavy parts twice a second, not 30 times
            busy = self.review and not getattr(self.source, "done", True)
            if self.dirty_words and (not busy or time.time() - self._scope_drawn > 0.5):
                self.dirty_words = False
                self._scope_drawn = time.time()
                self._draw_transcript()
                self._draw_scores()
                self._draw_vowels()
            self.live_btn.setEnabled(not self.follow)
            self._draw_status()
        except Exception as e:  # never let one bad frame kill the window
            import traceback
            traceback.print_exc()
            self.status.setText(f"error: {type(e).__name__}: {e}")

    def _draw_status(self):
        T = self.T
        w = self.whisper
        warn = []
        if self.words_note:
            ws = self.words_note
        elif w is None:
            ws = "Whisper off"
        elif w.error:
            ws = "Whisper failed"
            warn.append(f"Whisper failed: {html.escape(w.error)}")
        elif not w.ready.is_set():
            ws = f"Whisper '{w.model_name}' loading..."
        else:
            lat = np.median(self.p.whisper_secs) if self.p.whisper_secs else None
            ws = f"Whisper '{w.model_name}' ready" + (f", {lat:.2f} s per phrase" if lat else "")
            if self.review:
                if w.pending():
                    ws += f", {w.pending()} phrases waiting"
            elif hasattr(w, "lag") and w.lag() > 5:
                warn.append(f"Whisper is {w.lag():.0f} s behind (older phrases get skipped so it stays live)")
            if getattr(w, "restarts", 0):
                ws += f", restarted {w.restarts}x"
        if self.p.clip_recent > 0.001 and not self.review:
            warn.append(f"input clipping ({100 * self.p.clip_recent:.1f}% of samples at full scale): turn the "
                        "mic / VoiceMeeter gain down, clipping distorts the formants")
        mode = self.p.cfg.ceiling
        ceil = f"formant ceiling {self.p.ceiling_now():.0f} Hz" + (" (fitted to your voice)" if mode == "speaker" else
                                                                     " (per vowel)" if mode == "auto" else "")
        hist = f" (+{self.history_n} from history)" if self.history_n else ""
        text = (f"{html.escape(self.source.describe())} · {clock(self.p.now)} · {html.escape(ws)} · {ceil} · "
                f"{len(self.p.syllables)} syllables, {len(self.chart_tokens)} vowel points{hist}"
                + (" · PAUSED" if self.paused else ""))
        if self.note[0] and time.time() < self.note[1]:
            text += f"<br><span style='color:{T['ink']}'>{html.escape(self.note[0])}</span>"
        if warn:
            text += "".join(f"<br><span style='color:{T['critical']}'>⚠ {m}</span>" for m in warn)
        if text != getattr(self, "_status_text", None):
            self._status_text = text
            self.status.setText(text)

    def _pack(self, pool, items, sec_per_px, rows):
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

    # ---- timeline -----------------------------------------------------------------------------------
    def _tl_vb(self):
        return self.tl.getPlotItem().getViewBox()

    def _tl_range(self, now):
        if self.follow:
            t1 = max(now, self.span)
            return t1 - self.span, t1
        return tuple(self._tl_vb().viewRange()[0])

    def _tl_dragged(self, *_):
        """You dragged or zoomed the timeline: stop following, unless you brought it back to the newest
        audio (like scrolling a terminal back to the bottom)."""
        t0, t1 = self._tl_vb().viewRange()[0]
        was = self.follow
        self.follow = t1 >= self.p.now - 0.25
        if self.follow:
            self.span = float(np.clip(t1 - t0, 2.0, 120.0))
        self._tl_key = None
        if self.follow != was:
            self.dirty_words = True

    def jump_to(self, t, span=None):
        """Shows time t in the middle of the timeline (stops following)."""
        span = span or max(self.view[1] - self.view[0], 2.0)
        self.follow = False
        self._tl_vb().setXRange(t - span / 2, t + span / 2, padding=0)
        self._tl_key = None

    def _draw_timeline(self):
        p, d = self.p, self.p.det
        now = p.now
        self.trace.pull(d)
        for s in p.syllables[len(self._syl_t):]:
            self._syl_t.append(s.t_peak)
        for w in p.words[len(self._word_t):]:
            self._word_t.append(w.start)
        if d.ref is not None:
            thr = d.ref - d.p.silence_rel_db
            lo, hi = max(d.floor - 6, thr - 12), d.ref + 6
            if self.tl_yr is None or abs(self.tl_yr[0] - lo) > 4 or abs(self.tl_yr[1] - hi) > 4:
                self.tl_yr = (lo, hi)
        if self.tl_yr is None:
            return
        t0, t1 = self._tl_range(now)
        self.view = (t0, t1)
        live_edge = t1 > now - 2.0
        key = (round(t0, 3), round(t1, 3), self.version, self.tl_yr,
               (len(p.syllables), self.trace.n) if live_edge else None)
        if key == self._tl_key:
            return   # a stretch from the past with nothing new in it: no need to redraw
        self._tl_key = key
        lo, hi = self.tl_yr
        tier = 0.18 * (hi - lo)
        y_syl, y_word = lo - 0.5 * tier, (hi + 0.35 * tier, hi + 0.8 * tier)
        self.tl.setYRange(lo - tier, hi + tier * 1.1, padding=0)
        if self.follow:
            self.tl.setXRange(t0, t1, padding=0)
        width_px = max(self._tl_vb().width(), 100.0)
        sec_per_px = (t1 - t0) / width_px
        tt, lv, th = self.trace.window(t0 - 0.2, t1 + 0.2, max_points=int(2 * width_px))
        self.env.setData(tt, lv)
        self.thr.setData(tt, th)

        i0, i1 = bisect.bisect_left(self._syl_t, t0 - 1.0), bisect.bisect_right(self._syl_t, t1 + 1.0)
        sy = [x for x in p.syllables[i0:i1] if x.status != "quiet"]
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
        self._pack(self.syl_text, labels, sec_per_px, [y_syl + 0.33 * tier, y_syl - 0.33 * tier])

        groups = {"pending": ([], []), "matched": ([], []), "extra": ([], [])}
        for x in sy:
            g = groups["matched" if x.status == "matched" else "extra" if x.status == "extra" else "pending"]
            g[0].append(x.t_peak)
            g[1].append(x.peak_db)
        self.nuc_pending.setData(*groups["pending"])
        self.nuc_match.setData(*groups["matched"])
        self.nuc_extra.setData(*groups["extra"])

        hit, miss, wl = [], [], []
        j0, j1 = bisect.bisect_left(self._word_t, t0 - 30.0), bisect.bisect_right(self._word_t, t1 + 0.5)
        for w in p.words[j0:j1]:
            if w.end < t0:
                continue
            ok = w.onset_err is not None and abs(w.onset_err) <= p.cfg.align.onset_tol_s
            judged = w.scored and not w.phrase_initial   # Whisper pins a phrase's first word to the clip start
            wl.append((w.start, _short(w.word) + ("" if not judged else (" ✓" if ok else " ✗")), False))
            if judged:
                (hit if ok else miss).extend([(w.start, hi + tier), (w.start, lo - tier)])
        self._pack(self.word_text, wl, sec_per_px, list(y_word))
        for item, pts in ((self.tick_hit, hit), (self.tick_miss, miss)):
            a = np.array(pts) if pts else np.zeros((0, 2))
            item.setData(a[:, 0], a[:, 1])
        self.vtl_view.setRegion((t0, t1))

    # ---- hover, click, playback ------------------------------------------------------------------------
    def _tl_mouse(self, pos):
        self.hover_scene = pos
        self._draw_hover()

    def _syllable_at(self, t):
        """The syllable whose span contains t, else the nearest nucleus within 0.25 s."""
        i = bisect.bisect_right(self._syl_t, t + 0.6)
        best = None
        for s in reversed(self.p.syllables[max(0, i - 10): i]):
            if s.status == "quiet":
                continue
            end = s.t_end if s.t_end is not None else s.t_peak + 0.08
            if s.t_onset <= t <= end:
                return s
            dist = abs(s.t_peak - t)
            if dist <= 0.25 and (best is None or dist < abs(best.t_peak - t)):
                best = s
        return best

    def _word_at(self, t):
        j = bisect.bisect_right(self._word_t, t)
        for w in reversed(self.p.words[max(0, j - 6): j]):
            if w.start <= t <= w.end:
                return w
        return None

    def _mouse_time(self):
        """Seconds under the mouse if it is over the timeline's plot area, else None."""
        pos, vb = self.hover_scene, self._tl_vb()
        if pos is None or not self.tl.underMouse() or not vb.sceneBoundingRect().contains(pos):
            return None
        return float(vb.mapSceneToView(pos).x())

    def _show_hover_hint(self):
        self.hover_info.setText(f"<span style='color:{self.T['muted']}'>Point at the timeline to see what was said "
                                "there and where that syllable sits on the vowel chart. Click a word to hear it"
                                + (" (pause first)." if not self.review else ".") +
                                " Drag or scroll the timeline to go back; Live (End) follows again.</span>")

    def _draw_hover(self):
        t = self._mouse_time()
        if t is None:
            if self._hover_key is not None:
                self._hover_key = None
                for it in (self.hover_line, self.vtl_line):
                    it.setVisible(False)
                self.hover_pt.setData([], [])
                self._show_hover_hint()
            return
        t = min(max(t, 0.0), self.p.now)
        key = (round(t, 3), self.version, self.axes_box.currentText())
        if key == self._hover_key:
            return
        self._hover_key = key
        T = self.T
        self.hover_line.setPos(t)
        self.hover_line.setVisible(True)
        self.vtl_line.setPos(t)
        self.vtl_line.setVisible(True)
        syl, word = self._syllable_at(t), self._word_at(t)
        parts = [f"<b>{clock(t, 2)}</b>"]
        if word is not None:
            if word.pron:
                ipa = "·".join(html.escape(x.ipa) for x in word.pron.syllables)
                ok = (f" <span style='color:{T['good_text']}'>✓</span>" if word.count_ok else
                      f" <span style='color:{T['critical']}'>✗ {word.n_det}/{word.n_exp} syllables</span>")
                parts.append(f"word <b>{html.escape(_short(word.word, 24))}</b> /{ipa}/{ok}")
            else:
                parts.append(f"word <b>{html.escape(_short(word.word, 24))}</b> (not in the dictionary)")
        elif self.p.words:
            parts.append(f"<span style='color:{T['muted']}'>no word here</span>")
        xy = None
        if syl is not None:
            F = syl.formants or [0, 0, 0, 0]
            lab = f"/{html.escape(syl.label)}/" if syl.label else STATUS_TEXT.get(syl.status, syl.status)
            line = f"syllable {lab}"
            if F[0] > 0:
                line += f": F1 {F[0]:.0f} · F2 {F[1]:.0f} · F3 {F[2]:.0f} · F4 {F[3]:.0f} Hz"
                xk, yk = AXES[self.axes_box.currentText()]
                i = {"F1": 0, "F2": 1, "F3": 2, "F4": 3}
                if F[i[xk]] > 0 and F[i[yk]] > 0:
                    xy = (F[i[xk]], F[i[yk]])
            if syl.f0 > 0:
                line += f" · pitch {syl.f0:.0f} Hz"
            parts.append(line)
            tok = self.tok_at.get(int(round(syl.t_peak * 1000)))
            if tok is not None:
                if tok["chart"] and not tok.get("outlier"):
                    parts.append(f"on the chart as <b>{html.escape(tok['ipa'])}</b> (ringed)")
                elif tok["chart"]:
                    parts.append(f"<span style='color:{T['muted']}'>left off the chart: far from your other "
                                 f"{html.escape(tok['ipa'])} tokens (probably a formant error)</span>")
                else:
                    parts.append(f"<span style='color:{T['muted']}'>not on the chart: "
                                 f"{html.escape(tok.get('reason') or '')}</span>")
        if xy:
            self.hover_pt.setData([xy[0]], [xy[1]])
        else:
            self.hover_pt.setData([], [])
        hint ="click: listen" if self._can_play() else "pause (space) to listen"
        self.hover_info.setText(" &nbsp;·&nbsp; ".join(parts) +
                                f" &nbsp;<span style='color:{T['muted']}'>({hint})</span>")

    def _can_play(self):
        if self.review or self.paused:
            return True
        return isinstance(self.source, FileSource) and not self.source.play

    def _tl_click(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        vb = self._tl_vb()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        t = float(vb.mapSceneToView(ev.scenePos()).x())
        w = self._word_at(t)
        if w is not None:
            a, b = w.start, w.end
        else:
            s = self._syllable_at(t)
            a, b = (s.t_onset, s.t_end if s.t_end is not None else s.t_peak + 0.1) if s is not None else (t, t + 1.0)
        self.play_range(a - 0.05, b + 0.05)

    def _vtl_click(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        vb = self.vtl.getPlotItem().getViewBox()
        if vb.sceneBoundingRect().contains(ev.scenePos()):
            self.jump_to(float(vb.mapSceneToView(ev.scenePos()).x()))

    def _transcript_link(self, url):
        try:
            self.jump_to(float(url.toString().split(":", 1)[1]))
        except (ValueError, IndexError):
            pass

    def _audio(self, a, b):
        """(samples, sample rate) of the session between a and b seconds."""
        get = getattr(self.source, "audio", None)
        if get is not None:
            return get(a, b)
        st = self.p.audio
        if a * SR >= st.start:
            return st.get(a, b), SR
        path = getattr(self.p, "wav_path", None)   # older than what is kept in memory: from the session file
        if path and os.path.exists(path):
            import soundfile as sf
            y, sr = sf.read(path, start=int(a * SR), stop=int(b * SR), dtype="float32", always_2d=True)
            return y[:, 0], sr
        return st.get(a, b), SR

    def play_range(self, a, b):
        if not self._can_play():
            self._note("Pause first (space) to listen: while live, the sound would go back into the microphone "
                       "and into your session.")
            return
        a, b = max(0.0, a), min(self.p.now, b)
        if b - a < 0.02:
            return
        try:
            import sounddevice as sd
            y, sr = self._audio(a, b)
            sd.stop()
            sd.play(np.clip(np.asarray(y, np.float32), -1, 1), int(sr))
            self.play = (a, a + len(y) / sr, time.perf_counter())
            self.play_btn.setText("■ Stop")
        except Exception as e:
            self._note(f"Could not play: {type(e).__name__}: {e}")

    def play_view(self):
        if self.play is not None:
            self.stop_audio()
            return
        t0, t1 = self.view
        self.play_range(max(t0, 0.0), min(t1, t0 + PLAY_MAX_S))

    def stop_audio(self):
        if self.play is not None:
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
        self.play = None
        self.play_line.setVisible(False)
        self.play_btn.setText("▶ Play view")

    def _draw_play(self):
        if self.play is None:
            return
        a, b, started = self.play
        t = a + time.perf_counter() - started
        if t > b:
            self.stop_audio()
            return
        self.play_line.setPos(t)
        self.play_line.setVisible(True)

    def open_other(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a recording, or any file in a saved session's folder", os.path.dirname(self.p.save_dir or "")
            or os.getcwd(), "Audio or session files (*.wav *.flac *.mp3 *.m4a *.ogg *.json *.csv);;All files (*)")
        if path:
            import subprocess
            subprocess.Popen(list(self.open_cmd) + ["--open", path, "--theme", self.theme])
            self._note(f"Opening {os.path.basename(path)} in a new window…")

    # ---- vowel chart ------------------------------------------------------------------------------------
    def _xy(self, d):
        xk, yk = AXES[self.axes_box.currentText()]
        return d.get(xk, 0.0), d.get(yk, 0.0)

    def _draw_cursor(self):
        xk, yk = AXES[self.axes_box.currentText()]
        idx = {"F1": 1, "F2": 2, "F3": 3, "F4": 4}
        now = self.p.now
        cur = [c for c in self.p.cursor if now - c[0] < 0.6 and c[idx[xk]] > 0 and c[idx[yk]] > 0]
        if cur:
            xs = [c[idx[xk]] for c in cur]
            ys = [c[idx[yk]] for c in cur]
            self.trail.setData(xs, ys)
            self.cursor_item.setData([xs[-1]], [ys[-1]])
            self._keep_in_view(xs[-1], ys[-1], xk, yk)
        else:
            self.trail.setData([], [])
            self.cursor_item.setData([], [])
        pend = [s for s in self.p.syllables[-12:] if s.status == "pending" and s.formants
                and now - s.t_peak < 4.0 and s.formants[idx[xk] - 1] > 0 and s.formants[idx[yk] - 1] > 0]
        self.pending_item.setData([s.formants[idx[xk] - 1] for s in pend], [s.formants[idx[yk] - 1] for s in pend])

    def _keep_in_view(self, x, y, xk, yk):
        """Grows the chart when the live cursor stays outside it for 3 updates (~0.1 s), so an extreme
        vowel you are practising stays visible; a single stray frame or an implausible value does not."""
        (lx, hx), (ly, hy) = PLAUSIBLE[xk], PLAUSIBLE[yk]
        (x0, x1), (y0, y1) = self.vc.getPlotItem().getViewBox().viewRange()
        if not (lx <= x <= hx and ly <= y <= hy) or (x0 <= x <= x1 and y0 <= y <= y1):
            self.cursor_out = 0
            return
        self.cursor_out += 1
        if self.cursor_out < 3:
            return
        self.cursor_out = 0
        px, py = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
        nx0, nx1, ny0, ny1 = min(x0, x - px), max(x1, x + px), min(y0, y - py), max(y1, y + py)
        self.vc.setXRange(nx0, nx1, padding=0)
        self.vc.setYRange(ny0, ny1, padding=0)
        last = getattr(self, "vc_last_fit", None)
        if last:
            self.vc_last_fit = (min(last[0], nx0), max(last[1], nx1), min(last[2], ny0), max(last[3], ny1))

    def _outline(self, item, pts_by_vowel):
        xk, yk = AXES[self.axes_box.currentText()]
        poly = [pts_by_vowel[v] for v in CORNERS if v in pts_by_vowel]
        if xk == "F2" and yk == "F1" and len(poly) >= 3:
            poly.append(poly[0])
            item.setData([a[0] for a in poly], [a[1] for a in poly])
        else:
            item.setData([], [])

    def _scope(self):
        """(vowel means, [(token, kept)] to draw, label) for what the chart should show now."""
        if self.scope_box.currentIndex() == 0:
            return self.space.means(), [(t, not t.get("outlier")) for t in self.chart_tokens[-200:]], ""
        t0, t1 = self.view
        sp = vowelspace.VowelSpace()
        toks = [(t, sp.add(t)) for t in self.chart_tokens if t0 <= t["t"] <= t1]
        return sp.means(), toks[-400:], f" ({clock(t0, 0)} – {clock(t1, 0)})"

    def _draw_vowels(self):
        T = self.T
        xk, yk = AXES[self.axes_box.currentText()]
        means, toks, scope_label = self._scope()
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
        spots, kept = [], [t for t, ok in toks if ok]
        for i, t in enumerate(kept):
            x, y = self._xy(t)
            if x > 0 and y > 0:
                spots.append({"pos": (x, y), "brush": pg.mkBrush(_color(T["you"], 40 + 170 * (i + 1) / max(len(kept), 1))),
                              "data": t})
        self.tokens_item.setData(spots)
        mw_pts = self._draw_gender(means)
        xs = [s["pos"][0] for s in spots] + [a[0] for a in tgt.values()] + [a[0] for a in drawn.values()] + mw_pts[0]
        ys = [s["pos"][1] for s in spots] + [a[1] for a in tgt.values()] + [a[1] for a in drawn.values()] + mw_pts[1]
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
        self.vc.setTitle(f"Your vowel space{scope_label}  vs  {self.target['name'] if self.target else 'no target'}",
                         color=T["ink"], size="10pt")
        self._draw_table(means, tv)
        self._draw_vtl(tv)

    def _draw_gender(self, means):
        """men -> women lines and your place along them. Returns the points drawn (for the chart's range)."""
        if self.gender is None:
            self.mw_lines.setData([], [])
            self.mw_ends.setData([], [])
            self.gender_pos = None
            return [], []
        men, women = (g["vowels"] for g in self.gender)
        segs = []
        for v in pronounce.CHART_VOWELS:
            if v in men and v in women:
                a, b = self._xy(men[v]), self._xy(women[v])
                if min(a) > 0 and min(b) > 0:
                    segs += [a, b]
        a = np.array(segs) if segs else np.zeros((0, 2))
        self.mw_lines.setData(a[:, 0], a[:, 1])
        self.mw_ends.setData(a[1::2, 0], a[1::2, 1])
        g = vowelspace.gender_position(means, men, women, min_n=vowelspace.SETTLED_N)
        self.gender_pos = g
        T = self.T
        if g["n_vowels"] == 0:
            self.mw_info.setText(f"<span style='color:{T['muted']}'>Dashed: men's → women's average per vowel "
                                 f"(square = women's end). Your place on them shows once a vowel has "
                                 f"{vowelspace.SETTLED_N} tokens.</span>")
        else:
            self.mw_info.setText(
                f"Your F1–F3 sit <b>{100 * g['position']:.0f}%</b> of the way from the men's to the women's averages "
                f"(0% men, 100% women; {g['n_vowels']} vowels), <b>{100 * g['off']:.0f}%</b> to the side of those "
                f"lines (typical speakers: 5–12%). <span style='color:{T['muted']}'>Formants only, and single "
                "speakers spread widely (hover the menu); pitch and voice quality count too.</span>")
        return list(a[:, 0]), list(a[:, 1])

    def _draw_table(self, means, tv):
        rows = [v for v in pronounce.CHART_VOWELS if v in means or v in tv]
        g = getattr(self, "gender_pos", None)
        cols = ["vowel", "n", "your F1", "target F1", "your F2", "target F2", "your F3", "target F3"]
        if g is not None:
            cols.append("m→w")
        if self.table.columnCount() != len(cols):
            self.table.setColumnCount(len(cols))
            self.table.setHorizontalHeaderLabels(cols)
            hdr = self.table.horizontalHeader()
            hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
            hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
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
            if g is not None:
                pv = g["per_vowel"].get(v)
                cell(r, 8, f"{100 * pv:.0f}%" if pv is not None else "–", few)

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

    def save_as_target(self):
        means, toks, scope_label = self._scope()
        if self.scope_box.currentIndex() == 0:
            use = [t for t in self.history + self.chart_tokens if not t.get("outlier")]
        else:
            use = [t for t, ok in toks if ok]
        if len({t["vowel"] for t in use}) < 3:
            self._note("Not enough vowels yet to save as a target (needs at least 3 different vowels).")
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Save my vowels as a target", "Name for this target:",
                                                  text=f"Me {datetime.now():%Y-%m-%d %H:%M}{scope_label}")
        if not ok or not name.strip():
            return
        tgt = vowelspace.target_from_tokens(use, name.strip(), source=f"own vowels saved from the live window "
                                            f"({len(use)} tokens{scope_label})")
        fn = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name.strip()).strip().replace(" ", "_")
        path = vowelspace.save_target(tgt, os.path.join(vowelspace.USER_TARGET_DIR, (fn or "target") + ".json"))
        if self.target_box.findData(path) < 0:
            self.target_box.addItem(name.strip(), path)
        self._note(f"Saved as target '{name.strip()}' ({path}). Pick it next time to compare with today.", 10)

    # ---- transcript and scores -------------------------------------------------------------------------
    def _draw_transcript(self):
        T = self.T
        out = []
        notes = {"exercise": "held vowel — shown live on the vowel chart, no words to transcribe",
                 "behind": "skipped: Whisper was behind, so it jumped to your newest phrase",
                 "timeout": "skipped: Whisper got stuck on this one and was restarted"}
        shown = [u for u in self.p.utts if u["status"] in ("done", "sent", "error") or u["status"] in notes]
        if self.follow:
            shown = shown[-10:]
        else:   # scrolled back: the phrases in view
            t0, t1 = self.view
            shown = [u for u in shown if u["t1"] >= t0 and u["t0"] <= t1][-25:]
        for u in shown:
            when = f"<a href='t:{(u['t0'] + u['t1']) / 2:.2f}' style='color:{T['muted']}'>{clock(u['t0'])}</a>"
            if u["status"] == "sent":
                out.append(f"<p style='color:{T['muted']}'>… transcribing {clock(u['t0'])}–{clock(u['t1'])}</p>")
                continue
            if u["status"] in notes:
                out.append(f"<p style='color:{T['muted']}'>{when} &nbsp;({notes[u['status']]})</p>")
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
            if not parts:
                parts.append(f"<span style='color:{T['muted']}'>(no words)</span>")
            out.append(f"<p>{when} &nbsp;" + " &nbsp; ".join(parts) + "</p>")
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
                ("  after a short pause", "onset_hit_after_pause", "%"),
                ("  in running speech", "onset_hit_connected", "%"),
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

    # ---- hover tips ---------------------------------------------------------------------------------------
    def _tok_tip(self, x, y, data):
        t = data
        return (f"{t['word']}  /{t['syllable']}/  vowel {t['ipa']}\nF1 {t['F1']:.0f}  F2 {t['F2']:.0f}  "
                f"F3 {t['F3']:.0f}  F4 {t['F4']:.0f} Hz\nat {clock(t['t'], 2)}, pitch {t['f0']:.0f} Hz")


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

    def audio(self, t0, t1):
        return self.y[max(0, int(t0 * self.sr)): max(0, int(t1 * self.sr))], self.sr

    def describe(self):
        state = "finished" if self.done else f"{self.pos / self.sr:.1f}/{len(self.y) / self.sr:.1f} s"
        return f"file: {self.name} ({state})"


class FastFileSource:
    """A recording (or a saved session's audio) run through the pipeline as fast as the computer allows,
    for reviewing it: the timeline fills in while you already look around. Works at most `budget_s` per
    window frame so the window stays responsive, and waits while Whisper has more than `max_waiting`
    phrases queued so the words keep up with the timeline. Long files are streamed from disk."""

    def __init__(self, path, name="", whisper=None, budget_s=0.03, max_waiting=6):
        import soundfile as sf
        self.path, self.name, self.whisper = path, name or os.path.basename(path), whisper
        self.budget_s, self.max_waiting = budget_s, max_waiting
        try:
            self.f = sf.SoundFile(path)
            self.sr, self.n, self.y = int(self.f.samplerate), int(self.f.frames), None
        except Exception:   # mp3/m4a etc.: decode it whole
            import librosa
            y, sr = librosa.load(path, sr=None, mono=True)
            self.f, self.y, self.sr, self.n = None, y.astype(np.float32), int(sr), len(y)
        self.pos, self.paused, self.done = 0, False, False
        self.t_start, self.work_s = None, 0.0

    def _read(self, a, b):
        a, b = max(0, int(a)), min(self.n, int(b))
        if b <= a:
            return np.zeros(0, np.float32)
        if self.y is not None:
            return self.y[a:b]
        self.f.seek(a)
        return self.f.read(b - a, dtype="float32", always_2d=True).mean(axis=1)

    def audio(self, t0, t1):
        return self._read(t0 * self.sr, t1 * self.sr), self.sr

    def start(self):
        self.t_start = time.perf_counter()
        return self

    def stop(self):
        if self.f is not None:
            try:
                self.f.close()
            except Exception:
                pass

    def set_paused(self, v):
        self.paused = v

    def waiting(self):
        w = self.whisper
        return w is not None and hasattr(w, "pending") and w.pending() > self.max_waiting

    def pump(self, pipeline):
        if self.paused or self.done:
            return
        t = time.perf_counter()
        step = max(1, int(self.sr * 0.25))
        while self.pos < self.n and time.perf_counter() - t < self.budget_s and not self.waiting():
            x = self._read(self.pos, self.pos + step)
            pipeline.feed(x)
            self.pos += len(x)
        self.work_s += time.perf_counter() - t
        if self.pos >= self.n:
            self.done = True
            pipeline.finish(wait=False)

    def describe(self):
        dur = self.n / self.sr
        if self.done:
            return f"{self.name}: all {clock(dur, 0)} processed"
        speed = (self.pos / self.sr) / max(time.perf_counter() - self.t_start, 1e-3) if self.t_start else 0
        return (f"{self.name}: {clock(self.pos / self.sr, 0)} of {clock(dur, 0)} processed ({speed:.0f}x real time"
                + (", waiting for Whisper" if self.waiting() else "") + ")")
