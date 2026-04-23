#!/usr/bin/env python3
"""Interactive GUI tool to extract y-values from benchmark graph images.

Workflow:
    1. Pass an image path on the CLI. The image is shown as the window
       background, letterboxed with aspect ratio preserved, rescaling on
       window resize. Scroll to zoom, middle-drag or Space+left-drag to pan.
    2. Click two points to calibrate the y-axis. Both points share the
       x-coordinate of the first click. Enter a y-value for each (e.g.
       `10^-2`, `1e-2`, `0.01`).
    3. After calibration the tool draws a labelled y-axis (linear or log)
       to the left of the calibration x.
    4. Hover for a live y-value readout. Click to add a measurement; an
       inline text field appears for the label. Enter commits, Esc cancels.
    5. Toggle linear/log, reset calibration, delete individual measurements,
       and save a CSV of (label, y_value, y_value_scientific).

Usage:
    uv run kld_extract.py path/to/image.png
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

AXIS_COLOUR = QColor(220, 60, 60)
MEASUREMENT_COLOUR = QColor(30, 120, 220)
HOVER_COLOUR = QColor(40, 40, 40, 200)
MIN_SCALE = 1e-2
MAX_SCALE = 200.0
ZOOM_STEP = 1.15


def parse_y_value(raw: str) -> float:
    """Parse strings like `10^-2`, `1e-2`, `0.01`, `2.5 x 10^-3`, `-3.14`."""
    s = raw.strip().replace(" ", "")
    if not s:
        raise ValueError("empty value")

    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(?:[x\*×])?10\^([+-]?\d+)", s)
    if m:
        return float(m.group(1)) * (10.0 ** int(m.group(2)))

    m = re.fullmatch(r"10\^([+-]?\d+)", s)
    if m:
        return 10.0 ** int(m.group(1))

    return float(s)


def format_sci(value: float, sig: int = 3) -> str:
    """Format as short scientific, e.g. `1.2e-2`, trimming trailing zeros."""
    if value == 0:
        return "0"
    if not math.isfinite(value):
        return str(value)
    formatted = f"{value:.{sig - 1}e}"
    mantissa, exp = formatted.split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    exp_int = int(exp)
    return f"{mantissa}e{exp_int:+d}"


def nice_linear_ticks(lo: float, hi: float, target: int = 8) -> list[float]:
    """Generate 'nice' (1/2/5 × 10^n) tick values covering [lo, hi] slightly extended."""
    if lo > hi:
        lo, hi = hi, lo
    span = hi - lo
    if span <= 0:
        return [lo]
    rough_step = span / max(target - 1, 1)
    exp = math.floor(math.log10(rough_step))
    base = 10**exp
    step = base
    for mult in (1, 2, 5, 10):
        step = mult * base
        if span / step <= target:
            break
    pad = step
    start = math.floor((lo - pad) / step) * step
    end = math.ceil((hi + pad) / step) * step
    ticks = []
    v = start
    while v <= end + step * 0.5:
        ticks.append(round(v / step) * step)
        v += step
    return ticks


class ImageCanvas(QWidget):
    """Image viewport with zoom/pan. Stores calibration and measurements in image pixel coords."""

    point_clicked = Signal(QPointF)  # emits image-coord position
    hover_moved = Signal(QPointF)  # emits image-coord position
    hover_left = Signal()

    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)

        # View transform: widget_pt = image_pt * scale + offset
        self.scale = 1.0
        self.offset = QPointF(0.0, 0.0)
        self._has_initial_fit = False

        # All stored in image pixel coords.
        self.calib_points: list[QPointF] = []
        self.calib_values: list[float] = []
        self.measurements: list[dict] = []
        self.axis_mode = "log"
        self.snap_x_img: float | None = None

        # Interaction state.
        self.hover_widget_pos: QPointF | None = None
        self.hover_y: float | None = None
        self._panning = False
        self._pan_last: QPointF | None = None
        self._space_held = False

    # ---------- view transform ----------

    def img_to_widget(self, p: QPointF) -> QPointF:
        return QPointF(p.x() * self.scale + self.offset.x(), p.y() * self.scale + self.offset.y())

    def widget_to_img(self, p: QPointF) -> QPointF:
        return QPointF(
            (p.x() - self.offset.x()) / self.scale, (p.y() - self.offset.y()) / self.scale
        )

    def image_widget_rect(self) -> QRectF:
        top_left = self.img_to_widget(QPointF(0, 0))
        bottom_right = self.img_to_widget(QPointF(self.pixmap.width(), self.pixmap.height()))
        return QRectF(top_left, bottom_right)

    def fit_to_window(self) -> None:
        if self.pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        pw, ph = self.pixmap.width(), self.pixmap.height()
        s = min(self.width() / pw, self.height() / ph)
        self.scale = s
        self.offset = QPointF((self.width() - pw * s) / 2, (self.height() - ph * s) / 2)
        self._has_initial_fit = True
        self.update()

    def zoom_at(self, widget_pos: QPointF, factor: float) -> None:
        new_scale = max(MIN_SCALE, min(MAX_SCALE, self.scale * factor))
        if new_scale == self.scale:
            return
        img_pt = self.widget_to_img(widget_pos)
        self.scale = new_scale
        # keep img_pt pinned under cursor
        self.offset = QPointF(
            widget_pos.x() - img_pt.x() * new_scale,
            widget_pos.y() - img_pt.y() * new_scale,
        )
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        if not self._has_initial_fit:
            self.fit_to_window()
        super().resizeEvent(event)

    # ---------- configuration setters ----------

    def set_axis_mode(self, mode: str) -> None:
        self.axis_mode = mode
        self.update()

    def set_calibration(self, points: list[QPointF], values: list[float]) -> None:
        self.calib_points = list(points)
        self.calib_values = list(values)
        self.update()

    def set_measurements(self, measurements: list[dict]) -> None:
        self.measurements = measurements
        self.update()

    def set_snap_x(self, x_img: float | None) -> None:
        self.snap_x_img = x_img
        self.update()

    # ---------- axis math (operates on image-y for calibration points) ----------

    def pixel_to_y(self, img_y: float) -> float | None:
        if len(self.calib_points) != 2 or len(self.calib_values) != 2:
            return None
        p1, p2 = self.calib_points
        v1, v2 = self.calib_values
        if p1.y() == p2.y():
            return None
        t = (img_y - p1.y()) / (p2.y() - p1.y())
        if self.axis_mode == "log":
            if v1 <= 0 or v2 <= 0:
                return None
            log_y = math.log10(v1) + t * (math.log10(v2) - math.log10(v1))
            return 10**log_y
        return v1 + t * (v2 - v1)

    def y_to_pixel(self, y: float) -> float | None:
        if len(self.calib_points) != 2 or len(self.calib_values) != 2:
            return None
        p1, p2 = self.calib_points
        v1, v2 = self.calib_values
        if self.axis_mode == "log":
            if v1 <= 0 or v2 <= 0 or y <= 0:
                return None
            lv1, lv2, ly = math.log10(v1), math.log10(v2), math.log10(y)
            if lv1 == lv2:
                return None
            t = (ly - lv1) / (lv2 - lv1)
        else:
            if v1 == v2:
                return None
            t = (y - v1) / (v2 - v1)
        return p1.y() + t * (p2.y() - p1.y())

    # ---------- input ----------

    def _pan_button(self, event: QMouseEvent) -> bool:
        return event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and self._space_held
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._pan_button(event):
            self._panning = True
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        pos = event.position()
        img_pos = self.widget_to_img(pos)
        if not self._point_in_image(img_pos):
            return
        if self.snap_x_img is not None:
            img_pos = QPointF(self.snap_x_img, img_pos.y())
        self.point_clicked.emit(img_pos)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning and (
            event.button() == Qt.MiddleButton or event.button() == Qt.LeftButton
        ):
            self._panning = False
            self._pan_last = None
            self.setCursor(Qt.OpenHandCursor if self._space_held else Qt.ArrowCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._panning and self._pan_last is not None:
            delta = pos - self._pan_last
            self._pan_last = pos
            self.offset = QPointF(self.offset.x() + delta.x(), self.offset.y() + delta.y())
            self.update()
            return

        img_pos = self.widget_to_img(pos)
        if self._point_in_image(img_pos):
            self.hover_widget_pos = pos
            self.hover_y = self.pixel_to_y(img_pos.y())
            self.hover_moved.emit(img_pos)
        else:
            self.hover_widget_pos = None
            self.hover_y = None
            self.hover_left.emit()
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and not self._space_held:
            self.fit_to_window()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = ZOOM_STEP ** (delta / 120.0)
        self.zoom_at(event.position(), factor)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            if not self._panning:
                self.setCursor(Qt.OpenHandCursor)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            if not self._panning:
                self.setCursor(Qt.ArrowCursor)
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self.hover_widget_pos = None
        self.hover_y = None
        self.hover_left.emit()
        self.update()
        super().leaveEvent(event)

    def _point_in_image(self, img_pt: QPointF) -> bool:
        return 0 <= img_pt.x() <= self.pixmap.width() and 0 <= img_pt.y() <= self.pixmap.height()

    # ---------- painting ----------

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        target = self.image_widget_rect()
        if not self.pixmap.isNull():
            painter.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))

        self._draw_snap_guide(painter, target)
        self._draw_calibration_points(painter)
        self._draw_axis(painter, target)
        self._draw_measurements(painter)
        self._draw_hover(painter)
        painter.end()

    def _draw_snap_guide(self, painter: QPainter, target: QRectF) -> None:
        if self.snap_x_img is None:
            return
        x_w = self.img_to_widget(QPointF(self.snap_x_img, 0)).x()
        pen = QPen(AXIS_COLOUR, 1, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(x_w, target.top()), QPointF(x_w, target.bottom()))

    def _draw_calibration_points(self, painter: QPainter) -> None:
        pen = QPen(AXIS_COLOUR, 2)
        painter.setPen(pen)
        painter.setBrush(AXIS_COLOUR)
        for p in self.calib_points:
            wp = self.img_to_widget(p)
            painter.drawEllipse(wp, 5, 5)

    def _draw_axis(self, painter: QPainter, target: QRectF) -> None:
        if len(self.calib_points) != 2 or len(self.calib_values) != 2:
            return
        p1, p2 = self.calib_points
        v1, v2 = self.calib_values
        x_w = self.img_to_widget(QPointF(p1.x(), 0)).x()

        pen = QPen(AXIS_COLOUR, 2)
        painter.setPen(pen)
        painter.drawLine(QPointF(x_w, target.top()), QPointF(x_w, target.bottom()))

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        fm = painter.fontMetrics()

        minor_ticks = self._compute_minor_ticks(v1, v2) if self.axis_mode == "log" else []
        pen_minor = QPen(AXIS_COLOUR, 1)
        painter.setPen(pen_minor)
        for v in minor_ticks:
            py_img = self.y_to_pixel(v)
            if py_img is None:
                continue
            y_w = self.img_to_widget(QPointF(0, py_img)).y()
            if y_w < target.top() or y_w > target.bottom():
                continue
            painter.drawLine(QPointF(x_w - 4, y_w), QPointF(x_w, y_w))

        major_ticks = self._compute_major_ticks(v1, v2)
        pen_major = QPen(AXIS_COLOUR, 2)
        painter.setPen(pen_major)
        for v, label in major_ticks:
            py_img = self.y_to_pixel(v)
            if py_img is None:
                continue
            y_w = self.img_to_widget(QPointF(0, py_img)).y()
            if y_w < target.top() or y_w > target.bottom():
                continue
            painter.drawLine(QPointF(x_w - 8, y_w), QPointF(x_w, y_w))
            tw = fm.horizontalAdvance(label)
            th = fm.height()
            painter.drawText(QPointF(x_w - 10 - tw, y_w + th / 3), label)

    def _compute_major_ticks(self, v1: float, v2: float) -> list[tuple[float, str]]:
        lo, hi = (v1, v2) if v1 < v2 else (v2, v1)
        if self.axis_mode == "log":
            if lo <= 0 or hi <= 0:
                return []
            e_lo = math.floor(math.log10(lo)) - 1
            e_hi = math.ceil(math.log10(hi)) + 1
            return [(10**e, f"10^{e}") for e in range(e_lo, e_hi + 1)]
        ticks = nice_linear_ticks(lo, hi)
        return [(v, self._format_linear_label(v)) for v in ticks]

    def _format_linear_label(self, v: float) -> str:
        if v == 0:
            return "0"
        av = abs(v)
        if av >= 1e4 or av < 1e-3:
            return format_sci(v)
        return f"{v:g}"

    def _compute_minor_ticks(self, v1: float, v2: float) -> list[float]:
        lo, hi = (v1, v2) if v1 < v2 else (v2, v1)
        if lo <= 0 or hi <= 0:
            return []
        e_lo = math.floor(math.log10(lo)) - 1
        e_hi = math.ceil(math.log10(hi)) + 1
        out = []
        for e in range(e_lo, e_hi + 1):
            base = 10**e
            for mult in range(2, 10):
                out.append(mult * base)
        return out

    def _draw_measurements(self, painter: QPainter) -> None:
        if not self.measurements:
            return
        calibrated = len(self.calib_points) == 2 and len(self.calib_values) == 2
        axis_x_w = (
            self.img_to_widget(QPointF(self.calib_points[0].x(), 0)).x() if calibrated else None
        )

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        fm = painter.fontMetrics()

        for m in self.measurements:
            img_p = QPointF(m["px"], m["py"])
            w = self.img_to_widget(img_p)

            if axis_x_w is not None:
                painter.setPen(QPen(MEASUREMENT_COLOUR, 1, Qt.DashLine))
                painter.drawLine(QPointF(axis_x_w, w.y()), QPointF(w.x(), w.y()))

            painter.setPen(Qt.NoPen)
            painter.setBrush(MEASUREMENT_COLOUR)
            painter.drawEllipse(w, 5, 5)

            label = m["label"]
            y_val = self.pixel_to_y(m["py"])
            text = label if y_val is None else f"{label}  {format_sci(y_val)}"
            painter.setPen(QPen(MEASUREMENT_COLOUR))
            painter.drawText(QPointF(w.x() + 8, w.y() + fm.height() / 3), text)

    def _draw_hover(self, painter: QPainter) -> None:
        if self.hover_widget_pos is None or self.hover_y is None:
            return
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text = format_sci(self.hover_y)
        tw = fm.horizontalAdvance(text) + 8
        th = fm.height() + 4
        rect = QRectF(self.hover_widget_pos.x() + 12, self.hover_widget_pos.y() - th - 2, tw, th)
        if rect.right() > self.width():
            rect.moveLeft(self.hover_widget_pos.x() - 12 - tw)
        if rect.top() < 0:
            rect.moveTop(self.hover_widget_pos.y() + 12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(HOVER_COLOUR)
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(QPen(QColor(240, 240, 240)))
        painter.drawText(rect.adjusted(4, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, text)


class LabelPopup(QLineEdit):
    committed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setPlaceholderText("label, Enter to commit, Esc to cancel")
        self.setFixedWidth(220)
        self.returnPressed.connect(self._commit)

    def _commit(self) -> None:
        text = self.text().strip()
        self.committed.emit(text)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, image_path: Path):
        super().__init__()
        self.image_path = image_path
        self.setWindowTitle(f"kld_extract — {image_path.name}")
        self.resize(1200, 800)

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            raise RuntimeError(f"failed to load image: {image_path}")

        self.canvas = ImageCanvas(pixmap, self)
        self.canvas.point_clicked.connect(self._on_point_clicked)
        self.canvas.hover_moved.connect(self._on_hover_moved)
        self.canvas.hover_left.connect(self._on_hover_left)

        # Calibration and measurements in image pixel coords.
        self.calib_points: list[QPointF] = []
        self.calib_values: list[float] = []
        self.measurements: list[dict] = []
        self._next_id = 0
        self.popup: LabelPopup | None = None
        self.pending_img_pos: QPointF | None = None

        self._build_sidebar()
        self._update_status()

    def _build_sidebar(self) -> None:
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        v = QVBoxLayout(sidebar)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        v.addWidget(self.status_label)

        self.hover_label = QLabel("hover: —")
        v.addWidget(self.hover_label)

        v.addWidget(
            QLabel("Scroll: zoom at cursor\nMiddle-drag or Space+drag: pan\nDouble-click: fit")
        )

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Axis:"))
        self.log_btn = QRadioButton("log")
        self.lin_btn = QRadioButton("linear")
        self.log_btn.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.log_btn)
        group.addButton(self.lin_btn)
        self.log_btn.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.log_btn)
        mode_row.addWidget(self.lin_btn)
        mode_row.addStretch()
        v.addLayout(mode_row)

        self.fit_btn = QPushButton("Fit to window")
        self.fit_btn.clicked.connect(self.canvas.fit_to_window)
        v.addWidget(self.fit_btn)

        self.reset_btn = QPushButton("Reset calibration")
        self.reset_btn.clicked.connect(self._reset_calibration)
        v.addWidget(self.reset_btn)

        v.addWidget(QLabel("Measurements:"))
        self.list_widget = QListWidget()
        v.addWidget(self.list_widget, 1)

        self.delete_btn = QPushButton("Delete selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        v.addWidget(self.delete_btn)

        self.save_btn = QPushButton("Save CSV…")
        self.save_btn.clicked.connect(self._save_csv)
        v.addWidget(self.save_btn)

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.canvas, 1)
        h.addWidget(sidebar)
        self.setCentralWidget(central)

    def _update_status(self) -> None:
        n = len(self.calib_points)
        if n == 0:
            txt = "Calibration: click point 1 on the y-axis."
        elif n == 1:
            txt = "Calibration: click point 2 (snaps to point 1's x)."
        else:
            v1 = format_sci(self.calib_values[0])
            v2 = format_sci(self.calib_values[1])
            txt = f"Calibrated: y1={v1}, y2={v2}. Click on the image to measure."
        self.status_label.setText(txt)
        self.canvas.set_snap_x(self.calib_points[0].x() if n == 1 else None)

    def _on_mode_changed(self) -> None:
        mode = "log" if self.log_btn.isChecked() else "linear"
        if (
            mode == "log"
            and len(self.calib_values) == 2
            and any(v <= 0 for v in self.calib_values)
        ):
            QMessageBox.warning(
                self,
                "Invalid for log",
                "Calibration values must both be > 0 for log mode. Reset calibration first.",
            )
            self.log_btn.blockSignals(True)
            self.lin_btn.setChecked(True)
            self.log_btn.blockSignals(False)
            return
        self.canvas.set_axis_mode(mode)
        self._refresh_list()

    def _on_point_clicked(self, img_pos: QPointF) -> None:
        if self.popup is not None:
            return
        if len(self.calib_points) < 2:
            self._prompt_calibration_value(img_pos)
        else:
            self._begin_measurement(img_pos)

    def _prompt_calibration_value(self, img_pos: QPointF) -> None:
        n = len(self.calib_points)
        while True:
            text, ok = QInputDialog.getText(
                self,
                f"Calibration point {n + 1}",
                f"y-value for point {n + 1} (e.g. 10^-2, 1e-2, 0.01):",
            )
            if not ok:
                return
            try:
                value = parse_y_value(text)
            except ValueError:
                QMessageBox.warning(self, "Invalid", f"Could not parse: {text!r}")
                continue
            if not math.isfinite(value):
                QMessageBox.warning(self, "Invalid", "Value must be finite.")
                continue
            if self.canvas.axis_mode == "log" and value <= 0:
                QMessageBox.warning(self, "Invalid", "Log mode requires value > 0.")
                continue
            if n == 1 and value == self.calib_values[0]:
                QMessageBox.warning(self, "Invalid", "Second value must differ from first.")
                continue
            break

        self.calib_points.append(img_pos)
        self.calib_values.append(value)
        self.canvas.set_calibration(self.calib_points, self.calib_values)
        self._update_status()
        self._refresh_list()

    def _reset_calibration(self) -> None:
        self.calib_points = []
        self.calib_values = []
        self.canvas.set_calibration([], [])
        self._update_status()
        self._refresh_list()

    def _begin_measurement(self, img_pos: QPointF) -> None:
        self.pending_img_pos = img_pos
        popup = LabelPopup(self.canvas)
        popup.committed.connect(self._on_label_committed)
        popup.cancelled.connect(self._on_label_cancelled)
        widget_pos = self.canvas.img_to_widget(img_pos)
        px = min(int(widget_pos.x()) + 10, self.canvas.width() - popup.width() - 4)
        py = min(int(widget_pos.y()) + 10, self.canvas.height() - popup.height() - 4)
        popup.move(max(px, 0), max(py, 0))
        popup.show()
        popup.setFocus()
        self.popup = popup

    def _on_label_committed(self, text: str) -> None:
        text = text.strip()
        if not text:
            QMessageBox.warning(self, "Empty label", "Label cannot be empty.")
            if self.popup is not None:
                self.popup.setFocus()
                self.popup.selectAll()
            return
        if any(m["label"] == text for m in self.measurements):
            QMessageBox.warning(self, "Duplicate", f"Label {text!r} already used.")
            if self.popup is not None:
                self.popup.setFocus()
                self.popup.selectAll()
            return
        pos = self.pending_img_pos
        self._close_popup()
        if pos is None:
            return
        self.measurements.append(
            {"id": self._next_id, "label": text, "px": pos.x(), "py": pos.y()}
        )
        self._next_id += 1
        self.canvas.set_measurements(self.measurements)
        self._refresh_list()

    def _on_label_cancelled(self) -> None:
        self.pending_img_pos = None
        self._close_popup()

    def _close_popup(self) -> None:
        if self.popup is not None:
            self.popup.deleteLater()
            self.popup = None
        self.canvas.setFocus()

    def _on_hover_moved(self, img_pos: QPointF) -> None:
        y = self.canvas.pixel_to_y(img_pos.y())
        self.hover_label.setText(f"hover: {format_sci(y) if y is not None else '—'}")

    def _on_hover_left(self) -> None:
        self.hover_label.setText("hover: —")

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for m in self.measurements:
            y = self.canvas.pixel_to_y(m["py"])
            y_str = format_sci(y) if y is not None else "—"
            item = QListWidgetItem(f"{m['label']}  {y_str}")
            item.setData(Qt.UserRole, m["id"])
            self.list_widget.addItem(item)

    def _delete_selected(self) -> None:
        items = self.list_widget.selectedItems()
        if not items:
            return
        ids_to_remove = {item.data(Qt.UserRole) for item in items}
        self.measurements = [m for m in self.measurements if m["id"] not in ids_to_remove]
        self.canvas.set_measurements(self.measurements)
        self._refresh_list()

    def _save_csv(self) -> None:
        if not self.measurements:
            QMessageBox.information(self, "Nothing to save", "No measurements to export.")
            return
        if len(self.calib_points) != 2:
            QMessageBox.warning(self, "Not calibrated", "Calibrate the axis before saving.")
            return
        default = str(self.image_path.with_name(f"{self.image_path.stem}-extracted.csv"))
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default, "CSV (*.csv)")
        if not path:
            return
        rows = []
        for m in self.measurements:
            y = self.canvas.pixel_to_y(m["py"])
            if y is None:
                continue
            y_rounded = float(f"{y:.3g}")
            rows.append(
                {
                    "label": m["label"],
                    "y_value": y_rounded,
                    "y_value_scientific": format_sci(y),
                }
            )
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "y_value", "y_value_scientific"])
            writer.writeheader()
            writer.writerows(rows)
        QMessageBox.information(self, "Saved", f"Wrote {len(rows)} row(s) to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract y-values from a benchmark graph image.")
    parser.add_argument("image", type=Path, help="path to the image file")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MainWindow(args.image)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
