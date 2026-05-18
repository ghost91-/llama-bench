import math
import re
from typing import Literal


AxisMode = Literal["linear", "log"]
CsvRow = dict[str, str | float]


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
    """Generate 'nice' (1/2/5 x 10^n) tick values covering [lo, hi] slightly extended."""
    if lo > hi:
        lo, hi = hi, lo
    span = hi - lo
    if span <= 0:
        return [lo]
    rough_step = span / max(target - 1, 1)
    exp = math.floor(math.log10(rough_step))
    base = 10.0**exp
    step = base
    for mult in (1, 2, 5, 10):
        step = mult * base
        if span / step <= target:
            break
    pad = step
    start = math.floor((lo - pad) / step) * step
    end = math.ceil((hi + pad) / step) * step
    ticks: list[float] = []
    v = start
    while v <= end + step * 0.5:
        ticks.append(round(v / step) * step)
        v += step
    return ticks


def pixel_to_y(
    img_y: float,
    p1_y: float,
    p2_y: float,
    v1: float,
    v2: float,
    axis_mode: AxisMode,
) -> float | None:
    if p1_y == p2_y:
        return None
    t = (img_y - p1_y) / (p2_y - p1_y)
    if axis_mode == "log":
        if v1 <= 0 or v2 <= 0:
            return None
        log_y = math.log10(v1) + t * (math.log10(v2) - math.log10(v1))
        return 10.0**log_y
    return v1 + t * (v2 - v1)


def y_to_pixel(
    y: float,
    p1_y: float,
    p2_y: float,
    v1: float,
    v2: float,
    axis_mode: AxisMode,
) -> float | None:
    if axis_mode == "log":
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
    return p1_y + t * (p2_y - p1_y)


def major_ticks(v1: float, v2: float, axis_mode: AxisMode) -> list[tuple[float, str]]:
    lo, hi = (v1, v2) if v1 < v2 else (v2, v1)
    if axis_mode == "log":
        if lo <= 0 or hi <= 0:
            return []
        e_lo = math.floor(math.log10(lo)) - 1
        e_hi = math.ceil(math.log10(hi)) + 1
        return [(10.0**e, f"10^{e}") for e in range(e_lo, e_hi + 1)]
    ticks = nice_linear_ticks(lo, hi)
    return [(v, format_linear_label(v)) for v in ticks]


def format_linear_label(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e4 or av < 1e-3:
        return format_sci(v)
    return f"{v:g}"


def minor_log_ticks(v1: float, v2: float) -> list[float]:
    lo, hi = (v1, v2) if v1 < v2 else (v2, v1)
    if lo <= 0 or hi <= 0:
        return []
    e_lo = math.floor(math.log10(lo)) - 1
    e_hi = math.ceil(math.log10(hi)) + 1
    out: list[float] = []
    for e in range(e_lo, e_hi + 1):
        base = 10.0**e
        for mult in range(2, 10):
            out.append(mult * base)
    return out


def measurement_csv_row(label: str, y: float) -> CsvRow:
    return {
        "label": label,
        "y_value": float(f"{y:.3g}"),
        "y_value_scientific": format_sci(y),
    }
