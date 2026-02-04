# practica_IGM_MC_Bezier.py
# Requisitos: numpy, pandas, matplotlib
# Uso (solo ejercicio 2 con SVG):
#   python practica_IGM_MC_Bezier.py --skip-mc --svg "/ruta/al/logo.svg" --samples-per-seg 220
# Si el SVG no define viewBox, se usa width/height; se invierte Y para dibujar en ejes cartesianos.

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


# -----------------------------
# Ejercicio 1 (se mantiene igual)
# -----------------------------


@dataclass(frozen=True)
class Box3D:
    low: np.ndarray
    high: np.ndarray

    @property
    def volume(self) -> float:
        d = self.high - self.low
        return float(d[0] * d[1] * d[2])


def make_box(name: str) -> Box3D:
    name = name.lower().strip()
    if name == "cube":
        return Box3D(low=np.array([-3.0, -3.0, -3.0]), high=np.array([3.0, 3.0, 3.0]))
    if name == "tight":
        return Box3D(low=np.array([-2.5, -2.0, -0.5]), high=np.array([2.5, 2.0, 0.5]))
    raise ValueError("box debe ser 'cube' o 'tight'")


def sample_uniform_in_box(rng: np.random.Generator, box: Box3D, n: int) -> np.ndarray:
    u = rng.random((n, 3), dtype=float)
    return box.low + u * (box.high - box.low)


def inside_toroid_sqrt(pts: np.ndarray, R: float, r: float) -> np.ndarray:
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    return (np.sqrt(x * x + y * y) - R) ** 2 + z * z <= r * r


def inside_toroid_poly(pts: np.ndarray, R: float, r: float) -> np.ndarray:
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    lhs = (x * x + y * y + z * z + R * R - r * r) ** 2
    rhs = 4.0 * (R * R) * (x * x + y * y)
    return lhs <= rhs


def inside_sphere_S1(pts: np.ndarray, rs: float) -> np.ndarray:
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    return (x - 2.0) ** 2 + y * y + z * z <= rs * rs


def inside_sphere_S2(pts: np.ndarray, rs: float) -> np.ndarray:
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    return (x + 2.0) ** 2 + y * y + z * z <= rs * rs


def exact_volumes(R: float, r: float, rs: float) -> Dict[str, float]:
    Vt = 2.0 * (math.pi**2) * R * (r**2)
    Vs = (4.0 / 3.0) * math.pi * (rs**3)
    return {"V_T_exact": Vt, "V_S_exact": Vs}


def volume_mc(Vbox: float, count: int, N: int) -> float:
    return Vbox * (count / N)


def se_mc(Vbox: float, count: int, N: int) -> float:
    p = count / N
    return Vbox * math.sqrt(p * (1.0 - p) / N)


def run_ex1(
    N: int,
    seed: int,
    box_name: str,
    torus_form: str,
    batch: int,
    R: float = 1.5,
    r: float = 0.5,
    rs: float = 0.5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    box = make_box(box_name)
    Vbox = box.volume

    torus_form = torus_form.lower().strip()
    if torus_form not in {"sqrt", "poly"}:
        raise ValueError("torus-form debe ser 'sqrt' o 'poly'")

    c_T = c_S1 = c_S2 = c_TS1 = c_TS2 = c_union = 0
    xs, vt_series, vs1_series, vunion_series = [], [], [], []

    done = 0
    while done < N:
        n = min(batch, N - done)
        pts = sample_uniform_in_box(rng, box, n)

        mT = (
            inside_toroid_sqrt(pts, R=R, r=r)
            if torus_form == "sqrt"
            else inside_toroid_poly(pts, R=R, r=r)
        )
        mS1 = inside_sphere_S1(pts, rs=rs)
        mS2 = inside_sphere_S2(pts, rs=rs)

        mTS1 = mT & mS1
        mTS2 = mT & mS2
        mU = mT | mS1 | mS2

        c_T += int(mT.sum())
        c_S1 += int(mS1.sum())
        c_S2 += int(mS2.sum())
        c_TS1 += int(mTS1.sum())
        c_TS2 += int(mTS2.sum())
        c_union += int(mU.sum())
        done += n

    ev = exact_volumes(R=R, r=r, rs=rs)
    Vt_exact, Vs_exact = ev["V_T_exact"], ev["V_S_exact"]

    VT = volume_mc(Vbox, c_T, N)
    VS1 = volume_mc(Vbox, c_S1, N)
    VS2 = volume_mc(Vbox, c_S2, N)
    VTS1 = volume_mc(Vbox, c_TS1, N)
    VTS2 = volume_mc(Vbox, c_TS2, N)
    VU = volume_mc(Vbox, c_union, N)

    df = pd.DataFrame(
        [
            {"objeto": "T (toroide sólido)", "V_hat": VT, "V_exact": Vt_exact},
            {"objeto": "S1 (esfera derecha)", "V_hat": VS1, "V_exact": Vs_exact},
            {"objeto": "S2 (esfera izquierda)", "V_hat": VS2, "V_exact": Vs_exact},
            {"objeto": "T∩S1", "V_hat": VTS1, "V_exact": np.nan},
            {"objeto": "T∩S2", "V_hat": VTS2, "V_exact": np.nan},
            {"objeto": "T∪S1∪S2", "V_hat": VU, "V_exact": np.nan},
        ]
    )
    df["N"] = N
    df["box"] = box_name
    df["V_box"] = Vbox
    df["abs_error"] = df["V_hat"] - df["V_exact"]
    df["rel_error"] = df["abs_error"] / df["V_exact"]
    df.loc[df["V_exact"].isna(), ["abs_error", "rel_error"]] = np.nan

    df["SE"] = [
        se_mc(Vbox, c_T, N),
        se_mc(Vbox, c_S1, N),
        se_mc(Vbox, c_S2, N),
        se_mc(Vbox, c_TS1, N),
        se_mc(Vbox, c_TS2, N),
        se_mc(Vbox, c_union, N),
    ]
    df["CI95_low"] = df["V_hat"] - 1.96 * df["SE"]
    df["CI95_high"] = df["V_hat"] + 1.96 * df["SE"]

    Vincl_excl = VT + VS1 + VS2 - VTS1 - VTS2
    Vs = 0.5 * (VS1 + VS2)
    Vinters = 0.5 * (VTS1 + VTS2)
    Vsym = VT + 2.0 * Vs - 2.0 * Vinters

    print("\nResultados Monte Carlo:")
    print(df.to_string(index=False, float_format=lambda x: f"{x: .6g}"))
    print("\nComprobaciones:")
    print(f"VS1 - VS2 = {VS1 - VS2: .6g}")
    print(f"V(T∩S1) - V(T∩S2) = {VTS1 - VTS2: .6g}")
    print(f"Vunión - IE = {VU - Vincl_excl: .6g}")
    print(f"Vunión - simetría = {VU - Vsym: .6g}")

    if plot_convergence:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xs, vt_series, label="VT (MC)")
        ax.plot(xs, vs1_series, label="VS1 (MC)")
        ax.plot(xs, vunion_series, label="Vunión (MC)")
        ax.axhline(Vt_exact, linestyle="--", label="VT exacto")
        ax.axhline(Vs_exact, linestyle="--", label="VS exacto")
        ax.set_xlabel("N acumulado")
        ax.set_ylabel("Volumen estimado")
        ax.set_title(f"Convergencia (box={box_name}, torus-form={torus_form}, N={N})")
        ax.legend()
        ax.grid(True)
        plt.show()

    return df


# -----------------------------
# Ejercicio 2: Logo desde SVG
# -----------------------------


def de_casteljau(ctrl: np.ndarray, t: float) -> np.ndarray:
    p = np.array(ctrl, dtype=float, copy=True)
    m = p.shape[0]
    for k in range(1, m):
        p[: m - k] = (1.0 - t) * p[: m - k] + t * p[1 : m - k + 1]
    return p[0]


def sample_bezier(ctrl: np.ndarray, n: int = 200) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n, dtype=float)
    return np.vstack([de_casteljau(ctrl, float(t)) for t in ts])


def plot_torus_and_spheres(R: float = 1.5, r: float = 0.5, rs: float = 0.5) -> None:
    """Visualiza el toro y las dos esferas S1 y S2 en 3D."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Generar el toro
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, 2 * np.pi, 60)
    u, v = np.meshgrid(u, v)

    x_torus = (R + r * np.cos(v)) * np.cos(u)
    y_torus = (R + r * np.cos(v)) * np.sin(u)
    z_torus = r * np.sin(v)

    # Generar esfera S1 (centro en x=2)
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)

    x_s1 = rs * np.sin(phi) * np.cos(theta) + 2.0
    y_s1 = rs * np.sin(phi) * np.sin(theta)
    z_s1 = rs * np.cos(phi)

    # Generar esfera S2 (centro en x=-2)
    x_s2 = rs * np.sin(phi) * np.cos(theta) - 2.0
    y_s2 = rs * np.sin(phi) * np.sin(theta)
    z_s2 = rs * np.cos(phi)

    # Dibujar las superficies
    ax.plot_surface(x_torus, y_torus, z_torus, alpha=0.6, color="blue", label="Toro")
    ax.plot_surface(x_s1, y_s1, z_s1, alpha=0.7, color="red")
    ax.plot_surface(x_s2, y_s2, z_s2, alpha=0.7, color="green")

    # Configurar ejes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Toro (R={R}, r={r}) y Esferas S1, S2 (rs={rs})")

    # Leyenda manual (plot_surface no soporta label directamente)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", alpha=0.6, label="Toro"),
        Patch(facecolor="red", alpha=0.7, label="S1 (x=2)"),
        Patch(facecolor="green", alpha=0.7, label="S2 (x=-2)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Ajustar aspecto
    ax.set_box_aspect([1, 1, 0.3])

    plt.tight_layout()
    plt.show()


def _tokenize_path(d: str) -> List[str]:
    return re.findall(
        r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d
    )


def _line_to_cubic(P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
    C1 = P0 + (P1 - P0) / 3.0
    C2 = P0 + 2.0 * (P1 - P0) / 3.0
    return np.vstack([P0, C1, C2, P1])


def _quad_to_cubic(P0: np.ndarray, Q: np.ndarray, P1: np.ndarray) -> np.ndarray:
    C1 = P0 + (2.0 / 3.0) * (Q - P0)
    C2 = P1 + (2.0 / 3.0) * (Q - P1)
    return np.vstack([P0, C1, C2, P1])


def _arc_to_cubics(
    P0: np.ndarray,
    rx: float,
    ry: float,
    phi_deg: float,
    large_arc: int,
    sweep: int,
    P1: np.ndarray,
) -> List[np.ndarray]:
    if rx == 0.0 or ry == 0.0:
        return [_line_to_cubic(P0, P1)]

    phi = math.radians(phi_deg % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    x1, y1 = P0[0], P0[1]
    x2, y2 = P1[0], P1[1]

    dx = (x1 - x2) / 2.0
    dy = (y1 - y2) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    rx_abs, ry_abs = abs(rx), abs(ry)
    lam = (x1p * x1p) / (rx_abs * rx_abs) + (y1p * y1p) / (ry_abs * ry_abs)
    if lam > 1.0:
        s = math.sqrt(lam)
        rx_abs *= s
        ry_abs *= s

    num = (
        rx_abs * rx_abs * ry_abs * ry_abs
        - rx_abs * rx_abs * y1p * y1p
        - ry_abs * ry_abs * x1p * x1p
    )
    den = rx_abs * rx_abs * y1p * y1p + ry_abs * ry_abs * x1p * x1p
    num = max(0.0, num)
    coef = math.sqrt(num / den) if den != 0.0 else 0.0
    if large_arc == sweep:
        coef = -coef

    cxp = coef * (rx_abs * y1p) / ry_abs
    cyp = coef * (-ry_abs * x1p) / rx_abs

    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    def _angle(u: Tuple[float, float], v: Tuple[float, float]) -> float:
        dot = u[0] * v[0] + u[1] * v[1]
        det = u[0] * v[1] - u[1] * v[0]
        return math.atan2(det, dot)

    ux = (x1p - cxp) / rx_abs
    uy = (y1p - cyp) / ry_abs
    vx = (-x1p - cxp) / rx_abs
    vy = (-y1p - cyp) / ry_abs

    theta1 = _angle((1.0, 0.0), (ux, uy))
    dtheta = _angle((ux, uy), (vx, vy))

    if sweep == 0 and dtheta > 0:
        dtheta -= 2.0 * math.pi
    if sweep == 1 and dtheta < 0:
        dtheta += 2.0 * math.pi

    nseg = max(1, int(math.ceil(abs(dtheta) / (math.pi / 2.0))))
    delta = dtheta / nseg

    cubics: List[np.ndarray] = []
    for i in range(nseg):
        a0 = theta1 + i * delta
        a1 = a0 + delta
        t = (4.0 / 3.0) * math.tan((a1 - a0) / 4.0)

        p0u = np.array([math.cos(a0), math.sin(a0)])
        p3u = np.array([math.cos(a1), math.sin(a1)])
        c1u = p0u + t * np.array([-math.sin(a0), math.cos(a0)])
        c2u = p3u - t * np.array([-math.sin(a1), math.cos(a1)])

        def _map(upt: np.ndarray) -> np.ndarray:
            x = rx_abs * upt[0]
            y = ry_abs * upt[1]
            xr = cos_phi * x - sin_phi * y + cx
            yr = sin_phi * x + cos_phi * y + cy
            return np.array([xr, yr], dtype=float)

        P0s = _map(p0u)
        C1s = _map(c1u)
        C2s = _map(c2u)
        P3s = _map(p3u)
        cubics.append(np.vstack([P0s, C1s, C2s, P3s]))

    return cubics


@dataclass
class Subpath:
    cubics: List[np.ndarray]  # cada uno (4,2)
    closed: bool


def parse_svg_path_to_cubics(d: str) -> List[Subpath]:
    tokens = _tokenize_path(d)
    i = 0
    cmd = None

    cur = np.array([0.0, 0.0], dtype=float)
    start = np.array([0.0, 0.0], dtype=float)

    last_c2: Optional[np.ndarray] = None
    last_q: Optional[np.ndarray] = None

    subpaths: List[Subpath] = []
    current_cubics: List[np.ndarray] = []
    current_closed = False

    def _flush_subpath():
        nonlocal current_cubics, current_closed
        if current_cubics:
            subpaths.append(Subpath(cubics=current_cubics, closed=current_closed))
        current_cubics = []
        current_closed = False

    def _read_float() -> float:
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[AaCcHhLlMmQqSsTtVvZz]", tok):
            cmd = tok
            i += 1
        if cmd is None:
            raise ValueError("Path inválido: falta comando inicial.")

        rel = cmd.islower()
        c = cmd.lower()

        if c == "m":
            x = _read_float()
            y = _read_float()
            pt = np.array([x, y], dtype=float)
            cur = cur + pt if rel else pt
            start = cur.copy()
            _flush_subpath()
            last_c2 = None
            last_q = None

            while i < len(tokens) and not re.fullmatch(
                r"[AaCcHhLlMmQqSsTtVvZz]", tokens[i]
            ):
                x = _read_float()
                y = _read_float()
                pt = np.array([x, y], dtype=float)
                nxt = cur + pt if rel else pt
                current_cubics.append(_line_to_cubic(cur, nxt))
                cur = nxt
                last_c2 = current_cubics[-1][2].copy()
                last_q = None

        elif c == "l":
            while i < len(tokens) and not re.fullmatch(
                r"[AaCcHhLlMmQqSsTtVvZz]", tokens[i]
            ):
                x = _read_float()
                y = _read_float()
                pt = np.array([x, y], dtype=float)
                nxt = cur + pt if rel else pt
                current_cubics.append(_line_to_cubic(cur, nxt))
                cur = nxt
                last_c2 = current_cubics[-1][2].copy()
                last_q = None

        elif c == "h":
            while i < len(tokens) and not re.fullmatch(
                r"[AaCcHhLlMmQqSsTtVvZz]", tokens[i]
            ):
                x = _read_float()
                nxt = cur + np.array([x, 0.0]) if rel else np.array([x, cur[1]])
                current_cubics.append(_line_to_cubic(cur, nxt))
                cur = nxt
                last_c2 = current_cubics[-1][2].copy()
                last_q = None

        elif c == "v":
            while i < len(tokens) and not re.fullmatch(
                r"[AaCcHhLlMmQqSsTtVvZz]", tokens[i]
            ):
                y = _read_float()
                nxt = cur + np.array([0.0, y]) if rel else np.array([cur[0], y])
                current_cubics.append(_line_to_cubic(cur, nxt))
                cur = nxt
                last_c2 = current_cubics[-1][2].copy()
                last_q = None

        elif c == "c":
            while i < len(tokens) and not re.fullmatch(
                r"[AaCcHhLlMmQqSsTtVvZz]", tokens[i]
            ):
                x1, y1, x2, y2, x, y = (
                    _read_float(),
                    _read_float(),
                    _read_float(),
                    _read_float(),
                    _read_float(),
                    _read_float(),
                )
                C1 = np.array([x1, y1], dtype=float)
                C2 = np.array([x2, y2], dtype=float)
                P1 = np.array([x, y], dtype=float)
                if rel:
                    C1 = cur + C1
                    C2 = cur + C2
                    P1 = cur + P1
                cubic = np.vstack([cur, C1, C2, P1])
                current_cubics.append(cubic)
                cur = P1
                last_c2 = C2.copy()
                last_q = None

        elif c == "s":
            while i < len(tokens) and not re.fullmatch(
                r"[AaCcHhLlMmQqSsTtVvZz]", tokens[i]
            ):
                x2, y2, x, y = (
                    _read_float(),
                    _read_float(),
                    _read_float(),
                    _read_float(),
                )
                C2 = np.array([x2, y2], dtype=float)
                P1 = np.array([x, y], dtype=float)
                if rel:
                    C2 = cur + C2
                    P1 = cur + P1

                if last_c2 is not None:
                    C1 = cur + (cur - last_c2)
                else:
                    C1 = cur.copy()

                cubic = np.vstack([cur, C1, C2, P1])
                current_cubics.append(cubic)
                cur = P1
                last_c2 = C2.copy()
                last_q = None

        elif c == "q":
            while i < len(tokens) and not re.fullmatch(
                r"[AaCcHhLlMmQqSsTtVvZz]", tokens[i]
            ):
                x1, y1, x, y = (
                    _read_float(),
                    _read_float(),
                    _read_float(),
                    _read_float(),
                )
                Q = np.array([x1, y1], dtype=float)
                P1 = np.array([x, y], dtype=float)
                if rel:
                    Q = cur + Q
                    P1 = cur + P1
                cubic = _quad_to_cubic(cur, Q, P1)
                current_cubics.append(cubic)
                cur = P1
                last_q = Q.copy()
                last_c2 = cubic[2].copy()

        elif c == "t":
            while i < len(tokens) and not re.fullmatch(
                r"[AaCcHhLlMmQqSsTtVvZz]", tokens[i]
            ):
                x, y = _read_float(), _read_float()
                P1 = np.array([x, y], dtype=float)
                if rel:
                    P1 = cur + P1
                if last_q is not None:
                    Q = cur + (cur - last_q)
                else:
                    Q = cur.copy()
                cubic = _quad_to_cubic(cur, Q, P1)
                current_cubics.append(cubic)
                cur = P1
                last_q = Q.copy()
                last_c2 = cubic[2].copy()

        elif c == "a":
            while i < len(tokens) and not re.fullmatch(
                r"[AaCcHhLlMmQqSsTtVvZz]", tokens[i]
            ):
                rx, ry = _read_float(), _read_float()
                phi = _read_float()
                large = int(_read_float())
                sweep = int(_read_float())
                x, y = _read_float(), _read_float()
                P1 = np.array([x, y], dtype=float)
                if rel:
                    P1 = cur + P1
                cubs = _arc_to_cubics(cur, rx, ry, phi, large, sweep, P1)
                for cb in cubs:
                    current_cubics.append(cb)
                    last_c2 = cb[2].copy()
                cur = P1
                last_q = None

        elif c == "z":
            if np.linalg.norm(cur - start) > 1e-12:
                current_cubics.append(_line_to_cubic(cur, start))
                last_c2 = current_cubics[-1][2].copy()
            cur = start.copy()
            current_closed = True
            last_c2 = None
            last_q = None
            _flush_subpath()

        else:
            raise ValueError(f"Comando SVG no soportado: {cmd}")

    _flush_subpath()
    return subpaths


def read_svg_geometry(
    svg_path: str,
) -> Tuple[List[Subpath], float, float, Optional[Tuple[float, float, float, float]]]:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    def _get_float_attr(name: str) -> Optional[float]:
        v = root.attrib.get(name)
        if v is None:
            return None
        v = v.strip()
        v = re.sub(r"[a-zA-Z%]+$", "", v)
        try:
            return float(v)
        except ValueError:
            return None

    width = _get_float_attr("width")
    height = _get_float_attr("height")
    viewBox = root.attrib.get("viewBox")

    vb = None
    if viewBox:
        parts = [p for p in re.split(r"[,\s]+", viewBox.strip()) if p]
        if len(parts) == 4:
            vb = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))

    paths = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "path":
            d = el.attrib.get("d")
            if d:
                paths.append(d)

    if not paths:
        raise ValueError("SVG sin <path d='...'> detectable.")

    subpaths_all: List[Subpath] = []
    for d in paths:
        subpaths_all.extend(parse_svg_path_to_cubics(d))

    if vb is not None:
        vb_x, vb_y, vb_w, vb_h = vb
        W = vb_w
        H = vb_h
    else:
        if width is None or height is None:
            xs, ys = [], []
            for sp in subpaths_all:
                for cb in sp.cubics:
                    xs.extend(cb[:, 0].tolist())
                    ys.extend(cb[:, 1].tolist())
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            W = maxx - minx
            H = maxy - miny
            vb = (minx, miny, W, H)
        else:
            W, H = width, height

    return subpaths_all, float(W), float(H), vb


def transform_subpaths_for_plot(
    subpaths: List[Subpath],
    W: float,
    H: float,
    vb: Optional[Tuple[float, float, float, float]],
) -> List[Subpath]:
    if vb is not None:
        vb_x, vb_y, vb_w, vb_h = vb
    else:
        vb_x = 0.0
        vb_y = 0.0
        vb_w = W
        vb_h = H

    def _xf(p: np.ndarray) -> np.ndarray:
        x = p[0] - vb_x
        y = p[1] - vb_y
        y = vb_h - y
        return np.array([x, y], dtype=float)

    out: List[Subpath] = []
    for sp in subpaths:
        cubs = []
        for cb in sp.cubics:
            cubs.append(np.vstack([_xf(cb[0]), _xf(cb[1]), _xf(cb[2]), _xf(cb[3])]))
        out.append(Subpath(cubics=cubs, closed=sp.closed))
    return out


def plot_svg_logo_bezier(
    subpaths: List[Subpath],
    title: str,
    samples_per_seg: int = 220,
    show_fill: bool = True,
    show_controls: bool = True,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    verts: List[Tuple[float, float]] = []
    codes: List[int] = []

    for sp in subpaths:
        poly = []
        for j, cb in enumerate(sp.cubics):
            pts = sample_bezier(cb, n=samples_per_seg)
            if j != len(sp.cubics) - 1:
                pts = pts[:-1]
            poly.append(pts)

            if show_controls:
                ax.plot(cb[:, 0], cb[:, 1], linestyle=":", linewidth=1.0, alpha=0.85)
                ax.scatter(cb[:, 0], cb[:, 1], s=14, alpha=0.9)

        if poly:
            poly_pts = np.vstack(poly)
            ax.plot(poly_pts[:, 0], poly_pts[:, 1], linewidth=2.0)

            verts.append((poly_pts[0, 0], poly_pts[0, 1]))
            codes.append(Path.MOVETO)
            for k in range(1, poly_pts.shape[0]):
                verts.append((poly_pts[k, 0], poly_pts[k, 1]))
                codes.append(Path.LINETO)
            verts.append((poly_pts[0, 0], poly_pts[0, 1]))
            codes.append(Path.CLOSEPOLY)

    if show_fill and verts:
        path = Path(verts, codes)
        patch = PathPatch(path, alpha=0.25)
        try:
            patch.set_fillrule("evenodd")
        except Exception:
            pass
        ax.add_patch(patch)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True)
    plt.show()


def modify_one_segment(
    subpaths: List[Subpath],
    subpath_idx: int = 0,
    seg_idx: int = 0,
    delta: Tuple[float, float] = (15.0, -10.0),
) -> List[Subpath]:
    out = []
    d = np.array(delta, dtype=float)
    for si, sp in enumerate(subpaths):
        cubs = [cb.copy() for cb in sp.cubics]
        if si == subpath_idx and 0 <= seg_idx < len(cubs):
            cubs[seg_idx][1] += d
            cubs[seg_idx][2] -= 0.6 * d
        out.append(Subpath(cubics=cubs, closed=sp.closed))
    return out


def run_ex2_from_svg(svg_path: str, samples_per_seg: int) -> None:
    subpaths_raw, W, H, vb = read_svg_geometry(svg_path)
    subpaths = transform_subpaths_for_plot(subpaths_raw, W=W, H=H, vb=vb)

    plot_svg_logo_bezier(
        subpaths,
        title="Logo desde SVG como spline Bézier cúbica (De Casteljau) con control points",
        samples_per_seg=samples_per_seg,
        show_fill=True,
        show_controls=True,
    )

    subpaths_mod = modify_one_segment(
        subpaths, subpath_idx=0, seg_idx=2, delta=(18.0, -12.0)
    )
    plot_svg_logo_bezier(
        subpaths_mod,
        title="Variante tras modificar control points (impacto geométrico)",
        samples_per_seg=samples_per_seg,
        show_fill=True,
        show_controls=True,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for sp in subpaths:
        poly = []
        for j, cb in enumerate(sp.cubics):
            pts = sample_bezier(cb, n=samples_per_seg)
            if j != len(sp.cubics) - 1:
                pts = pts[:-1]
            poly.append(pts)
        if poly:
            pts = np.vstack(poly)
            ax.plot(pts[:, 0], pts[:, 1], linewidth=2.0)
    for sp in subpaths_mod:
        poly = []
        for j, cb in enumerate(sp.cubics):
            pts = sample_bezier(cb, n=samples_per_seg)
            if j != len(sp.cubics) - 1:
                pts = pts[:-1]
            poly.append(pts)
        if poly:
            pts = np.vstack(poly)
            ax.plot(pts[:, 0], pts[:, 1], linestyle="--", linewidth=2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Comparativa: original (continua) vs modificada (discontinua)")
    ax.grid(True)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=600000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--box", type=str, choices=["cube", "tight"], default="cube")
    parser.add_argument(
        "--torus-form", type=str, choices=["sqrt", "poly"], default="sqrt"
    )
    parser.add_argument("--batch", type=int, default=100000)
    parser.add_argument("--skip-mc", action="store_true")
    parser.add_argument("--skip-bezier", action="store_true")
    parser.add_argument(
        "--svg", type=str, default="", help="Ruta al logo SVG (para ejercicio 2)."
    )
    parser.add_argument(
        "--samples-per-seg", type=int, default=220, help="Muestras por segmento Bézier."
    )
    args = parser.parse_args()

    if not args.skip_mc:
        run_ex1(
            N=args.N,
            seed=args.seed,
            box_name=args.box,
            torus_form=args.torus_form,
            batch=max(1, args.batch),
        )

    if not args.skip_bezier:
        if not args.svg:
            raise ValueError(
                "Para el ejercicio 2 con logo SVG se requiere --svg con la ruta del archivo."
            )
        run_ex2_from_svg(args.svg, samples_per_seg=max(30, args.samples_per_seg))


if __name__ == "__main__":
    main()
