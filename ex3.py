from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path


# ============================================================================
# DATOS DEL CIRCUITO: Curvas Bezier cubicas preprocesadas desde
# RaceCircuitInterlagos.svg
# Cada curva tiene 4 puntos de control: [P0, C1, C2, P3]
# ============================================================================

SUBPATH_1_CLOSED = True
SUBPATH_1_CUBICS = [
    [[115.55, 366.5416], [93.927, 366.3961], [54.085, 351.3476], [40.03, 337.0556]],
    [[40.03, 337.0556], [25.636, 322.4186], [3.2, 241.5526], [3.2, 204.3056]],
    [[3.2, 204.3056], [3.2, 194.2866], [16.614, 134.1896], [25.122, 106.0946]],
    [[25.122, 106.0946], [26.2046, 102.5196], [31.893, 81.3696], [37.763, 59.0946]],
    [[37.763, 59.0946], [50.082, 12.3446], [52.866, 5.9136], [61.701, 3.8036]],
    [[61.701, 3.8036], [68.3795, 2.2087], [70.487, 3.1893], [78.269, 11.5087]],
    [[78.269, 11.5087], [88.383, 22.3207], [91.135, 22.9267], [101.562, 16.6396]],
    [[101.562, 16.6396], [130.208, -0.6324], [170.349, 14.3471], [179.152, 45.5946]],
    [[179.152, 45.5946], [218.142, 184.0046], [236.312, 256.9946], [233.252, 262.9846]],
    [
        [233.252, 262.9846],
        [227.852, 273.5826],
        [191.042, 280.7666],
        [171.142, 275.1096],
    ],
    [
        [171.142, 275.1096],
        [159.892, 271.9136],
        [156.162, 267.4546],
        [101.542, 191.8646],
    ],
    [[101.542, 191.8646], [79.621, 161.5346], [79.625, 161.5346], [62.499, 162.2846]],
    [[62.499, 162.2846], [41.668, 163.1946], [35.053, 170.4546], [30.782, 197.0846]],
    [[30.782, 197.0846], [25.735, 228.5646], [28.751, 235.3746], [43.352, 225.4546]],
    [[43.352, 225.4546], [56.752, 216.3545], [65.011, 215.8695], [72.329, 223.7554]],
    [[72.329, 223.7554], [81.5174, 233.6574], [79.5066, 239.6194], [60.241, 259.5874]],
    [[60.241, 259.5874], [44.627, 275.7744], [35.302, 300.0184], [41.89, 307.2984]],
    [[41.89, 307.2984], [45.8157, 311.6362], [50.3499, 308.993], [61.869, 295.6554]],
    [[61.869, 295.6554], [84.245, 269.7464], [99.533, 264.9154], [118.904, 277.6324]],
    [
        [118.904, 277.6324],
        [130.046, 284.9472],
        [165.146, 340.5304],
        [162.412, 346.5304],
    ],
    [
        [162.412, 346.5304],
        [159.482, 352.9604],
        [138.332, 363.0984],
        [121.622, 366.0854],
    ],
    [
        [121.622, 366.0854],
        [119.822, 366.4064],
        [117.782, 366.5514],
        [115.552, 366.5363],
    ],
    [
        [115.552, 366.5363],
        [115.5513, 366.5381],
        [115.5507, 366.5398],
        [115.55, 366.5416],
    ],
]


# ============================================================================
# Estructura de datos
# ============================================================================


@dataclass
class Subpath:
    """Representa un subpath compuesto por curvas Bezier cubicas."""

    cubics: List[np.ndarray]
    closed: bool


def get_circuit_subpaths() -> List[Subpath]:
    """Retorna los subpaths del circuito de Interlagos."""
    cubics = [np.array(cb, dtype=float) for cb in SUBPATH_1_CUBICS]
    return [Subpath(cubics=cubics, closed=SUBPATH_1_CLOSED)]


# ============================================================================
# Algoritmo de De Casteljau
# ============================================================================


def de_casteljau(ctrl: np.ndarray, t: float) -> np.ndarray:
    """
    Evalua una curva Bezier en el parametro t usando el algoritmo de De Casteljau.

    Args:
        ctrl: Array (n, 2) con n puntos de control
        t: Parametro en [0, 1]

    Returns:
        Punto (x, y) en la curva
    """
    p = np.array(ctrl, dtype=float, copy=True)
    n = p.shape[0]
    for k in range(1, n):
        p[: n - k] = (1.0 - t) * p[: n - k] + t * p[1 : n - k + 1]
    return p[0]


def sample_bezier(ctrl: np.ndarray, n_samples: int = 200) -> np.ndarray:
    """
    Muestrea n puntos uniformemente a lo largo de una curva Bezier.

    Args:
        ctrl: Array (4, 2) con puntos de control de la cubica
        n_samples: Numero de puntos a muestrear

    Returns:
        Array (n_samples, 2) con los puntos de la curva
    """
    t_values = np.linspace(0.0, 1.0, n_samples, dtype=float)
    return np.vstack([de_casteljau(ctrl, t) for t in t_values])


# ============================================================================
# Visualizacion
# ============================================================================


def plot_circuit(
    subpaths: List[Subpath],
    title: str,
    samples_per_segment: int = 220,
    show_fill: bool = False,
    show_control_points: bool = True,
) -> None:
    """
    Dibuja el circuito usando curvas Bezier.

    Args:
        subpaths: Lista de subpaths con curvas Bezier
        title: Titulo del grafico
        samples_per_segment: Puntos por segmento Bezier
        show_fill: Si mostrar el relleno
        show_control_points: Si mostrar los puntos de control
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    vertices: List[Tuple[float, float]] = []
    codes: List[int] = []

    for sp in subpaths:
        polygon_points = []

        for j, cubic in enumerate(sp.cubics):
            curve_points = sample_bezier(cubic, n_samples=samples_per_segment)

            if j != len(sp.cubics) - 1:
                curve_points = curve_points[:-1]
            polygon_points.append(curve_points)

            if show_control_points:
                ax.plot(
                    cubic[:, 0],
                    cubic[:, 1],
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.7,
                    color="gray",
                )
                ax.scatter(cubic[:, 0], cubic[:, 1], s=18, alpha=0.8, zorder=5)

        if polygon_points:
            poly = np.vstack(polygon_points)
            ax.plot(poly[:, 0], poly[:, 1], linewidth=2.0)

            vertices.append((poly[0, 0], poly[0, 1]))
            codes.append(Path.MOVETO)
            for k in range(1, poly.shape[0]):
                vertices.append((poly[k, 0], poly[k, 1]))
                codes.append(Path.LINETO)
            vertices.append((poly[0, 0], poly[0, 1]))
            codes.append(Path.CLOSEPOLY)

    if show_fill and vertices:
        path = Path(vertices, codes)
        patch = PathPatch(path, alpha=0.25, facecolor="blue", edgecolor="none")
        try:
            patch.set_fillrule("evenodd")
        except AttributeError:
            pass
        ax.add_patch(patch)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def modify_control_points(
    subpaths: List[Subpath],
    subpath_idx: int = 0,
    segment_idx: int = 0,
    delta: Tuple[float, float] = (15.0, -10.0),
) -> List[Subpath]:
    """
    Modifica los puntos de control de un segmento especifico.

    Args:
        subpaths: Lista de subpaths original
        subpath_idx: Indice del subpath a modificar
        segment_idx: Indice del segmento dentro del subpath
        delta: Desplazamiento (dx, dy) a aplicar

    Returns:
        Nueva lista de subpaths con la modificacion
    """
    d = np.array(delta, dtype=float)
    result = []

    for si, sp in enumerate(subpaths):
        cubics = [cb.copy() for cb in sp.cubics]
        if si == subpath_idx and 0 <= segment_idx < len(cubics):
            cubics[segment_idx][1] += d
            cubics[segment_idx][2] -= 0.6 * d
        result.append(Subpath(cubics=cubics, closed=sp.closed))

    return result


def plot_comparison(
    original: List[Subpath],
    modified: List[Subpath],
    samples_per_segment: int = 220,
) -> None:
    """Dibuja comparacion entre circuito original y modificado."""
    fig, ax = plt.subplots(figsize=(10, 10))

    for sp in original:
        points = []
        for j, cubic in enumerate(sp.cubics):
            pts = sample_bezier(cubic, n_samples=samples_per_segment)
            if j != len(sp.cubics) - 1:
                pts = pts[:-1]
            points.append(pts)
        if points:
            poly = np.vstack(points)
            ax.plot(poly[:, 0], poly[:, 1], linewidth=2.0, label="Original")

    for sp in modified:
        points = []
        for j, cubic in enumerate(sp.cubics):
            pts = sample_bezier(cubic, n_samples=samples_per_segment)
            if j != len(sp.cubics) - 1:
                pts = pts[:-1]
            points.append(pts)
        if points:
            poly = np.vstack(points)
            ax.plot(
                poly[:, 0],
                poly[:, 1],
                linestyle="--",
                linewidth=2.0,
                label="Modificado",
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Comparativa: original (continua) vs modificado (discontinua)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Demostracion
# ============================================================================


def run_demo(samples_per_segment: int = 220) -> None:
    """Ejecuta la demostracion completa del circuito con curvas Bezier."""
    subpaths = get_circuit_subpaths()

    plot_circuit(
        subpaths,
        title="Circuito de Interlagos como spline Bezier cubica (De Casteljau)",
        samples_per_segment=samples_per_segment,
        show_fill=False,
        show_control_points=True,
    )

    subpaths_modified = modify_control_points(
        subpaths,
        subpath_idx=0,
        segment_idx=2,
        delta=(18.0, -12.0),
    )
    plot_circuit(
        subpaths_modified,
        title="Interlagos tras modificar puntos de control",
        samples_per_segment=samples_per_segment,
        show_fill=False,
        show_control_points=True,
    )

    plot_comparison(subpaths, subpaths_modified, samples_per_segment)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizacion del circuito de Interlagos mediante curvas Bezier cubicas"
    )
    parser.add_argument(
        "--samples-per-seg",
        type=int,
        default=420,
        help="Puntos de muestreo por segmento Bezier (default: 420)",
    )
    args = parser.parse_args()

    run_demo(samples_per_segment=max(30, args.samples_per_seg))


if __name__ == "__main__":
    main()
