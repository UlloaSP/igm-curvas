# ex2___2.py
# Visualización INTERACTIVA de logo mediante curvas Bézier cúbicas (algoritmo de De Casteljau)
# Permite arrastrar puntos de control y ver el impacto en tiempo real
# Requisitos: numpy, matplotlib

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


# ============================================================================
# DATOS DEL LOGO: Curvas Bézier cúbicas preprocesadas
# Cada curva tiene 4 puntos de control: [P0, C1, C2, P3]
# ============================================================================

# Subpath 1: Silueta principal (23 curvas)
SUBPATH_1_CLOSED = True
SUBPATH_1_CUBICS = [
    [[208.00003, 495.4], [209.20003, 495.6], [210.40003, 495.8], [211.60003, 496.0]],
    [[211.60003, 496.0], [190.70003, 486.1], [162.60003, 465.6], [143.80003, 451.1]],
    [[143.80003, 451.1], [113.40003, 427.8], [89.70003, 393.2], [69.80003, 358.2]],
    [[69.80003, 358.2], [153.90003, 403.9], [231.60003, 410.6], [248.30003, 406.4]],
    [[248.30003, 406.4], [248.50003, 406.3], [240.60003, 360.9], [211.60003, 322.8]],
    [[211.60003, 322.8], [181.10003, 282.9], [129.40003, 250.4], [129.40003, 250.4]],
    [[129.40003, 250.4], [148.30003, 248.2], [167.30003, 247.8], [186.10003, 249.6]],
    [[186.10003, 249.6], [218.30003, 252.6], [251.40003, 262.0], [277.60003, 284.3]],
    [[277.60003, 284.3], [289.20003, 294.2], [310.40003, 323.3], [308.10003, 319.1]],
    [[308.10003, 319.1], [295.60003, 287.9], [293.80003, 252.4], [295.80003, 218.5]],
    [[295.80003, 218.5], [296.90003, 198.6], [299.00003, 178.9], [304.90003, 160.1]],
    [[304.90003, 160.1], [309.90003, 144.0], [317.20003, 128.9], [325.50003, 114.8]],
    [
        [325.50003, 114.8],
        [328.23336, 112.63333],
        [330.9667, 110.46667],
        [333.70003, 108.3],
    ],
    [[333.70003, 108.3], [333.80003, 109.0], [325.80003, 220.1], [382.20003, 275.0]],
    [[382.20003, 275.0], [484.80003, 374.8], [598.20003, 279.9], [598.20003, 279.9]],
    [[598.20003, 279.9], [598.20003, 279.9], [593.70003, 355.1], [508.60003, 391.6]],
    [[508.60003, 391.6], [712.40003, 410.4], [668.30003, 289.6], [668.30003, 289.6]],
    [[668.30003, 289.6], [668.30003, 289.6], [738.10003, 319.4], [727.40003, 404.5]],
    [[727.40003, 404.5], [717.70003, 481.5], [642.40003, 499.8], [626.70003, 502.8]],
    [[626.70003, 502.8], [613.50003, 524.5], [513.50003, 689.6], [346.90003, 624.7]],
    [[346.90003, 624.7], [166.30003, 554.4], [20.00003, 571.1], [20.00003, 571.1]],
    [[20.00003, 571.1], [41.50003, 546.4], [66.70003, 526.5], [94.10003, 512.4]],
    [[94.10003, 512.4], [129.70003, 494.0], [169.40003, 488.6], [208.00003, 495.4]],
]

# Subpath 2: Ojo pequeño (4 curvas)
SUBPATH_2_CLOSED = True
SUBPATH_2_CUBICS = [
    [[522.00003, 511.3], [528.30003, 509.2], [534.70003, 507.1], [541.10003, 505.1]],
    [[541.10003, 505.1], [540.60003, 499.0], [536.30003, 494.2], [531.10003, 494.2]],
    [[531.10003, 494.2], [525.60003, 494.2], [521.00003, 499.6], [521.00003, 506.3]],
    [[521.00003, 506.3], [521.10003, 508.1], [521.40003, 509.8], [522.00003, 511.3]],
]

# Subpath 3: Ojo grande (9 curvas)
SUBPATH_3_CLOSED = True
SUBPATH_3_CUBICS = [
    [[481.90003, 508.9], [481.90003, 514.0], [482.80003, 518.8], [484.50003, 523.3]],
    [[484.50003, 523.3], [488.30003, 522.4], [492.10003, 521.5], [495.70003, 520.3]],
    [[495.70003, 520.3], [499.50003, 519.1], [503.20003, 517.8], [507.00003, 516.5]],
    [[507.00003, 516.5], [505.90003, 513.4], [505.20003, 510.0], [505.20003, 506.4]],
    [[505.20003, 506.4], [505.20003, 491.0], [516.80003, 478.5], [531.10003, 478.5]],
    [[531.10003, 478.5], [543.70003, 478.5], [554.10003, 488.2], [556.50003, 501.0]],
    [[556.50003, 501.0], [559.20003, 500.4], [561.80003, 499.7], [564.50003, 499.2]],
    [[564.50003, 499.2], [560.10003, 480.7], [543.60003, 466.9], [523.80003, 466.9]],
    [[523.80003, 466.9], [500.60003, 466.9], [481.90003, 485.7], [481.90003, 508.9]],
]


# ============================================================================
# Estructura de datos
# ============================================================================


@dataclass
class Subpath:
    """Representa un subpath compuesto por curvas Bézier cúbicas."""

    cubics: List[np.ndarray]  # Lista de arrays (4, 2) con puntos de control
    closed: bool


def get_logo_subpaths() -> List[Subpath]:
    """Retorna los subpaths del logo como estructuras de datos."""
    subpaths_data = [
        (SUBPATH_1_CUBICS, SUBPATH_1_CLOSED),
        (SUBPATH_2_CUBICS, SUBPATH_2_CLOSED),
        (SUBPATH_3_CUBICS, SUBPATH_3_CLOSED),
    ]

    result = []
    for cubics_list, closed in subpaths_data:
        cubics = [np.array(cb, dtype=float) for cb in cubics_list]
        result.append(Subpath(cubics=cubics, closed=closed))

    return result


# ============================================================================
# Algoritmo de De Casteljau
# ============================================================================


def de_casteljau(ctrl: np.ndarray, t: float) -> np.ndarray:
    """
    Evalúa una curva Bézier en el parámetro t usando el algoritmo de De Casteljau.

    Args:
        ctrl: Array (n, 2) con n puntos de control
        t: Parámetro en [0, 1]

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
    Muestrea n puntos uniformemente a lo largo de una curva Bézier.

    Args:
        ctrl: Array (4, 2) con puntos de control de la cúbica
        n_samples: Número de puntos a muestrear

    Returns:
        Array (n_samples, 2) con los puntos de la curva
    """
    t_values = np.linspace(0.0, 1.0, n_samples, dtype=float)
    return np.vstack([de_casteljau(ctrl, t) for t in t_values])


# ============================================================================
# Editor Interactivo de Bézier
# ============================================================================


class InteractiveBezierEditor:
    """
    Editor interactivo para manipular puntos de control de curvas Bézier.
    
    Permite arrastrar cualquier punto de control con el mouse y ver
    el impacto en la curva en tiempo real.
    """

    def __init__(
        self,
        subpaths: List[Subpath],
        samples_per_segment: int = 100,
        pick_radius: float = 15.0,
    ):
        """
        Inicializa el editor interactivo.

        Args:
            subpaths: Lista de subpaths con curvas Bézier
            samples_per_segment: Puntos por segmento para renderizado
            pick_radius: Radio en píxeles para seleccionar puntos
        """
        self.subpaths = subpaths
        self.samples_per_segment = samples_per_segment
        self.pick_radius = pick_radius

        # Estado de interacción
        self.selected_point: Optional[Tuple[int, int, int]] = None  # (subpath, curve, point)
        self.dragging = False

        # Elementos gráficos
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.curve_lines: List[plt.Line2D] = []
        self.control_lines: List[plt.Line2D] = []
        self.control_points: List[plt.PathCollection] = []
        self.fill_patch: Optional[PathPatch] = None

        # Conectar eventos
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        # Dibujo inicial
        self._setup_plot()
        self._draw_all()

    def _setup_plot(self) -> None:
        """Configura el gráfico inicial."""
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(
            "Editor Interactivo de Bézier\n"
            "Arrastra los puntos de control para modificar las curvas",
            fontsize=12,
        )
        self.ax.grid(True, alpha=0.3)

        # Instrucciones
        self.ax.text(
            0.02, 0.98,
            "• Click y arrastra puntos de control\n"
            "• Los cambios se aplican en tiempo real\n"
            "• Cierra la ventana para terminar",
            transform=self.ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    def _get_all_control_points(self) -> np.ndarray:
        """Obtiene todos los puntos de control como un array."""
        points = []
        for sp in self.subpaths:
            for cubic in sp.cubics:
                for pt in cubic:
                    points.append(pt)
        return np.array(points)

    def _find_nearest_point(self, x: float, y: float) -> Optional[Tuple[int, int, int]]:
        """
        Encuentra el punto de control más cercano a las coordenadas dadas.

        Returns:
            Tupla (subpath_idx, curve_idx, point_idx) o None si no hay punto cercano
        """
        min_dist = float("inf")
        nearest = None

        for si, sp in enumerate(self.subpaths):
            for ci, cubic in enumerate(sp.cubics):
                for pi, pt in enumerate(cubic):
                    dist = np.sqrt((pt[0] - x) ** 2 + (pt[1] - y) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (si, ci, pi)

        # Convertir distancia de datos a píxeles para comparar con pick_radius
        if nearest is not None:
            # Obtener la transformación de datos a display
            transform = self.ax.transData
            pt_data = self.subpaths[nearest[0]].cubics[nearest[1]][nearest[2]]
            pt_display = transform.transform(pt_data)
            click_display = transform.transform([x, y])
            dist_pixels = np.sqrt(
                (pt_display[0] - click_display[0]) ** 2
                + (pt_display[1] - click_display[1]) ** 2
            )
            if dist_pixels <= self.pick_radius:
                return nearest

        return None

    def _on_press(self, event) -> None:
        """Maneja el evento de click del mouse."""
        if event.inaxes != self.ax or event.button != 1:
            return

        self.selected_point = self._find_nearest_point(event.xdata, event.ydata)
        if self.selected_point is not None:
            self.dragging = True

    def _on_release(self, event) -> None:
        """Maneja el evento de soltar el mouse."""
        self.dragging = False
        self.selected_point = None

    def _on_motion(self, event) -> None:
        """Maneja el movimiento del mouse."""
        if not self.dragging or self.selected_point is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return

        si, ci, pi = self.selected_point

        # Actualizar posición del punto de control
        self.subpaths[si].cubics[ci][pi] = np.array([event.xdata, event.ydata])

        # Mantener continuidad C0: si movemos P0 o P3, actualizar curva adyacente
        n_curves = len(self.subpaths[si].cubics)
        if pi == 0:  # Movimos P0
            # Actualizar P3 de la curva anterior
            prev_ci = (ci - 1) % n_curves
            self.subpaths[si].cubics[prev_ci][3] = np.array([event.xdata, event.ydata])
        elif pi == 3:  # Movimos P3
            # Actualizar P0 de la curva siguiente
            next_ci = (ci + 1) % n_curves
            self.subpaths[si].cubics[next_ci][0] = np.array([event.xdata, event.ydata])

        # Redibujar
        self._draw_all()

    def _draw_all(self) -> None:
        """Redibuja todas las curvas y puntos de control."""
        # Limpiar elementos anteriores
        for line in self.curve_lines:
            line.remove()
        for line in self.control_lines:
            line.remove()
        for scatter in self.control_points:
            scatter.remove()
        if self.fill_patch is not None:
            self.fill_patch.remove()

        self.curve_lines = []
        self.control_lines = []
        self.control_points = []

        vertices: List[Tuple[float, float]] = []
        codes: List[int] = []

        colors = plt.cm.tab10.colors

        for si, sp in enumerate(self.subpaths):
            color = colors[si % len(colors)]
            polygon_points = []

            for ci, cubic in enumerate(sp.cubics):
                # Muestrear la curva
                curve_points = sample_bezier(cubic, n_samples=self.samples_per_segment)

                if ci != len(sp.cubics) - 1:
                    curve_points = curve_points[:-1]
                polygon_points.append(curve_points)

                # Dibujar líneas de control (polígono de control)
                (line,) = self.ax.plot(
                    cubic[:, 0],
                    cubic[:, 1],
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.6,
                    color="gray",
                )
                self.control_lines.append(line)

                # Dibujar puntos de control
                # P0 y P3 en un color, C1 y C2 en otro
                scatter_endpoints = self.ax.scatter(
                    cubic[[0, 3], 0],
                    cubic[[0, 3], 1],
                    s=60,
                    c=[color],
                    alpha=0.9,
                    zorder=10,
                    edgecolors="black",
                    linewidths=1,
                )
                scatter_controls = self.ax.scatter(
                    cubic[[1, 2], 0],
                    cubic[[1, 2], 1],
                    s=40,
                    c="orange",
                    alpha=0.9,
                    zorder=10,
                    marker="s",
                    edgecolors="black",
                    linewidths=0.5,
                )
                self.control_points.extend([scatter_endpoints, scatter_controls])

            if polygon_points:
                poly = np.vstack(polygon_points)
                (curve_line,) = self.ax.plot(
                    poly[:, 0], poly[:, 1], linewidth=2.0, color=color
                )
                self.curve_lines.append(curve_line)

                # Construir path para relleno
                vertices.append((poly[0, 0], poly[0, 1]))
                codes.append(Path.MOVETO)
                for k in range(1, poly.shape[0]):
                    vertices.append((poly[k, 0], poly[k, 1]))
                    codes.append(Path.LINETO)
                vertices.append((poly[0, 0], poly[0, 1]))
                codes.append(Path.CLOSEPOLY)

        # Aplicar relleno
        if vertices:
            path = Path(vertices, codes)
            self.fill_patch = PathPatch(
                path, alpha=0.15, facecolor="blue", edgecolor="none"
            )
            try:
                self.fill_patch.set_fillrule("evenodd")
            except AttributeError:
                pass
            self.ax.add_patch(self.fill_patch)

        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Muestra el editor interactivo."""
        plt.tight_layout()
        plt.show()

    def get_modified_subpaths(self) -> List[Subpath]:
        """Retorna los subpaths con las modificaciones aplicadas."""
        return self.subpaths


# ============================================================================
# Visualización estática (del código original)
# ============================================================================


def plot_logo(
    subpaths: List[Subpath],
    title: str,
    samples_per_segment: int = 220,
    show_fill: bool = True,
    show_control_points: bool = True,
) -> None:
    """
    Dibuja el logo usando curvas Bézier.

    Args:
        subpaths: Lista de subpaths con curvas Bézier
        title: Título del gráfico
        samples_per_segment: Puntos por segmento Bézier
        show_fill: Si mostrar el relleno
        show_control_points: Si mostrar los puntos de control
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    vertices: List[Tuple[float, float]] = []
    codes: List[int] = []

    for sp in subpaths:
        polygon_points = []

        for j, cubic in enumerate(sp.cubics):
            # Muestrear la curva
            curve_points = sample_bezier(cubic, n_samples=samples_per_segment)

            # Evitar duplicar el último punto excepto en el último segmento
            if j != len(sp.cubics) - 1:
                curve_points = curve_points[:-1]
            polygon_points.append(curve_points)

            # Dibujar puntos de control
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

            # Construir path para relleno
            vertices.append((poly[0, 0], poly[0, 1]))
            codes.append(Path.MOVETO)
            for k in range(1, poly.shape[0]):
                vertices.append((poly[k, 0], poly[k, 1]))
                codes.append(Path.LINETO)
            vertices.append((poly[0, 0], poly[0, 1]))
            codes.append(Path.CLOSEPOLY)

    # Aplicar relleno
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


def plot_comparison(
    original: List[Subpath],
    modified: List[Subpath],
    samples_per_segment: int = 220,
) -> None:
    """Dibuja comparación entre logo original y modificado."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Dibujar original (línea continua)
    for sp in original:
        points = []
        for j, cubic in enumerate(sp.cubics):
            pts = sample_bezier(cubic, n_samples=samples_per_segment)
            if j != len(sp.cubics) - 1:
                pts = pts[:-1]
            points.append(pts)
        if points:
            poly = np.vstack(points)
            ax.plot(poly[:, 0], poly[:, 1], linewidth=2.0, color="blue", label="Original")

    # Dibujar modificado (línea discontinua)
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
                color="red",
                label="Modificado",
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Comparativa: original (azul) vs modificado (rojo discontinuo)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# Demostración interactiva
# ============================================================================


def run_interactive_demo(samples_per_segment: int = 100) -> None:
    """
    Ejecuta la demostración interactiva del logo con curvas Bézier.
    
    Permite arrastrar puntos de control y ver el impacto en tiempo real.
    Al cerrar la ventana, muestra una comparación con el original.
    """
    # Obtener subpaths originales (para comparación)
    original_subpaths = get_logo_subpaths()

    # Crear copia para edición interactiva
    editable_subpaths = get_logo_subpaths()

    print("=" * 60)
    print("EDITOR INTERACTIVO DE CURVAS BÉZIER")
    print("=" * 60)
    print("• Arrastra cualquier punto de control con el mouse")
    print("• Los puntos naranjas (cuadrados) son C1 y C2")
    print("• Los puntos de colores (círculos) son P0 y P3")
    print("• La curva se actualiza en tiempo real")
    print("• Cierra la ventana cuando termines para ver la comparación")
    print("=" * 60)

    # Ejecutar editor interactivo
    editor = InteractiveBezierEditor(
        editable_subpaths,
        samples_per_segment=samples_per_segment,
    )
    editor.show()

    # Obtener los subpaths modificados
    modified_subpaths = editor.get_modified_subpaths()

    # Mostrar comparación final
    print("\nMostrando comparación final...")
    plot_comparison(original_subpaths, modified_subpaths, samples_per_segment=200)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Editor INTERACTIVO de logo mediante curvas Bézier cúbicas"
    )
    parser.add_argument(
        "--samples-per-seg",
        type=int,
        default=100,
        help="Puntos de muestreo por segmento Bézier (default: 100)",
    )
    args = parser.parse_args()

    run_interactive_demo(samples_per_segment=max(30, args.samples_per_seg))


if __name__ == "__main__":
    main()
