# practica_IGM_MC_Bezier.py
# Requisitos: numpy, pandas, matplotlib
# Ejecución ejemplo:
#   python practica_IGM_MC_Bezier.py --N 800000 --seed 123 --box cube --torus-form sqrt
#   python practica_IGM_MC_Bezier.py --N 800000 --seed 123 --box tight --torus-form poly

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Box3D:
    low: np.ndarray  # (3,)
    high: np.ndarray  # (3,)

    @property
    def volume(self) -> float:
        d = self.high - self.low
        return float(d[0] * d[1] * d[2])


def make_box(name: str) -> Box3D:
    name = name.lower().strip()
    if name == "cube":
        return Box3D(low=np.array([-3.0, -3.0, -3.0]), high=np.array([3.0, 3.0, 3.0]))
    if name == "tight":
        # Caja más ajustada (opcional) que contiene toroide (R=1.5,r=0.5) y esferas en x=±2 con rs=0.5:
        # x ∈ [-2.5, 2.5], y ∈ [-2.0, 2.0], z ∈ [-0.5, 0.5]
        return Box3D(low=np.array([-2.5, -2.0, -0.5]), high=np.array([2.5, 2.0, 0.5]))
    raise ValueError("box debe ser 'cube' o 'tight'")


def sample_uniform_in_box(rng: np.random.Generator, box: Box3D, n: int) -> np.ndarray:
    # Uniforme en caja axis-aligned: X = low + U*(high-low), U ~ U([0,1]^3)
    u = rng.random((n, 3), dtype=float)
    return box.low + u * (box.high - box.low)


def inside_toroid_sqrt(pts: np.ndarray, R: float, r: float) -> np.ndarray:
    # Fórmula del enunciado:
    # (sqrt(x^2+y^2) - R)^2 + z^2 <= r^2
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    return (np.sqrt(x * x + y * y) - R) ** 2 + z * z <= r * r


def inside_toroid_poly(pts: np.ndarray, R: float, r: float) -> np.ndarray:
    # Forma equivalente sin raíz (enunciado):
    # (x^2 + y^2 + z^2 + R^2 - r^2)^2 <= 4 R^2 (x^2 + y^2)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    lhs = (x * x + y * y + z * z + R * R - r * r) ** 2
    rhs = 4.0 * (R * R) * (x * x + y * y)
    return lhs <= rhs


def inside_sphere_S1(pts: np.ndarray, rs: float) -> np.ndarray:
    # (x-2)^2 + y^2 + z^2 <= rs^2
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    return (x - 2.0) ** 2 + y * y + z * z <= rs * rs


def inside_sphere_S2(pts: np.ndarray, rs: float) -> np.ndarray:
    # (x+2)^2 + y^2 + z^2 <= rs^2
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    return (x + 2.0) ** 2 + y * y + z * z <= rs * rs


def exact_volumes(R: float, r: float, rs: float) -> Dict[str, float]:
    # V_T exacto = 2 π^2 R r^2 ; V_S exacto = (4/3) π rs^3
    Vt = 2.0 * (math.pi**2) * R * (r**2)
    Vs = (4.0 / 3.0) * math.pi * (rs**3)
    return {"V_T_exact": Vt, "V_S_exact": Vs}


def volume_mc(Vbox: float, count: int, N: int) -> float:
    return Vbox * (count / N)


def se_mc(Vbox: float, count: int, N: int) -> float:
    # Error estándar del estimador Bernoulli: Vbox * sqrt(p(1-p)/N), con p = count/N
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
    plot_convergence: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    box = make_box(box_name)
    Vbox = box.volume

    torus_form = torus_form.lower().strip()
    if torus_form not in {"sqrt", "poly"}:
        raise ValueError("torus-form debe ser 'sqrt' o 'poly'")

    c_T = 0
    c_S1 = 0
    c_S2 = 0
    c_TS1 = 0
    c_TS2 = 0
    c_union = 0

    xs = []
    vt_series = []
    vs1_series = []
    vunion_series = []

    done = 0
    while done < N:
        n = min(batch, N - done)
        pts = sample_uniform_in_box(rng, box, n)

        if torus_form == "sqrt":
            mT = inside_toroid_sqrt(pts, R=R, r=r)
        else:
            mT = inside_toroid_poly(pts, R=R, r=r)

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

        if plot_convergence:
            xs.append(done)
            vt_series.append(volume_mc(Vbox, c_T, done))
            vs1_series.append(volume_mc(Vbox, c_S1, done))
            vunion_series.append(volume_mc(Vbox, c_union, done))

    ev = exact_volumes(R=R, r=r, rs=rs)
    Vt_exact = ev["V_T_exact"]
    Vs_exact = ev["V_S_exact"]

    VT = volume_mc(Vbox, c_T, N)
    VS1 = volume_mc(Vbox, c_S1, N)
    VS2 = volume_mc(Vbox, c_S2, N)
    VTS1 = volume_mc(Vbox, c_TS1, N)
    VTS2 = volume_mc(Vbox, c_TS2, N)
    VU = volume_mc(Vbox, c_union, N)

    rows = [
        {"objeto": "T (toroide sólido)", "V_hat": VT, "V_exact": Vt_exact},
        {"objeto": "S1 (esfera derecha)", "V_hat": VS1, "V_exact": Vs_exact},
        {"objeto": "S2 (esfera izquierda)", "V_hat": VS2, "V_exact": Vs_exact},
        {"objeto": "T∩S1", "V_hat": VTS1, "V_exact": np.nan},
        {"objeto": "T∩S2", "V_hat": VTS2, "V_exact": np.nan},
        {"objeto": "T∪S1∪S2", "V_hat": VU, "V_exact": np.nan},
    ]

    df = pd.DataFrame(rows)
    df["N"] = N
    df["box"] = box_name
    df["V_box"] = Vbox
    df["abs_error"] = df["V_hat"] - df["V_exact"]
    df["rel_error"] = df["abs_error"] / df["V_exact"]
    df.loc[df["V_exact"].isna(), ["abs_error", "rel_error"]] = np.nan

    ses = [
        se_mc(Vbox, c_T, N),
        se_mc(Vbox, c_S1, N),
        se_mc(Vbox, c_S2, N),
        se_mc(Vbox, c_TS1, N),
        se_mc(Vbox, c_TS2, N),
        se_mc(Vbox, c_union, N),
    ]
    df["SE"] = ses
    df["CI95_low"] = df["V_hat"] - 1.96 * df["SE"]
    df["CI95_high"] = df["V_hat"] + 1.96 * df["SE"]

    Vincl_excl = VT + VS1 + VS2 - VTS1 - VTS2
    Vs = 0.5 * (VS1 + VS2)
    Vinters = 0.5 * (VTS1 + VTS2)
    Vsym = VT + 2.0 * Vs - 2.0 * Vinters

    print("\nResultados Monte Carlo (con fórmulas del enunciado):")
    print(df.to_string(index=False, float_format=lambda x: f"{x: .6g}"))

    print("\nComprobaciones pedidas:")
    print(f"VS1 - VS2 = {VS1 - VS2: .6g}")
    print(f"V(T∩S1) - V(T∩S2) = {VTS1 - VTS2: .6g}")
    print(f"Vunión - (VT + VS1 + VS2 - V(T∩S1) - V(T∩S2)) = {VU - Vincl_excl: .6g}")
    print(f"Vunión - (VT + 2VS - 2Vinters) = {VU - Vsym: .6g}")

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
        ax.set_title(
            f"Convergencia por lotes (box={box_name}, torus-form={torus_form}, N={N})"
        )
        ax.legend()
        ax.grid(True)
        plt.show()

    return df


def de_casteljau(ctrl: np.ndarray, t: float) -> np.ndarray:
    p = np.array(ctrl, dtype=float, copy=True)
    m = p.shape[0]
    for k in range(1, m):
        p[: m - k] = (1.0 - t) * p[: m - k] + t * p[1 : m - k + 1]
    return p[0]


def sample_bezier(ctrl: np.ndarray, n: int = 250) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n, dtype=float)
    return np.vstack([de_casteljau(ctrl, float(t)) for t in ts])


def make_letter_S_segments() -> List[np.ndarray]:
    # Spline a trozos con Bézier cúbicas (4 puntos de control por segmento).
    # Se fuerza continuidad C0 y se sugiere continuidad C1 alineando tangentes en uniones.
    segs: List[np.ndarray] = []

    p0 = np.array([-1.0, 1.2])
    p1 = np.array([-0.2, 1.6])
    p2 = np.array([0.9, 1.3])
    p3 = np.array([0.6, 0.6])
    segs.append(np.vstack([p0, p1, p2, p3]))

    q0 = p3
    q1 = q0 + (q0 - p2)
    q2 = np.array([-0.7, 0.6])
    q3 = np.array([-0.5, 0.1])
    segs.append(np.vstack([q0, q1, q2, q3]))

    r0 = q3
    r1 = r0 + (r0 - q2)
    r2 = np.array([0.8, 0.0])
    r3 = np.array([0.5, -0.6])
    segs.append(np.vstack([r0, r1, r2, r3]))

    s0 = r3
    s1 = s0 + (s0 - r2)
    s2 = np.array([-0.6, -1.0])
    s3 = np.array([-1.0, -1.3])
    segs.append(np.vstack([s0, s1, s2, s3]))

    return segs


def plot_spline_with_controls(segments: List[np.ndarray], title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, ctrl in enumerate(segments):
        curve = sample_bezier(ctrl, n=300)
        ax.plot(curve[:, 0], curve[:, 1], label=f"trazo {i + 1}")

        ax.plot(ctrl[:, 0], ctrl[:, 1], linestyle=":", linewidth=1.0)
        ax.scatter(ctrl[:, 0], ctrl[:, 1], s=25)

    # Resaltado “cerca de los extremos” en uniones consecutivas:
    # Para cúbica: ... P2-P3 | Q0-Q1 ..., donde P3=Q0 en continuidad C0.
    for i in range(len(segments) - 1):
        A = segments[i]
        B = segments[i + 1]
        pts = np.vstack([A[2], A[3], B[1]])  # P2, P3(=Q0), Q1
        ax.plot(pts[:, 0], pts[:, 1], linewidth=2.5)
        ax.scatter(pts[:, 0], pts[:, 1], s=45)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.show()


def run_ex2() -> None:
    segs = make_letter_S_segments()
    plot_spline_with_controls(
        segs, "Spline Bézier (letra S) con puntos de control y uniones"
    )

    # Impacto de cambios en puntos de control: se modifica un control interior de un trazo intermedio
    segs_mod = [s.copy() for s in segs]
    segs_mod[1][2] = segs_mod[1][2] + np.array(
        [-0.25, 0.35]
    )  # modifica Q2 del segundo trazo
    plot_spline_with_controls(
        segs_mod, "Spline Bézier tras modificar un punto de control (impacto)"
    )

    # Superposición para visualizar cambio de forma de manera inmediata
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ctrl in segs:
        c = sample_bezier(ctrl, n=300)
        ax.plot(c[:, 0], c[:, 1])
    for ctrl in segs_mod:
        c = sample_bezier(ctrl, n=300)
        ax.plot(c[:, 0], c[:, 1], linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Comparativa: original (línea continua) vs modificada (línea discontinua)"
    )
    ax.grid(True)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=600000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--box", type=str, choices=["cube", "tight"], default="cube")
    parser.add_argument(
        "--torus-form",
        type=str,
        choices=["sqrt", "poly"],
        default="sqrt",
        help="Usar fórmula del enunciado con raíz (sqrt) o la equivalente sin raíz (poly).",
    )
    parser.add_argument("--batch", type=int, default=100000)
    parser.add_argument("--skip-mc", action="store_true")
    parser.add_argument("--skip-bezier", action="store_true")
    args = parser.parse_args()

    if not args.skip_mc:
        run_ex1(
            N=args.N,
            seed=args.seed,
            box_name=args.box,
            torus_form=args.torus_form,
            batch=max(1, args.batch),
            plot_convergence=True,
        )

    if not args.skip_bezier:
        run_ex2()


if __name__ == "__main__":
    main()
