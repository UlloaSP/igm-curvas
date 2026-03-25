from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Box3D:
    low: np.ndarray
    high: np.ndarray

    @property
    def volume(self) -> float:
        d = self.high - self.low
        return float(d[0] * d[1] * d[2])


def make_box() -> Box3D:
    return Box3D(low=np.array([-3.0, -3.0, -3.0]), high=np.array([3.0, 3.0, 3.0]))


def sample_uniform_in_box(rng: np.random.Generator, box: Box3D, n: int) -> np.ndarray:
    # Muestreo uniforme en una caja alineada con los ejes.
    u = rng.random((n, 3), dtype=float)
    return box.low + u * (box.high - box.low)


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
    # Error estandar del estimador Bernoulli Vbox * sqrt(p(1-p)/N).
    p = count / N
    return Vbox * math.sqrt(p * (1.0 - p) / N)


def run_ex1(
    N: int, seed: int, batch: int, R: float = 1.5, r: float = 0.5, rs: float = 0.5
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    box = make_box()
    Vbox = box.volume

    c_T = 0
    c_S1 = 0
    c_S2 = 0
    c_TS1 = 0
    c_TS2 = 0
    c_union = 0

    done = 0
    while done < N:
        n = min(batch, N - done)
        pts = sample_uniform_in_box(rng, box, n)

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
        {"objeto": "T (toroide solido)", "V_hat": VT, "V_exact": Vt_exact},
        {"objeto": "S1 (esfera derecha)", "V_hat": VS1, "V_exact": Vs_exact},
        {"objeto": "S2 (esfera izquierda)", "V_hat": VS2, "V_exact": Vs_exact},
        {"objeto": "T inter S1", "V_hat": VTS1, "V_exact": np.nan},
        {"objeto": "T inter S2", "V_hat": VTS2, "V_exact": np.nan},
        {"objeto": "T union S1 union S2", "V_hat": VU, "V_exact": np.nan},
    ]

    df = pd.DataFrame(rows)
    df["N"] = N
    df["box"] = "cube"
    df["V_box"] = Vbox
    df["abs_error"] = (df["V_hat"] - df["V_exact"]).abs()
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

    print("\nResultados Monte Carlo:")
    print(df.to_string(index=False, float_format=lambda x: f"{x: .6g}"))

    print("\nComprobaciones de consistencia:")
    print(f"VS1 - VS2 = {VS1 - VS2: .6g}")
    print(f"V(T inter S1) - V(T inter S2) = {VTS1 - VTS2: .6g}")
    print(
        f"Vunion - (VT + VS1 + VS2 - V(T inter S1) - V(T inter S2)) = {VU - Vincl_excl: .6g}"
    )
    print(f"Vunion - (VT + 2VS - 2Vinters) = {VU - Vsym: .6g}")

    return df


def plot_torus_and_spheres(
    R: float = 1.5,
    r: float = 0.5,
    rs: float = 0.5,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Visualiza el toro y las dos esferas S1 y S2 en 3D."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, 2 * np.pi, 60)
    u, v = np.meshgrid(u, v)

    x_torus = (R + r * np.cos(v)) * np.cos(u)
    y_torus = (R + r * np.cos(v)) * np.sin(u)
    z_torus = r * np.sin(v)

    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)

    x_s1 = rs * np.sin(phi) * np.cos(theta) + 2.0
    y_s1 = rs * np.sin(phi) * np.sin(theta)
    z_s1 = rs * np.cos(phi)

    x_s2 = rs * np.sin(phi) * np.cos(theta) - 2.0
    y_s2 = rs * np.sin(phi) * np.sin(theta)
    z_s2 = rs * np.cos(phi)

    ax.plot_surface(x_torus, y_torus, z_torus, alpha=0.6, color="blue", label="Toro")
    ax.plot_surface(x_s1, y_s1, z_s1, alpha=0.7, color="red")
    ax.plot_surface(x_s2, y_s2, z_s2, alpha=0.7, color="green")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Toro (R={R}, r={r}) y Esferas S1, S2 (rs={rs})")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", alpha=0.6, label="Toro"),
        Patch(facecolor="red", alpha=0.7, label="S1 (x=2)"),
        Patch(facecolor="green", alpha=0.7, label="S2 (x=-2)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_box_aspect([1, 1, 0.3])

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=600000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch", type=int, default=100000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio donde guardar la figura generada",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir la ventana de Matplotlib",
    )
    args = parser.parse_args()

    run_ex1(N=args.N, seed=args.seed, batch=max(1, args.batch))

    output_path = None
    if args.output_dir is not None:
        output_path = args.output_dir / "ex1_toro_y_esferas.png"

    plot_torus_and_spheres(output_path=output_path, show=not args.no_show)


if __name__ == "__main__":
    main()
