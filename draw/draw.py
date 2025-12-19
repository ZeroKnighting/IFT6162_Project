import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from dataclasses import dataclass
from typing import Optional


def load_trajs_npz(path: Path):
    z = np.load(path, allow_pickle=False)

    # meta
    meta = {}
    for k in z.files:
        if k.startswith("meta__"):
            meta[k.replace("meta__", "")] = z[k].item() if z[k].shape == () else z[k]

    # trajs
    trajs = {}
    prefixes = sorted({k.split("__")[0] for k in z.files if "__" in k and not k.startswith("meta__")})

    for p in prefixes:
        qspill = z[f"{p}__qspill_cfs"]
        trajs[p] = {
            "h_ft": z[f"{p}__h_ft"],
            "c": z[f"{p}__c"],
            "qout_cfs": z[f"{p}__qout_cfs"],
            "theta": z[f"{p}__theta"],
            "qspill_cfs": None if qspill.size == 0 else qspill,
        }

    return trajs, meta


@dataclass
class ControllerTraj:
    h_ft: np.ndarray
    c: np.ndarray
    qout_cfs: np.ndarray
    theta: np.ndarray
    qspill_cfs: Optional[np.ndarray] = None


def load_trajs_npz_as_dataclass(path: Path):
    raw_trajs, meta = load_trajs_npz(path)
    trajs = {}
    for k, d in raw_trajs.items():
        trajs[k] = ControllerTraj(**d)
    return trajs, meta

def _get_traj(trajs: dict, name: str):
    """Allow both 'Stochastic MPC' and 'Stochastic_MPC' / '-' style keys."""
    if name in trajs:
        return trajs[name]
    alt = name.replace(" ", "_").replace("-", "_")
    if alt in trajs:
        return trajs[alt]
    # try reverse
    for k in trajs.keys():
        if k.replace(" ", "_").replace("-", "_") == alt:
            return trajs[k]
    raise KeyError(f"Missing traj '{name}'. Available keys: {list(trajs.keys())}")


def _read_rain_numeric_col(csv_path: Path) -> np.ndarray:
    """
    Robust rainfall reader:
    - CSV may have 'Time' column or headers
    - we pick the column with the most numeric values
    """
    rain = pd.read_csv(csv_path)
    rain_num = rain.apply(pd.to_numeric, errors="coerce")
    best_col = rain_num.notna().sum().idxmax()
    prep = rain_num[best_col].to_numpy(dtype=float)
    prep = np.nan_to_num(prep, nan=0.0)
    return prep


def _cum_load_kg(c_mgL: np.ndarray, q_m3s: np.ndarray, dt_s: float) -> np.ndarray:
    # kg per step: c(mg/L)*q(m3/s)*dt(s)*1e-3
    return np.cumsum(c_mgL * q_m3s * dt_s * 1e-3)


def apply_matlab_like_style(font_scale: float = 0.75):
    import matplotlib as mpl

    base = 20 * font_scale
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,

        "font.size": base,
        "axes.labelsize": base * 1.05,
        "axes.titlesize": base * 1.05,
        "xtick.labelsize": base * 0.90,
        "ytick.labelsize": base * 0.90,
        "legend.fontsize": base * 0.95,

        "axes.linewidth": 1.0,
    })


def plot_benchmark_9x3_event_with_inset(
    out_png: Path,
    trajs: dict,
    qin_t: np.ndarray,
    cin_t: np.ndarray,
    cin_f: np.ndarray,
    rainfall_csv: Path,
    Ts: float = 1.0,
    start: int = 25220,
    Nstep: int = 26275,
    Tfull: int = 34000,
    hlimit_m: float = 3.0,
    inset_src=(25300, 25550, 2.85, 3.45),
    inset_dst=(25900, 26250, 1.70, 2.90),
):
    apply_matlab_like_style(font_scale=0.65)

    # --- conversions (your python trajs are ft/cfs; MATLAB plot is m/m3s) ---
    ft_to_m = 1.0 / 3.281
    cfs_to_m3s = 0.028316846592
    dt_s = 15 * 60  # 900s

    passive = _get_traj(trajs, "Passive")
    smpc    = _get_traj(trajs, "Stochastic MPC")
    rbc_c   = _get_traj(trajs, "RBC-Concentration")
    rbc_q   = _get_traj(trajs, "RBC-Outflow")
    rbc_b   = _get_traj(trajs, "RBC-Both")

    lao_tv  = _get_traj(trajs, "LAO_terminal_value_MPC")
    lao_pr  = _get_traj(trajs, "LAO_pruned_MPC")

    prep0 = _read_rain_numeric_col(rainfall_csv)
    prep = np.concatenate([np.zeros(48, dtype=float), prep0.astype(float)])
    if prep.size < Tfull:
        prep = np.pad(prep, (0, Tfull - prep.size), constant_values=0.0)
    else:
        prep = prep[:Tfull]

    def _clip(arr, n): return np.asarray(arr)[:n]
    qin_t = _clip(qin_t, Tfull)
    cin_t = _clip(cin_t, Tfull)
    cin_f = _clip(cin_f, Tfull)

    h_p = _clip(passive.h_ft, Tfull) * ft_to_m
    h_s = _clip(smpc.h_ft,    Tfull) * ft_to_m
    h_c = _clip(rbc_c.h_ft,   Tfull) * ft_to_m
    h_q = _clip(rbc_q.h_ft,   Tfull) * ft_to_m
    h_b = _clip(rbc_b.h_ft,   Tfull) * ft_to_m
    h_tv = _clip(lao_tv.h_ft, Tfull) * ft_to_m
    h_pr = _clip(lao_pr.h_ft, Tfull) * ft_to_m

    q_p = _clip(passive.qout_cfs, Tfull) * cfs_to_m3s
    q_s = _clip(smpc.qout_cfs,    Tfull) * cfs_to_m3s
    q_c = _clip(rbc_c.qout_cfs,   Tfull) * cfs_to_m3s
    q_q = _clip(rbc_q.qout_cfs,   Tfull) * cfs_to_m3s
    q_b = _clip(rbc_b.qout_cfs,   Tfull) * cfs_to_m3s
    q_tv = _clip(lao_tv.qout_cfs, Tfull) * cfs_to_m3s
    q_pr = _clip(lao_pr.qout_cfs, Tfull) * cfs_to_m3s

    C_p = _clip(passive.c, Tfull)
    C_s = _clip(smpc.c,    Tfull)
    C_c = _clip(rbc_c.c,   Tfull)
    C_q = _clip(rbc_q.c,   Tfull)
    C_b = _clip(rbc_b.c,   Tfull)
    C_tv = _clip(lao_tv.c, Tfull)
    C_pr = _clip(lao_pr.c, Tfull)

    th_p = np.ones(Tfull, dtype=float)
    th_s = _clip(smpc.theta,  Tfull)
    th_c = _clip(rbc_c.theta, Tfull)
    th_q = _clip(rbc_q.theta, Tfull)
    th_b = _clip(rbc_b.theta, Tfull)
    th_tv = _clip(lao_tv.theta, Tfull)
    th_pr = _clip(lao_pr.theta, Tfull)

    def _spill_m3s(tr):
        if getattr(tr, "qspill_cfs", None) is None or tr.qspill_cfs is None:
            return np.zeros(Tfull, dtype=float)
        x = _clip(tr.qspill_cfs, Tfull) * cfs_to_m3s
        return x

    sp_c = _spill_m3s(rbc_c)
    sp_q = _spill_m3s(rbc_q)
    sp_b = _spill_m3s(rbc_b)
    sp_tv = _spill_m3s(lao_tv)
    sp_pr = _spill_m3s(lao_pr)

    x_full = (np.arange(1, Tfull + 1) * Ts).astype(float)
    x_idx  = np.arange(1, Tfull + 1)

    col_passive = "black"
    col_blue    = "#0072BD"   # Stochastic MPC
    col_yellow  = "#EDB120"
    col_orange  = "#D95319"
    col_green   = "#77AC30"
    col_inflow  = "#0000a7"
    col_truepol = "#c1272d"
    col_falsepol= "#7E2F8E"

    col_lao_tv  = "#4DBEEE"   
    col_lao_pr  = "#7E2F8E"  

    fig = plt.figure(figsize=(15, 9), dpi=150)
    height_ratios = [0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    gs = GridSpec(
        9, 3, figure=fig,
        wspace=0.42, hspace=0.6,
        height_ratios=height_ratios
    )

    charlbl = [f"({chr(ord('a') + i)})" for i in range(8)]

    axA = fig.add_subplot(gs[0, :])
    axA.plot(x_full, h_p, "-",  color=col_passive, linewidth=2)
    axA.plot(x_full, np.full_like(x_full, hlimit_m), "--", color=(0.7, 0.7, 0.7), linewidth=2)
    axA.plot(x_full, h_c, "-.", color=col_yellow, linewidth=2)
    axA.plot(x_full, h_q, ":",  color=col_orange, linewidth=2)
    axA.plot(x_full, h_b, "--", color=col_green, linewidth=2)
    axA.plot(x_full, h_s, "-",  color=col_blue, linewidth=3)
    axA.plot(x_full, h_tv, "-", color=col_lao_tv, linewidth=3)
    axA.plot(x_full, h_pr, "-", color=col_lao_pr, linewidth=3)
    axA.axvspan(0, start * Ts, color="0.7", alpha=0.5)
    axA.axvspan(Nstep * Ts, Tfull * Ts, color="0.7", alpha=0.5)
    axA.set_ylim(0, 1.2 * hlimit_m)
    axA.set_xlim(1 * Ts, Tfull * Ts)
    month_ticks = np.array([20, 2929, 5617, 8593, 11473, 14449, 17329, 20305, 23281, 26161, 29137, 32017], dtype=float) * Ts
    axA.set_xticks(month_ticks)
    axA.set_xticklabels(["Jan.","Feb.","Mar.","Apr.","May","Jun.","Jul.","Aug.","Sep.","Oct.","Nov.","Dec."])
    axA.text(0.005, 0.80, charlbl[0], transform=axA.transAxes)

    axB = fig.add_subplot(gs[1:3, 0])
    axB.plot(x_idx[:Nstep], qin_t[:Nstep], "-", color=col_inflow, linewidth=3)
    axB.set_ylabel(r"Inflow (m$^{3}$/s)")
    axB.set_ylim(0, 2)
    axB.set_xlim(start, Nstep)
    axB.tick_params(labelbottom=False)
    axB.text(0.02, 0.90, charlbl[1], transform=axB.transAxes)

    axB2 = axB.twinx()
    axB2.spines["right"].set_position(("axes", 1.08))
    axB2.yaxis.set_label_coords(1.16, 0.5)
    axB2.tick_params(axis="y", pad=2, colors="blue")

    bar_x = np.arange(start, Nstep + 1)
    bar_y = prep[start-1:Nstep] * 25.4 * 4.0
    axB2.bar(bar_x, bar_y, width=1.0, color="blue", align="center")
    axB2.set_ylabel("Precipitation (mm/hr)", color="blue", labelpad=6)
    axB2.tick_params(axis="y", colors="blue", labelsize=plt.rcParams["ytick.labelsize"]*0.9)
    ymax = 2 * float(np.max(bar_y) if np.max(bar_y) > 0 else 1.0)
    axB2.set_ylim(0, ymax)
    axB2.invert_yaxis()

    axC = fig.add_subplot(gs[3:5, 0])
    axC.plot(x_idx[:Nstep], cin_t[:Nstep], "-", color=col_truepol, linewidth=3, label="True pollutograph")
    axC.plot(np.arange(start, Nstep + 1), cin_f[start-1:Nstep], ":", color=col_falsepol, linewidth=3, label="False pollutograph")
    axC.set_ylabel("TSS (mg/L)")
    axC.set_xlim(start, Nstep)
    axC.set_ylim(0, 1.5 * float(np.max(cin_t[:Nstep]) if np.max(cin_t[:Nstep]) > 0 else 1.0))
    axC.tick_params(labelbottom=False)
    axC.legend(frameon=False)
    axC.text(0.02, 0.90, charlbl[2], transform=axC.transAxes)

    axD = fig.add_subplot(gs[1:5, 1])
    sl = slice(start-1, Nstep)
    x_zoom = (np.arange(start, Nstep + 1) * Ts).astype(float)

    axD.plot(x_zoom, h_p[sl], "-",  color=col_passive, linewidth=4)
    axD.plot(x_zoom, np.full_like(x_zoom, hlimit_m), "--", color=(0.7,0.7,0.7), linewidth=4)
    axD.plot(x_zoom, h_c[sl], "-.", color=col_yellow, linewidth=4)
    axD.plot(x_zoom, h_q[sl], ":",  color=col_orange, linewidth=4)
    axD.plot(x_zoom, h_b[sl], "--", color=col_green, linewidth=4)
    axD.plot(x_zoom, h_s[sl], "-",  color=col_blue, linewidth=5)
    axD.plot(x_zoom, h_tv[sl], "-", color=col_lao_tv, linewidth=4)
    axD.plot(x_zoom, h_pr[sl], "-", color=col_lao_pr, linewidth=4)
    axD.set_ylabel("Pond height (m)")
    axD.set_ylim(0, 1.25 * hlimit_m)
    axD.set_xlim(start * Ts, Nstep * Ts)
    axD.tick_params(labelbottom=False)
    hmax = axD.plot(x_zoom, np.full_like(x_zoom, hlimit_m),
                    "--", color=(0.7,0.7,0.7), linewidth=4,
                    label="Maximum allowable height")[0]
    axD.legend(handles=[hmax], frameon=False, loc="upper right")
    axD.text(0.02, 0.95, charlbl[3], transform=axD.transAxes)

    sx1, sx2, sy1, sy2 = inset_src
    dx1, dx2, dy1, dy2 = inset_dst

    xmin, xmax = axD.get_xlim()
    ymin, ymax = axD.get_ylim()
    fx1 = (dx1*Ts - xmin) / (xmax - xmin)
    fx2 = (dx2*Ts - xmin) / (xmax - xmin)
    fy1 = (dy1   - ymin) / (ymax - ymin)
    fy2 = (dy2   - ymin) / (ymax - ymin)
    fx1, fx2 = np.clip([fx1, fx2], 0.02, 0.98)
    fy1, fy2 = np.clip([fy1, fy2], 0.02, 0.98)

    axins = inset_axes(
        axD,
        width="100%",
        height="100%",
        bbox_to_anchor=(fx1, fy1, max(0.08, fx2-fx1), max(0.08, fy2-fy1)),
        bbox_transform=axD.transAxes,
        loc="lower left",
        borderpad=0.2,
    )

    axins.plot(x_zoom, h_p[sl], "-",  color=col_passive, linewidth=3)
    axins.plot(x_zoom, np.full_like(x_zoom, hlimit_m), "--", color=(0.7,0.7,0.7), linewidth=3)
    axins.plot(x_zoom, h_c[sl], "-.", color=col_yellow, linewidth=3)
    axins.plot(x_zoom, h_q[sl], ":",  color=col_orange, linewidth=3)
    axins.plot(x_zoom, h_b[sl], "--", color=col_green, linewidth=3)
    axins.plot(x_zoom, h_s[sl], "-",  color=col_blue, linewidth=4)
    axins.plot(x_zoom, h_tv[sl], "-", color=col_lao_tv, linewidth=3)
    axins.plot(x_zoom, h_pr[sl], "-", color=col_lao_pr, linewidth=3)

    axins.set_xlim(sx1 * Ts, sx2 * Ts)
    axins.set_ylim(sy1, sy2)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(axD, axins, loc1=1, loc2=3, fc="none", ec="0.2", lw=1.5)

    axE = fig.add_subplot(gs[1:5, 2])
    axE.plot(x_zoom, q_c[sl], "-.", color=col_yellow, linewidth=3)
    axE.plot(x_zoom, q_p[sl], "-",  color=col_passive, linewidth=4)
    axE.plot(x_zoom, q_b[sl], "--", color=col_green, linewidth=3)
    axE.plot(x_zoom, q_q[sl], ":",  color=col_orange, linewidth=4)
    axE.plot(x_zoom, q_s[sl], "-",  color=col_blue, linewidth=5)
    axE.plot(x_zoom, q_tv[sl], "-", color=col_lao_tv, linewidth=4)
    axE.plot(x_zoom, q_pr[sl], "-", color=col_lao_pr, linewidth=4)
    axE.set_ylabel(r"Outflow (m$^{3}$/s)")
    axE.set_xlim(start * Ts, Nstep * Ts)
    axE.tick_params(labelbottom=False)
    axE.text(0.02, 0.95, charlbl[4], transform=axE.transAxes)


    axF = fig.add_subplot(gs[5:9, 0])
    axF.plot(x_zoom, C_p[sl], "-",  color=col_passive, linewidth=4)
    axF.plot(x_zoom, C_c[sl], "-.", color=col_yellow, linewidth=4)
    axF.plot(x_zoom, C_q[sl], ":",  color=col_orange, linewidth=4)
    axF.plot(x_zoom, C_b[sl], "--", color=col_green, linewidth=4)
    axF.plot(x_zoom, C_s[sl], "-",  color=col_blue, linewidth=5)
    axF.plot(x_zoom, C_tv[sl], "-", color=col_lao_tv, linewidth=4)
    axF.plot(x_zoom, C_pr[sl], "-", color=col_lao_pr, linewidth=4)
    axF.set_ylabel("TSS concentration (mg/L)")
    axF.set_xlim(start * Ts, Nstep * Ts)
    axF.set_ylim(0, 1.2 * float(np.max(C_s[sl]) if np.max(C_s[sl]) > 0 else 1.0))
    axF.set_xticks(np.array([24770, 25250, 25730, 26210], dtype=float) * Ts)
    axF.set_xticklabels(["Sep. 17", "Sep. 22", "Sep. 27", "Oct. 1"])
    axF.text(0.02, 0.95, charlbl[5], transform=axF.transAxes)


    axG = fig.add_subplot(gs[5:9, 1])
    axG.plot(x_zoom, np.ones_like(x_zoom), "-", color=col_passive, linewidth=4)
    axG.plot(x_zoom, th_c[sl], "-.", color=col_yellow, linewidth=3)
    axG.plot(x_zoom, th_b[sl], "--", color=col_green, linewidth=3)
    axG.plot(x_zoom, th_q[sl], ":",  color=col_orange, linewidth=4)
    axG.plot(x_zoom, th_s[sl], "-",  color=col_blue, linewidth=5)
    axG.plot(x_zoom, th_tv[sl], "-", color=col_lao_tv, linewidth=4)
    axG.plot(x_zoom, th_pr[sl], "-", color=col_lao_pr, linewidth=4)
    axG.set_ylabel("Valve opening ratio")
    axG.set_xlim(start * Ts, Nstep * Ts)
    axG.set_ylim(0, 1)
    axG.set_xticks(np.array([24770, 25250, 25730, 26210], dtype=float) * Ts)
    axG.set_xticklabels(["Sep. 17", "Sep. 22", "Sep. 27", "Oct. 1"])
    axG.text(0.02, 0.95, charlbl[6], transform=axG.transAxes)

    axH = fig.add_subplot(gs[5:9, 2])
    load_p  = _cum_load_kg(C_p[sl],  q_p[sl],              dt_s)
    load_s  = _cum_load_kg(C_s[sl],  q_s[sl],              dt_s)
    load_c  = _cum_load_kg(C_c[sl],  q_c[sl] + sp_c[sl],   dt_s)
    load_q  = _cum_load_kg(C_q[sl],  q_q[sl] + sp_q[sl],   dt_s)
    load_b  = _cum_load_kg(C_b[sl],  q_b[sl] + sp_b[sl],   dt_s)
    load_tv = _cum_load_kg(C_tv[sl], q_tv[sl] + sp_tv[sl], dt_s)
    load_pr = _cum_load_kg(C_pr[sl], q_pr[sl] + sp_pr[sl], dt_s)

    axH.plot(x_zoom, load_p,  "-", color=col_passive, linewidth=4, label="Passive")
    axH.plot(x_zoom, load_s,  "-", color=col_blue,    linewidth=5, label="Stochastic MPC")
    axH.plot(x_zoom, load_tv, "-", color=col_lao_tv,  linewidth=5, label="LAO-TerminalValue MPC")
    axH.plot(x_zoom, load_pr, "-", color=col_lao_pr,  linewidth=5, label="LAO-Pruned MPC")
    axH.plot(x_zoom, load_c,  "-.", color=col_yellow, linewidth=4, label="RBC-Conc")
    axH.plot(x_zoom, load_q,  ":",  color=col_orange, linewidth=4, label="RBC-Outflow")
    axH.plot(x_zoom, load_b,  "--", color=col_green,  linewidth=4, label="RBC-Both")
    axH.set_ylabel("Cummulative Load (kg)")
    axH.set_xlim(start * Ts, Nstep * Ts)
    axH.set_xticks(np.array([24770, 25250, 25730, 26210], dtype=float) * Ts)
    axH.set_xticklabels(["Sep. 17", "Sep. 22", "Sep. 27", "Oct. 1"])
    axH.text(0.02, 0.95, charlbl[7], transform=axH.transAxes)

    handles, labels = axH.get_legend_handles_labels()
    fig.subplots_adjust(top=0.95)
    fig.legend(handles, labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.995),
               ncol=7, frameon=False)

    fig.savefig(out_png, pad_inches=0.08)
    plt.close(fig)
    print(f"[plot] saved: {out_png}")


base_dir = Path(__file__).resolve().parent

data_path = base_dir / "pond4_2021.xlsx"
if not data_path.exists():
    raise FileNotFoundError(f"Missing data file: {data_path}")

data1 = pd.read_excel(data_path)
data2 = pd.read_excel(data_path)

qin_t = data1["qin"].to_numpy(dtype=float)
cin_t = data1["cin"].to_numpy(dtype=float)

qin_f = data2["qin"].to_numpy(dtype=float)

EMC = float(12.0)  # mg/L
cin_f = np.full_like(qin_t, EMC, dtype=float)

MD_t = np.column_stack((qin_t, cin_t))
MD_f = np.column_stack((qin_f, cin_f))

hlimit_ft = 10.0
hlimit_m = hlimit_ft / 3.281

trajs_path = base_dir / "results_trajs_real_case.npz"
trajs, meta = load_trajs_npz_as_dataclass(trajs_path)

print("Available traj keys:", trajs.keys())

plot_benchmark_9x3_event_with_inset(
    out_png=base_dir / "benchmark_9x3_inset_with_lao.png",
    trajs=trajs,
    qin_t=qin_t,
    cin_t=cin_t,
    cin_f=cin_f,
    rainfall_csv=base_dir / "rainfall_15min_2021.csv",
    Ts=1.0,
    start=25220,
    Nstep=26275,
    Tfull=34000,
    hlimit_m=hlimit_m,
)
