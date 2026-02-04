#!/usr/bin/env python3
"""
Visualize PlotJuggler CSV exports for all 4 runs:
  - TS-LSTM
  - TS-CNN
  - PV-LSTM
  - PV-CNN
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAD2DEG = 180.0 / np.pi

# Joint order in your controller / commanded vector:
JOINTS = [
    ("Left Hip",   "/joint_states/left_hip_revolute_joint/position",   "/joint_states/left_hip_revolute_joint/velocity"),
    ("Right Hip",  "/joint_states/right_hip_revolute_joint/position",  "/joint_states/right_hip_revolute_joint/velocity"),
    ("Left Knee",  "/joint_states/left_knee_revolute_joint/position",  "/joint_states/left_knee_revolute_joint/velocity"),
    ("Right Knee", "/joint_states/right_knee_revolute_joint/position", "/joint_states/right_knee_revolute_joint/velocity"),
    ("Left Ankle", "/joint_states/left_ankle_revolute_joint/position", "/joint_states/left_ankle_revolute_joint/velocity"),
    ("Right Ankle","/joint_states/right_ankle_revolute_joint/position","/joint_states/right_ankle_revolute_joint/velocity"),
]


def traj_cols(point_idx: int):
    return [f"/trajectory_controller/joint_trajectory/points[{point_idx}]/positions[{i}]" for i in range(6)]


def load_and_prepare(csv_path: str, point_idx: int) -> pd.DataFrame:
    """
    Returns a dataframe with:
      _t (seconds from start)
      joint_states positions/velocities
      commanded positions for a chosen points[point_idx] (forward-filled)
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")  # auto-detect delimiter (tabs/commas)

    if "__time" not in df.columns:
        raise KeyError(f"{csv_path} missing '__time' column.")
    
    # --------------------------------------------------
    # Convert joint positions (rad -> deg)
    # --------------------------------------------------
    for col in df.columns:
        if col.endswith("/position") or "positions[" in col:
            df[col] = df[col] * RAD2DEG

    for col in df.columns:
        if col.endswith("/velocity"):
            df[col] = df[col] * RAD2DEG
    
    # time axis in seconds, normalized to start at 0
    t = pd.to_numeric(df["__time"], errors="coerce").to_numpy()
    t = t - np.nanmin(t)
    df["_t"] = t

    needed = [pos for _, pos, _ in JOINTS] + [vel for _, _, vel in JOINTS] + traj_cols(point_idx)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"{csv_path} missing required columns for point[{point_idx}]:\n" + "\n".join(missing)
        )

    # numeric conversion
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # commanded trajectory is sparse -> forward-fill (piecewise-constant hold)
    cmd_cols = traj_cols(point_idx)
    cmd = df[["_t"] + cmd_cols].copy()
    cmd[cmd_cols] = cmd[cmd_cols].ffill()

    act = df[["_t"] + [pos for _, pos, _ in JOINTS] + [vel for _, _, vel in JOINTS]].copy()

    out = pd.concat([act.set_index("_t"), cmd.set_index("_t")], axis=1).reset_index()
    out = out.rename(columns={"index": "_t"})
    return out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    e = a[m] - b[m]
    return float(np.sqrt(np.mean(e * e)))


def plot_actual_vs_commanded(df: pd.DataFrame, title: str, point_idx: int, plot_vel: bool):
    t = df["_t"].to_numpy()
    cmd_cols = traj_cols(point_idx)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    rmses = []
    for i, (joint_name, pos_col, _) in enumerate(JOINTS):
        ax = axes[i]
        cmd_col = cmd_cols[i]

        y_act = df[pos_col].to_numpy()
        y_cmd = df[cmd_col].to_numpy()

        ax.plot(t, y_act, label="Actual pos (joint_states)")
        ax.plot(t, y_cmd, label=f"Commanded pos (points[{point_idx}])")
        ax.set_title(joint_name)
        ax.grid(True)

        r = rmse(y_act, y_cmd)
        rmses.append(r)
        ax.text(0.01, 0.95, f"RMSE={r:.4f}", transform=ax.transAxes, va="top")

    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.supxlabel("Time (s)")
    fig.supylabel("Position (deg)")
    plt.tight_layout()
    plt.show()

    if plot_vel:
        fig2, axes2 = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        axes2 = axes2.flatten()
        for i, (joint_name, _, vel_col) in enumerate(JOINTS):
            ax = axes2[i]
            v = df[vel_col].to_numpy()
            ax.plot(t, v, label="Actual vel (joint_states)")
            ax.set_title(joint_name)
            ax.grid(True)
        axes2[0].legend(loc="best")
        fig2.suptitle(title + " — Velocities")
        fig2.supxlabel("Time (s)")
        fig2.supylabel("Velocity (deg/s)")
        plt.tight_layout()
        plt.show()

    return rmses


def plot_commanded_compare(dfs, labels, point_idx: int, title: str):
    """
    dfs: list[pd.DataFrame] already prepared
    labels: list[str] same length as dfs
    """
    cols = traj_cols(point_idx)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=False)
    axes = axes.flatten()

    for j, (joint_name, _, _) in enumerate(JOINTS):
        ax = axes[j]
        for df, lab in zip(dfs, labels):
            ax.plot(df["_t"].to_numpy(), df[cols[j]].to_numpy(), label=lab)
        ax.set_title(joint_name)
        ax.grid(True)

    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.supxlabel("Time from start (s)")
    fig.supylabel("Commanded position (deg)")
    plt.tight_layout()
    plt.show()


def plot_jointstates_compare(dfs, labels, title: str):
    """
    Compare ACTUAL joint_states positions for multiple runs on the same plots.
    dfs: list[pd.DataFrame] output of load_and_prepare()
    labels: list[str] same length
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=False)
    axes = axes.flatten()

    for j, (joint_name, pos_col, _) in enumerate(JOINTS):
        ax = axes[j]
        for df, lab in zip(dfs, labels):
            ax.plot(df["_t"].to_numpy(), df[pos_col].to_numpy(), label=lab)
        ax.set_title(joint_name)
        ax.grid(True)

    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.supxlabel("Time from start (s)")
    fig.supylabel("Actual position (deg)")
    plt.tight_layout()
    plt.show()



def print_rmse_block(run_name: str, rmses):
    print(f"\n{run_name}:")
    for (name, _, _), r in zip(JOINTS, rmses):
        print(f"  {name:10s}: {r:.6f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-lstm", default="misc/PlotJuggler_misc/plot_juggler_lstm.csv", help="TS-LSTM PlotJuggler CSV")
    ap.add_argument("--ts-cnn",  default="misc/PlotJuggler_misc/plot_juggler_cnn.csv",  help="TS-CNN PlotJuggler CSV")
    ap.add_argument("--pv-lstm", default="misc/PlotJuggler_misc/plot_juggler_lstm_pv.csv", help="PV-LSTM PlotJuggler CSV")
    ap.add_argument("--pv-cnn",  default="misc/PlotJuggler_misc/plot_juggler_cnn_pv.csv",  help="PV-CNN PlotJuggler CSV")
    ap.add_argument("--point", type=int, default=0, choices=[0, 1, 2],
                    help="Which JointTrajectory point index to use (0, 1, or 2)")
    ap.add_argument("--plot-vel", action="store_true", help="Also plot joint velocities")
    args = ap.parse_args()

    # Load
    df_ts_lstm = load_and_prepare(args.ts_lstm, args.point)
    df_ts_cnn  = load_and_prepare(args.ts_cnn,  args.point)
    df_pv_lstm = load_and_prepare(args.pv_lstm, args.point)
    df_pv_cnn  = load_and_prepare(args.pv_cnn,  args.point)

    print("\n=== RMSE (Actual position vs Commanded position) ===")
    print(f"(Using JointTrajectory points[{args.point}])")

    rmse_ts_lstm = plot_actual_vs_commanded(df_ts_lstm, "TS-LSTM: Actual vs Commanded", args.point, args.plot_vel)
    rmse_ts_cnn  = plot_actual_vs_commanded(df_ts_cnn,  "TS-CNN:  Actual vs Commanded", args.point, args.plot_vel)
    rmse_pv_lstm = plot_actual_vs_commanded(df_pv_lstm, "PV-LSTM: Actual vs Commanded", args.point, args.plot_vel)
    rmse_pv_cnn  = plot_actual_vs_commanded(df_pv_cnn,  "PV-CNN:  Actual vs Commanded", args.point, args.plot_vel)

    print_rmse_block("TS-LSTM", rmse_ts_lstm)
    print_rmse_block("TS-CNN ", rmse_ts_cnn)
    print_rmse_block("PV-LSTM", rmse_pv_lstm)
    print_rmse_block("PV-CNN ", rmse_pv_cnn)

    # Commanded-only comparisons
    plot_commanded_compare(
        [df_ts_lstm, df_ts_cnn],
        ["TS-LSTM commanded", "TS-CNN commanded"],
        args.point,
        f"Commanded trajectories: TS-LSTM vs TS-CNN (points[{args.point}])",
    )
    plot_commanded_compare(
        [df_pv_lstm, df_pv_cnn],
        ["PV-LSTM commanded", "PV-CNN commanded"],
        args.point,
        f"Commanded trajectories: PV-LSTM vs PV-CNN (points[{args.point}])",
    )
    plot_commanded_compare(
        [df_ts_lstm, df_pv_lstm],
        ["TS-LSTM commanded", "PV-LSTM commanded"],
        args.point,
        f"Commanded trajectories: TS-LSTM vs PV-LSTM (points[{args.point}])",
    )
    plot_commanded_compare(
        [df_ts_cnn, df_pv_cnn],
        ["TS-CNN commanded", "PV-CNN commanded"],
        args.point,
        f"Commanded trajectories: TS-CNN vs PV-CNN (points[{args.point}])",
    )
    # Actual joint_states comparison (ALL 4 on the same plots)
    plot_jointstates_compare(
        [df_ts_lstm, df_ts_cnn, df_pv_lstm, df_pv_cnn],
        ["TS-LSTM actual", "TS-CNN actual", "PV-LSTM actual", "PV-CNN actual"],
        "Actual joint_states positions: All models (deg)",
    )


if __name__ == "__main__":
    main()
