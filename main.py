# -*- coding: utf-8 -*-
"""
Лабораторная работа №1 по дисциплине «Методы оптимизации. Линейное
программирование». Вариант 1.

Задача: план производства продукции на станках, максимум прибыли
(5 продуктов, 4 станка).

Реализовано:
    * загрузка данных из CSV/JSON;
    * LP-модель: max c^T x при ограничениях по времени станков и спросу;
    * решение HiGHS (SciPy linprog) и интерпретация статуса;
    * независимая проверка допустимости и целевой функции на NumPy;
    * теневые цены мощностей станков и их проверка конечными разностями;
    * сценарии: базовый, дефицитный (M4 −20%), стрессовый (M1 и M4 −20%,
      спрос +10%);
    * диагностика несовместимости (эластичный фильтр) для контрактной
      перегрузки;
    * анализ диапазонов устойчивости и визуализация области допустимых
      решений для двух продуктов.

Запуск:
    python generate_data.py
    python main.py
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "results"
TOL = 1e-7


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
def load_data():
    products = []
    with open(DATA / "products.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            products.append({
                "id": int(row["product_id"]),
                "name": row["product"],
                "profit": float(row["profit_per_unit"]),
                "demand": float(row["max_demand"]),
                "times": np.array([float(row["time_m1"]), float(row["time_m2"]),
                                   float(row["time_m3"]), float(row["time_m4"])]),
            })
    machines = []
    with open(DATA / "machines.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            machines.append({"id": int(row["machine_id"]), "name": row["machine"],
                             "capacity": float(row["capacity_hours"])})
    with open(DATA / "params.json", encoding="utf-8") as f:
        params = json.load(f)
    return products, machines, params


def arrays(products, machines):
    c = np.array([p["profit"] for p in products])
    A = np.array([p["times"] for p in products]).T   # (machines, products)
    d = np.array([p["demand"] for p in products])
    b = np.array([m["capacity"] for m in machines])
    return c, A, d, b


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
def solve_lp(A, c, b, d, min_prod=None, disp=False):
    """max c^T x  ⇔  min -c^T x."""
    rows = [A, np.eye(len(c))]
    rhs = [b, d]
    if min_prod is not None:
        rows.append(-np.eye(len(c)))
        rhs.append(-np.asarray(min_prod, dtype=float))
    A_ub = np.vstack(rows)
    b_ub = np.concatenate(rhs)
    args = dict(A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, None)] * len(c),
                method="highs")
    if not disp:
        return linprog(-c, **args), A_ub, b_ub, ""
    old_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 1)
    try:
        res = linprog(-c, options={"disp": True}, **args)
    finally:
        sys.stdout.flush()
        os.dup2(old_fd, 1)
        os.close(old_fd)
        tmp.seek(0)
        log = tmp.read().decode("utf-8", errors="replace")
        tmp.close()
    return res, A_ub, b_ub, log


def verify(x, A, c, b, d, min_prod=None, tol=TOL):
    x = np.asarray(x, dtype=float)
    usage = A @ x
    cap_violation = np.maximum(0.0, usage - b)
    demand_violation = np.maximum(0.0, x - d)
    neg_violation = np.maximum(0.0, -x)
    min_violation = (np.maximum(0.0, np.asarray(min_prod) - x)
                     if min_prod is not None else np.zeros_like(x))
    return {
        "usage": usage,
        "cap_violation": cap_violation,
        "demand_violation": demand_violation,
        "neg_violation": neg_violation,
        "min_violation": min_violation,
        "max_violation": float(max(cap_violation.max(), demand_violation.max(),
                                   neg_violation.max(), min_violation.max())),
        "objective": float(c @ x),
        "feasible": bool(max(cap_violation.max(), demand_violation.max(),
                             neg_violation.max(), min_violation.max()) <= tol),
    }


def shadow_prices(res, A, c, b, d, min_prod=None):
    """Теневые цены мощностей станков и конечные разности."""
    base_obj = -res.fun
    marg = np.asarray(res.ineqlin.marginals)
    machine_sp = [-float(marg[m]) for m in range(len(b))]
    demand_sp = [-float(marg[len(b) + j]) for j in range(len(c))]

    fd_machines = []
    for m in range(len(b)):
        b2 = b.copy()
        b2[m] += 1.0
        r, _, _, _ = solve_lp(A, c, b2, d, min_prod)
        fd_machines.append(float(-r.fun - base_obj))

    fd_demand = []
    for j in range(len(c)):
        d2 = d.copy()
        d2[j] += 1.0
        r, _, _, _ = solve_lp(A, c, b, d2, min_prod)
        fd_demand.append(float(-r.fun - base_obj))
    return {"machines": machine_sp, "machines_fd": fd_machines,
            "demand": demand_sp, "demand_fd": fd_demand}


def active_signature(res, A, b, d, min_prod=None):
    if not res.success:
        return ("infeasible",)
    x = res.x
    sig = []
    for m in range(len(b)):
        if abs((A @ x)[m] - b[m]) <= 1e-6:
            sig.append(f"machine{m}")
    for j in range(len(x)):
        if abs(x[j] - d[j]) <= 1e-6:
            sig.append(f"demand{j}")
    if min_prod is not None:
        for j in range(len(x)):
            if abs(x[j] - min_prod[j]) <= 1e-6:
                sig.append(f"min{j}")
    return tuple(sig)


def elastic_filter_min(A, b, d, min_required):
    """Минимальные послабления контрактных требований при заданных мощностях."""
    n = len(d)
    zeros_mn = np.zeros((A.shape[0], n))
    A_ub = np.vstack([np.hstack([A, zeros_mn]),
                      np.hstack([np.eye(n), np.zeros((n, n))]),
                      np.hstack([-np.eye(n), -np.eye(n)])])
    b_ub = np.concatenate([b, d, -np.asarray(min_required, dtype=float)])
    c_obj = np.concatenate([np.zeros(n), np.ones(n)])
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0.0, None)] * n + [(0.0, None)] * n, method="highs")
    if not res.success:
        return {"status": int(res.status), "message": res.message}
    return {"status": int(res.status), "message": res.message,
            "x": res.x[:n], "slack": res.x[n:],
            "total_slack": float(res.x[n:].sum())}


# ---------------------------------------------------------------------------
# Сценарии
# ---------------------------------------------------------------------------
def run_scenarios(products, machines):
    c, A, d, b = arrays(products, machines)
    scenarios = {}
    res, A_ub, b_ub, log = solve_lp(A, c, b, d, None, disp=True)
    scenarios["base"] = {"res": res, "b": b.copy(), "d": d.copy(),
                         "min": None, "log": log}

    b_def = b.copy()
    b_def[3] *= 0.80
    res_def, _, _, log_def = solve_lp(A, c, b_def, d, None, disp=True)
    scenarios["deficit"] = {"res": res_def, "b": b_def, "d": d.copy(),
                            "min": None, "log": log_def}

    b_str = b.copy()
    b_str[0] *= 0.80
    b_str[3] *= 0.80
    d_str = d * 1.10
    res_str, _, _, log_str = solve_lp(A, c, b_str, d_str, None, disp=True)
    scenarios["stress"] = {"res": res_str, "b": b_str, "d": d_str,
                           "min": None, "log": log_str}

    # Контрактная перегрузка: жёсткие минимальные объёмы + дефицит мощностей.
    b_con = b.copy()
    for m in (0, 2, 3):
        b_con[m] *= 0.70
    d_con = d * 1.20
    min_con = np.full(len(c), 30.0)
    res_con, _, _, log_con = solve_lp(A, c, b_con, d_con, min_con, disp=True)
    scenarios["contract_overload"] = {"res": res_con, "b": b_con, "d": d_con,
                                      "min": min_con, "log": log_con}
    return scenarios, c, A, d, b


def scenario_record(key, sc, A, c, machines, products):
    res = sc["res"]
    rec = {"scenario": key, "status": int(res.status), "message": res.message}
    if res.success:
        verify_res = verify(res.x, A, c, sc["b"], sc["d"], sc["min"])
        rec.update({"x": res.x, "objective": float(-res.fun),
                    "usage": verify_res["usage"], "feasible": True,
                    "active": []})
        for m, mach in enumerate(machines):
            if abs(verify_res["usage"][m] - sc["b"][m]) <= 1e-6:
                rec["active"].append(f"мощность {mach['name']}")
        for j, prod in enumerate(products):
            if abs(res.x[j] - sc["d"][j]) <= 1e-6:
                rec["active"].append(f"спрос {prod['name']}")
        if sc["min"] is not None:
            for j, prod in enumerate(products):
                if abs(res.x[j] - sc["min"][j]) <= 1e-6:
                    rec["active"].append(f"контракт {prod['name']}")
    else:
        rec.update({"x": None, "objective": None, "usage": None,
                    "feasible": False, "active": []})
    return rec


# ---------------------------------------------------------------------------
# Чувствительность
# ---------------------------------------------------------------------------
def sensitivity_coefficients(A, c, b, d, base_sig, mult=None):
    if mult is None:
        mult = np.linspace(0.5, 1.5, 101)
    out = []
    for j in range(len(c)):
        same = []
        for t in mult:
            c2 = c.copy()
            c2[j] *= t
            res, _, _, _ = solve_lp(A, c2, b, d)
            same.append(active_signature(res, A, b, d) == base_sig)
        same = np.array(same)
        idx = int(np.argmin(np.abs(mult - 1.0)))
        lo = idx
        while lo > 0 and same[lo - 1]:
            lo -= 1
        hi = idx
        while hi < len(mult) - 1 and same[hi + 1]:
            hi += 1
        out.append({"product": j, "base": float(c[j]),
                    "t_low": float(mult[lo]), "t_high": float(mult[hi]),
                    "low": float(c[j] * mult[lo]),
                    "high": float(c[j] * mult[hi])})
    return out


def sensitivity_capacity(A, c, b, d, base_sig, mult=None):
    if mult is None:
        mult = np.linspace(0.70, 1.30, 121)
    out = []
    for m in range(len(b)):
        same = []
        for t in mult:
            b2 = b.copy()
            b2[m] *= t
            res, _, _, _ = solve_lp(A, c, b2, d)
            same.append(active_signature(res, A, b2, d) == base_sig)
        same = np.array(same)
        idx = int(np.argmin(np.abs(mult - 1.0)))
        lo = idx
        while lo > 0 and same[lo - 1]:
            lo -= 1
        hi = idx
        while hi < len(mult) - 1 and same[hi + 1]:
            hi += 1
        out.append({"machine": m, "base": float(b[m]),
                    "t_low": float(mult[lo]), "t_high": float(mult[hi]),
                    "low": float(b[m] * mult[lo]),
                    "high": float(b[m] * mult[hi])})
    return out


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------
def plot_plan(x, d, names, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(x)))
    bars = ax.bar(names, x, color=colors, edgecolor="white")
    ax.scatter(names, d, color="#C0392B", zorder=5, label="максимальный спрос")
    for bar, v in zip(bars, x):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}",
                ha="center", fontsize=9)
    ax.set_ylabel("Объём производства, ед.")
    ax.set_title("Оптимальный план производства")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_machines(usage, b, names, path):
    util = usage / b * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [plt.cm.RdYlGn_r(min(1.0, u / 100)) for u in util]
    bars = ax.bar(names, util, color=colors, edgecolor="white")
    for bar, u, used, cap in zip(bars, util, usage, b):
        ax.text(bar.get_x() + bar.get_width() / 2, u + 1.2,
                f"{used:.1f}/{cap:.0f}\n({u:.0f}%)", ha="center", fontsize=8.5)
    ax.axhline(100, color="#C0392B", ls="--", lw=1)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Загрузка, %")
    ax.set_title("Использование мощностей станков")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_scenarios(records, product_names, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    keys = ["base", "deficit", "stress"]
    titles = ["базовый", "дефицит M4 −20%", "стресс M1/M4 −20%, спрос +10%"]
    objs = [records[k]["objective"] if records[k]["objective"] is not None else 0
            for k in keys]
    bars = axes[0].bar(titles, objs, color=["#2F5D8A", "#F39C12", "#C0392B"], alpha=0.85)
    for bar, k in zip(bars, keys):
        rec = records[k]
        if rec["objective"] is None:
            bar.set_height(0.04 * max(objs + [1]))
        else:
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                         f"{rec['objective']:.0f}", ha="center", va="center",
                         color="white", fontsize=9)
    axes[0].set_ylabel("Прибыль, тыс. руб.")
    axes[0].set_title("Целевая функция по сценариям")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(alpha=0.3, axis="y")

    w = 0.38
    xpos = np.arange(len(product_names))
    for idx, (k, color, label) in enumerate(zip(keys, ["#2F5D8A", "#F39C12", "#C0392B"],
                                                ["базовый", "дефицит", "стресс"])):
        rec = records[k]
        if rec["x"] is not None:
            axes[1].bar(xpos + (idx - 1) * w, rec["x"], w, color=color, label=label)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(product_names, rotation=20, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Объём, ед.")
    axes[1].set_title("Перестройка плана")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_shadow(sp, machine_names, path):
    labels = machine_names
    values = sp["machines"]
    fd = sp["machines_fd"]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, values, w, label="двойственная оценка", color="#2F5D8A")
    ax.bar(x + w / 2, fd, w, label="конечная разность", color="#95A5A6")
    for i in range(len(x)):
        ax.text(x[i] - w / 2, values[i] + 0.25, f"{values[i]:.2f}",
                ha="center", fontsize=8.5)
        ax.text(x[i] + w / 2, fd[i] + 0.25, f"{fd[i]:.2f}",
                ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Прирост прибыли на +1 час")
    ax.set_title("Теневые цены мощностей станков")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_capacity_curve(A, c, b, d, m, path):
    mults = np.linspace(0.7, 1.3, 121)
    objs = []
    for t in mults:
        b2 = b.copy()
        b2[m] *= t
        r, _, _, _ = solve_lp(A, c, b2, d)
        objs.append(float(-r.fun) if r.success else np.nan)
    base_sig = active_signature(solve_lp(A, c, b, d)[0], A, b, d)
    stable = []
    for t in mults:
        b2 = b.copy()
        b2[m] *= t
        r, _, _, _ = solve_lp(A, c, b2, d)
        stable.append(active_signature(r, A, b2, d) == base_sig)
    stable = np.array(stable)
    idx = int(np.argmin(np.abs(mults - 1.0)))
    lo = idx
    while lo > 0 and stable[lo - 1]:
        lo -= 1
    hi = idx
    while hi < len(mults) - 1 and stable[hi + 1]:
        hi += 1
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mults * b[m], objs, color="#2F5D8A", lw=2)
    ax.axvline(mults[lo] * b[m], color="#27AE60", ls="--", lw=1.2,
               label=f"устойчиво до {mults[lo] * b[m]:.0f}")
    ax.axvline(mults[hi] * b[m], color="#C0392B", ls="--", lw=1.2,
               label=f"устойчиво после {mults[hi] * b[m]:.0f}")
    ax.set_xlabel("Мощность станка M4, ч")
    ax.set_ylabel("Оптимальная прибыль, тыс. руб.")
    ax.set_title("Зависимость прибыли от мощности M4 и диапазон устойчивости")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return float(mults[lo] * b[m]), float(mults[hi] * b[m])


def plot_feasible_region(A, c, b, d, x_opt, products, path):
    """Сечение области допустимых решений по продуктам A и B."""
    # фиксируем продукты C, D, E на оптимальных значениях
    fixed = x_opt[2:]
    b_eff = b - A[:, 2:] @ fixed
    a1 = A[:, 0]
    a2 = A[:, 1]
    x1 = np.linspace(0, d[0] * 1.05, 400)
    x2 = np.linspace(0, d[1] * 1.05, 400)
    X1, X2 = np.meshgrid(x1, x2)
    feasible = np.ones_like(X1, dtype=bool)
    for m in range(len(b)):
        feasible &= (a1[m] * X1 + a2[m] * X2 <= b_eff[m] + 1e-9)
    feasible &= (X1 <= d[0]) & (X2 <= d[1]) & (X1 >= 0) & (X2 >= 0)
    obj = c[0] * X1 + c[1] * X2

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.contourf(X1, X2, feasible.astype(float), levels=[0.5, 1.5],
                colors=["#AED6F1"], alpha=0.55)
    for m in range(len(b)):
        if abs(a2[m]) > 1e-9:
            x2_line = (b_eff[m] - a1[m] * x1) / a2[m]
            ax.plot(x1, x2_line, lw=1.0, label=f"станок M{m + 1}")
    ax.scatter([x_opt[0]], [x_opt[1]], color="#C0392B", s=120, marker="*",
               zorder=5, label="оптимум")
    ax.set_xlim(0, d[0] * 1.05)
    ax.set_ylim(0, d[1] * 1.05)
    ax.set_xlabel(products[0]["name"] + ", ед.")
    ax.set_ylabel(products[1]["name"] + ", ед.")
    ax.set_title("Сечение допустимой области (C, D, E зафиксированы)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_dashboard(x, d, A, b, product_names, machine_names, records, sp, summary, path):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)
    fig.suptitle("ЛР №1 · План производства на станках · LP-модель (вариант 1)",
                 fontsize=16, fontweight="bold", color="#12263A", y=0.98)

    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.bar(product_names, x, color="#2F5D8A", alpha=0.85)
    ax1.scatter(product_names, d, color="#C0392B", zorder=5, label="спрос")
    for bar, v in zip(bars, x):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}",
                 ha="center", fontsize=9)
    ax1.set_ylabel("Объём, ед.")
    ax1.set_title("Оптимальный план производства")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    lines = ["Сводка", "",
             f"Прибыль: {summary['objective']:.0f} тыс. руб.",
             f"Статус: Optimal", "",
             "Активные ограничения:"]
    lines += [f"  • {a}" for a in summary["active"]]
    lines += ["", f"Теневая цена M4: {sp['machines'][3]:.2f}"]
    ax2.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#EAF2FB", edgecolor="#2F5D8A"))

    ax3 = fig.add_subplot(gs[1, 0])
    usage = A @ x
    util = usage / b * 100
    ax3.bar(machine_names, util,
            color=[plt.cm.RdYlGn_r(min(1.0, u / 100)) for u in util], alpha=0.85)
    for i, u in enumerate(util):
        ax3.text(i, u + 2, f"{u:.0f}%", ha="center", fontsize=9)
    ax3.axhline(100, color="#C0392B", ls="--", lw=1)
    ax3.set_ylim(0, 118)
    ax3.set_ylabel("Загрузка, %")
    ax3.set_title("Мощности станков")
    ax3.grid(alpha=0.3, axis="y")

    ax4 = fig.add_subplot(gs[1, 1])
    keys = ["base", "deficit", "stress"]
    vals = [records[k]["objective"] if records[k]["objective"] is not None else 0
            for k in keys]
    bars = ax4.bar(["базовый", "дефицит", "стресс"], vals,
                   color=["#2F5D8A", "#F39C12", "#C0392B"], alpha=0.85)
    for bar, k in zip(bars, keys):
        rec = records[k]
        if rec["objective"] is None:
            bar.set_height(0.04 * max(vals + [1]))
        else:
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                     f"{rec['objective']:.0f}", ha="center", va="center",
                     color="white", fontsize=9)
    ax4.set_ylabel("Прибыль")
    ax4.set_title("Сценарии")
    ax4.tick_params(axis="x", rotation=12)
    ax4.grid(alpha=0.3, axis="y")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.bar(machine_names, sp["machines"], color="#2F5D8A", alpha=0.85)
    for i, v in enumerate(sp["machines"]):
        ax5.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=8.5)
    ax5.set_ylabel("Прирост на +1 ч")
    ax5.set_title("Теневые цены")
    ax5.tick_params(axis="x", rotation=15)
    ax5.grid(alpha=0.3, axis="y")

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------
def save_verification(path: str, products, machines, x, c, A, b, d,
                      ver, solver_obj):
    """Независимая проверка: таблица нарушений и пересчёт цели."""
    rows = []
    for m, mach in enumerate(machines):
        rows.append(["machine_capacity", mach["name"],
                     round(float(ver["usage"][m]), 6), round(float(b[m]), 6),
                     round(float(ver["cap_violation"][m]), 9),
                     "OK" if ver["cap_violation"][m] <= TOL else "FAIL"])
    for j, p in enumerate(products):
        rows.append(["demand", p["name"], round(float(x[j]), 6),
                     round(float(d[j]), 6),
                     round(float(ver["demand_violation"][j]), 9),
                     "OK" if ver["demand_violation"][j] <= TOL else "FAIL"])
        rows.append(["nonnegativity", p["name"], round(float(x[j]), 6), 0.0,
                     round(float(ver["neg_violation"][j]), 9),
                     "OK" if ver["neg_violation"][j] <= TOL else "FAIL"])
    delta = abs(float(ver["objective"]) - float(solver_obj))
    rows.append(["objective", "recomputed_vs_solver",
                 round(float(ver["objective"]), 6), round(float(solver_obj), 6),
                 round(delta, 9), "OK" if delta <= 1e-6 else "FAIL"])
    rows.append(["overall", "max_violation",
                 round(float(ver["max_violation"]), 9), TOL,
                 round(float(ver["max_violation"]), 9),
                 "OK" if ver["feasible"] else "FAIL"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check", "entity", "value", "limit", "violation", "status"])
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    products, machines, params = load_data()
    c, A, d, b = arrays(products, machines)
    names = [p["name"] for p in products]
    machine_names = [m["name"] for m in machines]

    scenarios, c, A, d, b = run_scenarios(products, machines)
    base = scenarios["base"]
    res = base["res"]
    if not res.success:
        raise SystemExit(f"Базовая задача не решена: {res.message}")

    base_rec = scenario_record("base", base, A, c, machines, products)
    verify_res = verify(base_rec["x"], A, c, b, d)
    save_verification(OUT / "verification.csv", products, machines,
                      base_rec["x"], c, A, b, d, verify_res,
                      base_rec["objective"])
    sp = shadow_prices(res, A, c, b, d)
    records = {"base": base_rec}
    for key in ["deficit", "stress", "contract_overload"]:
        records[key] = scenario_record(key, scenarios[key], A, c, machines, products)

    elastic = elastic_filter_min(A, scenarios["contract_overload"]["b"],
                                 scenarios["contract_overload"]["d"],
                                 scenarios["contract_overload"]["min"])

    base_sig = active_signature(res, A, b, d)
    sens_c = sensitivity_coefficients(A, c, b, d, base_sig)
    sens_b = sensitivity_capacity(A, c, b, d, base_sig)

    # ---------------- CSV ----------------
    with open(OUT / "lab1_plan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["product", "x", "demand", "profit_per_unit", "profit",
                    "machine_1", "machine_2", "machine_3", "machine_4"])
        for j, p in enumerate(products):
            w.writerow([p["name"], round(float(base_rec["x"][j]), 4), p["demand"],
                        p["profit"], round(float(base_rec["x"][j] * p["profit"]), 3),
                        *[round(float(v), 3) for v in p["times"]]])
        w.writerow(["ИТОГО", round(float(base_rec["x"].sum()), 4), "", "",
                    round(float(base_rec["objective"]), 3), "", "", "", ""])

    with open(OUT / "lab1_machines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["machine", "usage_hours", "capacity_hours", "utilization",
                    "shadow_price", "shadow_finite_diff"])
        for m, mach in enumerate(machines):
            w.writerow([mach["name"], round(float(verify_res["usage"][m]), 4),
                        mach["capacity"],
                        round(float(verify_res["usage"][m] / mach["capacity"]), 4),
                        round(sp["machines"][m], 6), round(sp["machines_fd"][m], 6)])

    with open(OUT / "lab1_scenarios.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "status", "feasible", "objective",
                    "active_constraints"] + [f"x_{n}" for n in names])
        for key, rec in records.items():
            row = [key, rec["status"], rec["feasible"], rec["objective"],
                   "; ".join(rec["active"])]
            if rec["x"] is not None:
                row += [round(float(v), 4) for v in rec["x"]]
            else:
                row += [""] * len(names)
            w.writerow(row)

    with open(OUT / "lab1_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "name", "base", "multiplier_low", "multiplier_high",
                    "value_low", "value_high"])
        for rec in sens_c:
            w.writerow(["coefficient", names[rec["product"]], round(rec["base"], 4),
                        round(rec["t_low"], 3), round(rec["t_high"], 3),
                        round(rec["low"], 4), round(rec["high"], 4)])
        for rec in sens_b:
            w.writerow(["capacity", machine_names[rec["machine"]],
                        round(rec["base"], 4), round(rec["t_low"], 3),
                        round(rec["t_high"], 3), round(rec["low"], 4),
                        round(rec["high"], 4)])

    with open(OUT / "lab1_elastic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["product", "required_min", "slack", "achieved_with_slack"])
        if elastic.get("status") == 0:
            for j, p in enumerate(products):
                w.writerow([p["name"], scenarios["contract_overload"]["min"][j],
                            round(float(elastic["slack"][j]), 4),
                            round(float(elastic["x"][j] + elastic["slack"][j]), 4)])

    summary = {
        "variant": 1,
        "student": "Якимов Максим Андреевич, Б24-517, № 31",
        "objective": float(base_rec["objective"]),
        "x": base_rec["x"].tolist(),
        "status": base_rec["message"],
        "active": base_rec["active"],
        "max_violation": verify_res["max_violation"],
        "feasible": verify_res["feasible"],
        "shadow_machines": sp["machines"],
        "shadow_machines_fd": sp["machines_fd"],
        "elastic_contract": {"status": elastic.get("status"),
                             "total_slack": elastic.get("total_slack"),
                             "slack": elastic.get("slack", []).tolist()
                             if hasattr(elastic.get("slack", []), "tolist") else elastic.get("slack")},
        "seed": params["seed"],
    }
    with open(OUT / "lab1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    with open(OUT / "solver.log", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\nБАЗОВЫЙ СЦЕНАРИЙ\n" + "=" * 70 + "\n")
        f.write(base["log"])
        f.write("\nДвойственные оценки ограничений:\n")
        f.write(np.array2string(res.ineqlin.marginals, precision=6) + "\n")
        f.write("\nДефицитный сценарий:\n" + scenarios["deficit"]["log"])
        f.write("\nСтрессовый сценарий:\n" + scenarios["stress"]["log"])
        f.write("\nКонтрактная перегрузка:\n" + scenarios["contract_overload"]["log"])

    # ---------------- Графики ----------------
    plot_plan(base_rec["x"], d, names, OUT / "lab1_plan.png")
    plot_machines(verify_res["usage"], b, machine_names, OUT / "lab1_machines.png")
    plot_scenarios(records, names, OUT / "lab1_scenarios.png")
    plot_shadow(sp, machine_names, OUT / "lab1_shadow.png")
    stable_m4 = plot_capacity_curve(A, c, b, d, 3, OUT / "lab1_sensitivity.png")
    plot_feasible_region(A, c, b, d, base_rec["x"], products,
                         OUT / "lab1_feasible_region.png")
    plot_dashboard(base_rec["x"], d, A, b, names, machine_names, records, sp,
                   summary, OUT / "lab1_dashboard.png")

    # ---------------- Консоль ----------------
    print("=" * 65)
    print("ЛР №1. Вариант 1: план производства на станках")
    print("=" * 65)
    print(f"Статус: {base_rec['message']}")
    print(f"Прибыль: {base_rec['objective']:.2f} тыс. руб.")
    print("\nПлан производства:")
    for name, v, dem in zip(names, base_rec["x"], d):
        print(f"  {name:<14} {v:8.2f} ед. (спрос {dem:.0f})")
    print("\nЗагрузка станков:")
    for m, mach in enumerate(machines):
        print(f"  {mach['name']}: {verify_res['usage'][m]:7.2f} / "
              f"{b[m]:.0f} ч ({verify_res['usage'][m] / b[m] * 100:5.1f}%)")
    print("\nНезависимая проверка:")
    print(f"  Максимальное нарушение: {verify_res['max_violation']:.2e}")
    print(f"  Допустимость: {verify_res['feasible']}")
    print("\nТеневые цены мощностей:")
    for name, v, fd in zip(machine_names, sp["machines"], sp["machines_fd"]):
        print(f"  {name}: {v:.3f} (конечная разность {fd:.3f})")
    print("\nСценарии:")
    for key in ["base", "deficit", "stress", "contract_overload"]:
        rec = records[key]
        obj = f"{rec['objective']:.0f}" if rec["objective"] is not None else "—"
        print(f"  {key:<18} status={rec['status']}  objective={obj}")
    if elastic.get("status") == 0:
        print(f"  эластичный фильтр: суммарный дефицит = {elastic['total_slack']:.2f}")
    print(f"\nДиапазон устойчивости по M4: [{stable_m4[0]:.0f}, {stable_m4[1]:.0f}] ч")
    print(f"\nФайлы сохранены в {OUT}")


if __name__ == "__main__":
    main()
