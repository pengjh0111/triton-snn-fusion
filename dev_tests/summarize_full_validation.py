# import json
# import csv
# from pathlib import Path

# ROOT = Path("full_validation")


# def safe_get(d, *keys, default=None):
#     cur = d
#     for k in keys:
#         if not isinstance(cur, dict) or k not in cur:
#             return default
#         cur = cur[k]
#     return cur


# def get_result(results, *names):
#     for name in names:
#         if name in results:
#             return results[name]
#     return None


# def mean_ms(result):
#     if isinstance(result, dict):
#         return result.get("mean_ms")
#     return None


# def p50_ms(result):
#     if isinstance(result, dict):
#         return result.get("p50_ms")
#     return None


# def p90_ms(result):
#     if isinstance(result, dict):
#         return result.get("p90_ms")
#     return None


# def find_kairos_case(results, graph_window):
#     candidates = [
#         f"kairos_single_step_loop_compile_w{graph_window}",
#         f"kairos_m_compile_w{graph_window}",
#     ]

#     for name in candidates:
#         if name in results:
#             return name, results[name]

#     for name, result in results.items():
#         if name.startswith("kairos") and f"w{graph_window}" in name:
#             return name, result

#     for name, result in results.items():
#         if name.startswith("kairos"):
#             return name, result

#     return None, None


# def get_case_stats(payload, case_name):
#     if not case_name:
#         return {}

#     return safe_get(
#         payload,
#         "fused_op_call_stats_by_case",
#         case_name,
#         default={},
#     ) or {}


# def speedup(base, target):
#     if base is None or target is None or target == 0:
#         return None
#     return base / target


# rows = []

# for summary_path in sorted(ROOT.glob("*/benchmark_summary_all.json")):
#     run_dir = summary_path.parent.name

#     with summary_path.open("r", encoding="utf-8") as f:
#         data = json.load(f)

#     for model_name, payload in data.items():
#         dtype = payload.get("dtype") or "unknown"
#         candidate_windows = payload.get("candidate_windows", [])
#         graph_window = candidate_windows[0] if candidate_windows else None

#         results = payload.get("results", {})
#         best = payload.get("best_case") or {}

#         single_eager = get_result(
#             results,
#             "baseline_single_step_mode_eager",
#             "baseline_s_eager",
#         )
#         single_compile = get_result(
#             results,
#             "baseline_single_step_mode_compile",
#             "baseline_s_compile",
#         )
#         multi_eager = get_result(
#             results,
#             "baseline_multi_step_mode_eager",
#             "baseline_m_eager",
#         )
#         multi_compile = get_result(
#             results,
#             "baseline_multi_step_mode_compile",
#             "baseline_m_compile",
#         )

#         kairos_case, kairos_result = find_kairos_case(
#             results,
#             graph_window,
#         )

#         kairos_stats = get_case_stats(payload, kairos_case)

#         best_case = best.get("case")
#         best_mean = best.get("mean_ms")

#         row = {
#             "run_dir": run_dir,
#             "model": model_name,
#             "dtype": dtype,
#             "graph_window": graph_window,

#             "best_case": best_case,
#             "best_ms": best_mean,

#             "single_step_eager_ms": mean_ms(single_eager),
#             "single_step_compile_ms": mean_ms(single_compile),

#             "multi_step_eager_ms": mean_ms(multi_eager),
#             "multi_step_compile_ms": mean_ms(multi_compile),

#             "kairos_case": kairos_case,
#             "kairos_ms": mean_ms(kairos_result),
#             "kairos_p50_ms": p50_ms(kairos_result),
#             "kairos_p90_ms": p90_ms(kairos_result),

#             "speedup_kairos_vs_single_compile": speedup(
#                 mean_ms(single_compile),
#                 mean_ms(kairos_result),
#             ),
#             "speedup_kairos_vs_single_eager": speedup(
#                 mean_ms(single_eager),
#                 mean_ms(kairos_result),
#             ),
#             "speedup_kairos_vs_multi_compile": speedup(
#                 mean_ms(multi_compile),
#                 mean_ms(kairos_result),
#             ),
#             "speedup_kairos_vs_multi_eager": speedup(
#                 mean_ms(multi_eager),
#                 mean_ms(kairos_result),
#             ),

#             "triton": kairos_stats.get("triton", 0),
#             "fallback": kairos_stats.get("fallback", 0),
#             "temporal_triton": kairos_stats.get("temporal_triton", 0),
#             "temporal_fallback": kairos_stats.get("temporal_fallback", 0),

#             "temporal_k3_s1_p1": kairos_stats.get("temporal_k3_s1_p1", 0),
#             "temporal_k3_s2_p1": kairos_stats.get("temporal_k3_s2_p1", 0),
#             "temporal_k7_s2_p3": kairos_stats.get("temporal_k7_s2_p3", 0),

#             "single_k3_s1_p1": kairos_stats.get("single_k3_s1_p1", 0),
#             "single_k3_s2_p1": kairos_stats.get("single_k3_s2_p1", 0),
#             "single_k7_s2_p3": kairos_stats.get("single_k7_s2_p3", 0),

#             "kernel_temporal_configs": kairos_stats.get(
#                 "kernel_temporal_configs",
#                 {},
#             ),
#             "fallback_reasons": kairos_stats.get(
#                 "fallback_reasons",
#                 {},
#             ),
#         }

#         rows.append(row)


# rows.sort(
#     key=lambda r: (
#         str(r["dtype"]),
#         str(r["model"]),
#         int(r["graph_window"] or -1),
#     )
# )

# csv_path = ROOT / "full_validation_summary.csv"
# json_path = ROOT / "full_validation_summary.json"
# md_path = ROOT / "full_validation_summary.md"

# with csv_path.open("w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#     writer.writeheader()
#     writer.writerows(rows)

# with json_path.open("w", encoding="utf-8") as f:
#     json.dump(rows, f, indent=2, sort_keys=True)

# headers = [
#     "dtype",
#     "model",
#     "window",
#     "best_case",
#     "single_compile",
#     "multi_compile",
#     "kairos_case",
#     "kairos",
#     "vs_single_compile",
#     "vs_multi_compile",
#     "fallback",
#     "temporal_triton",
#     "kernel_temporal_configs",
# ]

# with md_path.open("w", encoding="utf-8") as f:
#     f.write("| " + " | ".join(headers) + " |\n")
#     f.write("|" + "|".join(["---"] * len(headers)) + "|\n")

#     for r in rows:
#         f.write(
#             "| "
#             + " | ".join([
#                 str(r["dtype"]),
#                 str(r["model"]),
#                 str(r["graph_window"]),
#                 str(r["best_case"]),
#                 f"{r['single_step_compile_ms']:.3f}" if r["single_step_compile_ms"] is not None else "",
#                 f"{r['multi_step_compile_ms']:.3f}" if r["multi_step_compile_ms"] is not None else "",
#                 str(r["kairos_case"] or ""),
#                 f"{r['kairos_ms']:.3f}" if r["kairos_ms"] is not None else "",
#                 f"{r['speedup_kairos_vs_single_compile']:.3f}x" if r["speedup_kairos_vs_single_compile"] is not None else "",
#                 f"{r['speedup_kairos_vs_multi_compile']:.3f}x" if r["speedup_kairos_vs_multi_compile"] is not None else "",
#                 str(r["fallback"]),
#                 str(r["temporal_triton"]),
#                 str(r["kernel_temporal_configs"]),
#             ])
#             + " |\n"
#         )

# print(f"Wrote {csv_path}")
# print(f"Wrote {md_path}")
# print(f"Wrote {json_path}")

# print("\n=== Summary ===")
# for r in rows:
#     dtype = r.get("dtype") or "unknown"
#     model = r.get("model") or "unknown"
#     graph_window = r.get("graph_window")
#     kairos = r.get("kairos_ms")
#     single_compile = r.get("single_step_compile_ms")
#     multi_compile = r.get("multi_step_compile_ms")
#     sp_single = r.get("speedup_kairos_vs_single_compile")
#     sp_multi = r.get("speedup_kairos_vs_multi_compile")

#     print(
#         f"{dtype:>5s} {model:>8s} "
#         f"w={str(graph_window):>2s} "
#         f"kairos={kairos:.3f} ms " if kairos is not None else
#         f"{dtype:>5s} {model:>8s} w={str(graph_window):>2s} kairos=N/A "
#     )

#     print(
#         f"      single_compile="
#         f"{single_compile:.3f} ms" if single_compile is not None else
#         f"      single_compile=N/A",
#         f"multi_compile="
#         f"{multi_compile:.3f} ms" if multi_compile is not None else
#         f"multi_compile=N/A",
#         f"speedup_vs_single="
#         f"{sp_single:.3f}x" if sp_single is not None else
#         f"speedup_vs_single=N/A",
#         f"speedup_vs_multi="
#         f"{sp_multi:.3f}x" if sp_multi is not None else
#         f"speedup_vs_multi=N/A",
#         f"fallback={r['fallback']}",
#     )

import json
import csv
from pathlib import Path

ROOT = Path("full_validation")


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_result(results, *names):
    for name in names:
        if name in results:
            return results[name]
    return None


def mean_ms(result):
    if isinstance(result, dict):
        return result.get("mean_ms")
    return None


def p50_ms(result):
    if isinstance(result, dict):
        return result.get("p50_ms")
    return None


def p90_ms(result):
    if isinstance(result, dict):
        return result.get("p90_ms")
    return None


def find_kairos_case(results, graph_window):
    candidates = [
        f"kairos_single_step_loop_compile_w{graph_window}",
        f"kairos_m_compile_w{graph_window}",
    ]

    for name in candidates:
        if name in results:
            return name, results[name]

    for name, result in results.items():
        if name.startswith("kairos") and f"w{graph_window}" in name:
            return name, result

    for name, result in results.items():
        if name.startswith("kairos"):
            return name, result

    return None, None


def get_case_stats(payload, case_name):
    if not case_name:
        return {}

    return safe_get(
        payload,
        "fused_op_call_stats_by_case",
        case_name,
        default={},
    ) or {}


def speedup(base, target):
    if base is None or target is None or target == 0:
        return None
    return base / target


rows = []

for summary_path in sorted(ROOT.rglob("benchmark_summary_all.json")):
    run_dir = summary_path.parent.name
    model_dir = summary_path.parent.parent.name

    inferred_dtype = "unknown"
    inferred_window = None

    parts = run_dir.split("_")
    if len(parts) >= 2 and parts[1].startswith("w"):
        inferred_dtype = parts[0]
        try:
            inferred_window = int(parts[1][1:])
        except ValueError:
            inferred_window = None

    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for model_name, payload in data.items():
        dtype = payload.get("dtype") or inferred_dtype

        candidate_windows = payload.get("candidate_windows", [])
        graph_window = (
            candidate_windows[0]
            if candidate_windows
            else inferred_window
        )

        results = payload.get("results", {})
        best = payload.get("best_case") or {}

        single_eager = get_result(
            results,
            "baseline_single_step_mode_eager",
            "baseline_s_eager",
        )
        single_compile = get_result(
            results,
            "baseline_single_step_mode_compile",
            "baseline_s_compile",
        )
        multi_eager = get_result(
            results,
            "baseline_multi_step_mode_eager",
            "baseline_m_eager",
        )
        multi_compile = get_result(
            results,
            "baseline_multi_step_mode_compile",
            "baseline_m_compile",
        )

        kairos_case, kairos_result = find_kairos_case(
            results,
            graph_window,
        )

        kairos_stats = get_case_stats(payload, kairos_case)

        best_case = best.get("case")
        best_mean = best.get("mean_ms")

        row = {
            "model_dir": model_dir,
            "run_dir": run_dir,
            "model": model_name,
            "dtype": dtype,
            "graph_window": graph_window,

            "best_case": best_case,
            "best_ms": best_mean,

            "single_step_eager_ms": mean_ms(single_eager),
            "single_step_compile_ms": mean_ms(single_compile),

            "multi_step_eager_ms": mean_ms(multi_eager),
            "multi_step_compile_ms": mean_ms(multi_compile),

            "kairos_case": kairos_case,
            "kairos_ms": mean_ms(kairos_result),
            "kairos_p50_ms": p50_ms(kairos_result),
            "kairos_p90_ms": p90_ms(kairos_result),

            "speedup_kairos_vs_single_compile": speedup(
                mean_ms(single_compile),
                mean_ms(kairos_result),
            ),
            "speedup_kairos_vs_single_eager": speedup(
                mean_ms(single_eager),
                mean_ms(kairos_result),
            ),
            "speedup_kairos_vs_multi_compile": speedup(
                mean_ms(multi_compile),
                mean_ms(kairos_result),
            ),
            "speedup_kairos_vs_multi_eager": speedup(
                mean_ms(multi_eager),
                mean_ms(kairos_result),
            ),

            "triton": kairos_stats.get("triton", 0),
            "fallback": kairos_stats.get("fallback", 0),
            "temporal_triton": kairos_stats.get("temporal_triton", 0),
            "temporal_fallback": kairos_stats.get("temporal_fallback", 0),

            "temporal_k3_s1_p1": kairos_stats.get("temporal_k3_s1_p1", 0),
            "temporal_k3_s2_p1": kairos_stats.get("temporal_k3_s2_p1", 0),
            "temporal_k7_s2_p3": kairos_stats.get("temporal_k7_s2_p3", 0),

            "single_k3_s1_p1": kairos_stats.get("single_k3_s1_p1", 0),
            "single_k3_s2_p1": kairos_stats.get("single_k3_s2_p1", 0),
            "single_k7_s2_p3": kairos_stats.get("single_k7_s2_p3", 0),

            "kernel_temporal_configs": kairos_stats.get(
                "kernel_temporal_configs",
                {},
            ),
            "fallback_reasons": kairos_stats.get(
                "fallback_reasons",
                {},
            ),
        }

        rows.append(row)


rows.sort(
    key=lambda r: (
        str(r["dtype"]),
        str(r["model"]),
        int(r["graph_window"] or -1),
    )
)

csv_path = ROOT / "full_validation_summary.csv"
json_path = ROOT / "full_validation_summary.json"
md_path = ROOT / "full_validation_summary.md"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

with json_path.open("w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, sort_keys=True)

headers = [
    "dtype",
    "model",
    "window",
    "best_case",
    "single_compile",
    "multi_compile",
    "kairos_case",
    "kairos",
    "vs_single_compile",
    "vs_multi_compile",
    "fallback",
    "temporal_triton",
    "kernel_temporal_configs",
]

with md_path.open("w", encoding="utf-8") as f:
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")

    for r in rows:
        f.write(
            "| "
            + " | ".join([
                str(r["dtype"]),
                str(r["model"]),
                str(r["graph_window"]),
                str(r["best_case"]),
                f"{r['single_step_compile_ms']:.3f}" if r["single_step_compile_ms"] is not None else "",
                f"{r['multi_step_compile_ms']:.3f}" if r["multi_step_compile_ms"] is not None else "",
                str(r["kairos_case"] or ""),
                f"{r['kairos_ms']:.3f}" if r["kairos_ms"] is not None else "",
                f"{r['speedup_kairos_vs_single_compile']:.3f}x" if r["speedup_kairos_vs_single_compile"] is not None else "",
                f"{r['speedup_kairos_vs_multi_compile']:.3f}x" if r["speedup_kairos_vs_multi_compile"] is not None else "",
                str(r["fallback"]),
                str(r["temporal_triton"]),
                str(r["kernel_temporal_configs"]),
            ])
            + " |\n"
        )

print(f"Wrote {csv_path}")
print(f"Wrote {md_path}")
print(f"Wrote {json_path}")

print("\n=== Summary ===")
for r in rows:
    dtype = r.get("dtype") or "unknown"
    model = r.get("model") or "unknown"
    graph_window = r.get("graph_window")
    kairos = r.get("kairos_ms")
    single_compile = r.get("single_step_compile_ms")
    multi_compile = r.get("multi_step_compile_ms")
    sp_single = r.get("speedup_kairos_vs_single_compile")
    sp_multi = r.get("speedup_kairos_vs_multi_compile")

    print(
        f"{dtype:>5s} {model:>8s} "
        f"w={str(graph_window):>2s} "
        f"kairos={kairos:.3f} ms " if kairos is not None else
        f"{dtype:>5s} {model:>8s} w={str(graph_window):>2s} kairos=N/A "
    )

    print(
        f"      single_compile="
        f"{single_compile:.3f} ms" if single_compile is not None else
        f"      single_compile=N/A",
        f"multi_compile="
        f"{multi_compile:.3f} ms" if multi_compile is not None else
        f"multi_compile=N/A",
        f"speedup_vs_single="
        f"{sp_single:.3f}x" if sp_single is not None else
        f"speedup_vs_single=N/A",
        f"speedup_vs_multi="
        f"{sp_multi:.3f}x" if sp_multi is not None else
        f"speedup_vs_multi=N/A",
        f"fallback={r['fallback']}",
    )