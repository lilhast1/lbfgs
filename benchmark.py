#!/usr/bin/env python3
import os
import subprocess
import statistics
import argparse
import stat
from time import perf_counter
import html
from datetime import datetime
import json
import re

OUT_DIR = "benchmark"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RAW_JSON = f"{OUT_DIR}/benchmark_raw_{TS}.json"
HTML_REPORT = f"{OUT_DIR}/benchmark_report_{TS}.html"

# CPU baselines (ms) for speedup calculation
CPU_BASELINES_MS = {
    "Rastrigin": 2326.843,
    "Ackley": 2039.391,
    "Rosenbrock": 57395.111,
}

def find_exec_files(directory: str, debug=False):
    executable = stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    exes = []
    for filename in os.listdir(directory):
        full = os.path.join(directory, filename)
        if os.path.isfile(full):
            st = os.stat(full)
            if st.st_mode & executable:
                if debug:
                    print(full, oct(st.st_mode))
                exes.append(full)
    return exes


def parse_test_time_ms(output: str, debug=False):
    """
    Parses blocks like:
      Starting: Rosenbrock...
      ...
      Rosenbrock Final F: 4.291678e-12 (Target: 0)
      Time elapsed: 195.682 ms

    Returns: [{"name": "Rosenbrock", "time": 195.682, "final_f": 4.29e-12, "target": 0.0, "abs_err": ...}, ...]
    """
    tests = []
    current_name = None
    pending_final = None  # dict with final_f/target/abs_err waiting to attach to next Time elapsed

    # Robust regex: "<Name> Final F: <num> (Target: <num>)"
    # Accepts scientific notation, signs, decimal points
    final_re = re.compile(
        r"""^(?P<name>[A-Za-z0-9_]+)\s+Final\s+F:\s+(?P<final>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+
            \(\s*Target:\s*(?P<target>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)\s*$""",
        re.VERBOSE
    )

    for raw in output.splitlines():
        line = raw.strip()

        # Match "Starting:" anywhere (tolerate prefixes)
        if "Starting:" in line:
            rest = line.split("Starting:", 1)[1].strip()
            if rest.endswith("..."):
                rest = rest[:-3].strip()
            name = rest.split()[0].strip() if rest else None
            if name:
                current_name = name
                pending_final = None
            continue

        # Match "X Final F: ... (Target: ...)" anywhere
        m = final_re.match(line)
        if m:
            name = m.group("name")
            try:
                final_f = float(m.group("final"))
                target = float(m.group("target"))
            except Exception:
                if debug:
                    print(f"[parse] failed to parse final/target from: {line!r}")
                continue

            abs_err = abs(final_f - target)
            pending_final = {"final_f": final_f, "target": target, "abs_err": abs_err}

            # If name wasn't set by "Starting:", use this
            if not current_name:
                current_name = name
            continue

        # Match time line
        if "Time elapsed" in line:
            try:
                after_colon = line.split(":", 1)[1].strip()
                num_str = after_colon.split()[0].strip()
                t_ms = float(num_str)
            except Exception:
                if debug:
                    print(f"[parse] failed to parse time from: {line!r}")
                continue

            rec = {"name": current_name or "UNKNOWN", "time": t_ms}
            if pending_final:
                rec.update(pending_final)
            tests.append(rec)

            current_name = None
            pending_final = None

    if debug:
        print(f"[parse] parsed {len(tests)} tests: {tests}")
    return tests


def run_bin(exe_path: str, debug=False):
    try:
        t0 = perf_counter()
        result = subprocess.run([exe_path], capture_output=True, text=True, check=True)
        t1 = perf_counter()
        wall_ms = (t1 - t0) * 1e3

        if debug:
            print("Standard Output:")
            print(result.stdout)

        test_data = parse_test_time_ms(result.stdout, debug=debug)

        if result.stderr:
            print("\nStandard Error:")
            print(result.stderr)

        return (wall_ms, test_data)

    except subprocess.CalledProcessError as e:
        print(f"Error: execution failed with exit code {e.returncode}")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Standard Error:\n{e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def generate_html_report(runs_data, output_file):
    # test_name -> version -> list of dict samples: {"time":..., "final_f":..., "target":..., "abs_err":...}
    test_cases = {}
    versions = set()

    for run in runs_data:
        exe_path, _, tests = run
        version = os.path.basename(exe_path)
        versions.add(version)

        for t in tests:
            name = t.get("name", "UNKNOWN")
            tm = t.get("time", None)
            if tm is None:
                continue

            test_cases.setdefault(name, {}).setdefault(version, []).append(t)

    versions = sorted(versions)

    # averages (time + abs_err) per test/per version
    test_avgs = {}
    for test_name, per_ver in test_cases.items():
        test_avgs[test_name] = {}
        for v, samples in per_ver.items():
            times = [s["time"] for s in samples if "time" in s and s["time"] is not None]
            errs  = [s["abs_err"] for s in samples if "abs_err" in s and s["abs_err"] is not None]
            finals = [s["final_f"] for s in samples if "final_f" in s and s["final_f"] is not None]

            test_avgs[test_name][v] = {
                "time_avg": statistics.mean(times) if times else None,
                "time_med": statistics.median(times) if times else None,
                "abs_err_avg": statistics.mean(errs) if errs else None,
                "abs_err_med": statistics.median(errs) if errs else None,
                "final_f_avg": statistics.mean(finals) if finals else None,
                "samples": len(times),
                "err_samples": len(errs),
            }

    # total speedup across CPU-known tests
    cpu_known = list(CPU_BASELINES_MS.keys())
    cpu_total = sum(CPU_BASELINES_MS.values())
    total_gpu_by_version = {}

    for v in versions:
        ok = True
        s = 0.0
        for tn in cpu_known:
            if tn not in test_avgs or v not in test_avgs[tn] or test_avgs[tn][v]["time_avg"] is None:
                ok = False
                break
            s += test_avgs[tn][v]["time_avg"]
        if ok:
            total_gpu_by_version[v] = s

    def speedup_class(x):
        if x >= 2.0:
            return "fast"
        if x >= 1.0:
            return "baseline"
        return "slow"

    def fmt_sci(x):
        if x is None:
            return "n/a"
        # compact scientific formatting
        return f"{x:.3e}"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Benchmark Performance Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
      background: #f5f5f5;
    }}
    h1 {{
      color: #333;
      border-bottom: 3px solid #4CAF50;
      padding-bottom: 10px;
    }}
    h2 {{
      color: #555;
      margin-top: 30px;
    }}
    .summary-section {{
      background: white;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 30px;
    }}
    .test-section {{
      background: white;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
    }}
    th, td {{
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid #ddd;
    }}
    th {{
      background: #4CAF50;
      color: white;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.5px;
    }}
    tr:hover {{
      background: #f9f9f9;
    }}
    .speedup {{
      font-weight: 600;
      padding: 4px 8px;
      border-radius: 4px;
      display: inline-block;
      min-width: 60px;
      text-align: center;
    }}
    .speedup.fast {{
      background: #d4edda;
      color: #155724;
    }}
    .speedup.baseline {{
      background: #fff3cd;
      color: #856404;
    }}
    .speedup.slow {{
      background: #f8d7da;
      color: #721c24;
    }}
    .metric {{
      font-size: 14px;
      color: #666;
    }}
    .timestamp {{
      color: #888;
      font-size: 14px;
    }}
    code {{
      background: #eee;
      padding: 2px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>Benchmark Performance Report</h1>
  <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

  <div class="summary-section">
    <h2>CPU Baselines Used</h2>
    <p class="metric">
      Speedups are computed against these fixed CPU runtimes (ms):
      <code>Rastrigin={CPU_BASELINES_MS["Rastrigin"]}</code>,
      <code>Ackley={CPU_BASELINES_MS["Ackley"]}</code>,
      <code>Rosenbrock={CPU_BASELINES_MS["Rosenbrock"]}</code>.
    </p>
  </div>

  <div class="summary-section">
    <h2>Total Speedup (Ackley + Rastrigin + Rosenbrock)</h2>
    <p class="metric">CPU Total: <strong>{cpu_total:.3f} ms</strong></p>
    <table>
      <thead>
        <tr>
          <th>Version</th>
          <th>GPU Total Avg (ms)</th>
          <th>Total Speedup vs CPU</th>
        </tr>
      </thead>
      <tbody>
"""

    if not total_gpu_by_version:
        html_content += """
        <tr><td colspan="3">Not enough data to compute total speedup (missing one or more tests).</td></tr>
"""
    else:
        for v in versions:
            if v not in total_gpu_by_version:
                continue
            gpu_total = total_gpu_by_version[v]
            sp = cpu_total / gpu_total if gpu_total > 0 else 0.0
            html_content += f"""
        <tr>
          <td><strong>{html.escape(v)}</strong></td>
          <td>{gpu_total:.3f} ms</td>
          <td><span class="speedup {speedup_class(sp)}">{sp:.2f}x</span></td>
        </tr>
"""

    html_content += """
      </tbody>
    </table>
  </div>

  <h2>Per Test Case Performance</h2>
"""

    if not test_cases:
        html_content += """
  <div class="test-section">
    <h3>No per-test data parsed</h3>
    <p class="metric">
      Your stdout did not match the expected patterns. Ensure your program prints lines like:
      <code>Starting: Rosenbrock...</code> and <code>Rosenbrock Final F: 1.23e-4 (Target: 0)</code> and <code>Time elapsed: 123.45 ms</code>.
    </p>
  </div>
"""
    else:
        for test_name in sorted(test_cases.keys()):
            cpu_base = CPU_BASELINES_MS.get(test_name)

            html_content += f"""
  <div class="test-section">
    <h3>{html.escape(test_name)}</h3>
    <table>
      <thead>
        <tr>
          <th>Version</th>
          <th>GPU Avg Time (ms)</th>
          <th>Samples</th>
          <th>CPU Baseline (ms)</th>
          <th>Speedup vs CPU</th>
          <th>Avg |F - Target|</th>
          <th>Avg Final F</th>
        </tr>
      </thead>
      <tbody>
"""
            for v in versions:
                if test_name not in test_avgs or v not in test_avgs[test_name]:
                    continue

                stats_v = test_avgs[test_name][v]
                avg_ms = stats_v["time_avg"]
                samples = stats_v["samples"]
                abs_err_avg = stats_v["abs_err_avg"]
                final_f_avg = stats_v["final_f_avg"]

                if cpu_base is not None and avg_ms and avg_ms > 0:
                    sp = cpu_base / avg_ms
                    sp_html = f'<span class="speedup {speedup_class(sp)}">{sp:.2f}x</span>'
                    cpu_str = f"{cpu_base:.3f}"
                else:
                    sp_html = '<span class="speedup baseline">n/a</span>'
                    cpu_str = "n/a"

                html_content += f"""
        <tr>
          <td><strong>{html.escape(v)}</strong></td>
          <td>{avg_ms:.3f} ms</td>
          <td>{samples}</td>
          <td>{cpu_str}</td>
          <td>{sp_html}</td>
          <td>{fmt_sci(abs_err_avg)}</td>
          <td>{fmt_sci(final_f_avg)}</td>
        </tr>
"""

            html_content += """
      </tbody>
    </table>
  </div>
"""

    html_content += """
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML report generated: {output_file}")


def main(N, debug, directory):
    exes = find_exec_files(directory=directory, debug=debug)
    runs = []

    WARMUP_RUNS = 5

    for ex in exes:
        print(f"\n=== Executable: {os.path.basename(ex)} ===")

        # --------------------
        # Warmup runs (discarded)
        # --------------------
        print(f"Running {WARMUP_RUNS} warmup runs...")
        for i in range(WARMUP_RUNS):
            data = run_bin(ex, debug=False)
            if data is None:
                print(f"  Warmup {i+1}/{WARMUP_RUNS}: FAILED")
            else:
                print(f"  Warmup {i+1}/{WARMUP_RUNS}: OK")

        # --------------------
        # Measured runs
        # --------------------
        print(f"Running {N} measured runs...")
        for i in range(N):
            data = run_bin(ex, debug=debug)
            if data is None:
                print(f"  Run {i+1}/{N}: FAILED")
                continue

            wall_ms, test_data = data
            runs.append([ex, wall_ms, test_data])
            print(f"  Run {i+1}/{N}: recorded")

    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    generate_html_report(runs, HTML_REPORT)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L-BFGS benchmark suite")
    parser.add_argument("--runs", type=int, required=True, help="Number of runs of each executable")
    parser.add_argument("--dir", type=str, default="build", help="Directory containing executables")
    parser.add_argument("--mode", choices=["normal", "verbose"], default="normal", help="Console logs")
    args = parser.parse_args()

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    main(args.runs, args.mode == "verbose", args.dir)
