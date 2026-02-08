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

# Explicit executables to benchmark (all compared in a single run)
# NOTE: Ensure the filenames match exactly (especially the 16M one).
EXE_LIST = [
    "lbfgs_cublas_N4096.exe",
    "lbfgs_cublas_N65536.exe",
    "lbfgs_cublas_N1048576.exe",
    "lbfgs_cublas_N16777216.exe",
    "lbfgs_dotF64Kernel_N4096.exe",
    "lbfgs_dotF64Kernel_N65536.exe",
    "lbfgs_dotF64Kernel_N1048576.exe",
    "lbfgs_dotF64Kernel_N16777216.exe",
]

N_RE = re.compile(r"_N(\d+)\.exe$", re.IGNORECASE)

def exe_meta(exe_name_or_path: str):
    """
    Accepts either a basename or a full path.
    Returns (impl, N, label)

    impl: "cublas" | "dotF64Kernel" | "other"
    N: int | None
    label: basename without .exe
    """
    base = os.path.basename(exe_name_or_path)
    m = N_RE.search(base)
    N = int(m.group(1)) if m else None

    low = base.lower()
    if "cublas" in low:
        impl = "cublas"
    elif "dotf64kernel" in low:
        impl = "dotF64Kernel"
    else:
        impl = "other"

    label = base[:-4] if base.lower().endswith(".exe") else base
    return impl, N, label


def parse_test_time_ms(output: str, debug: bool = False):
    """
    Parses blocks like:
      Starting: Rosenbrock...
      ...
      Rosenbrock Final F: 4.291678e-12 (Target: 0)
      Time elapsed: 195.682 ms

    Returns: [{"name": "...", "time": 195.682, "final_f": ..., "target": ..., "abs_err": ...}, ...]
    """

    tests = []
    current_name = None
    pending_final = None  # dict with final_f/target/abs_err waiting to attach to next Time elapsed

    # Robust regex: "<Name> Final F: <num> (Target: <num>)"
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

            # Keep full label (helps when you embed dimension info in the test header)
            name = rest.strip() if rest else None

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


def run_bin(exe_path: str, debug: bool = False):
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
    """
    Generates an HTML report with:
      1) Dimension Summary: Avg runtime by N for cublas vs dotF64Kernel (per test)
      2) Per-test detailed table including avg/median time, samples, and optional accuracy stats
    No speedup calculations.
    """

    # test_name -> version(basename) -> list of dict samples
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

    # Compute averages/medians per test per version
    test_avgs = {}
    for test_name, per_ver in test_cases.items():
        test_avgs[test_name] = {}
        for v, samples in per_ver.items():
            times = [s.get("time") for s in samples if s.get("time") is not None]
            errs  = [s.get("abs_err") for s in samples if s.get("abs_err") is not None]
            finals = [s.get("final_f") for s in samples if s.get("final_f") is not None]

            test_avgs[test_name][v] = {
                "time_avg": statistics.mean(times) if times else None,
                "time_med": statistics.median(times) if times else None,
                "abs_err_avg": statistics.mean(errs) if errs else None,
                "abs_err_med": statistics.median(errs) if errs else None,
                "final_f_avg": statistics.mean(finals) if finals else None,
                "samples": len(times),
                "err_samples": len(errs),
            }

    # Dimension summary: test_name -> N -> impl -> time_avg
    by_dim = {}
    for test_name, per_ver in test_avgs.items():
        for ver, s in per_ver.items():
            impl, N, _ = exe_meta(ver)
            if N is None or s["time_avg"] is None:
                continue
            by_dim.setdefault(test_name, {}).setdefault(N, {})[impl] = s["time_avg"]

    def fmt_sci(x):
        if x is None:
            return "n/a"
        return f"{x:.3e}"

    def fmt_ms(x):
        if x is None:
            return "n/a"
        return f"{x:.3f}"

    # Sort versions nicely: by N, then impl, then name
    versions_sorted = sorted(
        versions,
        key=lambda v: (
            exe_meta(v)[1] if exe_meta(v)[1] is not None else 10**18,
            exe_meta(v)[0],
            v
        )
    )

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
      vertical-align: top;
    }}
    th {{
      background: #4CAF50;
      color: white;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.5px;
      position: sticky;
      top: 0;
    }}
    tr:hover {{
      background: #f9f9f9;
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
    .muted {{
      color: #999;
    }}
  </style>
</head>
<body>
  <h1>Benchmark Performance Report</h1>
  <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

  <div class="summary-section">
    <h2>Executables Compared (Single Run)</h2>
    <p class="metric">The benchmark compares the following executables in one run:</p>
    <ul>
"""

    for x in versions_sorted:
        impl, N, _ = exe_meta(x)
        html_content += f'      <li><code>{html.escape(x)}</code> <span class="muted">(impl={html.escape(impl)}, N={N if N is not None else "n/a"})</span></li>\n'

    html_content += """    </ul>
  </div>
"""

    # Dimension summary
    html_content += """
  <h2>Dimension Summary (Avg Time by N)</h2>
"""

    if not by_dim:
        html_content += """
  <div class="test-section">
    <h3>No dimension data</h3>
    <p class="metric">
      No N tags were parsed from executable names, or no per-test timing was parsed.
      Ensure filenames contain <code>_N123.exe</code> and stdout contains <code>Time elapsed: X ms</code>.
    </p>
  </div>
"""
    else:
        for test_name in sorted(by_dim.keys()):
            html_content += f"""
  <div class="test-section">
    <h3>{html.escape(test_name)} — Avg Time by N</h3>
    <table>
      <thead>
        <tr>
          <th>N</th>
          <th>cublas Avg (ms)</th>
          <th>dotF64Kernel Avg (ms)</th>
        </tr>
      </thead>
      <tbody>
"""
            for N in sorted(by_dim[test_name].keys()):
                cublas_t = by_dim[test_name][N].get("cublas")
                kern_t = by_dim[test_name][N].get("dotF64Kernel")

                html_content += f"""
        <tr>
          <td><strong>{N}</strong></td>
          <td>{fmt_ms(cublas_t)}</td>
          <td>{fmt_ms(kern_t)}</td>
        </tr>
"""
            html_content += """
      </tbody>
    </table>
  </div>
"""

    # Per-test detailed tables
    html_content += """
  <h2>Per Test Case Performance (Detailed)</h2>
"""

    if not test_cases:
        html_content += """
  <div class="test-section">
    <h3>No per-test data parsed</h3>
    <p class="metric">
      Your stdout did not match the expected patterns. Ensure your program prints lines like:
      <code>Starting: Rosenbrock...</code>,
      <code>Rosenbrock Final F: 1.23e-4 (Target: 0)</code>,
      <code>Time elapsed: 123.45 ms</code>.
    </p>
  </div>
"""
    else:
        for test_name in sorted(test_cases.keys()):
            html_content += f"""
  <div class="test-section">
    <h3>{html.escape(test_name)}</h3>
    <table>
      <thead>
        <tr>
          <th>Executable</th>
          <th>Impl</th>
          <th>N</th>
          <th>Avg Time (ms)</th>
          <th>Median (ms)</th>
          <th>Samples</th>
          <th>Avg |F - Target|</th>
          <th>Median |F - Target|</th>
          <th>Avg Final F</th>
        </tr>
      </thead>
      <tbody>
"""
            for v in versions_sorted:
                if test_name not in test_avgs or v not in test_avgs[test_name]:
                    continue

                s = test_avgs[test_name][v]
                avg_ms = s["time_avg"]
                med_ms = s["time_med"]
                samples = s["samples"]
                abs_err_avg = s["abs_err_avg"]
                abs_err_med = s["abs_err_med"]
                final_f_avg = s["final_f_avg"]

                impl, N, _ = exe_meta(v)

                html_content += f"""
        <tr>
          <td><strong>{html.escape(v)}</strong></td>
          <td>{html.escape(impl)}</td>
          <td>{N if N is not None else "n/a"}</td>
          <td>{avg_ms:.3f}</td>
          <td>{med_ms:.3f}</td>
          <td>{samples}</td>
          <td>{fmt_sci(abs_err_avg)}</td>
          <td>{fmt_sci(abs_err_med)}</td>
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


def main(N: int, debug: bool, directory: str):
    # Build absolute paths from EXE_LIST
    exes = [os.path.join(directory, x) for x in EXE_LIST]

    # Validate existence early
    missing = [x for x in exes if not os.path.exists(x)]
    if missing:
        print("ERROR: Missing executables:")
        for x in missing:
            print("  ", x)
        print("\nFix: put the listed .exe files into --dir, or update EXE_LIST.")
        return

    runs = []
    WARMUP_RUNS = 5

    for ex in exes:
        base = os.path.basename(ex)
        impl, Ndim, _ = exe_meta(base)

        print(f"\n=== Executable: {base} (impl={impl}, N={Ndim if Ndim is not None else 'n/a'}) ===")

        # Warmup runs (discarded)
        print(f"Running {WARMUP_RUNS} warmup runs (discarded)...")
        for i in range(WARMUP_RUNS):
            data = run_bin(ex, debug=False)
            if data is None:
                print(f"  Warmup {i+1}/{WARMUP_RUNS}: FAILED")
            else:
                print(f"  Warmup {i+1}/{WARMUP_RUNS}: OK")

        # Measured runs
        print(f"Running {N} measured runs...")
        for i in range(N):
            data = run_bin(ex, debug=debug)
            if data is None:
                print(f"  Run {i+1}/{N}: FAILED")
                continue

            wall_ms, test_data = data
            runs.append([ex, wall_ms, test_data])
            print(f"  Run {i+1}/{N}: recorded ({len(test_data)} tests parsed)")

    # Save raw JSON and HTML
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    generate_html_report(runs, HTML_REPORT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L-BFGS benchmark suite (dimension comparison, no speedup)")
    parser.add_argument("--runs", type=int, required=True, help="Number of measured runs of each executable")
    parser.add_argument("--dir", type=str, default="build", help="Directory containing executables")
    parser.add_argument("--mode", choices=["normal", "verbose"], default="normal", help="Console logs")
    args = parser.parse_args()

    debug = (args.mode == "verbose")
    main(args.runs, debug, args.dir)
