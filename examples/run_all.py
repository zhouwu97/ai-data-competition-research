"""从仓库根目录运行：python examples/run_all.py。失败时停止，不伪造成功记录。"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import numpy
import pandas
import sklearn
import matplotlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT = HERE / "results"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    RESULT.mkdir(exist_ok=True)
    env = os.environ.copy()
    # 本练习很小，单线程避免在多人服务器上无意义地开很多线程。
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    steps = ["00_make_data.py", "01_demand_forecast.py", "02_warehouse_eiq_abc.py",
             "03_dc_locations.py", "04_customer_segments.py", "05_kmeans_by_hand.py",
             "failure_cases.py"]
    log_path = RESULT / "run.log"
    step_seconds = {}
    with log_path.open("w", encoding="utf-8") as log:
        for step in steps:
            cmd = [sys.executable, str(HERE / step)]
            log.write(f"$ python examples/{step}\n")
            log.flush()
            started = time.perf_counter()
            proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            step_seconds[step] = round(time.perf_counter() - started, 4)
            log.write(proc.stdout + proc.stderr + f"exit_code={proc.returncode}\n"
                      f"elapsed_seconds={step_seconds[step]}\n\n")
            log.flush()
            print(proc.stdout.strip())
            if proc.returncode:
                raise RuntimeError(f"{step} 失败。先看 examples/results/run.log，再修复。")
    log_path.write_text(log_path.read_text().rstrip() + "\n", encoding="utf-8")
    summaries = {name: json.loads((RESULT / name / "summary.json").read_text())
                 for name in ["task01", "task02", "task03", "task04"]}
    claims = [
        {"id": "C01", "value": summaries["task01"]["windows"],
         "result": "examples/results/task01/summary.json", "keys": ["windows"]},
        {"id": "C02", "value": summaries["task02"]["total_quantity"],
         "result": "examples/results/task02/summary.json", "keys": ["total_quantity"]},
        {"id": "C03", "value": summaries["task03"]["weighted_distance_km"],
         "result": "examples/results/task03/summary.json", "keys": ["weighted_distance_km"]},
        {"id": "C04", "value": summaries["task04"]["silhouette_k3"],
         "result": "examples/results/task04/summary.json", "keys": ["silhouette_k3"]}]
    for claim in claims:
        claim["data_kind"] = "synthetic"
        claim["display_value"] = str(claim["value"]) if isinstance(claim["value"], int) else f"{claim['value']:.4f}"
    (RESULT / "claims.json").write_text(json.dumps(claims, ensure_ascii=False, indent=2) + "\n")
    report = "# 四类教学练习：实际执行结果\n\n"
    report += ("2026-10-08建立的原创教学练习。全部输入是合成数据，"
               "没有取得正式赛题数据，没有提交比赛或证明获奖效果。\n\n")
    report += "| 证据编号 | 本次结果 | 数值 | 结果文件 |\n| --- | --- | --- | --- |\n"
    descriptions = ["题一14天回测窗口数", "题二出库总件数",
                    "题三演示k=3的需求加权距离（公里）", "题四演示k=3的轮廓系数"]
    for claim, description in zip(claims, descriptions):
        relative = Path(claim["result"]).relative_to("examples/results").as_posix()
        report += f"| {claim['id']} | {description} | {claim['display_value']} | [{relative}]({relative}) |\n"
    report += ("\n这些量回答不同问题，不能据此排列四个赛题的获奖机会。"
               "选题表中的真实数据和评分证据仍是待取得状态。\n\n"
               "详细记录：[题一](task01/report.md)、[题二](task02/report.md)、"
               "[题三](task03/report.md)、[题四](task04/report.md)、"
               "[错误示范](failure_cases/report.md)。\n\n"
               "复查：[命令日志](run.log)、[版本与SHA256](run_manifest.json)、"
               "[结论对应数值](claims.json)。全部步骤实际返回0后，才会写入本汇总。\n")
    (RESULT / "report.md").write_text(report, encoding="utf-8")
    # 原始输入、源码和结果分别留摘要；摘要能发现文件变化，不能证明研究结论正确。
    input_files = sorted((HERE / "sample_inputs").glob("*.csv"))
    source_files = sorted(HERE.glob("*.py"))
    result_files = sorted(p for p in RESULT.rglob("*")
                          if p.is_file() and p.name != "run_manifest.json")
    manifest = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python examples/run_all.py", "data_kind": "synthetic",
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__,
                        "pandas": pandas.__version__, "scikit-learn": sklearn.__version__,
                        "matplotlib": matplotlib.__version__},
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in input_files},
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files},
        "result_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in result_files},
        "completed_steps": steps, "step_seconds": step_seconds}
    (RESULT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("所有练习完成。打开 examples/results/report.md 查看已执行结果和边界。")

if __name__ == "__main__":
    main()
