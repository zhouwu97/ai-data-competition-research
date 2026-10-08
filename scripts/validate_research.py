"""先核对版本与引用，再调用独立公式重算指定教学指标。"""
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json
import os
import re
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TEXT_FIELDS = ["id", "title", "competition", "type", "url", "priority", "value",
               "limitation", "next_action", "verification", "checked_at"]

def metadata_errors(data):
    errors, ids, urls = [], set(), set()
    if not isinstance(data, dict) or not isinstance(data.get("resources"), list):
        return ["catalog根对象必须含resources数组"]
    for index, item in enumerate(data["resources"]):
        if not isinstance(item, dict):
            errors.append(f"第{index}项不是对象")
            continue
        label = item.get("id", str(index))
        for field in TEXT_FIELDS:
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"{label}: {field}缺失、非字符串或空白")
        if any(not isinstance(item.get(f), str) or not item[f].strip() for f in TEXT_FIELDS):
            continue
        if item["id"] in ids or item["url"] in urls:
            errors.append(f"{label}: ID或URL重复")
        ids.add(item["id"])
        urls.add(item["url"])
        if item["priority"] not in ["P0", "P1", "P2"]:
            errors.append(f"{label}: priority无效")
        parts = urlsplit(item["url"])
        if parts.scheme not in ["http", "https"] or not parts.hostname:
            errors.append(f"{label}: URL缺少合法协议或主机")
        elif parts.hostname.endswith(".invalid"):
            errors.append(f"{label}: URL使用保留的.invalid无效域")
        value = item["checked_at"]
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError()
            date.fromisoformat(value)
        except ValueError:
            errors.append(f"{label}: checked_at必须为有效YYYY-MM-DD日期")
    return errors

def report_claim_rows(report):
    """只读现有四列结果表，按证据ID取行；正文中碰巧出现数字不算证据。"""
    records, errors, in_table, found = {}, [], False, False
    header = ["证据编号", "本次结果", "数值", "结果文件"]
    for line in report.splitlines():
        if not line.strip().startswith("|"):
            in_table = False
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if cells == header:
            in_table, found = True, True
            continue
        if not in_table or all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        if len(cells) != 4:
            errors.append("汇总报告的结果表应有4列")
            continue
        records.setdefault(cells[0], []).append(cells)
    if not found:
        errors.append("汇总报告缺少证据编号/本次结果/数值/结果文件表")
    return records, errors

def report_claim_errors(root, report_path, report, claims):
    records, errors = report_claim_rows(report)
    seen = set()
    for claim in claims:
        identifier = claim["id"]
        if identifier in seen:
            errors.append(f"{identifier}: 结论记录编号重复")
            continue
        seen.add(identifier)
        matching = records.get(identifier, [])
        if not matching:
            errors.append(f"{identifier}: 汇总报告缺少结果行")
            continue
        if len(matching) != 1:
            errors.append(f"{identifier}: 汇总报告结果编号重复")
            continue
        _, _, number, link = matching[0]
        displayed = claim["display_value"]
        if number != displayed:
            errors.append(f"{identifier}: 报告数值应为{displayed}，实际为{number}")
        # 显示精度由现有display_value决定，同时检查它不是另造的数字。
        if not isinstance(displayed, str) or not re.fullmatch(r"-?\d+(?:\.\d+)?", displayed):
            errors.append(f"{identifier}: 展示值必须为普通十进制数字")
        else:
            places = len(displayed.partition(".")[2])
            value = claim["value"]
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or f"{value:.{places}f}" != displayed):
                errors.append(f"{identifier}: 展示值与原始结论数值不符")
        match = re.fullmatch(r"\[[^\]]+\]\(([^\s()]+)\)", link)
        target = match.group(1) if match else ""
        parts = urlsplit(target)
        if (not target or target.startswith("/") or parts.scheme or parts.netloc
                or parts.query or parts.fragment
                or (report_path.parent / target).resolve() != (root / claim["result"]).resolve()):
            errors.append(f"{identifier}: 报告结果链接与证据文件不一致")
    for identifier in records.keys() - seen:
        errors.append(f"{identifier}: 汇总报告结果编号未登记")
    return errors

def evidence_errors(root):
    errors = []
    manifest_path = root / "examples/results/run_manifest.json"
    if not manifest_path.exists():
        return ["缺少教学运行记录；先运行python examples/run_all.py"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        return [f"教学本次运行状态不是success：{manifest.get('status', '未记录')}；查看report.md与run.log"]
    if manifest.get("data_kind") != "synthetic":
        errors.append("教学运行记录必须明确synthetic")
    for group in ["input_sha256", "source_sha256", "result_sha256"]:
        if not manifest.get(group):
            errors.append(f"运行记录缺少{group}")
        for name, expected in manifest.get(group, {}).items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                errors.append(f"产物缺失或越界：{name}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"文件变化未同步运行记录：{name}")
    claims_path = root / "examples/results/claims.json"
    if not claims_path.exists():
        return errors + ["缺少结论追溯表claims.json"]
    report_path = root / "examples/results/report.md"
    report = report_path.read_text(encoding="utf-8")
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    errors += report_claim_errors(root, report_path, report, claims)
    for claim in claims:
        path = (root / claim["result"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f"{claim['id']}: 证据文件缺失或越界")
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        try:
            for key in claim["keys"]:
                value = value[key]
            if value != claim["value"]:
                errors.append(f"{claim['id']}: 结论数值与证据不一致")
        except (KeyError, TypeError):
            errors.append(f"{claim['id']}: 证据字段缺失")
        if claim["data_kind"] != "synthetic":
            errors.append(f"{claim['id']}: 教学结论未标明合成数据")
    return errors

def main():
    errors = metadata_errors(json.loads((ROOT / "resources/catalog.json").read_text(encoding="utf-8")))
    errors += evidence_errors(ROOT)
    if errors:
        raise SystemExit("\n".join(errors))
    print("OK: 元数据、文件版本和4项报告引用一致。", flush=True)
    # 错误公式可以生成彼此一致的文件，所以这一步不再只比较JSON。
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for script in ["check_calculations.py", "check_public_research.py"]:
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], env=env)
        if result.returncode:
            raise SystemExit(result.returncode)
    print("指定计算已独立复核。检查范围见 tests/README.md。")

if __name__ == "__main__":
    main()
