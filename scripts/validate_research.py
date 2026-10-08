"""离线检查元数据、教学产物摘要与结论数值；不判断观点正确或外网可用。"""
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json
import re
import sys

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

def evidence_errors(root):
    errors = []
    manifest_path = root / "examples/results/run_manifest.json"
    if not manifest_path.exists():
        return ["缺少教学运行记录；先运行python examples/run_all.py"]
    manifest = json.loads(manifest_path.read_text())
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
    report = (root / "examples/results/report.md").read_text()
    for claim in json.loads(claims_path.read_text()):
        path = (root / claim["result"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f"{claim['id']}: 证据文件缺失或越界")
            continue
        value = json.loads(path.read_text())
        try:
            for key in claim["keys"]:
                value = value[key]
            if value != claim["value"]:
                errors.append(f"{claim['id']}: 结论数值与证据不一致")
        except (KeyError, TypeError):
            errors.append(f"{claim['id']}: 证据字段缺失")
        if claim["data_kind"] != "synthetic":
            errors.append(f"{claim['id']}: 教学结论未标明合成数据")
        if claim["display_value"] not in report:
            errors.append(f"{claim['id']}: 汇总报告缺少对应展示值")
    return errors

def main():
    errors = metadata_errors(json.loads((ROOT / "resources/catalog.json").read_text()))
    errors += evidence_errors(ROOT)
    if errors:
        raise SystemExit("\n".join(errors))
    print("OK: 非空元数据、日期、URL结构、教学文件摘要和4项结论数值检查通过。")
    print("未检查外网可用性、所有报告句子、赛规符合性或真实业务收益；仍需人工审核。")

if __name__ == "__main__":
    main()
