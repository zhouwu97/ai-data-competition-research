"""针对此前会漏过的元数据和产物错误，验证检查器确实失败。"""
from pathlib import Path
import copy
from contextlib import contextmanager
import hashlib
import json
import shutil
import tempfile
import unittest
from validate_research import ROOT, metadata_errors, evidence_errors

def sync_manifest(root):
    path = root / "examples/results/run_manifest.json"
    manifest = json.loads(path.read_text())
    for group in ["input_sha256", "source_sha256", "result_sha256"]:
        for name in manifest[group]:
            manifest[group][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))

@contextmanager
def research_copy():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        shutil.copytree(ROOT / "examples", root / "examples")
        manifest = json.loads((root / "examples/results/run_manifest.json").read_text())
        for name in manifest["source_sha256"]:
            target = root / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
        sync_manifest(root)
        yield root

class ResearchChecks(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT / "resources/catalog.json").read_text())

    def test_blank_evidence(self):
        data = copy.deepcopy(self.catalog)
        data["resources"][0]["verification"] = "  "
        self.assertTrue(any("verification" in x for x in metadata_errors(data)))

    def test_impossible_date(self):
        data = copy.deepcopy(self.catalog)
        data["resources"][0]["checked_at"] = "2026-02-30"
        self.assertTrue(any("checked_at" in x for x in metadata_errors(data)))

    def test_reserved_invalid_host(self):
        data = copy.deepcopy(self.catalog)
        data["resources"][0]["url"] = "https://competition.invalid/resource"
        self.assertTrue(any(".invalid" in x for x in metadata_errors(data)))

    def test_changed_result_is_detected(self):
        with research_copy() as root:
            path = root / "examples/results/task01/predictions.csv"
            path.write_text(path.read_text() + "changed\n")
            self.assertTrue(any("文件变化" in x for x in evidence_errors(root)))

    def test_false_claim_is_detected(self):
        with research_copy() as root:
            path = root / "examples/results/claims.json"
            claims = json.loads(path.read_text())
            claims[0]["value"] = 999
            path.write_text(json.dumps(claims))
            sync_manifest(root)
            self.assertTrue(any("结论数值与证据不一致" in x for x in evidence_errors(root)))

    def test_original_report_is_accepted(self):
        with research_copy() as root:
            self.assertEqual(evidence_errors(root), [])

    def test_wrong_report_row_is_detected_after_syncing_hash(self):
        with research_copy() as root:
            path = root / "examples/results/report.md"
            path.write_text(path.read_text().replace(
                "| C01 | 题一14天回测窗口数 | 4 |", "| C01 | 题一14天回测窗口数 | 999 |"))
            self.assertIn("4", path.read_text())  # 别处仍有4，不能拿它救C01。
            sync_manifest(root)
            self.assertTrue(any("C01: 报告数值" in x for x in evidence_errors(root)))

    def test_wrong_report_link_is_detected_after_syncing_hash(self):
        with research_copy() as root:
            path = root / "examples/results/report.md"
            path.write_text(path.read_text().replace(
                "[task01/summary.json](task01/summary.json)",
                "[task01/summary.json](task02/summary.json)"))
            sync_manifest(root)
            self.assertTrue(any("C01: 报告结果链接" in x for x in evidence_errors(root)))

    def test_missing_report_id_is_detected(self):
        with research_copy() as root:
            path = root / "examples/results/report.md"
            path.write_text("\n".join(line for line in path.read_text().splitlines()
                                      if not line.startswith("| C01 |")) + "\n")
            sync_manifest(root)
            self.assertTrue(any("C01: 汇总报告缺少结果行" in x for x in evidence_errors(root)))

    def test_duplicate_report_id_is_detected(self):
        with research_copy() as root:
            path = root / "examples/results/report.md"
            report = path.read_text()
            row = next(line for line in report.splitlines() if line.startswith("| C01 |"))
            path.write_text(report.replace(row, row + "\n" + row))
            sync_manifest(root)
            self.assertTrue(any("C01: 汇总报告结果编号重复" in x for x in evidence_errors(root)))

    def test_duplicate_claim_id_is_detected(self):
        with research_copy() as root:
            path = root / "examples/results/claims.json"
            claims = json.loads(path.read_text())
            claims.append(claims[0])
            path.write_text(json.dumps(claims))
            sync_manifest(root)
            self.assertTrue(any("C01: 结论记录编号重复" in x for x in evidence_errors(root)))

    def test_report_id_without_claim_is_detected(self):
        with research_copy() as root:
            path = root / "examples/results/claims.json"
            path.write_text(json.dumps(json.loads(path.read_text())[1:]))
            sync_manifest(root)
            self.assertTrue(any("C01: 汇总报告结果编号未登记" in x for x in evidence_errors(root)))

    def test_fraudulent_display_value_is_detected(self):
        with research_copy() as root:
            path = root / "examples/results/claims.json"
            claims = json.loads(path.read_text())
            claims[0]["display_value"] = "999"
            path.write_text(json.dumps(claims))
            report = root / "examples/results/report.md"
            report.write_text(report.read_text().replace(
                "| C01 | 题一14天回测窗口数 | 4 |", "| C01 | 题一14天回测窗口数 | 999 |"))
            sync_manifest(root)
            self.assertTrue(any("C01: 展示值与原始结论数值不符" in x for x in evidence_errors(root)))

if __name__ == "__main__":
    unittest.main()
