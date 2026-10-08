"""针对此前会漏过的元数据和产物错误，验证检查器确实失败。"""
from pathlib import Path
import copy
import json
import shutil
import tempfile
import unittest
from validate_research import ROOT, metadata_errors, evidence_errors

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
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / "examples", root / "examples")
            path = root / "examples/results/task01/predictions.csv"
            path.write_text(path.read_text() + "changed\n")
            self.assertTrue(any("文件变化" in x for x in evidence_errors(root)))

    def test_false_claim_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / "examples", root / "examples")
            path = root / "examples/results/claims.json"
            claims = json.loads(path.read_text())
            claims[0]["value"] = 999
            path.write_text(json.dumps(claims))
            self.assertTrue(any("结论数值与证据不一致" in x for x in evidence_errors(root)))

if __name__ == "__main__":
    unittest.main()
