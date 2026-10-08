"""按赛事、关键词和优先级缩小阅读范围，不自动推荐获奖路线。"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--competition", default="", help="例如长风杯、海豚杯")
parser.add_argument("--keyword", default="", help="在标题、用途和限制里搜索")
parser.add_argument("--priority", choices=["P0", "P1", "P2"])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
items = json.loads((root / "resources/catalog.json").read_text())["resources"]
selected = []
for item in items:
    if args.competition and args.competition not in item["competition"]:
        continue
    if args.priority and item["priority"] != args.priority:
        continue
    text = " ".join(str(item.get(key, "")) for key in ["title", "value", "limitation"])
    if args.keyword.casefold() not in text.casefold():
        continue
    selected.append(item)
for item in selected:
    print(f"{item['id']} {item['priority']} {item['title']}")
    print(f"  {item['url']}")
    print(f"  核查：{item['verification']}；限制：{item['limitation']}")
print(f"共{len(selected)}条；优先级针对用途，核查状态不代表已复现。")
