"""不用机器学习库，手算四个数字的K-means，观察“分组—求均值”的循环。"""
import csv
from pathlib import Path

def main():
    points = [1.0, 2.0, 8.0, 9.0]
    centers = [1.0, 2.0]  # 特意放得不好，让你看到中心会移动。
    previous_labels = None
    trace = []
    for step in range(1, 11):
        # 对每个点，找到距离最近的中心；相等时选编号更小的中心。
        labels = [min(range(2), key=lambda group: abs(point - centers[group]))
                  for point in points]
        updated = []
        for group in range(2):
            members = [point for point, label in zip(points, labels) if label == group]
            # 本例没有空组；若为空，保持旧中心只是此玩具程序的处理。
            updated.append(sum(members) / len(members) if members else centers[group])
        trace.append([step, centers[0], centers[1], ",".join(map(str, labels)),
                      updated[0], updated[1]])
        centers = updated
        if labels == previous_labels:
            break
        previous_labels = labels
    assert centers == [1.5, 8.5]
    folder = Path(__file__).resolve().parent / "results" / "kmeans_hand"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "trace.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["step", "old_center0", "old_center1", "labels",
                         "new_center0", "new_center1"])
        writer.writerows(trace)
    print("手算K-means完成：中心从1和2移动到1.5和8.5；见trace.csv。")

if __name__ == "__main__":
    main()
