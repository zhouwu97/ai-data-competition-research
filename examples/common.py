"""几个练习共用的文件读写与绘图设置，先不用逐行研究这个文件。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

# 不打开桌面窗口，直接保存图，云电脑和自己的电脑都能运行。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INPUT = HERE / "sample_inputs"
RESULT = HERE / "results"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.hashsalt": "competition-study"})

def output_dir(name):
    folder = RESULT / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def read_input(name, required, input_dir=None):
    path = (Path(input_dir) if input_dir is not None else INPUT) / name
    if not path.exists():
        if input_dir is not None:
            raise FileNotFoundError(f"缺少输入文件：{path}")
        raise FileNotFoundError(f"先运行 python examples/00_make_data.py；缺少 {path.name}")
    # 编号保留原字符串；001和1可能是不同客户，不能自动转成同一个整数。
    identifiers = {field: "string" for field in required
                   if field.endswith("_id") or field in ["region", "sku"]}
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=identifiers)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} 缺少字段：{sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(f"{name} 为空，或必需字段有缺失。请先解释和处理缺失。")
    return frame

def write_csv(frame, path):
    # index=False 避免把 pandas 的行号误当作业务字段写入 CSV。
    frame.to_csv(path, index=False, float_format="%.6f", encoding="utf-8", lineterminator="\n")

def write_json(value, path):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8", newline="\n")

def save_figure(fig, path):
    fig.tight_layout()
    # 去掉 SVG 内部生成日期，让相同输入的图更容易比较。
    fig.savefig(path, metadata={"Date": None})
    # 绘图库的SVG路径带行尾空格；规范化文本，不改变图形或计算。
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                    + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)

def assert_future_split(train_dates, valid_dates):
    # 这里专门检查“预测未来”的时间切分；它不是所有任务通用的切分规则。
    if pd.to_datetime(train_dates).max() >= pd.to_datetime(valid_dates).min():
        raise ValueError("训练日期没有严格早于验证日期：不能把这次比较称为未来回测。")

def assert_same_columns(training_columns, inference_columns):
    # 只检查输入字段及顺序；不能代替对 OOF 生成方式和预处理过程的审核。
    if list(training_columns) != list(inference_columns):
        raise ValueError("训练与推理字段或顺序不同，必须先统一接口。")
