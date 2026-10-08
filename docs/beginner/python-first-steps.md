# 怎样运行，怎样读代码

本仓库提供Python脚本，不需要先学Notebook。只读文档时可以直接在GitHub打开；运行时下载仓库ZIP、解压到自己的文件夹，再打开终端。

## 1. 文件夹与命令的位置

仓库根目录是有README.md、docs、examples、scripts的那层文件夹。终端里cd表示切换文件夹，例如Windows：

~~~powershell
cd C:\study\ai-data-competition-research-main
py -3.12 --version
~~~

路径应换成你实际解压的位置。下面推荐Python3.12；本次实际运行版本是3.12.14。尚未验证所有操作系统与Python版本。

## 2. Windows：先准备独立环境

~~~powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r examples\requirements.txt
.venv\Scripts\python.exe examples\05_kmeans_by_hand.py
.venv\Scripts\python.exe examples\run_all.py
.venv\Scripts\python.exe scripts\check_catalog.py
.venv\Scripts\python.exe scripts\validate_research.py
~~~

不用修改PowerShell执行策略，也不必先激活环境。每次直接调用.venv里的Python。若找不到py，先从[Python官网](https://www.python.org/downloads/)安装Python3.12，或使用已安装的对应解释器；不要把错误提示中的命令随意复制执行。

macOS / Linux：

~~~bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r examples/requirements.txt
.venv/bin/python examples/05_kmeans_by_hand.py
.venv/bin/python examples/run_all.py
.venv/bin/python scripts/check_catalog.py
.venv/bin/python scripts/validate_research.py
~~~

依赖安装需要联网。练习运行不下载数据、不需要GPU或付费API。虚拟环境依据：[Python官方中文说明](https://docs.python.org/zh-cn/3/tutorial/venv.html)。

## 3. 运行后看哪里？

examples/sample_inputs里是合成CSV输入；examples/results里是实际结果、报告、SVG图和日志。CSV可以用文本编辑器或表格软件看；直接双击SVG能看图。结果已经随仓库保存，你可以先读，再在自己电脑复现。

run_all会重新生成这些教学输入与结果，覆盖本练习同名文件。不要把自己的正式数据放进这些教学目录，应另建项目和配置。

## 4. 读代码先看五件事

| 写法 | 先这样理解 |
| --- | --- |
| import | 引入工具；比如pandas用来处理表 |
| def main() | 把主要工作放进一个函数 |
| pd.read_csv(...) | 读表；先检查文件和字段 |
| groupby(...).sum() | 按某个键分组，把每组数量相加 |
| merge(..., validate="one_to_one") | 按键合并；检查每边一条记录的约定 |
| fit / fit_predict | 学规则；不是自动证明结果有效 |
| to_csv(..., index=False) | 保存结果，不额外写行号 |
| assert | 预期必须成立，否则停止；不是业务正确性的完整证明 |
| if __name__ == "__main__" | 直接运行此文件时调用主要工作 |
| #之后的文字 | 注释，解释目的，不会被当成Python指令 |

先看每个练习的模块说明和main函数，再逐步研究。common.py是共用读写工具，第一遍可以跳过。不要一次修改多个地方；改前保存原结果，再比较相同输入和验证。

## 5. 常见报错怎样处理？

ModuleNotFoundError通常是当前解释器没有装所需依赖。确认安装和运行都使用同一个.venv里的Python。

FileNotFoundError先检查工作目录和输入文件；练习提示缺少输入时先运行00_make_data.py。

ValueError常表示输入违反已声明规则，例如日期缺失、字段不存在或时间切分不成立。应该查数据与口径，不是直接删掉报错检查。

把错误交给AI时，提供执行命令、最后的报错类型、相关代码与必要匿名样例；不要只说“跑不动”，也不要让AI把检查去掉就算修复。
