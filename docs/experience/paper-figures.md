# 论文配图：使用本机 MathModel

本仓库按用户要求，论文配图优先使用本机`@mathmodeldesktop`提供的技能和脚本。当前安装根目录为`C:\Users\haha\AppData\Local\Programs\@mathmodeldesktop`；另一台电脑可设置`MATHMODEL_HOME`指向自己的安装目录。这里只调用已安装工具，不修改程序安装文件。

先写一句图要说明的发现，再选择入口。统计图不能由模板里的模拟数字代替真实实验；流程图的箭头表达先后、依赖或决策，不用连线装饰无关内容。

| 图的任务 | 安装目录内的入口 | 应保留什么 |
| --- | --- | --- |
| 数据折线、柱状、误差或其他统计图 | `resources/builtin-skills/nature-figure/SKILL.md` | 实际结果文件、脚本、单位与评价范围，以及PNG和矢量版本 |
| 明确需要内置绘图模板 | `resources/builtin-skills/mathmodel-figure-templates/SKILL.md` | 模板名称、修改后的脚本与真实输入；模板默认模拟图只作演示 |
| 研究路线、框架或算法流程 | `resources/builtin-skills/paper-diagram/SKILL.md` | 内容JSON或其他源、可编辑`.drawio`与实际导出的预览 |

统一路由见安装目录内的`resources/builtin-skills/mma-figure/SKILL.md`。本仓库已有Python数据分析代码，数据图沿用对应Python结果；若用户选择R或其他明确工具，则按用户选择处理。

## 已核对的调用方式

在仓库根目录的PowerShell中，先定位安装目录，再调用其公开脚本。例如生成[两套数据的研究路线源文件](../../figures/research_decisions.drawio)：

```powershell
$mathModelRoot = if ($env:MATHMODEL_HOME) { $env:MATHMODEL_HOME } else { Join-Path $env:LOCALAPPDATA 'Programs\@mathmodeldesktop' }
$diagramSkill = Join-Path $mathModelRoot 'resources\builtin-skills\paper-diagram'
python (Join-Path $diagramSkill 'scripts\framework_3col.py') figures/research_decisions.json -o figures/research_decisions.drawio
python (Join-Path $diagramSkill 'scripts\check_layout.py') figures/research_decisions.drawio --strict
```

图中仓储与智能分仓是并列分支，各自在一致条件下验证，再决定保留或回退。它说明建议的研究组织方法，不表示图中所有策略都已取得收益。内容源为[research_decisions.json](../../figures/research_decisions.json)。

内置模板入口可这样列出可用样式：

```powershell
$templateSkill = Join-Path $mathModelRoot 'resources\builtin-skills\mathmodel-figure-templates'
python (Join-Path $templateSkill 'scripts\render_template.py') --list
```

实际调用前读对应技能说明，按照脚本参数执行。不要把桌面程序的存在写成已核验的命令行接口，也不要编造`mathmodel.exe`的参数。

## 导出和交付

流程图导出使用`paper-diagram/scripts/export_figure.py`，该脚本另需可用的`drawio`命令。缺少时应说明当前只有可编辑源，不能声称PNG/PDF已导出。内置`preview_html.py --no-serve`可生成预览页，但预览依赖外部diagrams.net，包含非公开业务材料时不直接使用这一在线预览。

数值图使用项目实际结果，保留PNG与SVG/PDF等所需版本；导出后打开图，检查文字、单位、图例、箭头和正文数字。结果图与概念图分别标清，正文解释发现和用途，不把所有说明塞进图题。

更新论文写作技能时，保留这项工具偏好；技能只负责选择正确入口，不能把未运行、未导出或模拟的内容写成实际成果。
