"""吉林二轮研究：重建记录量结构、配对时长诊断、图和报告，不训练新模型。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from common import plt, save_figure, write_json
from volume_structure import analyze as analyze_volume, EXPECTED_SHA
from duration_mechanism import analyze as analyze_duration


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(out, name):
    return pd.read_csv(out / name, encoding="utf-8")


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def clock_text(minutes):
    return f"{int(minutes)//60}:{int(minutes)%60:02d}"


def reference(out, target):
    # 换输出目录后，报告仍能找到仓库里的计划和证据说明；不同磁盘用公开来源链接。
    try:
        return Path(os.path.relpath(target.resolve(), out.resolve())).as_posix()
    except ValueError:
        return "https://github.com/zhouwu97/ai-data-competition-research/blob/main/" + target.relative_to(ROOT).as_posix()


def save_plot(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=300)
    save_figure(fig, out / f"{name}.svg")


def plots(out):
    from matplotlib.dates import DateFormatter, DayLocator

    daily = read_table(out, "volume_daily.csv")
    daily["date"] = pd.to_datetime(daily.date)
    shown = daily[(daily.clock == "delivery") & daily.date.between("2022-09-17", "2022-10-17")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    for ax, column, title, ylabel in zip(axes.flat[:3],
            ["records", "active_couriers", "records_per_active_courier"],
            ["A. Daily completed records", "B. Couriers observed each day", "C. Records per observed courier"],
            ["Records/day", "Distinct couriers/day", "Records/courier-day"]):
        ax.plot(shown.date, shown[column], color="#176B87", marker=".", linewidth=1.4)
        ax.axvspan(pd.Timestamp("2022-09-24"), pd.Timestamp("2022-09-30") + pd.Timedelta(hours=23),
                   color="#176B87", alpha=.09)
        ax.axvspan(pd.Timestamp("2022-10-01"), pd.Timestamp("2022-10-07") + pd.Timedelta(hours=23),
                   color="#D97732", alpha=.09)
        ax.axvline(pd.Timestamp("2022-10-03"), color="#555555", linestyle="--", linewidth=.8)
        ax.set(title=title, ylabel=ylabel)
        ax.xaxis.set_major_locator(DayLocator(interval=7))
        ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
        ax.grid(axis="y", alpha=.2)
    regions = read_table(out, "volume_region_comparisons.csv")
    regions = regions[(regions.clock == "delivery") & (regions.comparison == "primary_7d")]
    positions = np.arange(len(regions))
    ax = axes[1, 1]
    ax.bar(positions - .18, regions.before_records, width=.36, color="#176B87", label="Sep 24-30")
    ax.bar(positions + .18, regions.after_records, width=.36, color="#D97732", label="Oct 1-7")
    ax.set(xticks=positions, xticklabels=regions.region, title="D. All four regions decline", ylabel="Records/7 days")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.suptitle("Jilin file: fewer observed courier-days and fewer records on those days", fontsize=12)
    save_plot(fig, out, "volume_structure")

    groups = read_table(out, "duration_groups.csv")
    hours = groups[(groups.by == "accept_hour") & (groups.duration_band == "all")].copy()
    hours["hour"] = hours.group.astype(int)
    hours = hours[hours.test_records >= 50].sort_values("hour")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    values = hours.mean_improvement_minutes
    axes[0].bar(hours.hour, values, color=["#176B87" if v >= 0 else "#B74A4A" for v in values])
    axes[0].axhline(0, color="#444444", linewidth=.8)
    axes[0].set(title="A. Gain differs by acceptance hour", xlabel="Hour (groups with n >= 50)",
                ylabel="Baseline MAE - hourly MAE (min)", xticks=hours.hour)
    overall = groups[(groups.by == "overall") & (groups.duration_band == "all")].iloc[0]
    tail = groups[(groups.by == "overall") & (groups.duration_band == "over_360")].iloc[0]
    share_values = [100 * tail.test_records / overall.test_records, 100 * tail.share_of_total_improvement]
    axes[1].bar([0, 1], share_values, color=["#808B96", "#176B87"], width=.55)
    for pos, value in enumerate(share_values):
        axes[1].text(pos, value + 2, f"{value:.1f}%", ha="center", fontsize=10)
    axes[1].set(title="B. Post-hoc tail group (> 6 h)", ylabel="Share (%)", ylim=(0, 78),
                xticks=[0, 1], xticklabels=["Test records", "Net error reduction"])
    windows = groups[(groups.by == "window_start") & (groups.duration_band == "all")].sort_values("group")
    x = np.arange(len(windows))
    axes[2].plot(x, windows.baseline_mae_minutes, "o-", color="#808B96", label="Global median")
    axes[2].plot(x, windows.hour_mae_minutes, "o-", color="#176B87", label="Hourly median")
    axes[2].set(title="C. Same four earlier test windows", ylabel="MAE (min)", xticks=x,
                xticklabels=[label[5:] for label in windows.group])
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Hourly medians: small overall gain, with different gains and losses inside it", fontsize=12)
    save_plot(fig, out, "duration_gain")


def report(out, audit, volume, duration):
    evidence_link = reference(out, ROOT / "docs/research/evidence.md")
    plan_link = reference(out, HERE.parent / "plan.md")
    before_model_link = reference(out, HERE.parent / "before-model.md")
    main = volume["primary_comparison"]["delivery"]
    before, after, split = main["before"], main["after"], main["decomposition"]
    mix = main["region_productivity_decomposition"]
    region = read_table(out, "volume_region_comparisons.csv")
    region = region[(region.clock == "delivery") & (region.comparison == "primary_7d")]
    cohorts = read_table(out, "volume_courier_cohorts.csv")
    shared = cohorts[(cohorts.clock == "delivery") & (cohorts.comparison == "primary_7d") & (cohorts.cohort == "shared")].iloc[0]
    comparisons = read_table(out, "volume_comparisons.csv")
    comparisons = comparisons[comparisons.clock == "delivery"]
    groups = read_table(out, "duration_groups.csv")
    hours = groups[(groups.by == "accept_hour") & (groups.duration_band == "all")].copy()
    hours["hour"] = hours.group.astype(int)
    band_hours = groups[groups.by == "accept_hour"].copy()
    band_hours["hour"] = band_hours.group.astype(int)
    morning_tail = band_hours[(band_hours.duration_band == "over_360") & band_hours.hour.between(7, 9)]
    morning_regular = band_hours[(band_hours.duration_band == "up_to_360") & band_hours.hour.between(7, 9)]
    afternoon = band_hours[(band_hours.duration_band == "up_to_360") & band_hours.hour.between(13, 15)]
    windows = groups[(groups.by == "window_start") & (groups.duration_band == "all")].sort_values("group")
    regular_windows = groups[(groups.by == "window_start") & (groups.duration_band == "up_to_360")].sort_values("group")
    all_stats, tail, regular = duration["overall"], duration["over_360"], duration["up_to_360"]
    gains = 100 * all_stats["mean_improvement_minutes"] / all_stats["baseline_mae_minutes"]
    last_seen = read_table(out, "volume_last_seen.csv")
    stopped = int(last_seen[(last_seen.clock == "delivery") & (last_seen.last_observed_date == "2022-10-02")].couriers_active_in_september24_30.sum())
    profiles = read_table(out, "duration_hour_profiles.csv").set_index("accept_hour")
    contrasts = read_table(out, "duration_composition_contrasts.csv")
    courier13 = contrasts[(contrasts.grouping == "courier") & (contrasts.target_hour == 13)].iloc[0]
    day13 = contrasts[(contrasts.grouping == "courier_day") & (contrasts.target_hour == 13)].iloc[0]
    table = markdown_table(["完成日口径", "9月24—30日", "10月1—7日"], [
        ["文件记录数", before["records"], after["records"]],
        ["日均记录数", f'{before["records_per_day"]:.2f}', f'{after["records_per_day"]:.2f}'],
        ["比较期内出现过的配送员数", before["unique_couriers"], after["unique_couriers"]],
        ["配送员日总数", before["courier_days"], after["courier_days"]],
        ["日均活跃配送员数 A", f'{before["active_couriers_per_day"]:.2f}', f'{after["active_couriers_per_day"]:.2f}'],
        ["每配送员日记录数 P", f'{before["records_per_active_courier_day"]:.2f}', f'{after["records_per_active_courier_day"]:.2f}']])
    regional_table = markdown_table(["匿名区域", "前期条数", "后期条数", "对总降量的占比", "每配送员日条数：前→后"], [
        [r.region, int(r.before_records), int(r.after_records), f"{100*r.share_of_total_decline:.2f}%",
         f"{r.before_records_per_active_courier_day:.2f} → {r.after_records_per_active_courier_day:.2f}"] for r in region.itertuples()])
    sensitivity = markdown_table(["前期 / 后期", "条数变化", "A项占降量"], [
        [f"{r.before_start[5:]}—{r.before_end[5:]} / {r.after_start[5:]}—{r.after_end[5:]}",
         f"{r.change_pct:.2f}%", f"{100*r.active_share_of_delta:.2f}%"] for r in comparisons.itertuples()])
    hour_table = markdown_table(["接单小时", "对象数", "总体中位数MAE", "小时中位数MAE", "平均改善（分钟）"], [
        [r.hour, int(r.test_records), f"{r.baseline_mae_minutes:.2f}", f"{r.hour_mae_minutes:.2f}",
         f"{r.mean_improvement_minutes:+.2f}"] for r in hours[hours.hour.isin([7, 8, 9, 10, 13, 14, 15])].sort_values("hour").itertuples()])
    window_table = markdown_table(["14天窗口起点", "对象数", "全部对象平均改善", "≤6小时对象平均改善"], [
        [r.group, int(r.test_records), f"{r.mean_improvement_minutes:+.2f}",
         f'{regular_windows[regular_windows.group == r.group].mean_improvement_minutes.iloc[0]:+.2f}'] for r in windows.itertuples()])
    text = f'''# 吉林配送二轮研究：记录量突降发生在哪里，小时预测为什么改善

**记录量的下降主要发生在两期共同出现的配送员身上；时长预测的长尾改善可以被预测值移动精确解释。** 这推进了首轮的判断：现在可以定位文件结构如何变化，也能说明误差改善的算术来源；仍需业务或采样材料来解释作业原因。

本次是看过首轮结果后的描述性分析，未训练新模型。来源与字段适用范围集中见[证据说明]({evidence_link})；原来的[建模前计划]({plan_link})、[第一次改判]({before_model_link})及失败的顺序选模结果保留。

## 输入与比较口径

原始 `delivery_jl.csv` 有 {audit['rows']:,} 行、{audit['columns']} 列、{audit['couriers']} 名配送员、{audit['regions']} 个区域ID，无重复运单ID。接单日期覆盖 {audit['accept_start']} 至 {audit['accept_end']}，完成日期覆盖 {audit['delivery_start']} 至 {audit['delivery_end']}。核心时间和ID字段无缺失；接单GPS经纬度各缺 {audit['missing_fields']['accept_gps_lng']} 条，本次不用GPS推断路线。全部输入有 {audit['cross_day_records']} 条跨日完成；`ds` 与接单日期全部相同，与完成日期有 {audit['ds_finish_disagreements']} 条不同，因此完成量直接按完成时间统计。

主比较选9月24—30日和10月1—7日，两期各7天且星期构成相同。文件中每天均有记录；敏感性比较另列。城市整日没记录时保留缺失，不能视为零需求。匿名区域序号按原 `region_id` 的字符串排序，表示文件内分组，不对应已确认的行政区。

时长分析复用首轮四个14天窗口的 {duration['prediction_rows']:,} 行预测，配对成 {duration['paired_records']:,} 个相同对象。按窗口与样本序号回连原始行顺序，核对实际时长、接单日期和小时后，才计算组间差异。输入及代码摘要、运行环境见[summary.json](summary.json)。

## 记录量：每天出现的配送员少了，每人出现当天的条数也少了

{table}

一个配送员出现两天，计2个“配送员日”。例如每天10人、每人20条，日均是200条。这里 **A = 配送员日总数 / 天数，P = 总记录数 / 配送员日总数**，所以日均记录数严格等于 A×P。P是文件内的记录强度，不能当作员工真实效率；活跃指在文件中出现过，不是完整在岗人数。

日均记录数减少 {-split['delta_records_per_day']:.2f} 条（{-split['change_pct']:.2f}%）。将两个同时变化的因素对称分解：

`Δ日均记录数 = ΔA × (P前 + P后)/2 + ΔP × (A前 + A后)/2`

A项为 {split['active_contribution_per_day']:.2f} 条/天，占降量 {100*split['active_share_of_delta']:.2f}%；P项为 {split['productivity_contribution_per_day']:.2f} 条/天，占 {100*split['productivity_share_of_delta']:.2f}%。这是恒等式分解，给出数量来源，不把两项称为因果解释。逐项结果见[volume_comparisons.csv](volume_comparisons.csv)。

**前期有记录、后期整期无记录的配送员集合，并非主要降量来源。** 两期共同的13名配送员，记录从 {int(shared.before_records):,} 条降到 {int(shared.after_records):,} 条（减少 {int(shared.before_records-shared.after_records):,} 条），配送员日从 {int(shared.before_courier_days)} 降至 {int(shared.after_courier_days)}，每配送员日从 {shared.before_records_per_active_courier_day:.2f} 降到 {shared.after_records_per_active_courier_day:.2f}。前期独有5名的149条未再出现，后期独有4名带来178条，两类集合变化净增加29条。共同集合也可能有人在10月初停止出现；两期累计人数接近，无法替代每日出现频率的比较。见[配送员集合聚合](volume_courier_cohorts.csv)。

10月1、2、3日的记录数分别是237、155、34条，日活跃人数分别是12、13、5。9月末出现过的18名配送员中，{stopped} 名在整个文件中的最后记录集中在10月2日。这提示应优先向数据方核对这一天附近的采样覆盖和任务变化；不能仅凭“最后出现”说他们离职。见[逐日表](volume_daily.csv)和[最后出现日期分布](volume_last_seen.csv)。

![记录数、日活跃人数、每配送员日条数和匿名区域](volume_structure.png)

## 区域：四组均下降，重新分配区域权重不能解释记录强度下降

{regional_table}

region_4贡献最多的降量，但四个区域都减少。全输入的配送员ID均只关联一个区域，本次两期共同配送员没有换区域；区域贡献可以相加，不能把它说成已识别的地区业务损失。

再检查“只是低条数组的占比变多”这一解释：用各区域配送员日占比 w 加权区域内条数 p，城市 P = Σw×p。对称分解得到，P的变化 {mix['delta_records_per_active_courier_day']:.4f} 条/配送员日中，**区域权重变化贡献 {mix['region_weight_contribution']:+.4f}，区域内条数变化贡献 {mix['within_region_contribution']:+.4f}**。权重变化反而略微抵消下降，因此单纯的区域配比变化不足以解释它；这里仍没有控制区域内配送员和任务组成。明细见[区域对比](volume_region_comparisons.csv)。

按接单日重算，两期总条数仍是1,677和657，下降比例相同，A项占 {100*volume['primary_comparison']['accept']['decomposition']['active_share_of_delta']:.2f}%。这使“完成日期挪动造成下降”的解释缺少支持，但两种日期都来自同一份已完成记录，接单量也不等于完整新增需求。

{sensitivity}

三个同星期构成的7天比较均有下降，A项占比约57.6%—62.3%；6天比较的星期构成不同，仅作日期边界检查。比较窗口是事后选择，不能称为预先设计的变化检验。

## 时长：上午的长尾改善和下午的普通任务改善来自不同方向

总体MAE从 {all_stats['baseline_mae_minutes']:.2f} 降至 {all_stats['hour_mae_minutes']:.2f} 分钟，减少 {all_stats['mean_improvement_minutes']:.2f} 分钟（{gains:.2f}%）。两法误差相减后，总绝对误差减少 {all_stats['total_improvement_minutes']:,.1f} 分钟；它是预测误差累计差，不是节省的配送作业时间。

实际时长>6小时的 {tail['test_records']:,} 条占测试对象 {100*tail['test_records']/all_stats['test_records']:.2f}%，贡献净误差减少量的 {100*tail['share_of_total_improvement']:.2f}%，每条平均改善 {tail['mean_improvement_minutes']:.2f} 分钟，小时法的MAE仍有 {tail['hour_mae_minutes']:.2f} 分钟。≤6小时的 {regular['test_records']:,} 条每条仅平均改善 {regular['mean_improvement_minutes']:.2f} 分钟。长尾由真实标签事后划分，接单时不能读取该标签。见[配对分组结果](duration_groups.csv)。

**这61%的贡献，并不说明小时法预测出了极端慢单。** 所有预测值最高只有 {duration['maximum_prediction_minutes']:.0f} 分钟，长尾实际值均大于两预测。若实际值y大于两预测a、b，则 `|y-a| - |y-b| = b-a`；长尾每条误差变化恰好等于小时预测相对总体预测的位移，与它到底多慢无关。

例如8月14日起的窗口，8时总体预测167分钟、小时预测195分钟。316条长尾每条均改善28分钟，共8,848分钟；同小时预测只是对所有任务一起上移。位于167分钟以下的快任务则每条增加28分钟误差，两预测之间的对象另按实际距离变化。[duration_loss_geometry.csv](duration_loss_geometry.csv)逐窗口逐小时核对这条恒等式。

拆成时段后，7—9时的 {int(morning_tail.test_records.sum()):,} 条长尾占全部长尾 {100*duration['tail_accept_7_to_9_share']:.2f}%，合计改善 {morning_tail.total_improvement_minutes.sum():,.0f} 分钟；同时间的 {int(morning_regular.test_records.sum()):,} 条普通任务反而合计退步 {-morning_regular.total_improvement_minutes.sum():,.0f} 分钟。13—15时的 {int(afternoon.test_records.sum())} 条普通任务通过预测下移，合计改善 {afternoon.total_improvement_minutes.sum():,.0f} 分钟、平均 {afternoon.total_improvement_minutes.sum()/afternoon.test_records.sum():.2f} 分钟。因此不能把不同人群的改善归为一个已证实的慢单机制。

{hour_table}

## 可以继续验证的作业线索，以及目前稳定性的边界

实际时长中位数：8时 {profiles.loc[8,'duration_median_minutes']:.0f} 分钟、9时 {profiles.loc[9,'duration_median_minutes']:.0f} 分钟、13时 {profiles.loc[13,'duration_median_minutes']:.0f} 分钟。{duration['paired_records']:,} 个对象中，{duration['same_day_records']:,} 条当天完成，仅 {duration['cross_day_records']} 条跨日。故测试长尾主要是同日耗时长，不能简单归因于跨夜。

8、9、10时三个组的接单时钟中位数分别为 {clock_text(profiles.loc[8,'accept_clock_median_minutes'])}、{clock_text(profiles.loc[9,'accept_clock_median_minutes'])}、{clock_text(profiles.loc[10,'accept_clock_median_minutes'])}，完成时钟中位数却较接近：{clock_text(profiles.loc[8,'delivery_clock_median_minutes'])}、{clock_text(profiles.loc[9,'delivery_clock_median_minutes'])}、{clock_text(profiles.loc[10,'delivery_clock_median_minutes'])}。这是核查作业批次或等待的线索，尚不能认定存在午间完成截止。各时段完成时钟分布见[duration_completion_clock.csv](duration_completion_clock.csv)；这些是分别计算的时钟中位数，其差不能代替时长中位数。

下午对象组成集中：13时最大单一配送员占 {100*profiles.loc[13,'largest_courier_share']:.2f}%，最大区域占 {100*profiles.loc[13,'largest_region_share']:.2f}%；8时对应为 {100*profiles.loc[8,'largest_courier_share']:.2f}% 和 {100*profiles.loc[8,'largest_region_share']:.2f}%。所以先检查任务构成，不能直接把下午短时长归因于某种配送安排。

相同配送员内比较13时与8时，各时段各至少5条的共同组有 {int(courier13.supported_groups)} 个，覆盖13时对象 {100*courier13.target_coverage_share:.2f}%，按13时对象数加权的组内中位数差为 {courier13.weighted_within_group_median_difference_minutes:.2f} 分钟，全部共同组的13时中位数较短。把范围再缩到同配送员同一天、两时段各至少3条，只覆盖 {100*day13.target_coverage_share:.2f}% 的13时对象，组内差为 {day13.weighted_within_group_median_difference_minutes:.2f} 分钟。这给出一个局部线索：更换配送员ID本身不足以描述已覆盖组里的全部差异。它没有控制批次、路线、任务类型或等待，也没有证明接单小时的因果作用。两个组内统计与总体中位数差不是同一种估计量，不用差额估算组成“解释了多少”。见[组成对照](duration_composition_contrasts.csv)及[各小时画像](duration_hour_profiles.csv)。

{window_table}

四个窗口的总体误差均改善，但后两个改善较小；最后一个窗口的普通任务平均改善为 {regular_windows.iloc[-1].mean_improvement_minutes:+.4f} 分钟，已经退步。56个测试日期中5日退步：{', '.join(duration['dates_with_negative_improvement'])}。这仅描述先前已评分的四窗口，不是新增独立验证；这些窗口也是提出本次线索的材料。10月9日以后的记录稀疏且有缺日，不用它来声称后续稳定性。当前支持保留小时中位数为对照方法，还不支持稳定服务收益或部署结论。

![小时差异、长尾贡献与四窗口误差](duration_gain.png)

## 研究决定和能推翻它的证据

| 当前判断 | 下一份材料直接检验什么 | 哪种结果会改变判断 |
| --- | --- | --- |
| 记录下降主要在共同配送员的出现频率和记录强度中，不能归结为累计人数减少 | 核对10月2—3日前后的采样规则、完整每日接单/完成记录及在岗名单；重复上述分解 | 完整覆盖数据没有同样下降，则应把本文件异常优先解释为覆盖变化；有同样下降才继续讨论业务变化 |
| 小时中位数描述了时段差异，长尾贡献本身是预测位移的算术结果 | 取得预先未查看的连续后续数据，冻结总体/小时两基线、14天评价窗、30条回退阈值，分别记录7—9时与13—15时、普通和长尾的配对误差 | 总体改善消失或普通任务持续退步，则放弃稳定改善的判断；不能在看到成绩后改时段规则 |
| 下午较短包含同配送员内的差异，但批次/等待机制尚未确认 | 补充预测时可见的任务批次、路线或作业事件时间，在同配送员同日的可比任务上核查 | 差异随任务类型/批次匹配消失，说明小时主要代理任务结构；保留差异也需检验更细作业事件，不能仅凭相关性定因果 |

当前不新增更复杂预测模型。完成量仍保留最近均值MAE48.01条/天、顺序选模64.89条/天的失败对照；新分解没有使自动选模获胜。下一步先核对覆盖与任务事件，再判断特征或模型是否值得增加。本轮没有执行事前选模规则的新实验，也没有取得正式竞赛数据。

## 复现与结果阅读

从仓库根目录运行：

```bash
python research/lade-jilin/run.py --download
python research/lade-jilin/round2/run.py
```

第二条从同一原始CSV和首轮逐条预测重建本目录的表、两幅PNG/SVG及本报告；可用 `--data`、`--predictions`、`--output-dir` 指定相同固定版本的文件位置。文件SHA256证明输入版本对应；回连原始对象、配对重算误差和闭合分解分别核查指定计算。公开结果只含聚合量和匿名区域，不输出原始配送员/运单ID或坐标。

先看主比较及小时损失解释，再用链接CSV核对数值；所有实验边界与来源集中见[证据说明]({evidence_link})。
'''
    (out / "delivery_jl_eda_report.md").write_text(text, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE.parent / "data/delivery_jl.csv")
    parser.add_argument("--predictions", type=Path, default=HERE.parent / "results/duration_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if sha(args.data) != EXPECTED_SHA:
        raise ValueError("本轮只复现已记录的吉林固定版本；换版本应单独确定比较日期和研究口径。")
    first_summary = json.loads((args.predictions.parent / "summary.json").read_text(encoding="utf-8"))
    expected_parameters = {"horizon_days": 14, "windows": 4, "hour_min_records": 30, "year": 2022,
                           "observation_end": None, "evaluation_end": None}
    if first_summary["source_sha256"] != EXPECTED_SHA or first_summary["parameters"] != expected_parameters:
        raise ValueError("本轮解释针对首轮固定吉林版本和四个14天窗口，请先复现首轮结果。")
    raw = pd.read_csv(args.data, encoding="utf-8-sig", dtype={"order_id": "string", "courier_id": "string", "region_id": "string"})
    required = ["order_id", "courier_id", "region_id", "accept_time", "delivery_time"]
    if raw[required].isna().any().any() or raw.order_id.duplicated().any():
        raise ValueError("核心字段缺失或运单重复，不能继续按当前定义分析。")
    column_count = len(raw.columns)
    missing = {k: int(v) for k, v in raw.isna().sum().items() if v}
    for col in ["accept_time", "delivery_time"]:
        raw[col] = pd.to_datetime("2022-" + raw[col], format="%Y-%m-%d %H:%M:%S", errors="raise")
    raw["minutes"] = (raw.delivery_time - raw.accept_time).dt.total_seconds() / 60
    if (raw.minutes < 0).any():
        raise ValueError("负时长不能用于当前分析。")
    raw["accept_date"], raw["delivery_date"] = raw.accept_time.dt.normalize(), raw.delivery_time.dt.normalize()
    ds = pd.to_datetime("2022-" + raw.ds.astype(str).str.zfill(4), format="%Y-%m%d")
    audit = {"input_file": args.data.name, "source_sha256": sha(args.data), "bytes": args.data.stat().st_size,
             "rows": len(raw), "columns": column_count, "couriers": int(raw.courier_id.nunique()),
             "regions": int(raw.region_id.nunique()), "duplicate_orders": int(raw.order_id.duplicated().sum()),
             "missing_fields": missing, "cross_day_records": int((raw.accept_date != raw.delivery_date).sum()),
             "ds_finish_disagreements": int((ds != raw.delivery_date).sum()),
             "ds_accept_disagreements": int((ds != raw.accept_date).sum()),
             "accept_start": str(raw.accept_date.min().date()), "accept_end": str(raw.accept_date.max().date()),
             "delivery_start": str(raw.delivery_date.min().date()), "delivery_end": str(raw.delivery_date.max().date())}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    volume = analyze_volume(raw, out)
    duration = analyze_duration(raw, out, args.predictions)
    if list(duration["window_mean_improvement_minutes"]) != ["2022-08-14", "2022-08-28", "2022-09-11", "2022-09-25"]:
        raise ValueError("预测文件的窗口与本轮解释不一致。")
    plots(out)
    write_json({"analysis_kind": "post_hoc_descriptive_followup", "input_audit": audit,
                "predictions_file": args.predictions.name, "predictions_sha256": sha(args.predictions),
                "first_round_parameters": first_summary["parameters"],
                "code_sha256": {p.name: sha(p) for p in [Path(__file__), HERE / "volume_structure.py", HERE / "duration_mechanism.py"]},
                "plot_helpers_sha256": sha(ROOT / "examples/common.py"),
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "matplotlib": plt.matplotlib.__version__,
                "volume_summary": "volume_summary.json", "duration_summary": "duration_summary.json"}, out / "summary.json")
    report(out, audit, volume, duration)
    print("吉林二轮分析完成：记录量结构、配对时长诊断、两幅图及 delivery_jl_eda_report.md。")


if __name__ == "__main__":
    main()
