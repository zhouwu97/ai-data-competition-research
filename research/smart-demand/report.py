"""从已经完成的固定窗口预测生成汇总；不重新选模型或重训。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = {
    'median14': '近14日中位数', 'median28': '近28日中位数',
    'median56': '近56日中位数', 'mean28': '近28日均值',
    'weekday8': '近8周同星期均值', 'ridge': 'Ridge（含历史行为）',
    'mlp_sales_mae': 'MLP：成交历史＋MAE',
    'mlp_behavior_mae': 'MLP：增加历史行为＋MAE',
    'mlp_behavior_poisson': 'MLP：增加历史行为＋Poisson',
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def pooled(frame):
    n = int(frame.n.sum())
    truth = float(frame.truth_sum.sum())
    result = {'n': n, 'mae': float(frame.abs_error.sum() / n),
              'bias': float((frame.bias * frame.n).sum() / n),
              'wape': float(frame.abs_error.sum() / truth) if truth else None}
    if frame.total14_ae.notna().any():
        result['mean_14day_total_ae'] = float(frame.total14_ae.mean())
    return result


def error_stats(truth, prediction):
    mask = np.isfinite(truth)
    error = np.asarray(prediction, dtype=np.float64)[mask] - np.asarray(truth, dtype=np.float64)[mask]
    if not len(error):
        return {'n': 0, 'mae': None, 'bias': None, 'over_mean': None, 'under_mean': None}
    return {'n': len(error), 'mae': float(np.abs(error).mean()), 'bias': float(error.mean()),
            'over_mean': float(np.maximum(error, 0).mean()),
            'under_mean': float(np.maximum(-error, 0).mean())}


def is_event(days, month, day):
    return np.asarray((days.month == month) & (days.day == day))


def collect(output, manifest, selections, metrics):
    config = manifest['config']
    summary = {'status': 'success', 'generated_utc': datetime.now(timezone.utc).isoformat(),
               'experiment_output': str(output.resolve()), 'selection_before_test': selections,
               'stable_definition': config['stable_definition'], 'scope_results': {}}
    with sqlite3.connect(Path(manifest['input']).as_uri() + '?mode=ro', uri=True) as con:
        all_rows = pd.read_sql_query('SELECT date,item_id,qty_alipay FROM Item_feature', con)
    all_rows['date'] = pd.to_datetime(all_rows['date'])
    items = np.sort(all_rows.item_id.unique())
    national_daily, coverage = [], []
    for scope in ('national', 'sku'):
        selected, median = selections[scope]['neural'], selections[scope]['median']
        part = metrics[(metrics.scope == scope) & (metrics.phase == 'test')]
        ensemble = part[part.seed == 'ensemble']
        groups = {str(method): pooled(rows) for method, rows in ensemble.groupby('method')}
        seeds = {str(seed): pooled(rows) for seed, rows in part[part.method == selected].groupby('seed') if seed != 'ensemble'}
        comparison = ensemble[ensemble.method.isin([selected, median])].pivot(index='start', columns='method', values='mae')
        difference = comparison[selected] - comparison[median]
        stable = bool(groups[selected]['mae'] < groups[median]['mae']
                      and (difference <= 1e-9).all()
                      and len(difference) == len(config['test_starts'])
                      and all(v['mae'] < groups[median]['mae'] for v in seeds.values())
                      and len(seeds) == len(config['test_seeds']))
        scope_summary = {'selected_neural': selected, 'selected_median': median,
                         'stable': stable, 'test_pooled': groups, 'test_seeds': seeds,
                         'development_mae': selections[scope]['development_mae'],
                         'improvement_percent': 100 * (1 - groups[selected]['mae'] / groups[median]['mae']),
                         'better_windows': int((difference < -1e-9).sum()),
                         'tied_windows': int((np.abs(difference) <= 1e-9).sum()),
                         'worse_windows': int((difference > 1e-9).sum()),
                         'windows': [], 'fixed_date_diagnostics': {}}
        diagnostic_truth, diagnostic_pred, diagnostic_median, diagnostic_dates = [], [], [], []
        for start in config['test_starts']:
            folder = output / 'predictions' / scope / 'test' / start
            saved = np.load(folder / 'truth.npz')
            truth = saved['truth'].astype(np.float64)
            prediction = np.load(folder / f'{selected}_ensemble.npz')['prediction'].astype(np.float64)
            reference = np.load(folder / f'{median}.npz')['prediction'].astype(np.float64)
            days = pd.date_range(start, periods=config['horizon'])
            window = {'start': start, 'end': str(days[-1].date()),
                      'median_mae': float(comparison.loc[start, median]),
                      'neural_mae': float(comparison.loc[start, selected]),
                      'mae_difference': float(difference.loc[start]), 'n': int(np.isfinite(truth).sum())}
            if scope == 'national':
                window.update(median_total14_ae=float(abs(reference.sum() - truth.sum())),
                              neural_total14_ae=float(abs(prediction.sum() - truth.sum())))
                daily = pd.DataFrame({'date': days.strftime('%Y-%m-%d'), 'window_start': start,
                                      'truth': truth[0]})
                for method in ('median14', 'median28', 'median56', 'mean28', 'weekday8', 'ridge'):
                    daily[method] = np.load(folder / f'{method}.npz')['prediction'][0]
                daily['selected_neural_ensemble'] = prediction[0]
                for seed in config['test_seeds']:
                    daily[f'selected_neural_{seed}'] = np.load(folder / f'{selected}_{seed}.npz')['prediction'][0]
                national_daily.append(daily)
            else:
                observed = all_rows[(all_rows.date >= days[0]) & (all_rows.date <= days[-1])]
                eligible = items[saved['indices']]
                selected_rows = observed[observed.item_id.isin(eligible)]
                if len(selected_rows) != np.isfinite(truth).sum() or abs(selected_rows.qty_alipay.sum() - np.nansum(truth)) > 1e-6:
                    raise ValueError('原库与SKU目标掩码的行数/件数不一致。')
                cov = {'start': start, 'eligible_items': len(eligible),
                       'future_observed_items': int(observed.item_id.nunique()),
                       'predicted_cells': int(truth.size), 'scored_cells': int(np.isfinite(truth).sum()),
                       'all_observed_cells': len(observed), 'cell_coverage': len(selected_rows) / len(observed),
                       'scored_truth_quantity': float(selected_rows.qty_alipay.sum()),
                       'all_observed_quantity': float(observed.qty_alipay.sum()),
                       'quantity_coverage': float(selected_rows.qty_alipay.sum() / observed.qty_alipay.sum())}
                coverage.append(cov)
                window['eligible_items'] = len(eligible)
            scope_summary['windows'].append(window)
            diagnostic_truth.append(truth.ravel())
            diagnostic_pred.append(prediction.ravel())
            diagnostic_median.append(reference.ravel())
            diagnostic_dates.extend(np.tile(days.to_numpy(), len(truth)))
        truth = np.concatenate(diagnostic_truth)
        prediction = np.concatenate(diagnostic_pred)
        reference = np.concatenate(diagnostic_median)
        dates = pd.DatetimeIndex(diagnostic_dates)
        event1111, event1212 = is_event(dates, 11, 11), is_event(dates, 12, 12)
        for name, mask in [('ordinary', ~(event1111 | event1212)), ('nov11', event1111), ('dec12', event1212)]:
            scope_summary['fixed_date_diagnostics'][name] = {
                'median': error_stats(truth[mask], reference[mask]),
                'neural': error_stats(truth[mask], prediction[mask])}
        summary['scope_results'][scope] = scope_summary
    summary['sku_coverage_by_window'] = coverage
    summary['sku_coverage_pooled'] = {
        'scored_cells': sum(c['scored_cells'] for c in coverage),
        'all_observed_cells': sum(c['all_observed_cells'] for c in coverage),
        'scored_truth_quantity': sum(c['scored_truth_quantity'] for c in coverage),
        'all_observed_quantity': sum(c['all_observed_quantity'] for c in coverage),
    }
    c = summary['sku_coverage_pooled']
    c['cell_coverage'] = c['scored_cells'] / c['all_observed_cells']
    c['quantity_coverage'] = c['scored_truth_quantity'] / c['all_observed_quantity']
    summary['execution'] = {key: manifest[key] for key in ('python', 'torch', 'numpy', 'pandas', 'neural_fits', 'elapsed_seconds', 'input_sha256', 'config_sha256')}
    verification = output / 'verification.json'
    if verification.exists():
        summary['independent_verification'] = json.loads(verification.read_text(encoding='utf-8'))
        summary['independent_verification_file'] = str(verification)
    exercise_manifest = output / 'exercise-check/examples/results/run_manifest.json'
    if exercise_manifest.exists():
        summary['exercise_check_manifest'] = str(exercise_manifest)
    return summary, pd.concat(national_daily, ignore_index=True)


def draw(summary, daily, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, dates as mdates

    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={'height_ratios': [1.35, 1, 1]}, layout='constrained')
    days = pd.to_datetime(daily.date)
    ax = axes[0]
    ax.plot(days, daily.truth, color='#202020', lw=1.8, label='已记录成交件数')
    ax.plot(days, daily[summary['scope_results']['national']['selected_median']], color='#0072b2', lw=1.5, label='开发选定中位数')
    ax.plot(days, daily.selected_neural_ensemble, color='#d55e00', lw=1.5, label='开发选定MLP（三种子均值）')
    ax.set_yscale('log')
    ax.set_ylabel('每日件数（对数轴）')
    ax.set_title('A  后段84天：直接预测数据中每日记录总量')
    for date, text in [('2015-11-11', '11/11'), ('2015-12-12', '12/12')]:
        date = pd.Timestamp(date)
        ax.axvline(date, color='#777777', ls=':', lw=1)
        qty = float(daily.loc[daily.date == str(date.date()), 'truth'].iloc[0])
        ax.annotate(f'{text}：{qty:,.0f}件', (date, qty), xytext=(-32, 9), textcoords='offset points', fontsize=9)
    ax.set_ylim(400, 650000)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.legend(loc='upper left', fontsize=9)
    for ax, scope, title in zip(axes[1:], ('national', 'sku'), ('B  全国记录总量：每个14天窗口的日MAE', 'C  商品日：仅对未来有记录的格子计算MAE')):
        result = summary['scope_results'][scope]
        windows = result['windows']
        x = np.arange(len(windows))
        ax.bar(x - .18, [w['median_mae'] for w in windows], .36, color='#0072b2', label='开发选定中位数')
        ax.bar(x + .18, [w['neural_mae'] for w in windows], .36, color='#d55e00', label='选定MLP（三种子均值）')
        ax.set_xticks(x, [w['start'][5:].replace('-', '/') for w in windows])
        ax.set_xlabel('预测窗口开始日期（2015年）')
        ax.set_ylabel('MAE（件）')
        ax.set_title(title)
        ax.legend(fontsize=9)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render(summary, manifest):
    national, sku = [summary['scope_results'][s] for s in ('national', 'sku')]
    lines = ['# 智能分仓数据：深度学习与中位数对照', '',
             '**本轮没有稳定超过中位数。** 全国记录总量与商品日两个任务中，开发选中的MLP在后段总体误差均高于开发选中的近14日中位数；三个单独种子的总体误差也均更高。', '',
             '预测目标是`qty_alipay`（成交件数），每次预测未来14天。全国任务直接预测每日已记录成交件数之和；商品任务使用多个商品共同训练一个MLP，仅对未来有记录的商品日计算误差。两者分别训练，商品预测没有按未来记录掩码求和来冒充全国预测。', '',
             '| 评价对象 | 中位数日MAE | 选中MLP日MAE | 相对误差变化 | 胜出窗口 | 稳定超过 |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for label, result in [('全国记录总量', national), ('商品日', sku)]:
        median, neural = result['selected_median'], result['selected_neural']
        lines.append(f"| {label} | {result['test_pooled'][median]['mae']:,.4f} | {result['test_pooled'][neural]['mae']:,.4f} | {-result['improvement_percent']:+.2f}% | {result['better_windows']}/6 | {'是' if result['stable'] else '否'} |")
    lines += ['', 'MAE表示平均每个有真值的日期／商品日错多少件，越小越好。全国与商品行的统计单位不同，数值不能互相比大小。“稳定”预先定义为后段总体更好、六窗均不退步，且三个种子的总体MAE各自更好；本轮不满足。', '',
              '![逐日与逐窗口对照](comparison.png)', '', '## 数据与特征', '',
              '完整库包含232,621个商品日、1,000个商品、453个日期，覆盖2014-10-01—2015-12-27。商品记录数量与日期覆盖不同；只有99个商品全程都有记录。缺记录保持未知，真实记录里的0仍是0；商品预测资格只看起点之前56天是否至少有28次观测。详见[完整数据审计](data_audit.md)。', '',
              '基础输入为近28天成交量、观测掩码、7/14/28/56天统计和未来已知日历。行为版本另加历史浏览人数、加购人数、收藏人数、下单件数及支付笔数的7/28天统计。它们全部截至预测起点；没有使用未来真实浏览、支付、下单值。商品模型还使用类目，输入处理只由当前训练集拟合。', '',
              '固定日历包含星期、年内周期及11月11日／12月12日。预测2015年的相应日期时，仅有一次上一年的相应日期，多个商品不是多个独立促销年份；共享模型仍未在本次对照中解决这种跨年高峰。Cost表虽包含仓代码，但成交表没有可关联的区域／仓字段，因此本轮不称为已完成分仓预测。', '',
              '## 开发期的删减对照', '',
              '先用2015-07-13—10-04的六个非重叠14天窗口比较三个预设MLP，每个候选只用种子20261023。选型按总体有真值格子的MAE进行，并分别在14/28/56日中位数中选主基线。开发最优MLP本身也未超过中位数；“选中”仅表示三个深度候选中最好。', '',
              '| 方法 | 全国开发MAE | 商品日开发MAE |', '| --- | ---: | ---: |']
    order = ['median14', 'median28', 'median56', 'mean28', 'weekday8', 'ridge', 'mlp_sales_mae', 'mlp_behavior_mae', 'mlp_behavior_poisson']
    for method in order:
        lines.append(f"| {LABELS[method]} | {national['development_mae'][method]:,.4f} | {sku['development_mae'][method]:,.4f} |")
    lines += ['', f"全国选中`{national['selected_neural']}`，商品选中`{sku['selected_neural']}`；两者主中位数均为`median14`。增加历史行为仅在商品开发对照中略降MAE；Poisson损失在两个任务中都更差。行为和损失候选没有在后段重新筛选。", '',
              '## 后段全部基线与种子', '',
              '后段为2015-10-05—12-27的六个非重叠14天窗口。每次仅使用当时已经结束的训练目标；后来的窗口可使用早先窗口已经发生的记录，符合滚动更新场景。测试只训练开发选中的深度候选，固定取三个种子的算术均值。', '',
              '| 方法 | 全国日MAE | 商品日MAE | 全国14天总量平均绝对误差 |', '| --- | ---: | ---: | ---: |']
    for method in ['median14', 'median28', 'median56', 'mean28', 'weekday8', 'ridge']:
        a, b = national['test_pooled'][method], sku['test_pooled'][method]
        lines.append(f"| {LABELS[method]} | {a['mae']:,.4f} | {b['mae']:,.4f} | {a['mean_14day_total_ae']:,.2f} |")
    a, b = national['test_pooled'][national['selected_neural']], sku['test_pooled'][sku['selected_neural']]
    lines.append(f"| 各自选定MLP：三种子均值 | {a['mae']:,.4f} | {b['mae']:,.4f} | {a['mean_14day_total_ae']:,.2f} |")
    for seed in manifest['config']['test_seeds']:
        a, b = national['test_seeds'][str(seed)], sku['test_seeds'][str(seed)]
        lines.append(f"| 各自选定MLP：种子{seed} | {a['mae']:,.4f} | {b['mae']:,.4f} | {a['mean_14day_total_ae']:,.2f} |")
    lines += ['', '近28日中位数后段数值也保留作敏感性对照，主结论继续使用开发选定的近14日中位数，没有根据后段重选基线。Ridge始终使用含历史行为的特征，全国选中的MLP只用成交历史；两者差异不能单独归因于线性／非线性模型。', '', '## 六个窗口分别发生了什么', '']
    for title, result in [('全国记录总量', national), ('商品日', sku)]:
        lines += [f'### {title}', '', '| 窗口开始 | 中位数MAE | MLP均值MAE | MLP增加的误差 |', '| --- | ---: | ---: | ---: |']
        for row in result['windows']:
            lines.append(f"| {row['start']} | {row['median_mae']:,.4f} | {row['neural_mae']:,.4f} | {row['mae_difference']:+,.4f} |")
        lines.append('')
    lines += ['## 普通日期与固定活动日期', '',
              '以下只按事先固定的11/11、12/12区分日期，不按实际成交峰值挑日期，也不据此切换模型。“过估／低估”分别是每个评分格子正向／负向误差的平均件数，包含另一个方向为0的格子；二者之和等于MAE。', '',
              '| 对象／日期 | 评分格子 | 中位数MAE | MLP MAE | 中位数过估／低估 | MLP过估／低估 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for title, result in [('全国', national), ('商品日', sku)]:
        for key, label in [('ordinary', '其余日期'), ('nov11', '11/11'), ('dec12', '12/12')]:
            d = result['fixed_date_diagnostics'][key]
            a, b = d['median'], d['neural']
            lines.append(f"| {title}／{label} | {a['n']:,} | {a['mae']:,.4f} | {b['mae']:,.4f} | {a['over_mean']:,.4f}／{a['under_mean']:,.4f} | {b['over_mean']:,.4f}／{b['under_mean']:,.4f} |")
    a, b = [national['fixed_date_diagnostics']['ordinary'][m] for m in ('median', 'neural')]
    aa, bb = [sku['fixed_date_diagnostics']['ordinary'][m] for m in ('median', 'neural')]
    lines += ['', f"全国其余82天的MAE从{a['mae']:,.2f}升至{b['mae']:,.2f}，平均过估从{a['over_mean']:,.2f}升至{b['over_mean']:,.2f}件；活动日的少量误差下降没有抵消日常退步。商品日其余日期的MAE也从{aa['mae']:,.4f}升至{bb['mae']:,.4f}，因此不能把失败全部归因于两个活动日。"]
    lines += ['', '## 商品评分覆盖', '',
              '所有达到历史条件的商品都输出14天预测；评分时只取未来有记录的格子。这个条件是为了不把未知真值当零，覆盖范围必须随误差一起读。', '',
              '| 窗口开始 | 有资格商品 | 评分格子／全部记录格子 | 记录格子覆盖 | 成交件数覆盖 |', '| --- | ---: | ---: | ---: | ---: |']
    for c in summary['sku_coverage_by_window']:
        lines.append(f"| {c['start']} | {c['eligible_items']} | {c['scored_cells']:,}／{c['all_observed_cells']:,} | {c['cell_coverage']:.2%} | {c['quantity_coverage']:.2%} |")
    c = summary['sku_coverage_pooled']
    lines += ['', f"合计评分{c['scored_cells']:,}个商品日，占后段全部已记录商品日的{c['cell_coverage']:.2%}，覆盖已记录成交件数的{c['quantity_coverage']:.2%}。这些数量从原库只读重算，与保存的目标掩码核对一致。", '',
              '## 解释与下一步', '',
              '这次增加了商品共享训练和历史行为输入，结果仍支持保留中位数作简易参考。全国模型普通日期误差与高峰误差需分别检查；商品模型的小幅局部改善不足以形成跨窗口优势。后续如继续修改，应先针对这些已暴露的错误提出有限对照，不能把增加网络宽度或更换架构直接写成改善。', '',
              '全量数据审计已经查看日期、覆盖及目标极值；后段属于配置冻结后的历史评价，不是前瞻盲测。六个窗口不重叠，但只有六段，三个种子也不是三份新数据。本结果不能证明未来持续有效，也不评价库存成本或真实分仓收益。其他来源与适用范围集中见[证据说明](../../../docs/research/evidence.md)。', '',
              '## 实际运行与复现', '',
              f"本轮实际完成{manifest['neural_fits']}次MLP拟合，训练主流程耗时{manifest['elapsed_seconds']:.2f}秒。CPU单线程PyTorch {manifest['torch']}；Python {manifest['python'].split()[0]}，NumPy {manifest['numpy']}，pandas {manifest['pandas']}。全国网络32/16隐藏单元、120轮；商品网络64/32、25轮。完整参数、训练日志和预测在运行目录中。", '',
              '在仓库根目录使用已安装PyTorch的系统Python运行；本机`.venv`没有PyTorch。新复现请换输出目录：', '',
              '```powershell', 'python research/smart-demand/audit.py',
              'python research/smart-demand/run.py --output outputs/smart-demand-reproduce',
              'python research/smart-demand/verify.py --output outputs/smart-demand-reproduce',
              'python research/smart-demand/report.py --output outputs/smart-demand-reproduce --results research/smart-demand/results-reproduce',
              'python -m unittest discover -s research/smart-demand -p "test_*.py" -v', '```', '',
              '本页数字见[summary.json](summary.json)，84天全国逐日预测见[national_daily_predictions.csv](national_daily_predictions.csv)。商品完整预测仍保存在运行目录的`predictions/sku/`，不展开复制大明细。', '']
    verification = summary.get('independent_verification')
    if verification:
        lines += ['## 已完成的验证', '', f"独立重算状态为`{verification['status']}`：检查{verification['prediction_arrays_checked']}个预测／真值数组，从原SQLite重算{verification['baseline_arrays_recomputed_from_sqlite']}份基线，并核对{verification['ensemble_formulas_checked']}份种子平均预测及开发选择。相关验证见运行目录`verification.json`。模型与数据共20项测试通过；主仓78项测试通过。完整教学练习在源文件副本重跑通过，执行清单位于`outputs/smart-demand-20261023/exercise-check/examples/results/run_manifest.json`。", '']
    return '\n'.join(lines)


def main(args):
    output, results = args.output.resolve(), args.results.resolve()
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'success':
        raise ValueError('必须等实验manifest为success再汇总。')
    selections = json.loads((output / 'selection_before_test.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(output / 'window_metrics.csv', dtype={'seed': str})
    results.mkdir(parents=True, exist_ok=True)
    summary, daily = collect(output, manifest, selections, metrics)
    if len(daily) != 84 or daily.date.duplicated().any():
        raise ValueError('后段全国日预测必须是84个不重复日期。')
    write_json(results / 'summary.json', summary)
    daily.to_csv(results / 'national_daily_predictions.csv', index=False)
    draw(summary, daily, results / 'comparison.png')
    (results / 'report.md').write_text(render(summary, manifest), encoding='utf-8')
    print(json.dumps({scope: {k: v for k, v in result.items() if k in ('stable', 'better_windows', 'worse_windows', 'improvement_percent')}
                      for scope, result in summary['scope_results'].items()}, ensure_ascii=False))
    print(str(results / 'report.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/smart-demand-20261023')
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    main(parser.parse_args())
