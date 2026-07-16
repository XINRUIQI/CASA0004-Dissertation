# 发给 Claude 的文件清单

按下面打包即可。**必发**优先。

> 目标：向不了解项目的人讲清  
> **动机 → RQ+设计 → 大表里谁有用 → 同样数据换用法**  
> （7 分钟 / ≤4 页）。**通俗语言**；少提随机游走；多讲模态作用与大表 vs 分开学对比。

---

## 必发（3 个）

| # | 文件路径 | 作用 |
|---|----------|------|
| 1 | `00_admin/meeting_notes/MiniConference_Claude_prompt.md` | **主 Prompt（中英对照）** — 整份粘贴 |
| 2 | `00_admin/meeting_notes/MiniConference_briefing.md` | 锁定 4 页 + 可上屏数字 + 模型清单 |
| 3 | `00_admin/meeting_notes/Meeting04_prep_20260707.md` | 完整叙事 / 架构 / RQ |

---

## 讲稿（已按 4 页弧写好）

| # | 文件路径 | 作用 |
|---|----------|------|
| 4 | `00_admin/meeting_notes/MiniConference_spoken_script.md` | **今日口头讲稿**（略偏技术；~7 min） |
| — | `Meeting04_spoken_script_20260707.md` | 仅背景；勿当今日讲稿 |

---

## 选发：图

| 文件 | 建议 |
|------|------|
| Slide 2 示意图 | 让 Claude 画：RQ → 两种用法（大表分叉 M0–M4 + 分开学再合） |
| `05_outputs/baselines/Flat/M4_Flat/shap_m4.png` | **推荐**支撑 Slide 3（52/34/15） |
| fusion matrix / metrics csv | **不发** |

---

## 生成后必查（Claude 上次偏了，按这个打回）

1. Slide 1 是否**没有**甩 flat/fusion 黑话（应用「一张大表 / 分开学再合」）  
2. Slide 2 是否 **RQ 在最前**，过渡句在**页末**  
3. M0–M4 是否为 **M1 分叉**（M2=金融+卫星，M3=金融+航运）  
4. Slide 3–4 主叙事是否是 **航运 vs 卫星** + **大表 vs 专用学习器**，而不是「打不赢随机游走」  
5. 数字是否只用：SHAP 52/34/15；skill −6.7%→+0.11%（航运）；−7.0%→−2.4%（卫星）；−8.6%→−1.3%（全）  
6. 是否点名 **Ridge + XGB**（大表）与 **Mfinship / Mfinrs / Mfull / Mconcat**（分开学），没有编造模型  
7. 若被问「为何 Ridge/XGB」：答复是否为 **「按文献基准阶梯选的，不是赛马挑冠军」**  

---

## 推荐操作步骤

1. 打开 Claude  
2. **先粘贴** `MiniConference_Claude_prompt.md` 全文  
3. **再上传** `MiniConference_briefing.md` + `Meeting04_prep_20260707.md`  
4. 对照上面「必查」七条；任一不对就追加下面这句  

---

## 若输出又偏了，追加这句

```text
Revise hard. Match the locked arc exactly:

Slide 1 — plain language only: which data help? → is one big table enough? → does learn-each-then-combine work better? No unexplained “flat/fusion”.

Slide 2 — RQs FIRST, then data types, fair rules, two ways (big-table M0–M4 branching from M1; learn-each-then-combine). Transition only at the END.

Slide 3 — modality roles, not random-walk: shipping useful (XGB), satellite weak; SHAP ~52% / ~34% / ~15%. Ridge+XGB = literature benchmarks (not horse-race).

Slide 4 — shipping vs satellite under smarter use; same-combo big-table vs specialised learner using ONLY:
  fin+ship −6.7% → +0.11%; fin+sat −7.0% → −2.4%; all −8.6% → −1.3%.
  Smart weighting > naive concat. Soft-pedal random walk.

Do not invent models or percentages.
```
