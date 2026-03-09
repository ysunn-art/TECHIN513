# PsychoMouse 数据采集指南

## 1. 环境准备

```bash
# 安装依赖（仅需一个库）
pip install pynput
```

**鼠标设置**：确保你的鼠标 polling rate 设置为 1000 Hz（在鼠标驱动软件中设置，如 Logitech G Hub、Razer Synapse 等）。

**系统要求**：Windows 10/11 或 macOS 或 Linux，Python 3.8+。

---

## 2. 数据采集流程

### 2.1 启动采集

```bash
# 基本用法：指定参与者 ID，默认录制 3 小时
python collect_data.py --participant P01

# 自定义时长（分钟）和疲劳问卷间隔（分钟）
python collect_data.py --participant P01 --duration 180 --fatigue-poll 30

# 短时间测试（10分钟，每5分钟问一次）
python collect_data.py --participant P01 --duration 10 --fatigue-poll 5
```

### 2.2 采集过程中

1. **启动后**：脚本会先让你填一个基线疲劳评分（KSS 1-9分）
2. **正常使用电脑**：脚本在后台静默记录，你可以正常工作、游戏、浏览网页
3. **每 30 分钟**：终端会弹出疲劳评分提示，填完立刻继续
4. **每 30 秒**：终端会显示采集状态（事件数、频率等）
5. **结束时**：到时自动结束，或按 `Ctrl+C` 提前结束；结束前会再问一次疲劳评分

### 2.3 KSS 疲劳评分标准

| 分数 | 含义 |
|------|------|
| 1 | 极度清醒 |
| 2 | 非常清醒 |
| 3 | 清醒 |
| 4 | 比较清醒 |
| 5 | 既不清醒也不困 |
| 6 | 有一些困意 |
| 7 | 困，但不需要努力保持清醒 |
| 8 | 困，需要一些努力保持清醒 |
| 9 | 非常困，极力对抗睡意 |

---

## 3. 输出文件

每次采集会在 `data/<参与者ID>/<时间戳>/` 下生成三个文件：

```
data/
└── P01/
    └── 20260206_143000/
        ├── mouse_events.csv       # 原始鼠标数据
        ├── fatigue_reports.csv    # 主观疲劳评分记录
        └── session_meta.json      # 会话元数据
```

### mouse_events.csv 格式

| 列名 | 说明 |
|------|------|
| `timestamp_ms` | 高精度时间戳（毫秒） |
| `x` | 鼠标 X 坐标（像素） |
| `y` | 鼠标 Y 坐标（像素） |
| `event_type` | 事件类型：`move` / `click_down` / `click_up` / `scroll` |
| `button` | 按键名称（仅 click 事件有值） |
| `dx` | 水平滚动量（仅 scroll 事件） |
| `dy` | 垂直滚动量（仅 scroll 事件） |

### session_meta.json 示例

```json
{
  "participant_id": "P01",
  "session_id": "20260206_143000",
  "duration_minutes": 182.3,
  "effective_event_rate_hz": 723.5,
  "total_events": 7891234,
  "move_events": 7654321,
  "click_events": 23456,
  "scroll_events": 213457
}
```

---

## 4. 数据标注

采集完成后，用以下命令自动添加时间标签：

```bash
# 自动标注：前30分钟=Alert，30-120分钟=Transition，120分钟后=Fatigued
python collect_data.py label data/P01/20260206_143000

# 自定义标注时间点
python collect_data.py label data/P01/20260206_143000 --alert-min 30 --fatigued-min 120
```

这会生成 `mouse_events_labeled.csv`，在原始数据最后加一列 `label`。

---

## 5. 采集计划（3人分工）

### 目标：≥ 20 个 session，≥ 3 个参与者

| 参与者 | 负责人 | 目标 Sessions | 建议场景 |
|--------|--------|---------------|----------|
| P01 (Eason) | Eason | 7 sessions | 编程、浏览网页、文档编辑 |
| P02 (Phoenix) | Phoenix | 7 sessions | 编程、游戏、一般使用 |
| P03 (Spancer) | Spancer | 7 sessions | 游戏、浏览网页、文档编辑 |

### 每个 Session 的标准流程

```
1. 确保已休息充足（建议在精神好的时候开始）
2. 确认鼠标 polling rate 已设为 1000 Hz
3. 运行采集脚本，填写基线 KSS 评分
4. 正常使用电脑至少 3 小时
   - 前 30 分钟的数据会被标为 "Alert"
   - 2 小时后的数据会被标为 "Fatigued"
   - 中间为 "Transition"（训练时可选择排除）
5. 每次 KSS 提示时如实填写
6. 结束后检查 session_meta.json 中的 effective_event_rate_hz
   - 理想值：≥ 500 Hz
   - 如果太低，检查鼠标设置
7. 运行 label 命令添加标签
```

### 注意事项

- **不要在采集中途切换鼠标**或改变 DPI/polling rate
- **手动记录**你使用的鼠标型号、DPI 设置、polling rate 设置到 `session_meta.json` 的 `mouse_info` 字段
- **避免长时间离开电脑**（>5分钟不动鼠标），如果需要离开，按 `Ctrl+C` 先停止
- 不同的 session 尽量覆盖**不同的使用场景**（工作、游戏、浏览等）
- 每周至少采集 **2-3 个 session**

---

## 6. 快速验证

采集完一个 session 后，可以快速检查数据质量：

```python
import pandas as pd

# 加载数据
df = pd.read_csv("data/P01/20260206_143000/mouse_events.csv")

# 基本统计
print(f"总事件数: {len(df):,}")
print(f"时长: {(df['timestamp_ms'].iloc[-1] - df['timestamp_ms'].iloc[0]) / 1000 / 60:.1f} 分钟")
print(f"事件类型分布:\n{df['event_type'].value_counts()}")

# 检查采样率
move_df = df[df['event_type'] == 'move']
dt = move_df['timestamp_ms'].diff().dropna()
print(f"Move 事件平均间隔: {dt.mean():.2f} ms (≈{1000/dt.mean():.0f} Hz)")
print(f"Move 事件中位数间隔: {dt.median():.2f} ms (≈{1000/dt.median():.0f} Hz)")
```

---

## 常见问题

**Q: macOS 上运行报权限错误？**  
A: 需要在 System Preferences → Security & Privacy → Privacy → Accessibility 中添加终端应用的权限。

**Q: 实际采样率远低于 1000 Hz？**  
A: 这是正常的。pynput 的实际捕获率取决于 OS 事件系统。只要 ≥ 500 Hz 就可以使用。如果低于 200 Hz，请检查鼠标硬件设置。

**Q: 文件太大怎么办？**  
A: 3小时的数据大约 200-500 MB。后续处理时可以按需加载（用 pandas 的 `chunksize` 参数）。
