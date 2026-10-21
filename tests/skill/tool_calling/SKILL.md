---
name: tool_calling
description: 工具调用技能。只要用户的提示词涉及以下任意一种能力，就会触发本技能：数学计算、城市天气查询、旅游景点推荐、城市美食推荐、文本字符统计。触发后按 Step 依次调用 calculator、get_weather、recommend_attractions、recommend_food、word_count 五个工具并汇总结果。
---

# 工具调用技能（tool_calling）

## 使用时机
只要用户的提示词涉及以下**任意一种**能力（或明确要求"依次执行 / 综合处理 / 走一遍流水线"），就应触发本技能：
- 数学计算（加减乘除、括号、数值运算）
- 城市天气查询
- 城市旅游景点推荐
- 城市美食推荐
- 文本字符统计

## 执行步骤（每个工具一个 Step，必须严格按顺序执行）

### Step 1 — calculator（数学计算）
- 目标：完成用户要求的数学计算。
- 调用：`calculator(expression="...")`，例如 `calculator("2 + 2")`。
- 说明：用户未给出表达式时，默认使用 `2 + 2`。

### Step 2 — get_weather（城市天气）
- 目标：查询目标城市的天气。
- 调用：`get_weather(city="...")`，例如 `get_weather("北京")`。
- 说明：用户未给出城市时，默认使用 `北京`。

### Step 3 — recommend_attractions（旅游景点）
- 目标：为 Step 2 中确定的城市推荐旅游景点。
- 调用：`recommend_attractions(city="...")`，城市与 Step 2 保持一致。

### Step 4 — recommend_food（城市美食）
- 目标：为 Step 2 中确定的城市推荐好吃的美食。
- 调用：`recommend_food(city="...")`，城市与 Step 2 保持一致。

### Step 5 — word_count（文本字符统计）
- 目标：统计指定文本的字符数。
- 调用：`word_count(text="...")`，例如 `word_count("hello world")`。
- 说明：用户未给出文本时，默认使用 `hello world`。

## 输出格式
```
【计算】<Step 1 结果>
【天气】<Step 2 结果>
【景点】<Step 3 结果>
【美食】<Step 4 结果>
【字数】<Step 5 结果>
```