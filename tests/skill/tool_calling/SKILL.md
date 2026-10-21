---
name: tool_calling
description: 工具调用技能。只要用户的提示词涉及以下任意一种能力，就会触发本技能：数学计算、城市天气查询、旅游景点推荐、城市美食推荐、文本字符统计。触发后会依次调用 calculator、get_weather、recommend_attractions、recommend_food、word_count 五个工具并汇总结果。
---

# 工具调用技能（tool_calling）

## 使用时机
只要用户的提示词涉及以下**任意一种**能力（或明确要求"依次执行 / 综合处理 / 走一遍流水线"），就应触发本技能：
- 数学计算（加减乘除、括号、数值运算）
- 城市天气查询
- 城市旅游景点推荐
- 城市美食推荐
- 文本字符统计

## 执行步骤（必须严格按顺序执行）
1. 调用 `calculator` 工具进行计算（若用户未给出表达式，默认使用 `2 + 2`）。
2. 调用 `get_weather` 工具查询城市天气（若用户未给出城市，默认使用 `北京`）。
3. 获取到该城市后，调用 `recommend_attractions` 工具推荐该城市的旅游景点。
4. 同样针对该城市，调用 `recommend_food` 工具推荐该城市好吃的美食。
5. 调用 `word_count` 工具统计文本字符数（若用户未给出文本，默认使用 `hello world`）。
6. 将以上五个工具的结果按步骤顺序汇总，输出最终答案。

## 输出格式
```
【计算】<calculator 结果>
【天气】<get_weather 结果>
【景点】<recommend_attractions 结果>
【美食】<recommend_food 结果>
【字数】<word_count 结果>
```