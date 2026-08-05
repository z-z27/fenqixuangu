# v004c 阶段2.1 工程决策记录

> 本文件由 v004c 阶段2.1 研究验收收尾任务生成,记录研究验收结论与已知工程限制。

## 3.1 决策摘要

```text
Decision ID: V004C-ENG-DEFER-001
Decision: Accept stage2.1 research assets and defer LF checkout support
Status: ACCEPTED
```

## 3.2 研究验收结论

```text
阶段2.1数据结构：PASS
阶段2.1因子准入：PASS
Target-blind验证：PASS
禁止字段审计：PASS
精确重复审计：PASS
派生公式审计：PASS
当前Windows正式研究环境重建：PASS
```

## 3.3 冻结统计

```text
333行
197列
42个信号日
79个允许原始候选

PRIMARY_RAW 32
SENSITIVITY_RAW 21
DERIVE_ONLY 12
EXCLUDE_DEPRECATED 2
EXCLUDE_EXACT_DUPLICATE_ALIAS 11
EXCLUDE_ALL_MISSING 1
EXCLUDE_CONSTANT 0

13组精确重复
7个互斥组
6个预声明派生因子
```

## 3.4 已知工程限制

```text
当前阶段2.1正式支持环境为Windows 11和Git core.autocrlf=true。

阶段1manifest中的部分历史raw SHA基于CRLF工作区文件。
在core.autocrlf=false的LF checkout环境中，
阶段2.1输入raw SHA门可能fail closed。

该限制表现为拒绝构建，而不是错误接受被篡改数据。
因此不存在已知数据完整性降级，
但当前不声明Linux、macOS或LF checkout跨平台重建能力。
```

## 3.5 为什么不阻塞研究

必须明确区分:

```text
研究正确性
```

和:

```text
跨平台工程兼容性
```

记录事实:

```text
333行样本正确
Target7未参与准入
32/21/12成员确定
禁止字段未进入allowlist
13组精确重复已逐值确认
6个派生公式已独立重算
Windows正式环境能够重建
LF环境会fail closed
```

因此该限制不阻塞阶段2.2研究。

## 3.6 延期工程项

```text
Backlog ID: V004C-ENG-001
Title: Canonicalize stage1 provenance checks across LF and CRLF checkouts
Priority: Deferred
Blocks current research: No
```

重新开启该工程项的触发条件:

```text
项目进入Linux/macOS或CI运行
多人跨平台协作
公开发布数据集
正式部署模型
需要监管或外部审计
迁移到生产数据平台
```

## 3.7 不得错误声明

本文件不得声称:

```text
阶段2.1已支持所有操作系统
阶段2.1已支持所有Git换行配置
原始字节在LF与CRLF之间完全相同
跨平台问题已经修复
```

---

## 总体状态

```text
研究正确性：PASS
当前正式研究环境可复现性：PASS
跨平台LF checkout支持：DEFERRED_ENGINEERING
总体阶段2.1状态：RESEARCH_ACCEPTED_WITH_DEFERRED_ENGINEERING
```

不再为阶段2.1增加以下审计(LF/CRLF双平台构建矩阵、BOM对抗、孤立CR对抗、mixed
line endings对抗、quoted CSV内部CRLF对抗、branch/tag遮蔽攻击、轻量tag与annotated
tag强制区分、供应链级Git对象审计),仅登记为工程backlog,不阻塞阶段2.2研究。
