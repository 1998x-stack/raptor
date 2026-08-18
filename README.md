# raptor

> RAPTOR（递归抽象处理树状检索）实现 + 中文详解与可视化。

## 📖 项目简介

**RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval** 的实践化实现——通过**聚类 + 递归摘要**构建多层树状语义索引，让长文档检索既保留原始细节又能从高层的跨段落语义进行召回，是 RAG 场景的进阶检索方案。

原始实现版权归 InfiniFlow Authors（Apache-2.0）。本项目包含中文逐章详解、可视化页面与可运行示例。

## 🛠️ 技术栈

- Python ≥ 3.10，Apache-2.0 许可
- `numpy`、`scikit-learn`（聚类）、`umap-learn`（降维）

## 🚀 快速开始

```bash
pip install -e .
pip install -e .[dev]        # 可选：开发依赖
python raptor.py <文档>      # 构建 RAPTOR 树
```

## 📚 文档与示例

| 资源 | 说明 |
|------|------|
| `RAPTOR_详细中文文档.md` | 论文 + 实现的中文逐章详解 |
| `index.html` | 交互式可视化页面（深色主题 `base-dark.css`） |
| `examples/` | 可运行示例 |
| `RAPTOR.pdf` | 原始论文 PDF |

## 🗂️ 核心原理

1. 将文本段进行**语义聚类**（UMAP + 层次聚类）
2. 对每个聚类递归**摘要**
3. 向上逐层构建树，形成从「片段 → 摘要 → 高层语义」的多粒度检索结构

## 📌 适用场景

- 长文档 / 多文档 RAG 检索
- 需要同时支持「精确细节」与「全局语义」两类查询的场景