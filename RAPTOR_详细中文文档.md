# RAPTOR 详细中文技术文档

> **论文**：RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
> **发表**：ICLR 2024
> **代码来源**：InfiniFlow (RAGFlow) 开源项目
> **文件**：`raptor.py`

---

## 目录

1. [概述](#1-概述)
2. [核心概念](#2-核心概念)
3. [树形结构构建](#3-树形结构构建)
4. [聚类方法详解](#4-聚类方法详解)
5. [Psi 树构建器](#5-psi-树构建器)
6. [树遍历与检索](#6-树遍历与检索)
7. [代码实现详解](#7-代码实现详解)
8. [实验与性能](#8-实验与性能)
9. [幻觉分析](#9-幻觉分析)
10. [附录](#10-附录)

---

## 1. 概述

### 1.1 什么是 RAPTOR？

**RAPTOR**（Recursive Abstractive Processing for Tree-Organized Retrieval）是一种**基于树形结构的递归抽象处理与检索系统**。它通过递归地对文本块进行**聚类**（clustering）和**摘要**（summarization），构建一个多层级的树形结构，从而在不同抽象层次上捕捉文本信息，大幅提升大语言模型在长文档问答中的检索效果。

### 1.2 核心思想

传统的检索增强生成（RAG）方法通常只检索原始文本块（leaf-level chunks），而 RAPTOR 的核心创新在于：

- **多层级表示**：构建一个从底层原始文本到高层摘要的树形结构
- **跨片段综合**：通过摘要将分散在不同段落的信息整合到一起
- **分层检索**：检索时同时利用叶节点（细节）和内部节点（主题/摘要），根据不同问题类型自动选择合适的层次

### 1.3 为什么需要 RAPTOR？

| 问题 | RAPTOR 的解决方案 |
|------|-------------------|
| 长文档中信息分散 | 通过聚类将相关内容聚合，再由 LLM 摘要 |
| 主题级问题需要全局理解 | 高层节点包含跨段的综合信息 |
| 细节问题需要精确定位 | 叶节点保留原始文本细节 |
| 多跳推理（multi-hop） | 树结构自然支持从不同层获取信息片段 |

### 1.4 文件概览

本目录包含两个文件：

| 文件 | 描述 |
|------|------|
| `RAPTOR.pdf` | ICLR 2024 论文原文（23页），包含完整的方法论、实验和附录 |
| `raptor.py` | InfiniFlow RAGFlow 项目中的 Python 实现（754行），包含经典 RAPTOR 和 Psi 树两种构建器 |

---

## 2. 核心概念

### 2.1 树形结构概述

RAPTOR 构建的是一棵**多层级树**：

```
                    ┌──────────┐
                    │  Layer 2 │  ← 最高层摘要（最抽象，覆盖全文主题）
                    │  (Root)  │
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              │                     │
         ┌────┴─────┐         ┌────┴─────┐
         │  Layer 1 │         │  Layer 1 │  ← 中间层摘要（部分主题）
         └────┬─────┘         └────┬─────┘
              │                    │
        ┌─────┼─────┐        ┌────┼────┐
        │     │     │        │    │    │
    [chunk1][chunk2][chunk3][chunk4][chunk5]  ← Layer 0 叶节点（原始文本块）
```

**构建过程**：
1. **分块**：将长文档切分成文本块（chunks），每个块生成向量嵌入（embedding）
2. **降维 + 聚类**：使用 UMAP 降维，然后用 GMM 或 AHC 聚类
3. **摘要生成**：对每个簇内的文本块，用 LLM 生成摘要，摘要作为上一层的节点
4. **迭代**：对上一层节点重复步骤 2-3，直到只剩一个节点或无法继续聚类

### 2.2 关键组件

| 组件 | 作用 |
|------|------|
| **嵌入模型** (Embedding Model) | 将文本转为向量，用于相似度计算和聚类 |
| **UMAP** | 降维算法，将高维嵌入降到低维空间以提升聚类质量 |
| **GMM / AHC** | 聚类算法，将语义相近的文本块分组 |
| **LLM** | 对每个簇生成摘要（生成式摘要，而非抽取式） |
| **向量检索** | 基于向量相似度从树中检索相关节点 |

---

## 3. 树形结构构建

### 3.1 经典 RAPTOR 构建流程

```
输入: chunks = [(text_1, embedding_1), (text_2, embedding_2), ..., (text_n, embedding_n)]
输出: chunks_with_summaries, layers

算法:
  layers = [(0, n)]  # 初始层：从索引0到n的原始块
  start, end = 0, n

  while end - start > 1:   # 只要当前层有超过1个节点
      1. 提取当前层的嵌入 [embd for _, embd in chunks[start:end]]
      2. 用 UMAP 降维
      3. 对降维后的嵌入进行聚类 (GMM 或 AHC)
      4. 对每个簇，用 LLM 生成摘要:
         summary_text, summary_embedding = await summarize(cluster_indices)
         chunks.append((summary_text, summary_embedding))
      5. layers.append((end, len(chunks)))  # 记录新层边界
      6. start = end, end = len(chunks)     # 移到下一层
```

### 3.2 UMAP 降维

在聚类之前，RAPTOR 使用 **UMAP**（Uniform Manifold Approximation and Projection）将高维嵌入降维：

```python
n_neighbors = int((len(embeddings) - 1) ** 0.8)   # 邻居数随数据量自适应
n_components = min(12, len(embeddings) - 2)        # 目标维度，最多12维
reduced = umap.UMAP(
    n_neighbors=max(2, n_neighbors),
    n_components=n_components,
    metric="cosine"                                # 使用余弦距离
).fit_transform(embeddings)
```

**为什么需要降维？**
- 高维空间中的距离度量会发生"维度灾难"，所有点之间的距离趋于相近
- UMAP 在保持局部结构的同时降维，使聚类更加有效

### 3.3 两种树构建器

代码支持两种树构建模式：

| 构建器 | 常量 | 描述 |
|--------|------|------|
| **经典 RAPTOR** | `RAPTOR_TREE_BUILDER` | 基于聚类的自底向上构建，每层通过聚类+摘要生成上级节点 |
| **Psi 树** | `PSI_TREE_BUILDER` | 基于嵌入相似度排序+并查集构建合并树，更高效 |

---

## 4. 聚类方法详解

### 4.1 GMM（高斯混合模型）聚类

GMM 是默认的聚类方法，通过 BIC（贝叶斯信息准则）自动选择最佳簇数：

```python
def _get_optimal_clusters(self, embeddings, random_state):
    max_clusters = min(self._max_cluster, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))  # BIC 越低越好
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters
```

**软聚类（Soft Clustering）**：
- GMM 支持软分配：每个点可以属于多个簇（有一定概率）
- 通过 `threshold` 参数控制归属阈值
- 一个文本块如果与多个簇相关，可以贡献到多个摘要中

```python
gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
gm.fit(reduced_embeddings)
probs = gm.predict_proba(reduced_embeddings)
# threshold 默认为 0.1：概率 > 0.1 的簇都算归属
lbls = [np.where(prob > self._threshold)[0] for prob in probs]
```

### 4.2 AHC（层次凝聚聚类）

AHC（Agglomerative Hierarchical Clustering）是另一种聚类选项：

```python
def _get_clusters_ahc(self, embeddings):
    # 1. 先用 Ward 连接法构建完整层次树
    full_clust = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        compute_distances=True,
        linkage="ward"
    )
    full_clust.fit(embeddings)
    
    # 2. 通过"距离间隔"启发式确定最佳簇数
    #    在合并距离变化最大的地方切分
    distances = full_clust.distances_
    gaps = np.diff(distances)
    max_gap_idx = int(np.argmax(gaps))
    n_clusters = max(1, min(n - max_gap_idx - 1, self._max_cluster))
    
    # 3. 用确定的簇数重新聚类
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return clustering.fit_predict(embeddings)
```

**AHC 的迭代优化（Ajust Tree Nodes）**：
聚类后还进行了额外优化——将每个节点重新分配到最近的簇中心：

```python
def _adjust_tree_nodes(self, embeddings, labels, max_iter=5):
    for _ in range(max_iter):
        centroids = [embeddings[labels == lbl].mean(axis=0) for lbl in unique_labels]
        # 将每个点分配给它最近的质心
        new_labels = unique_labels[np.argmin(sq_dists, axis=1)]
        if converged:
            break
    return labels
```

### 4.3 聚类方法对比

| 特性 | GMM | AHC |
|------|-----|-----|
| 软/硬聚类 | 软聚类（概率分配） | 硬聚类（唯一分配） |
| 簇数选择 | BIC 自动选择 | 距离间隔启发式 |
| 迭代优化 | 无 | 有（_adjust_tree_nodes） |
| 复杂度 | O(n × k × d) | O(n² log n) |

---

## 5. Psi 树构建器

### 5.1 设计动机

经典 RAPTOR 每层都需要做 UMAP 降维 + 聚类 + LLM 摘要，当文档极长时效率较低。**Psi 树**通过以下方式优化：

- **一次构建**：基于原始嵌入的余弦相似度，一次性构建完整树结构
- **分桶策略**：对于大量叶子节点，先分桶再在每个桶内精确构建
- **并查集合并**：使用带秩并查集（Union-Find with Ranks）构建树边

### 5.2 Psi 树数据结构

```python
@dataclass
class _PsiTreeNode:
    index: int                          # 节点编号
    text: str = ""                      # 节点文本（叶节点=原始文本，内部节点=摘要）
    embedding: np.ndarray | None = None  # 向量嵌入
    children: list["_PsiTreeNode"] = []  # 子节点列表
    parent: "_PsiTreeNode | None" = None # 父节点引用
```

### 5.3 Psi 并查集（_PsiUnionFind）

这是 Psi 树的核心数据结构，支持带秩的合并操作：

```python
class _PsiUnionFind:
    def __init__(self, n):
        self._rank = [0 for _ in range(n)]        # 每个节点的秩
        self._parent_chains = [[] for _ in range(n)] # 父节点链
        self._node_ids = [[i] for i in range(n)]  # 每个节点的ID历史
        self._tree = [-1 for _ in range(2 * n - 1)] # 紧凑的子→父数组
        self._next_id = n                          # 下一个可用ID
    
    def union(self, i, j):
        """合并两个叶子节点，记录树边"""
        # 1. 找到各自的根
        root_i = self._find(i)[-1]
        root_j = self._find(j)[-1]
        if root_i == root_j:
            return False  # 已在同一树中
        
        # 2. 根据秩决定合并方向
        if self._rank[root_i] < self._rank[root_j]:
            # root_i 插入到 root_j 的父链中（在恰当位置）
            insert_point = chain[higher_rank_idx]
            self._ordered_extend(self._parent_chains[root_i], chain[higher_rank_idx:])
            self._build(root_i, root_j, insert_point=insert_point)
        elif self._rank[root_i] > self._rank[root_j]:
            # 对称处理
            ...
        else:
            # 秩相等：两个根合并，秩+1
            self._rank[root_i] += 1
            self._build(root_i, root_j)
        return True
```

### 5.4 Psi 树构建流程

```
1. 排名叶子对（Rank Leaf Pairs）
   - 计算所有叶子对之间的余弦相似度
   - 按相似度从高到低排序
   - 输出：ranked_pairs = [(i1, j1), (i2, j2), ...]

2. 并查集合并
   for each (i, j) in ranked_pairs:
       union_find.union(i, j)
       merges += 1
       if merges == n - 1:  # 树构建完成
           break

3. 从并查集的 tree 数组恢复树结构
   - tree[child_idx] = parent_idx
   - 构建 _PsiTreeNode 的父子关系

4. 分桶策略（处理大数据量）
   - 如果叶子数 > psi_exact_max_leaves (默认4096):
     将叶子分成多个桶（每桶最多 psi_bucket_size 个，默认1024）
     - 在每个桶内精确构建子树
     - 用子树的平均嵌入作为"原型"嵌入
     - 再对原型嵌入进行精确构建
```

### 5.5 桶内拆分算法

```python
def _split_psi_buckets(self, nodes):
    """通过 K-means 风格的迭代将大集合拆分成桶"""
    while groups:
        group = groups.pop()
        if len(group) <= self._psi_bucket_size:
            buckets.append(group)
            continue
        
        # 使用 K-means 风格迭代（5轮）拆分
        fanout = min(max(2, ceil(len(group) / psi_bucket_size)), len(group), 32)
        centers = group_embeddings[center_idx]
        
        for _ in range(5):
            # 分配: 每个点归到最近的质心
            labels = np.argmax(group_embeddings @ centers.T, axis=1)
            # 更新: 重新计算质心
            for center_id in range(fanout):
                centers[center_id] = group_embeddings[mask].mean(axis=0)
        
        # 根据最终分配拆分
        split_groups = [group[labels == cid] for cid in range(fanout)]
        groups.extend(split_groups)
```

### 5.6 Psi 树平衡（Rebalance）

构建完成后，检查每个内部节点的子节点数是否超过 `max_cluster`：

```python
def _rebalance_psi_tree(self, root, next_index):
    max_children = max(2, int(self._max_cluster or 2))
    
    def rebalance(node):
        for child in node.children:
            rebalance(child)
        
        while len(node.children) > max_children:
            # 将子节点分批，每批 max_children 个
            # 为每批创建新的父节点
            for start in range(0, len(node.children), max_children):
                batch = node.children[start:start + max_children]
                if len(batch) == 1:
                    grouped_children.append(batch[0])
                else:
                    grouped_children.append(self._create_psi_parent(next_index, batch))
                    next_index += 1
            node.children = grouped_children
```

### 5.7 Psi 层摘要生成

树结构构建完成后，自底向上逐层生成摘要：

```python
async def _build_psi_layers(self, chunks):
    layers = [(0, len(chunks))]
    root, _ = self._build_psi_structure(chunks)  # 构建树结构
    
    for layer_idx, nodes in enumerate(sorted(self._psi_layers(root).items())):
        # 并行对同一层的所有内部节点生成摘要
        tasks = [summarize_node(node) for node in nodes]
        results = await asyncio.gather(*tasks)
        
        for node in results:
            if node is not None:
                chunks.append((node.text, node.embedding))  # 追加到 chunks 列表
        
        layers.append((layer_start, len(chunks)))
    
    return chunks, layers
```

### 5.8 经典 RAPTOR vs Psi 树对比

| 特性 | 经典 RAPTOR | Psi 树 |
|------|------------|--------|
| 构建方式 | 逐层聚类 + 摘要 | 一次构建合并树 + 分层摘要 |
| 层数 | 取决于聚类结果 | 取决于合并树高度 |
| 簇大小 | 由聚类算法决定 | 由 max_cluster 参数控制 |
| 大数据量处理 | 受限于内存 | 分桶策略，支持更大数据集 |
| 精确度 | 完整 GMM/AHC | 近似（桶内精确，桶间近似） |
| LLM 调用次数 | 更多 | 可能更少 |

---

## 6. 树遍历与检索

### 6.1 树遍历检索（Tree Traversal）

论文算法 1 描述的方法：自顶向下逐层检索 top-k 节点。

```
算法: TRAVERSE_TREE(tree, query, k)
  1. S_current = tree.layer[0]          # 从根层开始
  2. for layer in range(tree.num_layers):
  3.     topk = []
  4.     for node in S_current:
  5.         score = dot_product(query_embedding, node.embedding)
  6.         topk.append((node, score))
  7.     S_layer = sorted(topk)[:k].nodes  # 选 top-k
  8.     S_current = S_layer               # 下一层的候选 = 这层的孩子
  9. return 所有层选中的节点集合
```

**工作方式**：
- 从最顶层开始，每层选 k 个最相关的节点
- 下一层只在这些节点的子节点中搜索
- 最终返回所有层选中节点的并集

### 6.2 折叠树检索（Collapsed Tree）

论文算法 2：将整棵树展平为一维列表，按相似度排序后按 token 预算截取。

```
算法: COLLAPSED_TREE(tree, query, k, max_tokens)
  1. flat_tree = flatten(tree)            # 展平所有节点
  2. 计算每个节点与 query 的相似度
  3. 按相似度降序排列
  4. result = []
  5. for node in sorted_nodes:
  6.     if total_tokens + node.token_size < max_tokens:
  7.         result.append(node)           # 加入结果直到 token 预算用尽
  8. return result
```

**两种检索方式对比**：

| 特性 | 树遍历 | 折叠树 |
|------|--------|--------|
| 结构利用 | 利用树结构逐层过滤 | 忽略结构，展平全搜索 |
| 效率 | 更高（每层只搜索 k 个节点） | 更低（需计算所有节点相似度） |
| 精确度 | 可能遗漏跨层相关节点 | 更完整（全局排序） |

### 6.3 各层节点的贡献

实验表明，RAPTOR 检索结果中 **18.5% 到 57%** 的节点来自非叶节点：

| 数据集 | DPR | SBERT | BM25 |
|--------|-----|-------|------|
| NarrativeQA | 57.36% | 36.78% | 34.96% |
| Quality | 32.28% | 24.41% | 32.36% |
| Qasper | 22.93% | 18.49% | 22.76% |

这表明中间层的摘要节点在检索中起着至关重要的作用。

---

## 7. 代码实现详解

### 7.1 类结构

```python
class RecursiveAbstractiveProcessing4TreeOrganizedRetrieval:
    """RAPTOR 摘要层构建器，支持经典和 Psi 树策略"""
    
    def __init__(
        self,
        max_cluster,          # 最大簇数（控制每层最多有多少节点）
        llm_model,            # LLM 模型（用于生成摘要）
        embd_model,           # 嵌入模型（用于向量化）
        prompt,               # 摘要提示词模板
        max_token=512,        # 摘要最大 token 数
        threshold=0.1,        # GMM 软聚类阈值
        max_errors=3,         # 最大错误容忍数
        tree_builder="raptor",    # 树构建器: "raptor" 或 "psi"
        clustering_method="gmm",  # 聚类方法: "gmm" 或 "ahc"
        psi_exact_max_leaves=4096,  # Psi: 精确构建的最大叶子数
        psi_bucket_size=1024,       # Psi: 每桶最大叶子数
    )
```

### 7.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_cluster` | - | 最大簇数。控制每层抽象后最多产生多少个节点 |
| `max_token` | 512 | LLM 生成的摘要最大 token 数 |
| `threshold` | 0.1 | GMM 软聚类阈值。概率 > 此值的簇被计入 |
| `max_errors` | 3 | 摘要生成最大连续错误数，超限则中止 |
| `tree_builder` | `"raptor"` | `"raptor"` 使用经典 GMM/AHC 聚类；`"psi"` 使用 Psi 合并树 |
| `clustering_method` | `"gmm"` | `"gmm"` 或 `"ahc"` |
| `psi_exact_max_leaves` | 4096 | 叶子数超过此值时分桶处理 |
| `psi_bucket_size` | 1024 | 每桶的最大叶子数 |

### 7.3 核心方法调用链

```
__call__(chunks, random_state, callback, task_id)  ← 入口
  │
  ├── [经典 RAPTOR]
  │   └── while end - start > 1:
  │       ├── UMAP 降维
  │       ├── _get_optimal_clusters() [GMM] 或 _get_clusters_ahc() [AHC]
  │       ├── _adjust_tree_nodes() [仅 AHC]
  │       └── 对每个簇 → summarize() → _summarize_texts()
  │
  └── [Psi 树]
      └── _build_psi_layers()
          ├── _build_psi_structure()
          │   └── _build_psi_structure_from_nodes()
          │       ├── _build_exact_psi_structure()  [≤4096 叶子]
          │       └── _build_bucketed_psi_structure() [>4096 叶子]
          │           ├── _split_psi_buckets()
          │           └── 对每个桶 → _build_exact_psi_structure()
          ├── _rebalance_psi_tree()
          └── 逐层并行 → summarize_node() → _summarize_texts()
```

### 7.4 LLM 摘要生成

```python
async def _summarize_texts(self, texts: list[str], callback=None, task_id=""):
    # 1. 对每个文本截断，控制在 LLM 上下文窗口内
    len_per_chunk = int((llm_max_length - max_token) / len(texts))
    cluster_content = "\n".join([truncate(t, max(1, len_per_chunk)) for t in texts])
    
    # 2. 调用 LLM 生成摘要（带缓存和重试）
    cnt = await self._chat(
        system="You're a helpful assistant.",
        history=[{
            "role": "user",
            "content": prompt.format(cluster_content=cluster_content)
        }],
        gen_conf={"max_tokens": max(max_token, 512)}
    )
    
    # 3. 清理特殊标记
    cnt = re.sub(r"^.*</think>", "", cnt, flags=re.DOTALL)
    
    # 4. 对摘要文本生成嵌入
    embds = await self._embedding_encode(cnt)
    
    return cnt, embds
```

**LLM 调用特性**：
- **缓存机制**：通过 `get_llm_cache` / `set_llm_cache` 避免重复调用
- **重试机制**：最多 3 次重试，每次间隔递增（1s, 2s）
- **超时控制**：单次调用 20 分钟超时
- **限流**：通过 `chat_limiter` 控制并发
- **嵌入缓存**：同理通过 `get_embed_cache` / `set_embed_cache` 缓存嵌入结果

### 7.5 摘要压缩比

论文实验统计：
- **平均压缩率**：72%（即摘要长度约为子节点总长度的 28%）
- **平均摘要长度**：131 tokens
- **平均子节点文本长度**：85.6 tokens
- **平均每父节点子节点数**：6.7 个

---

## 8. 实验与性能

### 8.1 基准测试结果

#### QuALITY 数据集（问答准确率）

| 模型 | 准确率 |
|------|--------|
| Longformer-base | 39.5% |
| DPR + DeBERTaV3-large | 55.4% |
| CoLISA (DeBERTaV3-large) | 62.3% |
| **RAPTOR + GPT-4** | **82.6%** |

#### QASPER 数据集（F-1 分数）

| 模型 | F-1 Match |
|------|-----------|
| LongT5 XL | 53.1% |
| CoLT5 XL | 53.9% |
| **RAPTOR + GPT-4** | **55.7%** |

#### NarrativeQA 数据集（METEOR 分数）

| 模型 | METEOR |
|------|--------|
| BiDAF | 3.7 |
| BM25 + BERT | 5.0 |
| 递归摘要（Wu et al., 2021）| 10.6 |
| Retriever + Reader | 11.1 |
| **RAPTOR + UnifiedQA** | **19.1** |

### 8.2 控制实验：聚类方法消融

| 配置 | QuALITY 准确率 |
|------|----------------|
| RAPTOR + SBERT + UnifiedQA（聚类） | 56.6% |
| 相邻块合并树 + SBERT + UnifiedQA | 55.8% |

聚类方法比简单的相邻块合并提升了 0.8 个百分点，证明了内容感知聚类的价值。

### 8.3 检索方法对比

| 检索器 | GPT-3 F-1 | GPT-4 F-1 | UnifiedQA F-1 |
|--------|-----------|-----------|---------------|
| 仅标题+摘要 | 25.2 | 22.2 | 17.5 |
| BM25 | 46.6 | 50.2 | 26.4 |
| DPR | 51.3 | 53.0 | 32.1 |
| **RAPTOR** | **53.1** | **55.7** | **36.6** |

### 8.4 计算成本

**线性扩展**：
- **Token 消耗**：与文档长度呈线性关系（R² 极高）
- **构建时间**：与文档长度呈线性关系
- **测试硬件**：Apple M1 Mac, 16GB RAM
- **测试范围**：12,500 ~ 78,000 tokens

这意味着 RAPTOR 在大规模语料库上具有良好的可扩展性。

---

## 9. 幻觉分析

论文对 RAPTOR 生成的摘要进行了幻觉（hallucination）分析：

### 9.1 分析方法
- 从 40 个故事中随机采样 150 个节点
- 人工标注每个节点是否包含幻觉
- 检查幻觉是否向父节点传播

### 9.2 发现

| 指标 | 结果 |
|------|------|
| 幻觉率 | **4%**（6/150 个节点） |
| 传播性 | 幻觉**不会**向更高层传播 |
| 严重程度 | 大多是轻微信息添加或外推，不影响主题理解 |
| 对 QA 的影响 | **无显著影响** |

### 9.3 幻觉示例

**子节点文本**：Bradley 被请求留在部落成为战士，但他拒绝并必须返回自己的国家。

**父节点摘要**（含幻觉）：摘要中称 Ajor 和 Co-Tan 是**姐妹**关系，但原文并未明确说明。

**结论**：在 RAPTOR 架构中，幻觉不是一个主要问题，且不会严重影响下游 QA 任务。

---

## 10. 附录

### 10.1 摘要提示词（Prompt）

```
System: You are a Summarizing Text Portal
User: Write a summary of the following, including as many key details as possible: {cluster_content}
```

### 10.2 数据集统计

| 数据集 | 平均摘要长度(tokens) | 平均子节点长度(tokens) | 平均子节点数/父节点 | 压缩率 |
|--------|---------------------|----------------------|-------------------|--------|
| 全部 | 131.0 | 85.6 | 6.7 | 28% |
| QuALITY | 124.4 | 87.9 | 5.7 | 28% |
| NarrativeQA | 129.7 | 85.5 | 6.8 | 27% |
| QASPER | 145.9 | 86.2 | 5.7 | 35% |

### 10.3 依赖关系

`raptor.py` 依赖以下组件：

```
numpy          → 数值计算
umap           → 降维
scikit-learn   → GMM + AHC 聚类

RAGFlow 内部依赖:
  api.db.services.task_service   → 任务取消检查
  common.connection_utils        → 超时装饰器
  common.exceptions              → 任务取消异常
  common.token_utils             → 文本截断
  common.misc_utils              → 线程池工具
  rag.graphrag.utils             → LLM/嵌入缓存, 限流器
  rag.utils.raptor_utils         → 常量定义
```

### 10.4 使用示例

```python
# 初始化 RAPTOR 处理器
raptor = RecursiveAbstractiveProcessing4TreeOrganizedRetrieval(
    max_cluster=10,
    llm_model=my_llm,
    embd_model=my_embedder,
    prompt="Write a summary of the following: {cluster_content}",
    max_token=512,
    threshold=0.1,
    tree_builder="raptor",       # 或 "psi"
    clustering_method="gmm",     # 或 "ahc"
)

# 构建 RAPTOR 树
chunks = [(text1, emb1), (text2, emb2), ..., (textN, embN)]
enriched_chunks, layers = await raptor(
    chunks=chunks,
    random_state=42,
    callback=progress_callback,
    task_id="doc_123"
)

# enriched_chunks 包含原始块 + 所有层的摘要块
# layers = [(0, 50), (50, 65), (65, 72), ...]  每层的起止索引
```

### 10.5 关键设计决策总结

1. **生成式摘要（Abstractive）优于抽取式（Extractive）**：LLM 能综合多个文本块的信息生成连贯摘要
2. **软聚类（GMM）优于硬聚类**：允许一个文本块对多个摘要做贡献
3. **UMAP 降维必要**：高维空间中聚类效果差，UMAP 保留局部结构
4. **缓存至关重要**：重复的 LLM 调用和嵌入计算通过缓存大幅降低计算成本
5. **Psi 树适合超大规模**：通过分桶+近似减少 O(n²) 相似度计算

---

> **参考资料**
> - 原论文：RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (ICLR 2024)
> - 代码仓库：InfiniFlow/RAGFlow — `rag/raptor.py`
> - 论文地址：https://arxiv.org/abs/2401.18059

---

*文档生成时间：2026年5月20日*
