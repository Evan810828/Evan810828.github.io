+++
title = "KL-Divergence in Generative Models Evaluation"
date = 2025-10-01
permalink = "/blogs/kl-generative-eval/"
draft = false
+++

- 数据真实分布：$p(x)$  
- 生成模型/参数化分布：$q_\theta(x)$  
- 目标：让 $q_\theta$ 尽可能贴近 p，通过可计算的指标来 **训练** 与 **评估**。

---

**自信息（surprisal）**：$I(x) = - \log p(x)$

**Shannon 熵（平均不确定性/最优编码平均码长下界）**：$
H(P) = \mathbb{E}_{x\sim p} \big[-\log p(x)\big]
$

**交叉熵（用“模型分布”去编码来自 p 的样本时的平均码长）**：$H(P, Q) = \mathbb{E}_{x\sim p} \big[-\log q(x)\big]$


---

## 从 Shannon 熵到 KL

在训练与评估时，真实分布 p 的密度值 $p(x)$ **不可得**，但模型分布 $q_\theta(x)$ 及其对数密度 **可计算**。因此，直接基于 $-\log p(x)$ 的衡量（即 $H(P)$）不可作为优化目标；我们考虑以 **交叉熵**：

$$
H(P,Q) = \mathbb{E}_{x\sim p}[-\log q(x)]
$$

这可以理解为把编码长度中的概率从 p 换成了 q，即**用模型分布 q 来编码真实数据 p 产生的样本**。交叉熵与熵的差正是 **KL 散度**：

$$
D_{\mathrm{KL}}(P\Vert Q) = H(P,Q) - H(P) = \mathbb{E}_{x\sim p}\!\left[\log\frac{p(x)}{q(x)}\right]
$$

### Motivation
生成模型 **无法直接访问** 或 **解析地表示** p，只能用 **参数化且可计算** 的 $q_\theta$ **去近似** $p$，并藉由 $-\log{q_\theta(x)}$ 的可计算性来进行训练与评估。当样本量充足、模型容量与优化得当时，$q_\theta$ 可以逼近 $p$。

---

**定义**：

$
D_{\mathrm{KL}}(P\Vert Q) = \sum_x p(x)\log \frac{p(x)}{q(x)} \quad \text{或} \quad 
D_{\mathrm{KL}}(P\Vert Q) = \mathbb{E}_{x\sim p}\!\left[\log\frac{p(x)}{q(x)}\right]
$

**性质**：

$
D_{\mathrm{KL}}(P\Vert Q) \ge 0, \quad \text{且当且仅当 } P=Q \text{ 时取 } 0
$

$
D_{\mathrm{KL}}(P\Vert Q) \neq D_{\mathrm{KL}}(Q\Vert P) \quad (\text{不对称})
$

**方向直觉**：

$
\text{前向 KL: } D_{\mathrm{KL}}(P\Vert Q) \;\;\Rightarrow\;\; \text{更惩罚 } p \text{ 大而 } q \text{ 小的区域（倾向覆盖模态）}
$

$
\text{反向 KL: } D_{\mathrm{KL}}(Q\Vert P) \;\;\Rightarrow\;\; \text{更惩罚 } q \text{ 大而 } p \text{ 小的区域（倾向寻找模态）}
$

---

### 最小化 KL 等价于最大化 MLE

由定义：

$$
D_{\mathrm{KL}}(P\Vert Q_\theta) = \mathbb{E}_{x\sim p}\big[\log p(x) - \log q_\theta(x)\big]
$$

将与参数无关的 $H(P)=\mathbb{E}_{p}[-\log p(x)]$ 视作常数，则

$$
\arg\min_{\theta} D_{\mathrm{KL}}(P\Vert Q_\theta)
\equiv
\arg\max_{\theta} \mathbb{E}_{x\sim p}\big[\log q_\theta(x)\big]
$$

以经验分布近似期望（样本 $\{x_i\}_{i=1}^n$ 来自 p）：

$$
\hat\theta_{\mathrm{MLE}}
= \arg\max_{\theta}\frac{1}{n}\sum_{i=1}^n \log q_\theta(x_i)
$$

也即最小化经验 **负对数似然（NLL）**：

$$
\hat\theta_{\mathrm{MLE}}
= \arg\min_{\theta}\frac{1}{n}\sum_{i=1}^n \big[-\log q_\theta(x_i)\big]
$$

