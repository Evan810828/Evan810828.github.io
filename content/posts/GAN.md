+++
title = "Generative Adversarial Networks (GAN)"
date = "2025-10-01T00:00:00-05:00"
draft = false
permalink = "/blogs/gan/"
math = true
+++

## 1. 舍弃对真实数据分布的拟合

传统生成模型通常尝试找到一个分布 \(P_\theta\)，使其尽可能接近真实数据分布 \(P_{\text{data}}\)。  
GANs 则换了一个视角：它们并不直接拟合概率密度，而是提出一个问题：  

**是否可以生成来自 \(P_\theta\) 的样本，使它们与来自 \(P_{\text{data}}\) 的样本无法区分？**
---

- 定义两个样本集合：
  - \(S_a = \{x \mid x \sim P_\theta\}\) —— 模型生成的样本  
  - \(S_b = \{x \mid x \sim P_{\text{data}}\}\) —— 真实分布的样本  
  
目标：判断 \(S_a\) 与 \(S_b\) 是否来自同一分布。 如果两者不可区分，则说明生成模型学到了足够接近真实分布的能力。  

- **生成器 \(G\)**：输入噪声 \(z\)，输出样本，目标是生成“以假乱真”的数据。  
$$
G_{\theta_g}: \mathbb{R}^P \to \mathbb{R}^N
$$
- **判别器 \(D\)**：输入样本，输出概率，表示该样本来自真实分布还是生成分布。  
$$
D_{\theta_d}: \mathbb{R}^N \to [0,1]
$$
二者形成对抗关系：
  - \(G\) 的目标：最大化判别器出错的概率；  
  - \(D\) 的目标：正确地区分真实样本和生成样本。


## 2. GAN 的训练过程等价于最小化数据分布与生成分布之间的 JS 散度


判别器目标：$
  D(x) = 1 \quad \text{if } x \sim P_{\text{data}}, \qquad 
  D(G(z)) = 0 \quad \text{if } z \sim P_Z \; (x \sim P_g)
  $

- 真实样本的判别准确率：  $\text{Accuracy}_{\text{real}} = D(x)$

- 生成样本的判别准确率：  $\text{Accuracy}_{\text{fake}} = 1 - D(G(z))$

**目标函数**：$\min_G \max_D V(D,G)$
其中
$$
V(D,G) = \mathbb{E}_{x \sim P_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim P_Z}[\log (1 - D(G(z)))]
$$

当固定生成器G的情况下，D的最优解为 $$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

当且仅当生成分布等于真实分布时，即：
  $$
  p_g(x) = p_{\text{data}}(x)
  $$
  生成器 \(G\) 达到全局最优。  

此时optimal value为$\frac{1}{2}$，也就是说判别器D做出完全随机的猜测。

---

### KL 散度定义：
  $$
  D_{KL}(p \parallel q) = \int p(x) \log \frac{p(x)}{q(x)} dx
  $$
在 GAN 中，我们考虑：

$$
\mathrm{KL}\!\left(p_{\text{data}} \Big\| \tfrac{p_{\text{data}} + p_g}{2}\right)
= \int p_{\text{data}}(x) 
\left( 
    \log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)} - \log \tfrac{1}{2}
\right) dx
$$

$$
= \int p_{\text{data}}(x) \Big( \log D_G^*(x) - \log D_{G^*}^*(x) \Big) dx
$$

$$
\mathrm{KL}\!\left(p_g \Big\| \tfrac{p_{\text{data}} + p_g}{2}\right)
= \int p_g(x) 
\left( 
    \log \frac{p_g(x)}{p_{\text{data}}(x)+p_g(x)} - \log \tfrac{1}{2}
\right) dx
$$

$$
= \int p_g(x) \Big( \log (1 - D_G^*(x)) - \log (1 - D_{G^*}^*(x)) \Big) dx
$$


两项 KL 相加等价于 GAN 的目标函数与最优判别器下的值函数之差：


$$
\mathrm{KL}\!\left(p_{\text{data}} \Big\| \tfrac{p_{\text{data}}+p_g}{2}\right) + \mathrm{KL}\!\left(p_g \Big\| \tfrac{p_{\text{data}}+p_g}{2}\right)
$$

$$
= \int p_{\text{data}}(x)\big(\log D_G^*(x) - \log D_{G^*}^*(x)\big)\,dx+ \int p_g(x)\big(\log(1-D_G^*(x)) - \log(1-D_{G^*}^*(x))\big)\,dx
$$

$$
= \mathbb{E}_{x\sim p_{\text{data}}}[\log D_G^*(x)]- \mathbb{E}_{x\sim p_{\text{data}}}[\log D_{G^*}^*(x)]+ \mathbb{E}_{x\sim p_g}[\log(1-D_G^*(x))]- \mathbb{E}_{x\sim p_g}[\log(1-D_{G^*}^*(x))]
$$

$$
= \Big(\mathbb{E}_{x\sim p_{\text{data}}}[\log D_G^*(x)]+ \mathbb{E}_{x\sim p_g}[\log(1-D_G^*(x))]\Big)- \Big(\mathbb{E}_{x\sim p_{\text{data}}}[\log D_{G^*}^*(x)]+ \mathbb{E}_{x\sim p_g}[\log(1-D_{G^*}^*(x))]\Big)
$$

$$
= V(G, D_G^*) - V(G^*, D_{G^*}^*)
$$


**Jensen–Shannon (JS) 散度**定义为：
$$
JS(p \parallel q) = \tfrac{1}{2} KL\Big(p \parallel \tfrac{p+q}{2}\Big) + \tfrac{1}{2} KL\Big(q \parallel \tfrac{p+q}{2}\Big)
$$

在 GAN 中：
  $$
  \min_G \Big\{ V(G, D^*_G) - V(G^*, D^*_{G^*}) \Big\} = \min_G 2 \, JS(p_{\text{data}} \parallel p_g)
  $$

当 \(p_g = p_{\text{data}}\) 时，JS 散度为 0，模型达到最优

## 3. GAN 中的 Mode Collapse


当生成分布 \(p_g\) 与真实分布 \(p_{\text{data}}\) **不重叠**时，JSD 达到常数最大值 \(\log 2\)，梯度消失，生成器无法获得改进的方向，导致训练停滞。 这导致在早期训练阶段（分布差异大时），GAN 很容易出现无法收敛的情况。  


  1. 训练一个 GAN（如 DCGAN）分别运行 **1, 10, 25 个 epoch**。  
  2. 固定该生成器 \(G\)。  
  3. 从零开始训练判别器 \(D\)。  
  4. 在判别器训练过程中，测量 **原始 GAN 损失下生成器的梯度范数**。  

![alt text](../files/image.png)

**训练 1 epoch 的生成器**
- 输出样本接近噪声，分布较分散。  
- 与真实分布存在部分 **模糊重叠**。  
- 判别器无法完全分离真假样本 → JSD 未饱和 → 生成器仍获得 **非零梯度**。  

**训练 10 epoch 的生成器**
- 学到一些结构特征，比 1 epoch 更接近真实分布。  
- 生成分布与真实分布依然有 **部分交集**。  
- 判别器虽逐渐变强，但生成器梯度 **不会完全消失**。  

**训练 25 epoch 的生成器**
- 容易出现 **模式崩溃 (mode collapse)**，生成器陷入某个狭小区域，不再探索整个数据分布，缺乏多样性。  
- 导致生成分布和真实分布几乎 **不重叠**。  
- 判别器可轻松分离真假 → JSD 饱和到常数 \(\log 2\) → 生成器梯度 **彻底消失**。  

---

从 JSD 的损失函数看，展开为两部分：
$$
JSD = \int p_{\text{data}}(x)\,\log \frac{2p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}\,dx + \int p_g(x)\,\log \frac{2p_g(x)}{p_{\text{data}}(x)+p_g(x)}\,dx
$$
$$
= \frac{1}{N}\sum_{n=1}^N \log D(x^n) + \frac{1}{M}\sum_{m=1}^M \log \big(1 - D(G(z^m))\big) \quad \text{(remove constants)}
$$

- 第一项 **Coverage** term 
  如果某区域真实分布 \(p_{\text{data}}(x)\) 很高，那么生成分布 \(p_g(x)\) 也应该高。 但这一项并 **不会直接出现在生成器的优化目标中**，因此生成器不会被强制覆盖所有模式。  

- 第二项 **Quality** term
  如果某区域生成分布 \(p_g(x)\) 高，而真实分布 \(p_{\text{data}}(x)\) 很低，会受到惩罚。 生成器主要优化这一项 → 倾向于集中在少量真实分布峰值上，来最小化惩罚。  


其中生成器部分对应 quality term，而 coverage term 并未被直接优化。这导致生成器只需专注于少量模式就能“骗过”判别器，而不是覆盖所有模式。  

