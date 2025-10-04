+++
title = "Variational Auto-Encoder"
date = 2025-10-01
permalink = "/blogs/vae/"
draft = false
math = true
+++

## 1. Dimensionality Reduction via Encoder-Decoder

\begin{align}
(e^*, d^\*) = \arg\min_{(e,d)\in E\times D}\; \epsilon\!\big(x,\; d(e(x))\big)
\end{align}

- $x \in \mathbb{R}^n$: 原始输入数据  
- $e: \mathbb{R}^n \to \mathbb{R}^m$（$m<n$）: 编码器，将 $x$ 映射到潜在空间  
- $d: \mathbb{R}^m \to \mathbb{R}^n$: 解码器，将潜在表示还原到原空间  
- $d(e(x))$: 重建结果  
- $\epsilon(\cdot,\cdot)$: 误差度量（如 MSE、MAE 或感知损失等）

###  Autoencoders

与 encoder–decoder 相同，但这里 encoder 和 decoder 都由神经网络实现，可以通过 BP 来训练整个模型  

+ 如果 encoder 和 decoder 只有一层 linear layer，并且没有非线性激活函数，那么 Autoencoder 的目标就近似等价于 PCA，不过 AE 不需要得到正交的向量
+ 而当 encoder 和 decoder 都非常深时，模型可能会记忆训练数据而不是学习有效的压缩表示；如果参数量足够大，甚至可以把所有数据挤到 1 维里去“记住”，而不是提取有意义的结构


## 2. VAE

**Motivation 1： 能否利用 Autoencoder 来生成新的内容？**

1. 先训练好 Encoder–Decoder 结构  
2. 在生成阶段，不再输入原始 $x$，而是在 latent space 中**采样**新的向量 $z'$，将 $z'$ 输入解码器，得到新的输出 $\hat{x}$ 

**Motivation 2： AE 只用到基本的 MSE 来约束整个重建过程，这种方式没有对 latent space 有直接的约束**

+ 相对的几何关系是否保留了（原始空间相近的点在 latent space 里也相近吗？）
+ 在 latent space 里取两点的中点，解码后得到的结果是否有意义？

**Solution**

VAE 加入 **regularization loss**，将 input 编码为 latent space 中的一个分布，使 latent space 平滑连续，采样有意义


**Encoder**

- 给定一组观测样本  
  $$
  \{x_k\}_{k=1}^N \sim X
  $$  
  其中 $X$ 是一个随机向量  

在 latent space 上假设一个分布  
  $$
  p(z) = \mathcal{N}(0, I)
  $$   
  - 无需优化参数，简单易用  
  - 在没有 $x$ 信息时，可将潜在空间视作噪声源（类似 GANs 的随机输入）  

**Decoder**

1. 从潜在分布采样  
   $$
   z \sim p(z)
   $$  
2. 根据 Conditional Distribution 生成观测  
   $$
   \hat{x} \sim p(x|z)
   $$  

一般设为高斯  
  $$
  p(x|z) = \mathcal{N}(f(z), cI)
  $$  
  - $f(\cdot)$：由神经网络参数化的确定性函数，作为均值  
  - $c$：控制方差  

---

假设我们知道生成分布的真实参数为 $\theta^*$  
为了生成一个类似真实数据的样本点 $x^{(i)}$，过程如下  

1. 从先验分布 $p_\theta(z)$ 中采样一个潜在变量 $z^{(i)}$  
2. 根据条件分布 $p_\theta(x \mid z=z^{(i)})$ 生成数据 $x^{(i)}$  
3. 最优参数 $\theta^*$ 是使生成的数据集概率最大的参数  

最大似然估计目标  
$$
\theta^* = \arg\max_\theta \prod_{i=1}^n p_\theta(x^{(i)})
$$

取对数变为 log-likelihood  
$$
\theta^* = \arg\max_\theta \sum_{i=1}^n \log p_\theta(x^{(i)})
$$

引入潜在变量，边际似然可写为  
$$
p_\theta(x^{(i)}) = \int p_\theta(x^{(i)}|z)p_\theta(z)\,dz
$$

似然难以求解 (Intractability of the Likelihood)  
- 在实际计算中，边际分布积分需要遍历所有可能的 $z$，代价极高  
- 因此 $p_\theta(x)$ 难以直接计算  
- 常见解决方案是使用后验分布 $p_\theta(z \mid x)$，根据贝叶斯公式表示为  
  $$
  p_\theta(z \mid x) = \frac{p_\theta(x, z)}{p_\theta(x)}
  $$


**Motivation 3：但 $p(x)$ 难以求解，导致 $p(z \mid x)$  intractable**

**Solution**

在 VAE 中，我们用另一个分布 $q(z \mid x)$ 来近似后验 $p(z \mid x)$  
- 设定为高斯分布  
  $$
  q(z \mid x) = \mathcal{N}(g(x), h(x))
  $$  
- $g(x)$ 输出均值，$h(x)$ 输出方差  
- 协方差矩阵通常假设为对角矩阵，简化计算

如何选择最优的近似分布  
- 目标是最小化近似分布 $q(z \mid x)$ 与真实后验 $p(z \mid x)$ 的差异  
- 差异度量采用 KL 散度  
  $$
  \min_{q} \; KL\big(q(z \mid x)\,\|\,p(z \mid x)\big)
  $$

  目标是最小化近似后验与真实后验的 KL 散度  
$$
(g^*, h^*) = \arg\min_{(g,h)} KL\big(q(z \mid x) \,\|\, p(z \mid x)\big)
$$



$$
= \arg\min_{(g,h)} \int q(z \mid x) \log \frac{q(z \mid x)}{p(z \mid x)} dz
$$

 
$$
= \arg\min_{(g,h)} \int q(z \mid x) \log \frac{q(z \mid x)}{p(x|z)p(z)} p(x) \, dz
$$

由于 $\log p(x)$ 与 $q(z \mid x)$ 无关，可以忽略  
$$
= \arg\min_{(g,h)} \int q(z \mid x)\Big[-\log p(x|z) + \log \frac{q(z \mid x)}{p(z)}\Big]dz
$$

$$
= \arg\min_{(g,h)} \Bigg(-\int q(z \mid x)\log p(x|z)\,dz + KL\big(q(z \mid x)\,\|\,p(z)\big)\Bigg)
$$

由于 $p(x \mid z)$ ~ $\mathcal{N}(f(z), cI)$，$-\log p(x \mid z) \approx \frac{\|x - f(z)\|^2}{2c}$

$$
= \arg\min_{(g,h)} \int q(z \mid x)\frac{\|x - f(z)\|^2}{2c}\,dz + KL\big(q(z \mid x)\,\|\,p(z)\big)
$$

这对应于 **ELBO（证据下界）** 的负值，最终的 loss = 重建误差 + KL 正则项

---

### Workflow Summary

![alt text](../files/image-1.png)

   
1. 数据集 D 提供样本 x
通过推断网络 $q_\phi(z \mid x)$ 将 x 映射为潜在分布而非单点
常用参数化：$q_\phi(z \mid x) = N(μ(x), diag(σ^2(x)))$

2. 设定先验 $p_\theta(z) = N(0, I)$
训练时用 $KL(q_\phi(z \mid x) || p_\theta(z))$ 约束 $ q_\phi$ 与先验接近以获得结构化的潜在空间

3. 从潜在分布采样 z 再经解码器得到生成分布 $p_\theta(x\mid z)$
常见设定：$p_\theta(x|z) = N(f_\theta(z), cI)$

4. 重建  $x → z ∼ q_\phi(z \mid x) → x̂ ∼ p_\theta(x \mid z)$ 目标是让 $x̂$ 接近 $x$
生成  $z ∼ p_\theta(z) → x̃ ∼ p_\theta(x \mid z)$ 无需给定 $x$

5. 训练目标：最大化 ELBO 等价于最小化以下损失
$L(x) = 𝐄_{q_\phi(z \mid x)}[−log p_\theta(x|z)] + KL(q_\phi(z \mid x) || p_\theta(z))$

---

**Motivation 4：采样过程无法通过 BP 进行更新**

- 编码器输出分布 $$ q_\phi(z \mid x) = N(μ_x, σ_x^2) $$
- 如果直接采样 $z ~ N(μ_x, σ_x^2)$，采样操作是不可导的  
- 不可导意味着无法进行反向传播，训练中断  

**Reparameterization Trick**

引入一个辅助噪声变量  
  $$
  \zeta \sim \mathcal{N}(0, I)
  $$  
将采样写成可导的形式  
  $$
  z = \mu_x + \sigma_x \odot \zeta
  $$  
随机性由 $ζ$ 提供，$μ_x$ 和 $σ_x$ 是神经网络输出，可导  