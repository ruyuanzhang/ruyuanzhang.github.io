# Variational Autoencoder (VA)

变分自动编码器（Variational Autoencoder, VA）是一种非线性潜变量模型，具有基于变分原理的高效梯度训练程序。在潜变量模型中，我们假定观察到的 $x$ 是由一些潜变量（未观察到的）$z$ 生成的；这些潜变量捕捉到了观察数据中的一些 "有趣 "结构，而这些结构并不能从观察数据本身立即看出。例如，一种称为独立成分分析的潜变量模型可用于将人们同时说话的录音中的单独语音信号分离出来。更正式地说，我们可以把潜变量模型看作是一个概率分布 $p(x|z)$，它描述了生成过程（即 $x$ 是如何从 $z$ 生成的）以及关于潜变量 $z$ 的先验值 $p(z)$。这相当于以下简单的图形模型
$$
z\rightarrow x \tag{1}
$$
#### 潜变量模型中的学习 

我们在这种模型中的目的是学习生成过程，即 $p(x|z)$（我们假设 $p(z)$是已知的）。一个好的$p(x|z)$ 会给观测到的$x$ 赋以高概率；因此，我们可以通过最大化观测数据的概率（即$p(x)$）来学习一个好的$p(x|z)$。假设 $p(x|z)$ 的参数为 $\theta$，我们需要解决以下优化问题
$$
{\rm \mathop{max}\limits_{\theta}} p_\theta(x) \tag{2}
$$
其中，$p_\theta(x)=\int_zp(z)p_\theta(x|z)$。这是一个很难的优化问题，因为它涉及 $z$的积分，但却有可能没有解析解。

#### 潜变量模型的后验推断 

目前，我们暂时抛开这个关于学习的问题，重点讨论另一个问题：潜变量模型的后验推断 $p(z|x)$。 我们很快就会看到，这个问题与学习问题密切相关，而且事实上还引出了解决这个问题的方法。给定 $p(z)$ 和 $p(x|z)$, 我们希望推断出后验分布  $p(z|x)$。 这通常相当困难，因为它涉及到对 $z$ 的积分,  $p(z|x) = \frac{p(x,z)}{\int_zp(x,z)}$。对于大多数潜变量模型来说，这个积分无法求出, 因此需要对 $p(z|x)$ 进行近似。例如，我们可以使用马尔科夫链蒙特卡洛技术对后验进行采样。不过，在这里我们将研究另一种基于变异推理的技术。变异推理将后验推理问题转化为优化问题， 变异推理将后验推理问题转化为优化问题，即找到一个尽可能接近  $p(z|x)$  的近似概率分布 $q(z|x)$  。这可以形式化为解决以下优化问题 
$$
{\rm \mathop{min}\limits_{\phi} KL} (q_\phi(z|x)||p(z|x)) \tag{3}
$$
其中，$\phi$ 表示近似值。 $q$ 和 $KL(q||p)$ 表示 $q$ and $p$ 之间的Kullback-Leibler发散，其值为 ${\rm KL}(q||p) = \int_xq(x) {\rm log} \frac{q(x)}{p(x)} $ 。然而，这个优化问题并不比我们原来的问题简单，因为它仍然要求我们估计 $p(z|x)$ 。让我们看看能否解决这个问题。根据 KL 的定义，我们可以写出
$$
\begin{aligned}
\mathrm{KL}(q_\phi(z|x)||p(z|x))& =\int_zq_\phi(z|x)\log\frac{q_\phi(z|x)}{p(z|x)}  \\
&=\int_zq_\phi(z|x)\log\frac{q_\phi(z|x)p(x)}{p(x,z)} \\
&=\int_zq_\phi(z|x)\log\frac{q_\phi(z|x)}{p(x,z)}+\int_zq_\phi(z|x)\log p(x) \\
&=-\mathcal{L}(\phi)+\log p(x)
\end{aligned}
$$

其中我们定义了
$$
\mathcal{L}(\phi)=\int_{z}q_{\phi}(z|x)\log\frac{p(x,z)}{q_{\phi}(z|x)} \tag{4}
$$
由于 $p(x)$ 与 $q_\phi(z|x)$ 无关，最小化 ${KL}(q_\phi(z|x)||p(z|x))$ 等于最大化 $\mathcal{L}(\phi)$。请注意，优化 $\mathcal{L}(\phi)$ 要容易得多，因为它只涉及 $p(x,z) = p(z)p(x|z)$ ，不涉及任何难解的积分。因此，我们可以通过求解下面的优化问题，对潜变量模型的后验进行变分推断
$$
{\rm \mathop{max}\limits_{\phi}} \ \ \mathcal{L}(\phi) \tag{5}
$$

#### 回到学习问题

上面的推导也提出了学习生成模型 $p(x|z)$ 的方法。我们可以看到，$\mathcal{L}(\phi)$ 实际上是观测数据 $p(x)$ 的对数概率的下限
$$
\begin{aligned}
\mathrm{KL}(q_\phi(z|x)||p(z|x))& =-\mathcal{L}(\phi)+\log p(x)  \\
\mathcal{L}(\phi)& =\log p(x)-\text{KL}(q_\phi(z|x)||p(z|x))  \\
\mathcal{L}(\phi)& \leq\log p(x) 
\end{aligned}
$$


其中我们使用了 KL 从不为负这一事实。现在，假设我们不做后验推断，而是固定 $q$，并学习生成模型 $p_\theta(x|z)$。那么 $ \mathcal{L}$ 现在是 $\theta$ 的函数，$\mathcal{L}(\theta) = \int_z q(z|x) {\rm log} \frac{p_\theta (x|z)p(z)}{q(z|x)}$ 。由于 $ \mathcal{L}$ 是 $log p(x)$ 的下限，我们可以最大化 $ \mathcal{L}$ 作为最大化  ${\rm log} p(x)$ 的近似。事实上，如果 $q(z|x) = p(z|x)$，则 KL 项为零，$\mathcal{L}(\theta) = \log p(x)$，即最大化 $ \mathcal{L}$  等于最大化 $p(x)$ 。这表明我们可以同时改变 $\phi$ 和 $\theta$ 来最大化 $ \mathcal{L}$ , 以达到同时学习 $q_\phi(z|x)$ 和 $p_\theta(x|z)$ 的目的:
$$
{\rm \mathop{max} \limits_{\theta, \phi}} \ \ L(\theta, \phi) \tag{6}
$$
在这里
$$
\begin{align*}
\mathcal{L}(\theta,\phi)&=\int_{z}q_{\phi}(z|x)\log\frac{p(z)p_{\theta}(x|z)}{q_{\phi}(z|x)}\\&= \mathbb{E}_q\left[\log\frac{p(z)p_\theta(x|z)}{q_\phi(z|x)}\right] 
\end{align*} \tag{7}
$$

#### 关于期望最大化（ expectation maximization, EM）的简单介绍

EM 可以看作是解决上述公式（6）中最大化问题的一种特殊策略。在 EM 中，E 步包括根据当前的 $\theta$（即后验  $p_\theta(x)$  ）计算最优 $q_\phi(z|x)$ 。在 M 步中，我们将最优 $q_\phi(z|x)$ 插入 $\mathcal{L}$，并相对于 $\theta$ 使其最大化。换句话说，EM 可以看作是一个坐标上升过程，它相对于 $\phi$ 和 $\theta$ 交替使  $\mathcal{L}$  最大化。

#### 解决公式 （6） 中的最大化问题

我们可以使用多种技术来解决上述最大化问题。在此，我们将重点讨论随机梯度上升，因为变分自动编码器使用了这种技术。在基于梯度的方法中，我们评估目标相对于模型参数的梯度，并沿着梯度方向迈出一小步。因此，我们需要估计 $\mathcal{L}(\theta,\phi)$ 的梯度。假设我们有一组样本 $z^{(l)}, l = 1...L$，我们可以对 $\mathcal{L}$ 进行如下蒙特卡罗估计
$$
\begin{gathered}\mathcal{L}(\theta,\phi)\approx\frac1L\sum_{l=1}^L\log p_\theta(x,z^{(l)})-\log q_\phi(z^{(l)}|x)\\\mathrm{where~}z^{(l)}\sim q_\phi(z|x)\end{gathered} \tag{8}
$$
并且 $p_\theta(x,z) = p(z)p_\theta(x|z)$ 。与 $\theta$ 有关的导数很容易估算，因为 $\theta$ 只出现在总和(sum)的内部。
$$
\begin{gathered}\nabla_\theta\mathcal{L}(\theta,\phi)\approx\frac1L\sum_{l=1}^L\nabla_\theta\log p_\theta(x,z^{(l)})\\\mathrm{where~}z^{(l)}\sim q_\phi(z|x)\end{gathered} \tag{9}
$$
相对而言，$\phi$ 的导数更难估算。我们不能简单地将梯度算子推入总和，因为用于估计 $ \mathcal{L}$ 的样本来自 $q_\phi(z|x)$，而 $q_\phi(z|x)$ 取决于 $\phi$ 。只要注意到 $∇_\phi \mathbb{E}q_\phi[f(z)] \neq \mathbb{E}q_\phi[∇_\phi f(z)]$ 就可以意识到这点, 其中 $f(z) = \log p_\theta(x,z^{(l)})-\log q_\phi (z^{(l)} |x)$ 。实际上，这种期望梯度的标准估计值方差过大，无法发挥作用（详见附录）。VA的一个主要贡献是对 $∇_\phi \mathcal{L}(\theta, \phi)$ 进行了更有效的估计，这依赖于所谓的重参化技巧。

#### 重参化技巧 

我们想估计 $\mathbb{E}q_{\phi(z|x)}[f(z)]$ 形式的期望梯度。问题在于，相对于 $\phi$ 的梯度难以估计，因为 $\phi$ 出现在期望的分布中。如果我们能以某种方式重写这个期望，使 $\phi$ 只出现在期望内部，我们就能简单地将梯度算子推入期望中。假设我们可以从噪声分布 $p(\epsilon)$ 中采样，并通过可微变换 $g_\phi(\epsilon, x)$  得到 $q_\phi(z|x)$的样本。
$$
z=g_\phi(\epsilon,x)\mathrm{~with~}\epsilon\sim p(\epsilon) \tag{10}
$$
那么，我们可以将期望值$\mathbb{E}q_{\phi(z|x)}[f(z)]$改写如下
$$
\mathbb{E}_{q_\phi(z|x)}[f(z)]=\mathbb{E}_{p(\epsilon)}[f(g_\phi(\epsilon,x))] \tag{11}
$$
假设我们有一组来自 $p(\epsilon)$ 的样本 $\epsilon^{(l)}, l = 1 ... L$，我们可以对 $\mathcal{L}(\theta, \phi)$ 进行蒙特卡罗估计
$$
\begin{aligned}\mathcal{L}(\theta,\phi)\approx\frac1L\sum_{l=1}^L\log p_\theta(x,z^{(l)})-\log q_\phi(z^{(l)}|x)\\\mathrm{where~}z^{(l)} =g_\phi(\epsilon^{(l)},x)\mathrm{~and~}\epsilon^{(l)}\sim p(\epsilon)\end{aligned} \tag{12}
$$


现在，$\phi$ 只出现在总和的内部，而$ \mathcal{L}$  相对于 $\phi$ 的导数可以用估计 $\theta$  的相同方法来估计。这实质上就是重参化的技巧，它大大降低了 $∇\phi\mathcal{L}(\theta, \phi)$ 估计值的方差，使得训练大型潜变量模型变得可行。我们可以为多种近似后验$q_\phi (z|x)$选择找到一个合适的噪声分布 $p(\epsilon)$ 和一个可变的变换 $g_\phi$（有关的方法，请参见原始论文[1]）。我们将在下文讨论VA时看到多元高斯分布的例子。 

#### 变分自动编码器（Variational Autoencoder, VA）

以上关于潜变量模型的讨论是一般性的，上述变异方法可以应用于任何潜变量模型。我们可以把VA看作是一种潜变量模型，它使用神经网络（特别是多层感知器）对近似后验 $q_\phi (z|x)$和生成模型 $p_\theta(x, z)$  进行建模。更具体地说，我们假设近似后验是一个多变量高斯分布，具有对角协方差矩阵。这个高斯分布的参数由一个以 $x$ 为输入的多层感知器（Multilayer Perceptron, MLP）计算得出。我们用两个非线性函数 $\mu_\phi$ 和 $\sigma_\phi$ 来表示这个 MLP，它们分别从 $x$ 映射到均值向量和标准偏差向量。
$$
q_\phi(z|x)=\mathcal{N}(z;\mu_\phi(x),\sigma_\phi(x)\mathbf{I}) \tag{13}
$$
对于生成模型 $p_\theta(x, z)$，我们假设 $p(z)$ 固定为单位多元高斯，即 $p(z) = \mathcal{N} (0, \mathbf{I})$。$p_\theta(x|z)$ 的形式取决于建模数据的性质。例如，对于实数 $x$，我们可以使用多元高斯分布；对于二元 $x$，我们可以使用伯努利分布。这里，我们假设 $x$ 是实数，$p_\theta(x|z)$是高斯分布。同样，我们假设 $p_\theta(x|z)$ 的参数由 MLP 计算得出。用两个非线性函数 $\mu_\theta$ 和 $\sigma_\theta$表示这个 MLP，它们分别从 $z$ 映射到均值向量和标准偏差向量。
$$
p_\theta(x|z)=\mathcal{N}(x;\mu_\theta(z),\sigma_\theta(z)\mathbf{I}) \tag{14}
$$
看看这个模型的网络结构，我们就知道为什么它被称为自动编码器了。
$$
x\xrightarrow{q_\phi(z|x)}z\xrightarrow{p_\theta(x|z)}x \tag{15}
$$
编码器 $q_\phi$ 将输入 $x$ 以概率方式映射到代码 $z$ ，解码器 $p_\theta$ 又将代码 $z$ 以概率方式映射回输入空间。

为了学习 $\theta$ 和 $\phi$ ，VA采用了上述变异方法。我们从 $q_\phi(z|x)$ 采样了 $z^{(l)},l = 1...L$ ，并利用这些样本获得变分下限 $\mathcal{L}(\theta, \phi)$ 的蒙特卡罗估计值，如公式 (8) 所示。然后，我们求出该下限相对于参数的导数，并在随机梯度上升过程中使用这些导数来学习 $\theta$ 和 $\phi$。如上所述，为了减小梯度估计值的方差，我们采用了重参化的技巧。我们希望使用噪声分布 $p(\epsilon)$ 和可微变换 $g_\phi$ 对多元高斯分布 $q_\phi(z|x)$ 进行重参化。我们假设从多元单位高斯中采样，即 $p(\epsilon)\sim\mathcal{N}(\epsilon;0,\mathbf{I})$。那么，如果我们让 $z = g_\phi(\epsilon, x) = \mu_\phi(x) + \epsilon\odot\sigma_\phi(x)$,  $z$ 将具有所需的分布 $q_\phi(z|x) ∼ \mathcal{N}(z; \mu_\phi(x), \sigma_\phi(x)) $（表示元素相乘）。因此，我们可以利用 $q_\phi$ 的这种重参化将变分下界重写如下
$$
\begin{aligned}\mathcal{L}(\theta,\phi)&\approx\frac1L\sum_{l=1}^L\log p_\theta(x,z^{(l)})-\log q_\phi(z^{(l)}|x)\\\mathrm{where~}z^{(l)}&=\mu_\phi(x)+\epsilon\odot\sigma_\phi(x)\mathrm{~and~}\epsilon^{(l)}\sim\mathcal{N}(\epsilon;0,\mathbf{I})\end{aligned} \tag{16}
$$
我们还可以做一个简化。将 $p_\theta(x，z)$ 明确写成 $p(z)p_\theta(x|z)$，我们可以看到
$$
\begin{aligned}
\mathcal{L}(\theta,\phi)& =\mathbb{E}_q\left[\log\frac{p(z)p_\theta(x|z)}{q_\phi(z|x)}\right] & &&&\hfill (17)\\  
&=\mathbb{E}_q\left[\log\frac{p(z)}{q_\phi(z|x)}\right]+\mathbb{E}_q\left[p_\theta(x|z)\right] \hfill& &&& (18)\\
&=-\mathrm{KL}(q_\phi(z|x)||p(z))+\mathbb{E}_q\left[p_\theta(x|z)\right]
& &&&\hfill (19)\\&& &&&\hfill (20)
\end{aligned}
$$
由于 $p(z)$ 和 $q_\phi(z|x)$ 都是高斯分布，因此 KL 项有一个封闭的表达式。将其插入，我们就得到了下面的变分下界表达式。
$$
\begin{aligned}\mathcal{L}(\theta,\phi)\approx\frac12\sum_{d=1}^D\left(1+\log(\sigma_{\phi,d}^2(x))-\mu_{\phi,d}^2(x)-\sigma_{\phi,d}^2(x)\right)+\frac1L\sum_{l=1}^L\log p_\theta(x|z^{(l)})\\\text{where }z^{(l)}=\mu_\phi(x)+\epsilon\odot\sigma_\phi(x)\text{ and }\epsilon^{(l)}\sim\mathcal{N}(\epsilon;0,\mathbf{I})\end{aligned} \tag{21}
$$
这里我们假设 $z$ 有 $D$ 维，并用 $\mu_{\phi ,d}$  和 $\sigma_{\phi,d}$ 表示 $z$ 的均值向量和标准偏差向量的第 $d$ 维。假设我们有一个包含 $N$ 个数据点的数据集，并随机抽取了 $M$ 个数据点。对每一个$x^{(i)}$, 小批量 $\{x^i\},i=1...M$ 的变分下界估计值是 $\mathcal{L}(\theta, \phi)$ 的简单平均值。
$$
\mathcal{L}(\theta,\phi;\{x^i\}_{i=1}^M)\approx\frac{N}{M}\sum_{i=1}^M\mathcal{L}(\theta,\phi;x^{(i)}) \tag{22}
$$
其中 $\mathcal{L}(\theta,\phi;x^{(i)})$  在公式 21 中给出。为了学习 $\theta$ 和 $\phi$，我们可以求出上述表达式的导数，并将其用于随机梯度上升过程。

## 附录

#### 估计$\nabla_\phi\mathbb{E}_{q_\phi}[f(z)]$ 

估计这种期望值梯度的一个常用方法是利用特征 $∇_\phi q_\phi = q_\phi∇_\phi log q_\phi$
$$
\begin{aligned}
\nabla_\phi\mathbb{E}_{q_\phi}[f(z)]& =\nabla_\phi\int_zq_\phi(z|x)f(z)  \\
&=\int_z\nabla_\phi q_\phi(z|x)f(z) \\
&=\int_zq_\phi(z|x)\nabla_\phi\log q(z|x)f(z) \\
&=\mathbb{E}_{q_\phi}[f(z)\nabla_\phi\log q(z|x)]
\end{aligned}
$$
这种估计器在文献中有多种名称，如 REINFORCE algorithm、score function estimator 或者likelihood ratio trick。给定来自 $q_\phi(z|x)$ 的样本 $z^{(l)},l = 1...L$ 时，我们可以使用该估计器对 $\mathcal{L}(\theta, \phi)$ 相对于 $\phi$ 的梯度进行无偏蒙特卡罗估计。然而，在实际应用中，它的方差过大，无法发挥作用。



