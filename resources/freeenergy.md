# What is free energy?

2020-07-17 by 张洳源

Karl Friston的理论是个很难懂的理论，曾经有在哥伦比亚大学的各领域专家聚集在一起也看不懂他的理论。https://www.lesswrong.com/posts/wpZJvgQ4HvJE2bysy/god-help-us-let-s-try-to-understand-friston-on-free-energy

我也没看到有什么中文靠谱的解释，大部分解释就是把Karl说的英文翻译成中文而已。

看懂Friston的自由能理论，需要一些预先知识。我个人建议先完全不要去看他的东西，但是需要搞清楚以下几个概念，

变分推断，贝叶斯推断里面的model evidence，生成模型等。

其中最关键的就是变分推断，我发现身边的心理学家或者神经科学家其实很少有人能弄懂变分推断的。我大概是2015年看到这个理论，当时读friston的paper也是完全抓瞎，直到我最近理解了变分推断，才算大概明白了自由能原理。

另外，个人不建议看Karl Friston的原文。Karl Friston是个天才，但是他写作有问题，不能把自己的观点用简单的语言表达出来。我个人建议看Sam Gershman的这篇解读，我觉得写得非常好。http://gershmanlab.webfactional.com/pubs/free_energy.pdf



我们就按照维基上对free energy principle的介绍，从贝叶斯变分推断讲起，慢慢推过去。

https://en.wikipedia.org/wiki/Free_energy_principleen.wikipedia.org

对于一般的贝叶斯推断来说，我们观察到$s$ 是sensory state, $\phi$是环境中生成$s$的hidden state。这里我选用的是和上面wikipedia中相同的符号，以方便读者理解。我们大脑认识这个世界，基本任务就是根据sensory state来对hidden state做推断，也就是求解后验概率$p(\phi|s)$。这是认知科学中所有贝叶斯模型的最基本思想。
$$
\begin{align} p(\phi|s)=\frac{p(s,\phi)}{p(s)} \tag{1} \\ p(s)=\frac{p(s,\phi)}{p(\phi|s)} \tag{2} \\ log(p(s)) = log(p(\phi,s)) - log(p(\phi|s)) \tag{3} \end{align}
$$
公式(1)是贝叶斯推断的基本形式，然后很容易推到公式(3)。

那么，下面就是变分推断的内容，要求解$p(\phi|s)$, 我们引入一个分布$q(\phi)$
$$
log(p(s)) = log(\frac{p(\phi,s)}{q(\phi)}) - log(\frac{p(\phi|s)}{q(\phi)}) \tag{4}
$$
注意无论$q(\phi)$是什么分布公式(4)都成立。然后两边求对于的$q(\phi)$期望，就有 
$$
\begin{align} \int{log(p(s))}q(\phi){d\phi} = \int{log(\frac{p(\phi,s)}{q(\phi)})}q(\phi){d\phi}-\int{log(\frac{p(\phi|s)}{q(\phi)})}q(\phi){d\phi} \tag{5} \\ log(p(s)) = \int{log(\frac{p(\phi,s)}{q(\phi)})}q(\phi){d\phi}-\int{log(\frac{p(\phi|s)}{q(\phi)})}q(\phi){d\phi} \tag{6}  \end{align}
$$
由于公式(5)的左边被积分项$log(p(s))$并不包含所$\phi$所以以做完积分不变，就得到公式(6)。然后，我们让等于$F(s)$公式(6)右边第一项的负数。第二项的负数，就是变分分布和所需$q(\phi)$要求解的后验概率$p(\phi|s)$的KL divergence。
$$
\begin{align} F(s) = -\int{log(\frac{p(\phi,s)}{q(\phi)})}q(\phi){d\phi} = -E_q(log(\frac{p(\phi,s)}{q(\phi)}) \tag{7} \\  D_{KL}[q(\phi)||p(\phi|s)] = -\int{log(\frac{p(\phi|s)}{q(\phi)})}q(\phi){d\phi} \tag{8} \end{align}
$$
那么公式(6)可以重新被表示成
$$
\begin{align} log(p(s)) = -F(s)+D_{KL}[q(\phi) || p(\phi|s)]  \tag{9} \\ F(s) =-log(p(s)) +D_{KL}[q(\phi) || p(\phi|s)]  \tag{10} \end{align}
$$
好了，记得整个推断的目的是求后验概率$p(\phi|s)$，因为这个太难求了，我们用变分分布来$q(\phi)$近似它，当和后$q(\phi)$和后验概率越接近$p(\phi|s)$，两者的$D_{KL}[q(\phi) || p(\phi|s)] $就越小(注意$D_{KL}[q(\phi) || p(\phi|s)]>=0 $ )。同时， 虽然我们不知道$-log(p(s))$是多少，但是肯定是个常数，因为给定一个sensory state，外部世界产生它的概率是恒定的，只不过我们求不出来。所以当$D_{KL}[q(\phi) || p(\phi|s)] $ 越小，公式(10)右边越小，公式左边也就越小。反之也$F(s)$成立，如果我们优化使得$F(s)$越小，那么$D_{KL}[q(\phi) || p(\phi|s)] $也就越小。



**这里$F(s)$就是free energy，minimize free energy就等价于minimize $D_{KL}[q(\phi) || p(\phi|s)] $**



以上内容，和大脑其实无关，我只不过重复了一遍机器学习里面变分推断的原理，是为了说明，minimize free energy其实就是贝叶斯推断中变分推断这一种特殊形式，目的就是求后验概率，只不过换了个名字而已。



**所以说，Karl Friston说的minimize free energy，不是什么玄学。就是认知神经科学当中经常说的贝叶斯推断，也就是说我们人脑在做贝叶斯推断时候，不过是用变分推断来求解后验概率的过程，这就是minimize free energy。**



我们继续来看一下https://en.wikipedia.org/wiki/Free_energy_principle上面给出的公式
$$
F(s,u) = -log(s|m) + D_{KL}[q(\phi|u)||p(\phi|s,m)] \tag{12}
$$
这个公式是不是很像我上面写的公式(10)？？但是细心的你也发现了一点区别，在上面公式(10)里面是没有Internal model $u$ 和 Generative (world) model $m$这两项的，这又是什么意思呢？



先解释Generative (world) model $m$ 。这代表是外部物理世界运行的规律，比如太阳从东边升起，从西边落下，总有一个客观规律存在。比较让人困惑的是，这个$m$ 和hidden state $\phi$到底是什么关系？**简单来说，$m$是外部世界运行的模型， $\phi$是这个模型的参数。$m$比$\phi$高一级，联合sensory state在一起，数学关系就是:
$$
p(s|m) = \int{p(s|\phi)p(\phi|m)}{d\phi} \tag{13}
$$
换句话说，sensory state并不是无边无际的，取决于外部物理世界的运行规律。所以严格来说$-log(p(s))$应该换成$-log(p(s|m))$。记住，虽然我们可能无法求得$p(s|m)$，但是它肯定还是个定值。



再来说Internal model $u$。这里就是Karl Friston理论上最大的贡献，他认为我们大脑是用变分分布来$q(\phi)$不断猜测和逼近实际的后验分布。但是这个 也不$q(\phi) $是随便来的，它取决于我们大脑内部对外部世界的认识，比如，实际情况是太阳从东边升起，从西边落下。但是大脑是完全无从得知这一客观真理的，有可能大脑就觉得太阳应该是从东南边升起，从西北边落下。所以 应该$q(\phi)$替换成 $q(\phi|u)$。



那么，如果做以上的替换，我们可以把上面公式(7)(8)(10)替换成以下形式
$$
\begin{align} F(s,u) = -\int{log(\frac{p(\phi,s|m)}{q(\phi|u)})}q(\phi|u){d\phi} = -E_q(log(\frac{p(\phi,s|m)}{q(\phi|u)})) \tag{14} \\ D_{KL}[q(\phi|u) || p(\phi|s,m)] =-\int{log(\frac{p(\phi|s,m)}{q(\phi|u)})}q(\phi|u){d\phi} \tag{15} \\  F(s,u)  = -log(s|m) + D_{KL}[q(\phi|u) || p(\phi|s,m)] \tag{16} \end{align}
$$
这就是Karl Friston minimize free energy理论的完整表达式，其中公式(16)和Wikipedia上面给出的公式(12)一致。

在Wikipedia上面还讲了minimize free energy理论和predictive coding, optimal control, active inference等多个概念的关系。我当然没有读完所有列出的论文，也绝不敢说理解了他说的每一句话，但是有了上面这个基础，去理解其他概念可能稍微容易点。

