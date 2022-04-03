# 拉普拉斯近似和对数模型证据

2021-11-21 by 张洳源

经常看到在贝叶斯建模的过程中用到拉普拉斯近似(laplacian approximation, LA)和对数模型证据(log model evidence, LME)。一直没有找到特别好的中文教程，有一些machine learning方面的介绍，与实际认知建模的联系不紧密。所以这里把一些概念理清。



## 最大后验概率估计

讲拉普拉斯近似和对数模型证据之前，首先需要明白的一个概念是最大后验概率估计(maximum a posteriori estimation，MAP)。我们先从最大似然估计(maximum likelihood estimation, MLE)开始。

假定数据$D$ 和模型$M$，以及关于模型的参数$\theta$ (注意这里的参数$\theta$ 可以是$d>1$维)。那么给定参数$\theta$，得到数据的似然函数为$p(D|\theta,M)$，那么MLE就是要找到$\hat{\theta}$, 满足$\hat{\theta}=\mathop{\arg\max}\limits_{\theta}[p(D|\theta, M)]$,也等价于最小化negative log likelihood,
$$
\hat{\theta}=\mathop{\arg\min}\limits_{\theta}[-log(p(D|\theta, M))]
$$
但是MLE是不考虑参数有先验的情况，在真实的情况下，其实我们是通过数据来推断参数，即求解参数的后验概率$p(\theta|D,M)$，根据贝叶斯公式，可以写成
$$
p(\theta|D,M)=\frac{p(D|\theta,M)*p(\theta|M)}{p(D|M)}
$$
那么求解参数就变成了后验概率$\hat{\theta}=\mathop{\arg\max}\limits_{\theta}[p(\theta|D,M)]$，因为$p(D|M)$是一个固定常数，就等价于求解$\hat{\theta}=\mathop{\arg\max}\limits_{\theta}[p(D|\theta,M)*p(\theta|M)]$。其中$p(\theta|M)$是关于参数$\theta$的先验分布，可以在具体的建模的过程中指定。那么其实等价于最小化negative log probability，即
$$
\hat{\theta}=\mathop{\arg\min}\limits_{\theta}[-log(p(D|\theta,M))-log(p(\theta|M))]
$$
和公式(1)比较，其实也就是在优化的过程中加上一个参数的先验的对数项 $-log(p(\theta|M)$。以上无论是MLE还是MAP的优化过程，目前都可以在matlab或者python里面利用相关数值方法(e.g., matlab里面的fminsearch或者fmincon)进行求解。



## 模型证据

模型证据(model evidence)，其实就是公式(2)中右边部分的分母，即$p(D|M)$。可以理解为，当我们不在乎参数$\theta$的取值，或者说参数$\theta$任意取一个值的时候，这个模型框架$M$，有多大程度上能解释数据$D$。模型证据可以有
$$
p(D|M)=\int_{\theta}p(D|\theta,M)*p(\theta|M)d\theta=\int_{\theta}p(D,\theta|M)d\theta
$$
现在的问题就是如何求解这个积分。



## 拉普拉斯近似求解模型证据

公式(4)中的积分一般情况下很难通过解析的方式求出，我们可以进行泰勒展开进行近似求解
$$
\begin{align*} 
p(D|M)&=\int_{\theta}p(D,\theta|M)d\theta\\ &=\int_{\theta}\exp \log(p(D,\theta|M)d\theta\\  &\approx \int_{\theta}\exp \Bigg\{ \log(p(D,\hat\theta|M) +\overbrace{\nabla\log(p(D,\hat\theta|M)}^{=0(\hat\theta\ is\ at\ the\ mode)}(\theta-\hat\theta)+\frac{1}{2}(\theta-\hat\theta)^T\overbrace{\nabla^2\log(p(D,\hat\theta|M)}^{=A (Hessian\ matrix)}(\theta-\hat\theta) \Bigg\}d\theta\\  &= \int_{\theta}\exp\log(p(D,\hat\theta|M) * \exp \Bigg\{ \frac{1}{2}(\theta-\hat\theta)^T\overbrace{\nabla^2\log(p(D,\hat\theta|M)}^{=-A (Hessian\ matrix)}(\theta-\hat\theta) \Bigg\}d\theta\\  
&=p(D,\hat\theta|M)*\int_{\theta}exp(\frac{1}{2}(\theta-\hat\theta)^T\overbrace{\nabla^2\log(p(D,\hat\theta|M)}^{=-A (Hessian\ matrix)}(\theta-\hat\theta) ) d\theta\\
&=p(D,\hat\theta|M)*\int_{\theta}exp(-\frac{1}{2}(\theta-\hat\theta)^T(A^{-1})^{-1}(\theta-\hat\theta) ) d\theta\\
&=p(D,\hat\theta|M)*(2\pi)^{\frac{d}{2}}*|A^{-1}|^{\frac{1}{2}}\overbrace{\int_{\theta}\frac{1}{(2\pi)^{\frac{d}{2}}*|A^{-1}|^{\frac{1}{2}}}exp(-\frac{1}{2}(\theta-\hat\theta)^T(A^{-1})^{-1}(\theta-\hat\theta) ) d\theta}^{integration\ of\ multivariate\ Gaussian\ distribution} \\
&=p(D,\hat\theta|M)*(2\pi)^{\frac{d}{2}}|A^{-1}|^{\frac{1}{2}}\\  
&=p(D,\hat\theta|M)*(2\pi)^{\frac{d}{2}}|A|^{-\frac{1}{2}}\\  
&=p(D|\hat\theta,M)p(\hat\theta|M)*(2\pi)^{\frac{d}{2}}|A|^{-\frac{1}{2}}  
\end{align*} \tag{6}
$$
以上的化简当中，从第二行到第三行，是将$log(p(D,\theta|M)$这个函数通过泰勒展开进行近似。$\hat\theta$是在公式(3)做MAP的一步已经求出来的最优参数。因为$\hat\theta$是极值，所以函数$log(p(D,\theta|M)$在$\hat\theta$这个地方的一阶导数为0，即泰勒展开的第二项$\nabla\log(p(D,\hat\theta|M)$为0，那么展开第二项就可以消掉。

最后得到的结果，$A=-\nabla^2\log(p(D,\hat\theta|M)$ 是$log(p(D,\theta|M)$这个函数在$\hat\theta$这个位置的二阶导数矩阵(i.e., Hessian矩阵)，$|A|$代表该矩阵的行列式。$d$是参数$\theta$的维度。而$p(D|\hat\theta,M)p(\hat\theta|M)$因为在前面MAP的过程中已经优化过，可以直接带入前面优化的函数值。那么求解模型证据的唯一难点就变成了求函数$log(p(D,\theta|M)$在$\hat\theta$这个位置的Hessian矩阵，在matlab里面可以用数值的方法来求解。同时，对数模型证据LME可以写为。
$$
log(p(D|M))=log(p(D|\hat\theta,M))+log(p(\hat\theta|M))+\frac{d}{2}log(2\pi)-\frac{1}{2}log(|A|)
$$

注意这里参数$\theta$的先验分布$p(\hat\theta|M)$是自己指定的，带有一定的主观性。

## 模型证据和贝叶斯因子

通常做模型比较的时候，LME越大说明该模型越好。同时，比较两个模型的时候，我们经常使用贝叶斯因子(Bayes Factor, BF)，两个模型$M_a$和$M_b$的贝叶斯因子为两个模型证据的比值
$$
BF=\frac{p(D|M_a)}{p(D|M_b)}\\
log(BF)=log(p(D|M_a))-log(p(D|M_b))
$$
在实际做模型的过程中，通常可以先求出两个模型的LME，然后再计算出BF。



## 对数模型证据和负变分自由能

在文献中经常看到一句话，LME即为负变分自由能(negative variational free energy)，那么一个模型越好，说明LME最大化，也就是负变分自由能(negative variational free energy)最大，也等价于最小化variational free energy，这个和Karl Friston的自由能理论是一致。我在[What is free energy?](./freeenergy.html) 一文中对两个关系进行了阐述。在这篇文章中，我们得到
$$
F=-log(p(D|M))+D_{KL}[q(\theta)||p(\theta|D,M)]
$$
在变分推断中，如果使用的变分分布$q(\theta)$和所需要求解的后验概率分布$p(\theta|D,M)$非常一致，那么两者的KL-divergence，即$D_{KL}[q(\theta)||p(\theta|D,M)]=0$，那么就有$F=-log(p(D|M))$。而F就是变分自由能，那么LME是负变分自由能就很好理解了。

