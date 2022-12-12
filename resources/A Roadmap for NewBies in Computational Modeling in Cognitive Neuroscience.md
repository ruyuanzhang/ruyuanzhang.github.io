# A Roadmap for NewBies in Computational Modeling in Cognitive Neuroscience

我经常受到很多学生的来信，问我如何自己自学或者想办法入门认知或者计算神经科学。有这样的问题的学生，大多是非心理学或者神经科学专业的学生，希望转行参与到这个领域。也有一些心理学或者神经科学专业的学生希望进一步提高自己在计算建模方面的能力，我一般的回答就是找一个做计算认知或者计算神经科学的老师开展一些研究，这是最快最直接的办法。但是很多人反应国内这方面的研究者太少，很难找到相关的人。

如果你实在找不动身边志同道合的人，有两个办法，请联系我(ruyuanzhangATgmail.com)，我可以把你拉入一些微信群来学习相关知识。其次，你可以按照以下路径进行自我探索。



1. 对于国内非心理学专业的同学来说，可能需要补充一些认知神经科学的基础知识，我推荐下面这本教材

   * [认知神经科学](https://book.douban.com/subject/5937126/)

   这本书可以当做小说来看，全面的了解有关脑认知的神经基础。如果是心理学或者神经科学专业的学生了解认知心理学和实验心理学，可以跳过这一步。

2. 数学课程请上以下一些基本课程。一般工程科学的数学基础都可以满足，但是有些心理学专业的学生可能这方面并不充足，但是起码保证上以下几门课
   * 线性代数
   * 微积分

3. 关于编程基础。在neuroscience里面编程基本上只需要两个工具Matlab和Python，我推荐两个课程
   * [B站中南大学Matlab课程](https://www.bilibili.com/video/av51366148?from=search&seid=3395198250382243694)
   * [B站小甲鱼0基础python教程](https://www.bilibili.com/video/av4050443/) (不需要学爬虫，GUI和Pygame等部分)
   * [B站Python numpy基础](https://www.bilibili.com/video/BV1U7411x76j)

   同时，在写代码中需要经常小组协作，所以git的用法也很关键，我推荐

   * [廖雪峰git教程](https://www.liaoxuefeng.com/wiki/896043488029600)

4. (可选)在认知实验，熟悉编写一个实验任务非常重要，目前主流的编写程序的软件有基于Matlab的Psychtoolbox, 基于Python的Psychopy和基于Java的JsPsych等，一般初学者推荐上来掌握Psychtoolbox
   * [北师大蒋挺老师的Psychtoolbox教程](https://zhuanlan.zhihu.com/p/45073723)
   * [B站Psychopy的教程](https://space.bilibili.com/357829140/channel/detail?cid=159082)

5. 完成了以上内容之后，可以开始上一些简单的机器学习入门课程，比如UCLA吴英年老师的Coursera的课程(Coursera上面的machine learning课程太简单。。)

   * [Yingnian Wu's Machine Learning Course](http://www.stat.ucla.edu/~ywu/teaching.html)

6. 如果以上的内容都完成了，可以开始进入一些计算神经科学的具体课程，首先推荐NeuroMatch Academy(NMA)的Computational Neuroscience的Summer Course

   * [NMA course](https://space.bilibili.com/534358980/channel/detail?cid=138741) 
   * [NMA github](https://github.com/NeuromatchAcademy/course-content)
7. 其次课程可以考虑Coursera上面的Computational Neuroscience
   * [Coursera Computational Neuroscience](https://www.coursera.org/learn/computational-neuroscience)
   
8. 有以上的内容做铺垫，我们可以进一步补充一些machine learning或者计算神经的更高阶课程，我推荐以下两本经典教科书，这两本书的内容并不需要完全掌握，很多时候是类似字典做查询的功能

   * [Reinforcement Learning by Sutton and Barto](http://incompleteideas.net/book/RLbook2020trimmed.pdf)，这本书有[中文网页版](https://rl.qiwihui.com/zh_CN/latest/) 
   * [Patten Recognition and Machine Learning中文版]()

   以及计算神经科学的经典教材。但是我不认为这本书是必读的，里面的很多内容其实非常艰深

   * Theoretical Neuroscience (可选)

9. 基于最近深度学习的火热，我觉得掌握深度学习的基本知识很有必要，我推荐基于Pytorch的教程
   
   * [动手学深度学习Pytorch版](https://tangshusen.me/Dive-into-DL-PyTorch/#/)
   
10. 如果你能走到这一步，我觉得往下系统的课程或者教科书已经不太重要的了，更多的是积累一些具体的topic上的经验。我列出一些比较有意思的话题，以及在认知神经科学上面的应用

    * Maximum Likelihood Estimation (MLE) and Maximum Posterior Estimation (MAP)
    * Markov Chain Monte-Carlo Simulation (MCMC)
    * Hierarchical Bayesian analysis 
    * Hidden Markov Model
    * Kalman Filter
    * Non-parametric Bayesian analysis
    * Variational Bayesian Methods
    * ...

    以及一些具体的认知计算模型的范例

    * Rescorlar-Wagner model in reinforcement learning
    * Drift-diffusion model
    * Visual working memory models
    * Cue-combination model
    * ...

在计算神经科学中，有一个大类比如Spiking Neural Networks, 因为我自己不是专家，就不做评价，有兴趣的人可以参考比如Xiao-Jing Wang老师的工作。



如果你掌握了以上所有的知识(大概率是不可能，我自己也几乎做不到)，还想进一步探索的话，欢迎联系我参与一些具体的实验研究和数据分析！