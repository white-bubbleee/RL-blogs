# 3 TRPO算法原理学习与实践

这篇文章是本人在学习TRPO算法的过程中，根据自己的理解所写，由于自己的数学基础较为薄弱，所以在学习的过程中比较吃力，对于TRPO算法的理论推导部分可能有错误的地方，希望能批评指正，一起进步！

## 一、TRPO算法

TRPO（Trust Region Policy Optimization, 信任区域梯度优化）算法在 2015年被提出，它在理论上能够保证策略学习的性能单调性，并在实际应用中取得了比策略梯度算法更好的效果。TRPO算法是强化学习中的一种策略梯度算法(Policy Gradient)算法，属于**on-policy**算法。

> 论文链接：[\[1502.05477\] Trust Region Policy Optimization (arxiv.org)](https://arxiv.org/abs/1502.05477)

### 1.TRPO算法与Actor-Critic算法区别

Actor-Critic算法在根据策略目标的梯度更新策略目标的参数过程中，其步长是固定的，而这种更新方式很有可能由于步长太长，策略突然显著变差，进而影响训练效果。

TRPO算法通过限制KL散度（或策略改变范围）来避免每次迭代中，策略参数过大的变化。 也就是说对比于Actor-Critic算法，TRPO算法每次优化策略时，让策略（参数）朝着优化的方向改进，同时将策略变化限制在一定的范围内，即，希望能够尽快上山，但又保证了不会掉下山。

![](E:\Document-manager-workspace\Obsidian-workspace\RL-blogs\图片\Pasted_image_20240117191131.png)



### 2.TRPO算法有关数学知识

#### 2.1 重要性采样（importance sampling）

> 有关参考文章：[重要性采样（Importance Sampling） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/41217212)

对于$f(x)$在概率分布$\pi(x)$下的期望是这样的：$E(f) = \int_{x}\pi(x)f(x)dx$

而当概率分布$\pi(x)$过于复杂，无法直接采样$f(x)$时（可以计算$f(x)$但是无法直接计算$E_{X \sim \pi(X)}(f(X)$, 比如使用神经网络拟合$f(x)$，无法直接求出期望），可以找到一个更简洁的分布$p(x)$，求$f(x)$在分布$p(x)$下的期望，使用大数定理得：$E(f) = \int_{x}p(x)f(x)dx \approx \frac{1}{N}\sum_{i=1}^{N}{f(x_i)}$

对该式子改写，由于$\pi(x)f(x) = p(x)\frac{\pi(x)}{p(x)}f(x)$，因此，$f(x)$在概率分布$\pi(x)$下的期望是：

$$
\begin{aligned} E(f) &= \int_{x}p(x)\frac{\pi(x)}{p(x)}f(x)dx \\ &\approx \frac{1}{N}\sum_{i=1}^{N}{\frac{\pi(x_i)}{p(x_i)}f(x_i)}, x_i \sim p(x), \frac{\pi(x_i)}{p(x_i)} \ is\ important \ weight \end{aligned}
$$

因此，$f(x)$关于$\pi(x)$分布下的期望，可以看做是$\frac{\pi(x)}{p(x)}f(x)$关于$p(x)$分布下的期望。

#### 2.2 Hessian矩阵

对于函数$F = f(x_1, x_2, ..., x_n)$, 其Hessian矩阵是**函数F关于x的二阶偏导**，即

$$
H(F)(x) = \frac{\mathrm{d}^{2} f }{\mathrm{d} x^{2}} = \begin{bmatrix} \frac{\partial^{2} f }{\partial x_1^2} & \frac{\partial^{2} f }{\partial x_1\partial x_2} & ... & \frac{\partial^{2} f }{\partial x_1\partial x_n}\\ \frac{\partial^{2} f }{\partial x_2\partial x_1} & \frac{\partial^{2} f }{\partial x^2} & ... & \frac{\partial^{2} f }{\partial x_2\partial x_n}\\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^{2} f }{\partial x_n\partial x_1} & \frac{\partial^{2} f }{\partial x_n\partial x_2} & ... & \frac{\partial^{2} f }{\partial x_n^2}\\ \end{bmatrix}
$$

#### 2.3 费歇尔信息矩阵（fisher information matrix， FIM）

假设观察到数据样本$x_1, x_2, ..., x_n \sim p(X, \theta)$ , $\theta$是目标参数且未知，利用极大似然法的思想：

首先计算似然函数，$L(X, \theta)=\prod_{i=1}^{n}p(x_i, \theta)$

对似然函数取对数求偏导，得评分函数$S(X, \theta) = \sum_{i=1}^{n}\frac{\partial logf(X_i, \theta)}{\partial \theta}$，该函数可以表示对$\theta$评估的好坏。

**(1)费歇尔信息**$I(\theta)$

定义：$I(\theta) = E[S(X, \theta)^{2}]$, 显然，$E[S(X, \theta)] = 0$_**(证明见下面)**_，所以可以得到：

$$
\begin{aligned} I(\theta) &= E[S(X, \theta)^{2}] - E[S(X, \theta)]^{2} \\ &= Var[S(X, \theta)]\\ &=E_{p(x|\theta)}[(S(X)-0)(S(X)-0)^T] \end{aligned}
$$

当假设$\theta$是向量时，费歇尔信息是一个矩阵——费歇尔信息矩阵$F$。

> 证明$E[S(X, \theta)] = 0$
> $$
> \begin{aligned}E[S(X, \theta)] &= E_{p(x|\theta)}[{\nabla log p(x|\theta)}]\\&=\int{\nabla logp(x|\theta) p(x|\theta) dx}\\&=\int \frac{\nabla p(x|\theta)}{p(x|\theta)}p(x|\theta)dx\\&=\int \nabla p(x|\theta)dx\\&=\nabla \int p(x|\theta)dx\\&=\nabla 1\\&=0 \end{aligned}
> $$

**(2)费歇尔信息矩阵**$F$

**费歇尔信息矩阵F的定义：**

$$
\begin {aligned} F &= E_{p(x|\theta)}[\nabla logp(x|\theta)\nabla logp(x|\theta)^T ]\\ &= E_{p(x|\theta)}[\nabla logp(x|\theta)^2 ] \end{aligned}
$$

对于训练样本$X = \left \{ x_1, x_2, ..., x_N\right \}$ , 其费歇尔信息矩阵为$F =\frac{1}{N} \sum_{i=1}^{N}[p(x_i|\theta)\nabla logp(x_i|\theta)\nabla logp(x_i|\theta)^T ]$

**(3)费歇尔信息矩阵和Hessian矩阵的关系**

$F = - E_{p(x|\theta)}[H_{log(p(x|\theta))}]$，所以:

$$
\begin{aligned} I(\theta)&=E(S(X, \theta)^2)\\ &=- E_{p(x|\theta)}[H_{log(p(x|\theta))}]\\ &= \end{aligned}
$$

**结论：对于**

> 参考文章：
>
> [费舍尔信息矩阵及自然梯度法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/228099600)
>
> [(3 条消息) 费雪信息 (Fisher information) 的直观意义是什么？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/26561604/answer/33275982)

#### 2.4 KL散度（KL divergence）

KL散度也叫相对熵，用于衡量两个随机变量的分布差距$D_{KL}(p||q) = \sum_{i}^{n}{ p(x_i) [log \frac {1}{q(x_i)}- log\frac {1}{p(x_i)} ] }$。具体来说，对于随机变量p和变量q来说，两者之间的KL散度有如下定义：

$$
D_{KL}(p||q)= \begin{cases} \sum_{i=1}^{n}{p(x)log\frac{p(x)}{q(x)}}, & p \ and \ q \ are \ discrete \ variables \\ \int_{-\infty}^{\infty} p(x)log\frac{p(x)}{q(x)} dx, & p \ and \ q \ are \ continuous \ variables \end{cases}
$$

而对于两个策略$p(x|\theta)$和$p(x|\theta+d)$，如果**二者之间变化很小**时，有如下结论：

对二者的KL散度做泰勒展开时，$D_{KL}(p(x|\theta)||p(x|\theta+d))\approx\frac{1}{2}d^{T}Hd$，其中$H= H_\theta[D_{KL}(p(x|\theta)||p(x|\theta+d))]$， 也就是两种策略KL散度的Hessian矩阵。

> 有关熵的介绍可以参见这位知乎作者的文章：
>
> [互信息(Mutual Information)浅尝辄止（一）：基础概念](https://zhuanlan.zhihu.com/p/240676850)

#### 2.5 自然梯度法

在欧式空间中，衡量二点之间的距离，使用的是L2范数，但是TRPO算法对于两个策略参数θ之间的距离关系，将策略分布之间的Fisher information matrix (FIM， 费歇尔信息矩阵)看成是[统计流形](https://www.zhihu.com/search?q=%E7%BB%9F%E8%AE%A1%E6%B5%81%E5%BD%A2\&search\_source=Entity\&hybrid\_search\_source=Entity\&hybrid\_search\_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A314354512%7D)上的黎曼度量，并用这个来衡量两个分布之间的距离，之后再用流形上的最速下降方向作为搜索方向，就是自然梯度法。

> 参考文章：[(3 条消息) 如何理解 natural gradient descent? - 知乎 (zhihu.com)](https://www.zhihu.com/question/266846405)

#### 2.6 共轭梯度法

> [共轭梯度法简介 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/178461470)

### 3.TRPO算法框架

首先，策略梯度算法的优化目标$J(\theta) = E_{\pi_\theta}[\sum_{t=0}^{\infty} \gamma^t r(s_t,a_t)]$

对于TRPO算法，优化目标是
![](E:\Document-manager-workspace\Obsidian-workspace\RL-blogs\图片\Pasted_image_20240117190048.png)
![](E:\Document-manager-workspace\Obsidian-workspace\RL-blogs\图片\Pasted_image_20240117190038.png)
![](E:\Document-manager-workspace\Obsidian-workspace\RL-blogs\图片\Pasted_image_20240117190108.png)

#### 3.1 算法计算流程
* 首先该算法是一个on-policy算法
* 算法每一次更新自身的策略参数时：
  * 先从memory中取出一个由当前策略$\pi_k$生成的轨迹样本
  * 计算奖励值$R$，利用广义优势函数估计GAE，计算$\hat A$
  * 再计算策略梯度$\hat g_k$
  * 然后使用共轭梯度法计算$\hat x_k=\hat H_k^{-1} \hat g_k$，其中$\hat H_k$用于度量当前
  * 然后计算符合KL散度要求（新的策略分布和当前的策略分布之间的KL散度上限是$\delta$）的策略梯度更新，$\theta_{k+1}=\theta_k+\alpha^j \sqrt{\frac{2\delta}{\hat x_k^T \hat H_k \hat x_k}} \hat x_k$，$j$是利用线搜索法得到的

#### 3.2 官方伪代码

关于TRPO算法的实现，官方文档给了如下的伪代码。

> 参考链接：[Trust Region Policy Optimization — Spinning Up documentation (openai.com)](https://spinningup.openai.com/en/latest/algorithms/trpo.html)

![](E:\Document-manager-workspace\Obsidian-workspace\RL-blogs\图片\Pasted_image_20240117190149.png)

TRPO算法伪代码
![](https://picx.zhimg.com/80/v2-d2a96f6460c4c5e03f16352ba50700df_720w.png?source=d16d100b)

#### 3.3 总结



## 二、代码实现(pytorch, gym-CartPole环境)

首先导入各种功能包

```python
import copy
import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm # 用于绘制进度条
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 定义pytorch框架训练时使用到的设备为gpu
```

### 1.策略网络和价值网络

```python
class PolicyNet(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.model = nn.Sequential(nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1) # 离散动作，所以使用softmax输出，因为是随机性策略，所以最后返回的action结果是一个概率分布，所以使用softmax
        )
    def forward(self, x):
        return self.model(x)
class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(QValueNet, self).__init__()
            self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出Q值
        )
    def forward(self, x):
        return self.model(x)
```

### 2.TRPO算法

算法在编程过程中使用到的技巧：

#### 2.1 计算广义优势GAE函数

```python
# 计算优势函数：使用广义优势估计
def compute_advantage(gamma, lmbda, td_delta):
    """
    :param gamma: 折扣因子
    :param lmbda: 加权因子
    :param td_delta: 时序差分误差
    :return:
    """
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
```

#### 2.2 TRPO类实现

```python
class TRPO(object):

    def __init__(self, hidden_dim, state_space, action_space, lmbda, kl_constraint, alpha, critic_lr, gamma):
        state_dim = state_space.shape[0]
        action_dim = action_space.n
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # 优势估计的加权分数
        self.kl_constraint = kl_constraint  # KL散度的阈值
        self.alpha = alpha  # 线性搜索法使用的超参数

    def take_action(self, state):  # 使用随机性策略
        state = torch.tensor([state], dtype=torch.float).to(device)
        probs = self.actor(state)
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        """
        计算hessian矩阵和一个向量v的乘积（Hv）：公式见《动手学强化学习》第11章TRPO算法的11.4节最后一个公式
        这里要计算的Hessian矩阵：策略之间平均KL距离的的Hessian矩阵
        :param states: 新的策略下所历经的各种状态
        :param old_action_dists:
        :param vector: 列向量v
        :return:
        """
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        # 计算新策略和旧策略的KL散度的平均值
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        # 计算KL散度关于旧策略的参数θ的梯度值，并把结果加入到计算图中
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        # 把KL散度的梯度值转置
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # 计算KL散度的梯度值转置*v
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        # 计算KL散度的梯度值转置*v 关于旧策略参数θ的梯度值
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        # 把输出结果转换成列向量？
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        # print(grad2_vector.shape) # shape (898)
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        """
        使用共轭梯度法计算出x=H-1g的x值
        :grad: 
        :states: 当前状态值
        :old_action_dists: 上一次策略下的动作列表 
        :return: 
        """
        x = torch.zeros_like(grad)  # 初始化x
        r = grad.clone()  # 初始化梯度值
        p = grad.clone()  # 初始化梯度更新方向 
        rdotr = torch.dot(r, r)  # rT*r的结果变量
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)  # 计算H*p
            alpha = rdotr / torch.dot(p, Hp)  # 计算步长
            x += alpha * p  # 更新迭代点
            r -= alpha * Hp  # 更新梯度
            new_rdotr = torch.dot(r, r)  # r'T*r'的结果变量, r'是上一次的r
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr  # 计算组合系数
            p = r + beta * p  # 计算共轭方向
            rdotr = new_rdotr  # 更新

        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        """
        计算新的策略目标
        :param states: 新的策略下的一系列状态
        :param actions: 新的策略下采取的一系列动作
        :param advantage: 旧的优势函数
        :param old_log_probs: 旧策略对新状态系列所计算出的动作log概率
        :param actor: 
        :return:
        """
        log_probs = torch.log(actor(states).gather(1, actions))  # 新的策略下计算出的动作log概率
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_pobs, old_action_dists, max_vec):
        """
        线性搜索法的实现
        :states: 
        :actions: 
        :advantage: 优势函数
        :old_log_pobs: 
        :old_action_dists: 
        :max_vec: 最大步长（见书上P112页的上方的公式），是一个固定值，提前计算出来存储好就可以
        :return: 合适的线性搜索结果的策略网络参数
        """

        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        ol_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_pobs, self.actor)

        for i in range(15):  # 线性搜索主循环
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            # 初始化一个新的策略actor
            new_actor = copy.deepcopy(self.actor)
            # 把线性搜索后得到的新参数更新到新策略actor网络里面
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            # 新策略所得到的动作action输出
            new_actions_dists = torch.distributions.Categorical(new_actor(states))
            # 计算当前策略θ和旧策略的θ之间的KL散度
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_actions_dists))
            # 计算
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_pobs, new_actor)
            # 根据参数是否满足要求，搜索并返回更新后的网络参数值
            if new_obj > ol_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        """
        # 更新策略actor网络
        :param states:
        :param actions:
        :param old_action_dists:
        :param old_log_probs:
        :param advantage:
        :return:
        """
        # 计算当前的策略目标
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        # 计算当前策略目标actor的梯度值
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        # 把策略目标的梯度值修改为向量形式，同时和上一次计算图断开
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
        # 计算H*x
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        # 计算出书上P112上方公式中的最大步长(平方根(2*δ/(xT*H*x)))
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))  # 这里为什么要加上1e-8??????
        # 利用线性搜索法计算出新的策略网络的参数
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists,
                                    descent_direction * max_coef)  # 线性搜索
        # 把计算出的网络参数更新至actor网络中
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())  # 用线性搜索后的参数更新策略

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(device)
        # 计算当前的目标Q值
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # 计算target_Q和cur_Q之间的差分值，即时序差分误差
        td_delta = td_target - self.critic(states)
        # 计算优势函数
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(device)
        # 计算旧的策略网络的log策略列表
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        # 计算出旧的actions分布
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())
        # 计算critic的loss函数: 
        critic_loss = torch.mean(torch.nn.functional.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数

        # 更新actor网络参数
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)
```

### 3.调用TRPO算法训练

```python
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

if __name__ == "__main__":
    num_episodes = 500  # 迭代次数
    hidden_dim = 128  # actor和critic网络超参数
    gamma = 0.98  # 折扣因子
    lmbda = 0.95  # 优势函数估计时用到的加权因子
    critic_lr = 1e-2  # critic的学习率
    kl_constraint = 0.0005  # KL散度阈值
    alpha = 0.5  # 线性搜索超参数

    env_name = 'CartPole-v0'  # 仿真环境
    env = gym.make(env_name)
    torch.manual_seed(0)
    agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda, kl_constraint, alpha, critic_lr, gamma)
    return_list = train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()
```

### 4.训练结果
![](https://picx.zhimg.com/80/v2-c455222829d07061f8763a5e2621f08e_720w.png?source=d16d100b)

![](https://picx.zhimg.com/80/v2-a9f870e9f20db482aaf4744e9dfc30fb_720w.png?source=d16d100b)

![](https://picx.zhimg.com/80/v2-8eb4d5b11ab9318cbe4a78fb3e7ee06e_720w.png?source=d16d100b)



## 三、连续动作环境的代码（Pendulum环境）



## 四、使用Stable Baselines3实现TRPO算法 gym-CartPole环境

### 1.Stable baselines3

Stable baselines3是个深度强化学习的工具包，stable\_baselines3能够快速完成强化学习算法的搭建训练和评估，包括保存，录视频等等你需要在进行深度强化实验方面要用到的函数和方法。

> stable baselines3的官方文档：[stable baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) 还有stable baselines3 contrib的官方文档：[ Stable Baselines3 Contrib](https://sb3-contrib.readthedocs.io/en/master/) stable baseline3中有个rl-zoo，里面有许多环境的最优参数[RL Baselines3 Zoo](https://stable-baselines3.readthedocs.io/en/master/guide/rl\_zoo.html) 关于stable baselines3的安装我放在文章的后面了，有需要的话可以参考。

### 2.代码实现

```python
"""首先定义环境"""
import gym
ENV_NAME = "CartPole-v0"
env = gym.make(ENV_NAME)
# from stable_baselines3.common.env_checker import check_env
# check_env(env) # 检查定义环境是否符合SB3的要求,要是自定义的环境可以检查一下，gym的环境一般来说没必要检查

"""创建TRPO智能体"""
from sb3_contrib import TRPO
model = TRPO(
    "MlpPolicy",
    env,
    tensorboard_log='./logs', # tensorboard保存数据的文件夹设置
    # 网络结构和学习率都暂时使用TRPO默认值
    verbose=0,    # 训练时不打印信息
    # device='cuda'  # 这个参数也可以不设置，sb3会在训练的时候自动选择gpu（如果有的话）
)

"""训练TRPO智能体"""
STEPS = 100
N_STEPS = 5
iter = 0
for _ in range(N_STEPS) True:# 这里可以自己设置循环次数
    iter += 1
    model.learn(
        total_timesteps=STEPS,
        tb_log_name="TRPO_logs",
        reset_num_timesteps=False, # 每100次训练都接着上一个100次的训练结果开始
    )
    # 每100次保存一下模型, 这个模型数据比较少，一般1分钟以内就训练完成了，所以也没必要保存中间model结果
    # model.save(f"model/TRPO_CartPole_{int(iter*STEPS)}")
    
## 如果训练中断了，比如说iter=21时代码中断了，可以使用如下操作接着训练
"""
model = TRPO.load("model/TRPO_CartPole_200000")
STEPS = 100
N_STEPS = 100
iter = 20
for _ in range(N_STEPS-iter):
    iter += 1
    model.learn(
        total_timesteps=STEPS,
        tb_log_name="TRPO_logs",
        reset_num_timesteps=False, # 每10000次训练都接着上一个10000次的训练结果开始
    )
    model.save(f"model/TRPO_CartPole_{int(iter*STEPS)}")
"""

"""评估训练后的智能体"""
from stable_baselines3.common.evaluation import evaluate_policy
# Evaluate the agent
mean_reward, std_reward = evaluate_policy(
    model,
    env, # 获取模型的训练环境
    n_eval_episodes=10,
    render=True # 渲染图像
)
print(f"mean_reward:{mean_reward}")
print(f"std_reward:{std_reward}")
```

可以使用tensorboard查看训练的过程，在该python文件同路径下，打开终端输入`tensorboard --logdir=./logs`

### 3.效果展示
![](E:\Document-manager-workspace\Obsidian-workspace\RL-blogs\图片\214560100236846.png)

## 附录

### 1.安装stable-baselines3

`pip install stable-baselines3==1.7.0 -i http://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com`

### 2.安装Stable Baselines3 contrib

`pip install sb3-contrib==1.7.0 -i http://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com`

## 参考文章

如果对你有帮助，请随手 **点个赞** 或 **点喜欢**！

\=======================================================

欢迎【**关注作者、私信作者】**。我们一起交流一起进步。

\=======================================================
