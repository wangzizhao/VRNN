

$$
\mathcal{L}_{VAE} = -\text{KL}(q_\phi(z|x)||p_\theta(z))+\mathop{\mathbb{E}}_{q_\phi(z|x)}[\log p_\theta(x|z)]\\
\mathcal{L}_{VRNN} = \mathbb{E}_{q(z_{1:T}|x_{1:T})}[\sum_{t=1}^T(-\text{KL}(q_\phi(z_t|z_{1:t-1}, x_{1:t})||p(z_t|z_{1:t-1}, x_{1:t-1}))+\log p_\theta(x_t|z_{1:t}, x_{1:t-1}))]\\
\mathcal{L}_{IWAE} = \mathbb{E}_{z_1, \dots, z_k \sim q_\phi(z|x)}[\log \frac{1}{k} \sum_{i=1}^k \frac{p_\theta(x|z_i)p(z_i)}{q_\phi(z_i|x)}]\\
\log p(x) \ge \mathcal{L}_{IWAE,k+1} \ge \mathcal{L}_{IWAE,k}\\
\lim_{k \rightarrow \infty} \mathcal{L}_{IWAE,k} = \log p(x)
$$

