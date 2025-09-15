# Understanding Muon Compared to Other Stochastic Optimizers

Gaining an intuition for what **Muon** brings compared to other stochastic optimizers, such as **Adam**, can be challenging. This section aims to provide some behavorial insights by visualizing how Muon behaves differently and what makes it a more exploratory and robust optimizer.  
The code used for this part is available on GitHub: [Understanding Muon](https://github.com/GAD-cell/explainable-muon).

---

## A Simple Illustrative Problem

Consider a simple a function:  
$$
f : [0,1]^3 \longrightarrow \mathbb{R}^3, \quad
f(x, y, z) =
\begin{pmatrix}
P_1(x, y, z) \\
P_2(x, y, z) \\
P_3(x, y, z)
\end{pmatrix}
$$  

where each $P_i(x, y, z)$ is a degree-2, 3-dimensional polynomial. Choosing three input and output dimensions allows us to easily visualize the model’s vector outputs in 3D.

We define our model as $f_\theta:[0,1]^{3}\rightarrow\mathbb{R}^{3}$, with $\theta \in \mathbb{R}^{n},n\in\{9,12\}$, and aim to solve:  
$$
\min_{\theta \in \mathbb{R}^n} \mathcal{L}(f_\theta, f),
$$
where $\mathcal{L}$ is the quadratic loss.

Concretely, the model consists of a linear layer (with or without bias i.e, n=9 or n=12) and a ReLU activation function (the ReLU nonlinearity is added to better mimic conditions of neural network training, even though the underlying target is polynomial). Our goal is to optimize this model for the problem above.

---

## On Model Capacity

Note that the network described above is **under-parameterized** and **mis-specified** for the posed problem. The best solution, which would almost surely converge to zero, is the following model:  

$$
f_\theta : [0,1]^{9} \longrightarrow \mathbb{R}^3, \quad
f_\theta(x^2, y^2, z^2, xy, xz, yz, x, y, z) =
\begin{pmatrix}
P_{1_\theta}(x, y, z) \\
P_{2_\theta}(x, y, z) \\
P_{3_\theta}(x, y, z)
\end{pmatrix},
\quad \theta \in \mathbb{R}^{24}.
$$  

Here, the model directly learns the features of the three polynomials (i.e., the monomials of degree ≤ 2).  

But the objective in this section is different: we intentionally place ourselves in **realistic conditions**. In practice, most non-convex optimization problems (e.g., deep learning, LLM training) are carried out on networks that are **sub-optimal in architecture**. They cannot perfectly represent the data distribution, either because they are **under-parameterized** (lacking sufficient capacity to represent the function) or **mis-specified**. This reflects real-world training scenarios much better than providing the exact optimal model.  

---

## Hypothesis: Why Muon Helps

**Hypothesis regarding Muon:**  
When the network is **under-parameterized**/**mis-specified** (as is almost always the case in non-convex deep learning tasks[^1]) and/or **poorly conditioned** (i.e regions of the parameter space where gradient propagation is weak or highly anisotropic[^2]), Muon can retain **exploration capacity**, while Adam and other stochastic optimizers are constrained to explore only the directions that are analytically “easy” to access.  

In other words:
- Adam and standard optimizers tend to align updates with dominant gradient directions, which in ill-conditioned networks means ignoring “rare” but important directions【Jordan et al., 2024】[^3].  
- Muon, through its orthonormalization of updates, maintains update energy across multiple directions, enabling it to explore parts of the parameter space that Adam would under-utilize.  
- This makes Muon especially effective on **poorly conditioned networks**, where conditioning problems cause slower convergence or loss of expressivity for other optimizers.  

---

## Experimental Setup

To test this idea, we compare performance under increasing levels of conditioning difficulty. Specifically, we test the following networks:
(calculer le condition number lambda max / lambda min)
**Case 1**  
$$f_{\theta_1}(X) = \mathrm{linear}_1(\mathrm{linear}_1(X))$$  

**Case 2**  
$$f_{\theta_2}(X) = \mathrm{linear}_1(\mathrm{ReLU}(\mathrm{linear}_1(X)))$$  

**Case 3**  
$$f_{\theta_3}(X) = \mathrm{ReLU}(\mathrm{linear}_1(\mathrm{ReLU}(\mathrm{linear}_1(X))))$$  

**Case 4**  
$$f_{\theta_4}(X) = \mathrm{ReLU}(\mathrm{linear}_1(\mathrm{ReLU}(\mathrm{linear}_1(X)))) \quad (\text{bias} = \text{False})$$  

These correspond to increasingly **mis-specified/ poorly conditioned** models for the problem:
- Case (1) is essentially linear and well-conditioned, but mis-specified.  
- Cases (2) and (3) introduce ReLU nonlinearities, which not only create asymmetric gradient propagation but also truncate half of the input space (since negative polynomial values are mapped to zero). This deliberate loss of information increases the anisotropy of the optimization landscape, making the problem more poorly conditioned.  
- Case (4) removes biases, further reducing flexibility and worsening conditioning.  

---

## Results

| Model         | Adam test loss    | Muon test loss | $\Delta_{loss}$ |
|---------------|-------------------------|-----------------------|-----------------|
| $f_{\theta_1}$   | 1.56 | 1.56 | 0.0  |
| $f_{\theta_2}$   | 1.19 | 1.14 | 0.05 |
| $f_{\theta_3}$   | 1.11 | 0.65 | 0.46 |
| $f_{\theta_4}$   | 2.03 | 0.89 | 1.14 |

We observe:
- On the well-conditioned linear case, Muon and Adam behave identically.  
- As conditioning worsens (cases 2–4), Muon consistently outperforms Adam, with a substantial gap in the most poorly conditioned model (case 4).  

This supports the hypothesis: Muon provides **robust exploration** under poor conditioning, while Adam is restricted to dominant gradient directions.  

Note that all experiments were run with the same random seed to ensure faire comparison between muon and adam (same starting weights, train/test polynomial samples)

---
## **Visual interpretation and exploration measure**
[^1]: Simsek et al. (2023) formalize the **under-parameterized regime** as when a “student” network with $n$ hidden neurons attempts to approximate a “teacher” network with $k>n$ neurons.  
[^2]: See Sutskever et al. (2013), “On the importance of initialization and momentum in deep learning,” on how poor conditioning slows gradient descent.  
[^3]: Jordan et al. (2024) show that gradient updates in Adam/SGD often have nearly low-rank structure, dominated by a few directions. Muon orthogonalizes these updates to amplify rare but important directions.  





Below is a visualization of how the linear layer matrix $M \in \mathbb{R}^{3 \times 3}$ evolves during training with Adam and Muon. Each color tracks represents one row of the linear layer, giving a view of how the optimizer explores directions within the 3D output space of the layer.   

<div align="center" style="border: 2px solid #333; padding: 10px; display: inline-block;">

  <figure style="margin: 20px 0;">
    <img src="/blog_media/f1.png" alt="" style="width:90%; height:auto;">
      <figcaption style="font-size:14px; font-style:italic; margin-top:5px;">
    Case 1
  </figcaption>
  </figure>

  <figure style="margin: 20px 0;">
    <img src="/blog_media/f2.png" alt="" style="width:90%; height:auto;">
  <figcaption style="font-size:14px; font-style:italic; margin-top:5px;">
    Case 2
  </figcaption>
  </figure>

  <figure style="margin: 20px 0;">
    <img src="/blog_media/f3.png" alt="" style="width:90%; height:auto;">
  <figcaption style="font-size:14px; font-style:italic; margin-top:5px;">
    Case 3
  </figcaption>
  </figure>

  <figure style="margin: 20px 0;">
    <img src="/blog_media/f4.png" alt="" style="width:90%; height:auto;">
  <figcaption style="font-size:14px; font-style:italic; margin-top:5px;">
    Case 4
  </figcaption>
  </figure>

  <figcaption style="font-size:14px; font-style:italic; margin-top:5px;">
    Figure 1 : Vectors directions across training steps, greater dispersion indicates broader exploration.
  </figcaption>
</div>

Even though it is visually apparent that **Adam** explores less in poorly conditioned cases, while **Muon** explores the space more extensively,  
a more quantitative way to measure the amount of exploration is to compute the log cumulative angular deviation of each vector at each step, defined as:

$$
\forall T\in[1,n\_steps],\space \Delta_{exploration}^T=log(\sum_{t=1}^{T}\sum_{v_j\in \text{rows of M}} arccos(\frac{\vec{v_j^t}.\vec{v_{j}^{t-1}}}{||\vec{v_j^t}||.||\vec{v_{j}^{t-1}}||}))
$$

Below is a representation of the $\Delta_{exploration}^T$ through the steps for all 4 cases (higher values means more exploration):
<p align="center">
 <img src="/blog_media/cumulative_delta_angles_all.png" alt=""  style="width: 90%; height: auto;"><br>
</p>

## Discussion of Results

The visualizations highlight a clear difference in the exploration behavior of Adam and Muon optimizers.


  1.  Trajectory Space Exploration (Figure 1)
  • With Adam (left), trajectories remain confined to narrow subspaces, showing limited angular deviation between consecutive steps. This suggests that Adam tends to favor a more constrained path during training, which can reduce the optimizer’s ability to fully explore the parameter space.
  • In contrast, Muon (right) exhibits much broader trajectories across the space, with frequent and larger angular deviations. This indicates a more exploratory dynamic that allows Muon to probe different directions of the optimization landscape.
  2.  Cumulative Angular Change (Figure 2)
  • The log cumulative difference of trajectory angles quantifies these observations. For all cases, Muon consistently accumulates larger angular changes than Adam, particularly in the early and middle stages of training.
  • This trend demonstrates that Muon maintains higher levels of exploration over time, while Adam rapidly converges to more stable, lower-variance trajectories.
  • Importantly, the gap is most pronounced for functions where the optimization landscape is less well-conditioned, reinforcing the idea that Muon is better suited to adapt to challenging geometries.
  3.  Implications
  • The higher exploration of Muon could help avoid premature convergence and improve generalization by encouraging the optimizer to sample a broader range of solutions.
  • Adam’s more conservative trajectory might be advantageous in smooth, well-conditioned settings, but it risks stagnation when the landscape is complex or highly anisotropic.


# Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```


# How to Use

## Training
Run the following command to train both models (Adam and Muon) and save results:
```bash
python train_muon_vs_adam.py --mode train
```
