# Notes for Double Descent Paper

## Notes on Papers

### [Belkin et al., 2019]

The focus the machine learning is on the problem of prediction: given some training examples $(\mathbf{x_1}, y_1),... , (\mathbf{x_n}, y_n)$ where $\mathbf{x_i}$ is the $d$-dimention vector of features and $y_i$ is a value (for regression) or a label (for classification), we want to learn a model $h_n: \mathbb{R}^d \to \mathbb{R}$ such that it can predict the output value or label for any input $\mathbf{x}$ as accurate as possible.

The model $h_n$ is commonly chosen from some function class $\mathcal{H}$. For example, $\mathcal{H}$ can be a class of neural networks with a certain archietecture derived by empirical risk minimization (ERM). In ERM, the model is taken to be a function $h \in \mathcal{H}$ that minimizes the empirical (or training) risk $\frac{1}{n}\sum_n^{i=1} L(h(\mathbf{x}_i), y_i)$, where $L$ is the loss function.

A good model should have a good generalizability, meaning that it can still perform accurately on new data, unseen in training. To study the performance on new data, we typically assume training examples $(\mathbf{x}_i, y_i)$ are sampled randomly from a probability distribution $P$ over $\mathbb{R}^d \times \mathbb{R}$ and evaluate the model $h_n$ on some new test examples drawn independently from $P$.

The challenge stems from the mis-match between the goals of minimizing the empirical risk and minimizing the true (or test) risk $\mathbb{E}_{(\mathbf{x}, y)\sim P}[L(h(\mathbf{x}, y))]$

Traditional wisdom in machine learning suggests controlling the **capacity** of function class $\mathcal{H}$ (also called model complexity) based on the bias-variance trade-off. A more complicated model, such as a deeper NN, corresponds to a large $\mathcal{H}$. In this case, the empirical risk minimizer may overfit spurious pattern in the trianing data, resulting in a poor accuracy on new data. On the contrary, if $\mathcal{H}$ is small, all models in $\mathcal{H}$ may underfit the training data and hence will not predict well on new data either. Therefore, the classical thinking is concerned with finding the "sweet spot" between underfitting and overfitting. The control of the function class capacity may be explicit, via the choice of $\mathcal{H}$ (e.g., picking a simpler, smaller nerual network architecture), OR it may be implicit, using all sorts of regularization methods. When the suitable balance is achieved, the performance of $h_n$ on the training data is said to generatlize to the population $P$.



## Presentation structure

- Quick review on the classic bais-variance trade-off
  - some concepts
  - definition and example
- Introduction to double descent
  - some concepts
  - Recent research on this topic
- A simple experiment to see the double descent
  - setup
  - Result
- Why does this matter?
- Plan for the next
- References

