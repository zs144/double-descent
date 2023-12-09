# Draft

## Abstract



## 1. Introduction

### 1.1 The traditional view of bias and variance tradeoff

Bias and variance tradeoff is one of the most important concepts in the classic machine learning. This rule regulates model test risks with different model complexity. To formalize these concepts, suppose we are given $n$ training samples $(\mathbf{X}_i, y_i) \in \mathbb{R}^d \times \mathbb{R}$, where $\mathbf{X}_i$ is the features for one data point and $y_i$ is the corresponding response. Our learning task is to train a model $f_n: \mathbb{R}^d \to \mathbb{R}$ such that we can accurately predict response $y$ given the features $\mathbf{X}$ of any input data points. In practice, the strategy is that we want to maximize the test accuracy while maintaining the training accuracy at a high level. The test accuracy, or relatively the test risk, indicates how well the model can generalize on unseen data, which is not something we can directly control as the training accruracy. We define $l$ as the function to quantify the loss between predictions and the ground truth. Specifically, during training, we use empirical risk minimization (ERM) or some other derivative alogrithm to find a proper model $f$ in some function class $\mathcal{H}$ such that it has the minimal empirical risk (also called training loss) $\frac{1}{n}\sum_{i=1}^nl(f(\mathbf{X}_i), y_i)$. Although the training set and testing set are generated on the same distribution $P$, the concern is that the $f$ derived by ERM may not simultaneously minimize the true risk (also called test risk) $\mathbb{E}[l(\mathbf{X}), y|(\mathbf{X}, y) \sim P]$. The conventional bias and variance tradeoff believes there are two stages when we look at the change of training risk and test risk: underfitting and overfitting. If the model is too simple (i.e., a small model class $\mathcal{H}$ capacity), all participants in $\mathcal{H}$ tend to underfit the training set, resulting in both a low training risk and a low test risk. On the other hand, if the model is overly complicated (i.e., a big model class $\mathcal{H}$ capacity), then models will overfit the training data and generlize badly on new samples, so the test risk is still high this time but the training risk is optimized. Generally, the training risk decreases as the model becomes more complex, whereas the test risk reduces first and then bounce back. Why we have a U-shaped curve for the change of test risk? This is becasue it can be decomposed into three parts: an increasing bias, a decreasing variance, and a constant irreducible error \ref{neville2008bias}. Therefore, between the two stages, there exists a balanced point where the model is neither too simple or two complicated, leading to the ideal training risk and test risk. Another widespreading rule of thumb is that: don't overfit the model to a zero training risk because it will definately harm its performance during testing.



## 2. Theoretical analysis



## 3. Experiment



## 4. Results



## 5. Conclusion and Discussion



## References