---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
#### Time Trends


<a id='index-6'></a>
The model $ y_t = a t + b $ is known as a *linear time trend*.

We can represent this model in the linear state space form by taking


<a id='equation-lss-ltt'></a>
$$
A
= \begin{bmatrix}
    1 & 1  \\
    0 & 1
  \end{bmatrix}
\qquad
C
= \begin{bmatrix}
        0 \\
        0
  \end{bmatrix}
\qquad
G
= \begin{bmatrix}
        a & b
  \end{bmatrix} \tag{4}
$$

and starting at initial condition $ x_0 = \begin{bmatrix} 0 & 1\end{bmatrix}' $.

In fact, it’s possible to use the state-space system to represent polynomial trends of any order.

For instance, we can represent the model $ y_t = a t^2 + bt + c $ in the linear state space form by taking

$$
A
= \begin{bmatrix}
    1 & 1 & 0 \\
    0 & 1 & 1 \\
    0 & 0 & 1
  \end{bmatrix}
\qquad
C
= \begin{bmatrix}
        0 \\
        0 \\
        0
  \end{bmatrix}
\qquad
G
= \begin{bmatrix}
        2a & a + b & c
  \end{bmatrix}
$$

and starting at initial condition $ x_0 = \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}' $.

It follows that

$$
A^t =
\begin{bmatrix}
 1 & t & t(t-1)/2 \\
 0 & 1 & t \\
 0 & 0 & 1
\end{bmatrix}
$$

Then $ x_t^\prime = \begin{bmatrix} t(t-1)/2 &t & 1 \end{bmatrix} $. You can now confirm that $ y_t = G x_t $ has the correct form.
<!-- #endregion -->
