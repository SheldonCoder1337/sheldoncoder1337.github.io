---
author: jiale cai
date: 2024-05-04 12:05:25
---

Problem: Decompose the matrix $\mathrm{A}=\begin{pmatrix}5&3 \\\ 0&-4\end{pmatrix}$ using Singular Value Decomposition(SVD). Please show detail calculation steps.

**Step1:** calculate $AA^T$ and $A^TA$

$$A = \begin{pmatrix}5&3 \\\ 0&-4\end{pmatrix}, \text{then } A^T = \begin{pmatrix}5&0 \\\ 3&-4\end{pmatrix}$$

$$AA^T = \begin{pmatrix}5&3 \\\ 0&-4\end{pmatrix}\begin{pmatrix}5&0 \\\ 3&-4\end{pmatrix} = \begin{pmatrix}34&-12 \\\ -12&-16\end{pmatrix}$$

$$A^TA = \begin{pmatrix}5&0 \\\ 3&-4\end{pmatrix}\begin{pmatrix}5&3 \\\ 0&-4\end{pmatrix} = \begin{pmatrix}25&15 \\\ 15&25\end{pmatrix}$$

**Step2:** calculate $\lambda_1, \lambda_2$ and $S$

$$\begin{aligned}
|AA^{T}-\lambda E|=0
    &\Rightarrow
    \left|\left(\begin{matrix}{34}&{-12} \\\ {-12}&{16} \end{matrix}\right)-\lambda\left(\begin{matrix}{1}&{0} \\\ {0}&{1} \end{matrix}\right)\right|=0  
\end{aligned}$$

$$\Rightarrow
\left|\begin{matrix}{34-\lambda}&{-12} \\\ {-12}&{16-\lambda} \end{matrix}\right|=0\Rightarrow(34-\lambda)(16-\lambda)-12\times12=0
$$

$$\Rightarrow
\lambda^{2}-50\lambda+400=0\Rightarrow(\lambda-40)(\lambda-10)=0
$$

Eigenvalues: $\lambda_1=40, \lambda_2=10$

Singular values: $\sigma_1=\sqrt{\lambda_1}=\sqrt{40}=2\sqrt{10},\sigma_2=\sqrt{\lambda_2}=\sqrt{10}$

Diagonal matrix S:  $S=\begin{pmatrix}\sigma_1&0 \\\ 0&\sigma_1\end{pmatrix}=\begin{pmatrix}2\sqrt{10}&0 \\\ 0&\sqrt{10}\end{pmatrix}$

**Step3:** Finding $U$ $(AA^{T}-\lambda E)x=0\Rightarrow U=(u_{1},u_{2})=\begin{pmatrix}-\frac{2}{\sqrt{5}}&\frac{1}{\sqrt{5}} \\\ \frac{1}{\sqrt{5}}&\frac{2}{\sqrt{5}}\end{pmatrix}$

$$\begin{aligned}
For~\lambda_{1}&=40,
(AA^{T}-\lambda_{1}E)x_{1}=0
\Rightarrow
\left(\left(\begin{matrix}{34}&{-12} \\\ {-12}&{16} \end{matrix}\right)-40\left(\begin{matrix}{1}&{0} \\\ {0}&{1} \end{matrix}\right)\right)x_1=0
&\end{aligned}
$$

$$\Rightarrow
\left(\begin{matrix}{-6}&{-12} \\\ {-12}&{-24} \end{matrix}\right)x_1=0
\Rightarrow
x_{1}=\binom{-2a}{a}\Longrightarrow u_{1}=\frac{x_{1}}{||x_{1}||}=\begin{pmatrix}-\frac{2}{\sqrt{5}} \\\ \frac{1}{\sqrt{5}}\end{pmatrix}
$$

$$\begin{aligned}
For~\lambda_{2}&=10,
(AA^{T}-\lambda_{2}E)x_{2}=0
\Rightarrow
\left(\left(\begin{matrix}{34}&{-12} \\\ {-12}&{16} \end{matrix}\right)-10\left(\begin{matrix}{1}&{0} \\\ {0}&{1} \end{matrix}\right)\right)x_2=0
\end{aligned}
$$

$$
\Rightarrow
\left(\begin{matrix}{24}&{-12} \\\ {-12}&{6} \end{matrix}\right)x_2=0
\Rightarrow
x_{2}=\binom{a}{2a}\Longrightarrow u_{2}=\frac{x_{2}}{||x_{2}||}=\begin{pmatrix}\frac{1}{\sqrt{5}} \\\ \frac{2}{\sqrt{5}}\end{pmatrix}
$$

**Step4:** Finding $V$ $(AA^{T}-\lambda E)x=0\Rightarrow V=(v_{1},v_{2})=\begin{pmatrix}\frac{1}{\sqrt{2}}&\frac{1}{\sqrt{2}} \\\ \frac{1}{\sqrt{2}}&-\frac{1}{\sqrt{2}}\end{pmatrix}$

$$\begin{aligned}
For~\lambda_{1}&=40,
(AA^{T}-\lambda_{1}E)x_{3}=0
\Rightarrow
\left(\left(\begin{matrix}{25}&{15} \\\ {15}&{25} \end{matrix}\right)-40\left(\begin{matrix}{1}&{0} \\\ {0}&{1} \end{matrix}\right)\right)x_3=0
\end{aligned}
$$

$$
\Rightarrow
\left(\begin{matrix}{-15}&{15} \\\ {15}&{-15} \end{matrix}\right)x_3 =0
\Rightarrow
x_{3}=\binom{c}{c}\Longrightarrow v_{1}=\frac{x_{3}}{||x_{3}||}=\begin{pmatrix}\frac{1}{\sqrt{2}} \\\ \frac{1}{\sqrt{2}}\end{pmatrix}
$$

$$\begin{aligned}
For~\lambda_{2}&=10,
(AA^{T}-\lambda_{2}E)x_{4}=0
\Rightarrow
\left(\left(\begin{matrix}{25}&{15} \\\ {15}&{25}  \end{matrix}\right)-10\left(\begin{matrix}{1}&{0} \\\ {0}&{1} \end{matrix}\right)\right)x_4=0
\end{aligned}
$$

$$
\Rightarrow
\left(\begin{matrix}{15}&{15} \\\ {15}&{15} \end{matrix}\right)x_4=0
\Rightarrow
x_{4}=\binom{d}{-d}\Longrightarrow v_{2}=\frac{x_{4}}{||x_{4}||}=\begin{pmatrix}\frac{1}{\sqrt{2}} \\\ -\frac{1}{\sqrt{2}}\end{pmatrix}
$$

**Step5:** Complete SVD

$$
\left(\begin{matrix}{5}&{3} \\\ {0}&{-4} \end{matrix}\right) =
\left(\begin{matrix}{-\frac{2}{\sqrt{5}}}&{\frac{1}{\sqrt{5}}} \\\ {\frac{1}{\sqrt{5}}}&{\frac{2}{\sqrt{5}}} \end{matrix}\right)
\left(\begin{matrix}{2\sqrt{10}}&{0} \\\ {0}&{\sqrt{10}} \end{matrix}\right)
\left(\begin{matrix}{\frac{1}{\sqrt{2}}}&{\frac{1}{\sqrt{2}}} \\\ {\frac{1}{\sqrt{2}}}&{-\frac{1}{\sqrt{2}}} \end{matrix}\right)
$$
