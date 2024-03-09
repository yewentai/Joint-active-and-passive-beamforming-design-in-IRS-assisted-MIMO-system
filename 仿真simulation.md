MISO MIMO MU-MIMO

number of antenna, spacing of antenna, position

distance, pathloss

channel: LoS/NLoS, K(Richian factor)

noise variance, radio power, others

单小区、单基站、基站多天线、天线间距半波长

多用户、用户单天线、用户位于小区边缘

单反射面、反射面元素、元素间距半波长

下行链路

基站-用户 直达径被阻挡/没被阻挡

基站-用户、基站-反射面、反射面-用户 分别如何建模？存在LoS信号？

## Overview of Millimeter Wave Communications for Fifth-Generation (5G) Wireless Networks—With a Focus on Propagation Models

The CI path loss model accounts for the frequency dependency of path loss by using a CI reference distance based on Friis' law as given by
$$
P L^{\mathrm{CI}}\left(f_{c}, d_{3 \mathrm{D}}\right)[\mathrm{dB}]=\mathrm{FSPL}\left(f_{c}, 1 \mathrm{~m}\right)+10 n \log _{10}\left(d_{3 \mathrm{D}}\right)+\chi_{\sigma}^{\mathrm{CI}}
$$
where $\chi_{\sigma}^{\mathrm{CI}}$ is the shadow fading (SF) that is modeled as a zero-mean Gaussian random variable with a standard deviation in $\mathrm{dB}$

$n$ is the Path Loss Exponent (PLE)  

$d_{3 \mathrm{D}}>1 \mathrm{~m}$ is the distance between transmitter and the receiver

$\operatorname{FSPL}(f, 1 \mathrm{~m})$ is the free space path loss (FSPL) at frequency $f_{c}$ in GHz at $1 \mathrm{~m}$ and is calculated by [19], [85]
$$
\begin{aligned}
\operatorname{FSPL}\left(f_{c}, 1 \mathrm{~m}\right) &=20 \log _{10}\left(\frac{4 \pi f_{c} \times 10^{9}}{c}\right) \\
&=32.4+20 \log _{10}\left(f_{c}\right)[\mathrm{dB}]
\end{aligned}
$$
where $c$ is the speed of light, $3 \times 10^{8} \mathrm{~m} / \mathrm{s}$. 



## Deep Reinforcement Learning Based Intelligent Reflecting Surface Optimization for MISO Communication Systems

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1u88rz0hfj20gm08y0tf.jpg" alt="截屏2022-05-02 18.16.57" style="zoom:50%;" />

#### number of antenna, spacing of antenna, position

The BS employs a uniform linear array (ULA) with $M$ antenna elements, 

the IRS is deployed with $N=N_{x} \times N_{y}$ passive phase shifters, where $N_{x}$ and $N_{y}$ are the number of passive units in each row and column. 

$M=10$, $N=50\left(N_{x}=10, N_{y}=5\right)$ , The distances between two adjacent antenna elements at BS and IRS are both half of the carrier frequency. 

The distance between the BS and IRS is $51 \mathrm{~m} $. The user moves on a line in parallel to that connects the BS and IRS, and the vertical distance between these two lines is $1.5 \mathrm{~m}$. 

#### distance, pathloss

The path loss is modeled as $P L=P L_{0}-10 \xi \log _{10}\left(\frac{d}{D_{0}}\right) \mathrm{dB}$, where $P L_{0}=-30 \mathrm{~dB}, D_{0}=1 \mathrm{~m}, \xi$ is the path loss exponent, and $d$ is the BS-user horizontal distance.

The path loss exponents of the BS-IRS, BS-user, and IRS-user links are set to $\xi_{b i}=2$, $\xi_{b u}=\xi_{i u}=2.8$, respectively.  

#### channel: LoS/NLoS, K(Richian factor)

All channels are assumed to be **quasistatic frequency flat-fading** and available at both the BS and IRS. The channels of the BS-user, IRS-user, and BS-IRS links are denoted as $\mathbf{h}_{d} \in \mathbb{C}^{M \times 1}, \mathbf{h}_{r} \in \mathbb{C}^{N \times 1}$, and $\mathbf{G} \in \mathbb{C}^{N \times M}$, respectively.

BS-users Rayleigh fading
$$
\mathbf{h}_{d}=\sqrt{P L_{d}} \tilde{\mathbf{h}}_{d},
$$
where $\widetilde{\mathbf{h}}_{d} \in \mathbb{C}^{M \times 1}$ contains independent and identical (i.i.d) distributed $\mathcal{C N}(0,1)$ elements. 

BS-IRS and IRS-users Rician fading
$$
\begin{aligned}
\mathbf{G} =\sqrt{P L_{G}}\left(\sqrt{\frac{K_{1}}{K_{1}+1}} \overline{\mathbf{G}}+\sqrt{\frac{1}{K_{1}+1}} \widetilde{\mathbf{G}}\right) \\
\mathbf{h}_{r}=\sqrt{P L_{r}}\left(\sqrt{\frac{K_{2}}{K_{2}+1}} \overline{\mathbf{h}}_{r}+\sqrt{\frac{1}{K_{2}+1}} \widetilde{\mathbf{h}}_{r}\right)
\end{aligned}
$$
where $K_{1}$ and $K_{2}$ are the Rician- $K$ factors

$\widetilde{\mathbf{G}} \in \mathbb{C}^{N \times M}$ and $\widetilde{\mathbf{h}}_{r} \in \mathbb{C}^{N \times 1}$ are the random components with i.i.d and $\mathcal{C} \mathcal{N}(0,1)$ distributed elements.
$$
\begin{aligned}
\overline{\mathbf{G}} &=\left[\mathbf{a}_{N_{x}}^{H}\left(\theta_{\mathrm{AoA}, \mathrm{h}}\right) \otimes \mathbf{a}_{N_{y}}^{H}\left(\theta_{\mathrm{AoA}, \mathrm{v}}\right)\right] \mathbf{a}_{M}\left(\theta_{\mathrm{AoD}, \mathrm{b}}\right), \\
\overline{\mathbf{h}}_{r} &=\mathbf{a}_{N_{y}}^{H}\left(\theta_{\mathrm{AoD}, \mathrm{v}}\right) \otimes \widetilde{\mathbf{a}}_{N_{x}}^{H}\left(\theta_{\mathrm{AoD}, \mathrm{v}}, \theta_{\mathrm{AoD}, \mathrm{h}}\right)
\end{aligned}
$$
with
$$
\begin{gathered}
\mathbf{a}_{i}(\theta)=\left[1, e^{-j 2 \pi \frac{d}{\lambda} \sin (\theta)}, \cdots, e^{-j 2 \pi(i-1) \frac{d}{\lambda} \sin (\theta)}\right] \\
\widetilde{\mathbf{a}}_{N_{x}}\left(\theta_{\mathrm{AoD}, \mathrm{v}}, \theta_{\mathrm{AoD}, \mathrm{h}}\right)=\left[1, e^{-j 2 \pi \frac{d}{\lambda} \phi}, \cdots, e^{-j 2 \pi\left(N_{x}-1\right) \frac{d}{\lambda} \phi}\right]
\end{gathered}
$$
where $\phi=\cos \left(\theta_{\mathrm{AoD}, \mathrm{v}}\right) \sin \left(\theta_{\mathrm{AoD}, \mathrm{h}}\right), \theta_{\mathrm{AoA} / \mathrm{D}, \mathrm{h} / \mathrm{v}}$ represent the angles of arrival/departure in horizontal/vertical directions at the IRS, and $\theta_{\mathrm{AoD}, \mathrm{b}}$ is the angle of departure at the BS.

#### noise variance, radio power, others

$P_{\max }=5 \mathrm{dBm}$

$\sigma^{2}=-80 \mathrm{dBm}$

The penetration loss of $5 \mathrm{~dB}$ is assumed in both BS-user link and IRS-user link. 

> Indicates the fading of radio signals from an indoor terminal to a base station due to obstruction by a building

The antenna gain of $0 \mathrm{dBi}$ is assumed at both the BS and user, and that of the IRS is $5 \mathrm{dBi}$. 

#### target

the received signal at the user is
$$
y=\left(\mathbf{h}_{r}^{H} \mathbf{\Phi} \mathbf{G}+\mathbf{h}_{d}^{H}\right) \mathbf{b} s+n,
$$
where $\boldsymbol{\Phi}=\operatorname{diag}\left(e^{j \theta_{1}}, e^{j \theta_{2}}, \cdots e^{j \theta_{N}}\right)$ is the phase shift matrix at the $\operatorname{IRS}$

$\operatorname{diag}\left(a_{1}, \cdots, a_{N}\right)$ denotes a diagonal matrix with $a_{1}, \cdots, a_{N}$ as its diagonal entries, $\theta_{i} \in[0,2 \pi]$ represents the phase shift of the $i$-th element on the $\operatorname{IRS}, \mathbf{b} \in \mathbb{C}^{M \times 1}$ is thebeamforming vector at the BS with the constraint $\|\mathbf{b}\|^{2} \leq$ $P_{\max }, P_{\max }$ is the maximum transmit power of the BS

$s$ is the transmitted signal satisfying $\mathbb{E}\left[s^{2}\right]=1, n \sim \mathcal{C} \mathcal{N}\left(0, \sigma^{2}\right)$ is the noise. 

the received SNR can be obtained as
$$
\gamma=\left|\left(\mathbf{h}_{r}^{H} \mathbf{\Phi} \mathbf{G}+\mathbf{h}_{d}^{H}\right) \mathbf{b}\right|^{2} / \sigma^{2}
$$

#### Algorithm

Note that, for a fixed phase shift matrix $\Phi$, the optimal beamforming method that maximizes the received SNR is the maximum-ratio transmission (MRT) [8], i.e.,
$$
\mathbf{b}^{*}=\sqrt{P_{\max }} \frac{\left(\mathbf{h}_{r}^{H} \mathbf{\Phi} \mathbf{G}+\mathbf{h}_{d}^{H}\right)^{H}}{\left\|\mathbf{h}_{r}^{H} \mathbf{\Phi} \mathbf{G}+\mathbf{h}_{d}^{H}\right\|}
$$
The optimization problem for the phase shift matrix $\Phi$ to maximize $\gamma$ can be formulated as
$$
\begin{array}{rll}
(\mathrm{P} 1): & \max _{\boldsymbol{\Phi}} & \left\|\mathbf{h}_{r}^{H} \mathbf{\Phi} \mathbf{G}+\mathbf{h}_{d}^{H}\right\|^{2}, \\
\text { s.t. } & \left|\boldsymbol{\Phi}_{i, i}\right|=1, \quad \forall i=1,2, \cdots, N,
\end{array}
$$
where $\Phi_{i, i}$ is the $i$-th diagonal element of $\Phi$. Note that (P1) is a NP-hard problem owing to the non-convexity of the objective function and the unit modulus constraints. A SDR method was proposed in [5] to solve this problem. However, it is computational expensive with complexity of $O\left((N+1)^{6}\right)[6]$. In this letter, we focus on the design of the phase shift matrix $\Phi$, we propose a robust DRL based framework to deal with (P1) efficiently, which will be described in the next section.

## Achievable Rate Optimization For MIMO Systems With Reconfigurable Intelligent Surfaces

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1u899spp9j20dg09g3yx.jpg" alt="截屏2022-05-02 18.17.31" style="zoom:50%;" />

#### number of antenna, spacing of antenna, position

$N_{t}=8,N_{\text {ris }}=225, N_{r}=4, s_{t}=s_{r}=\lambda / 2=7.5 \mathrm{~cm}, s_{\text {ris }}=\lambda / 2=7.5 \mathrm{~cm}$

The RIS elements are placed in a $15 \times 15$ square formation so that the area of the RIS is slightly larger than $1 \mathrm{~m}^{2}$.

#### distance, pathloss

$D=500 \mathrm{~m}$

The path loss exponent of the direct link, whose value is influenced by the obstacle present, is denoted by $\alpha_{\mathrm{DIR}}=3$.

 $f= 2 \mathrm{GHz}$ (i.e., $\lambda=15 \mathrm{~cm}$ )

The FSPL for the direct link is given by $\beta_{\mathrm{DIR}}=(4 \pi / \lambda)^{2} d_{0}^{\alpha_{\mathrm{DIR}}}$

The FSPL for the indirect link can be computed according to [8], [26], [27, Eqn. (18.13.6)] as
$$
\beta_{\text {INDIR }}^{-1}=\frac{\lambda^{4}}{256 \pi^{2}} \frac{\left(\cos \gamma_{1}+\cos \gamma_{2}\right)^{2}}{d_{1}^{2} d_{2}^{2}},
$$
where $d_{1}=\sqrt{d_{\mathrm{ris}}^{2}+l_{t}^{2}}$ is the distance between the transmit array midpoint and the RIS center, and $d_{2}=\sqrt{\left(D-d_{\text {ris }}\right)^{2}+l_{r}^{2}}$ is the distance between the RIS center and the receive array midpoint. Also, $\gamma_{1}$ is the angle between the incident wave direction from the transmit array midpoint to the RIS center and the vector normal to the RIS, and $\gamma_{2}$ is the angle between the vector normal to the RIS and the reflected wave direction from the RIS center to the receive array midpoint. Therefore, we have $\cos \gamma_{1}=l_{t} / d_{1}$ and $\cos \gamma_{2}=l_{r} / d_{2}$, which finally gives
$$
\beta_{\text {INDIR }}^{-1}=\frac{\lambda^{4}}{256 \pi^{2}} \frac{\left(l_{t} / d_{1}+l_{r} / d_{2}\right)^{2}}{d_{1}^{2} d_{2}^{2}}
$$

#### channel: LoS/NLoS, K(Richian factor)


$$
\mathbf{H}=\mathbf{H}_{\mathrm{DIR}}+\mathbf{H}_{\mathrm{INDIR}}
$$
where $\mathbf{H}_{\text {DIR }} \in \mathbb{C}^{N_{r} \times N_{t}}$ represents the direct link between the transmitter and the receiver, and $\mathbf{H}_{\text {INDIR }} \in \mathbb{C}^{N_{r} \times N_{t}}$ represents the indirect link between the transmitter and the receiver (i.e., via the RIS). 

Adopting the Rician fading channel model, the direct link channel matrix is given by
$$
\mathbf{H}_{\mathrm{DIR}}=\frac{\sqrt{\beta_{\mathrm{DIR}}^{-1}}}{\sqrt{K+1}}\left(\sqrt{K} \mathbf{H}_{\mathrm{D}, \mathrm{LOS}}+\mathbf{H}_{\mathrm{D}, \mathrm{NLOS}}\right),
$$
where $H_{\mathrm{D}, \mathrm{LOS}}(r, t)=e^{-j 2 \pi d_{r, t} / \lambda}$ and $d_{r, t}$ is the distance between the $t$-th transmit and the $r$-th receive antenna. 

The elements of $\mathbf{H}_{\mathrm{D}, \mathrm{NLOS}}$ are independent and identically distributed (i.i.d.) according to $\mathcal{C N}(0,1)$. 



The Rician factor $K$ is chosen from the interval $[0,+\infty)$.

We assume that the far-field model is valid for signal transmission via the RIS (i.e., for the indirect link), and thus $\mathbf{H}_{\text {INDIR }}$ can be written as
$$
\mathbf{H}_{\text {INDIR }}=\sqrt{\beta_{\text {INDIR }}^{-1}} \mathbf{H}_{2} \mathbf{F}(\boldsymbol{\theta}) \mathbf{H}_{1},
$$
where $\mathbf{H}_{1} \in \mathbb{C}^{N_{\text {ris }} \times N_{t}}$ represents the channel between the transmitter and the RIS, $\mathbf{H}_{2} \in \mathbb{C}^{N_{r} \times N_{\text {ris }}}$ represents the channel between the RIS and the receiver, and $\beta_{\text {INDIR }}^{-1}$ represents the overall FSPL for the indirect link. Signal reflection from the RIS is modeled by the matrix $\mathbf{F}(\boldsymbol{\theta})=\operatorname{diag}(\boldsymbol{\theta}) \in$ $\mathbb{C}^{N_{\text {ris }} \times N_{\text {ris }}}$, where $\boldsymbol{\theta}=\left[\theta_{1}, \theta_{2}, \ldots, \theta_{N_{\text {ris }}}\right]^{T} \in \mathbb{C}^{N_{\text {ris }} \times 1}$. In this paper, similar to related works [9], [24], we assume that the signal reflection from any RIS element is ideal, i.e., without any power loss. In other words, we may write $\theta_{l}=e^{j \phi_{l}}$ for $l=1,2, \ldots, N_{\text {ris }}$, where $\phi_{l}$ is the phase shift induced by the $l$-th RIS element. Equivalently, we may write
$$
\left|\theta_{l}\right|=1, \quad l=1,2, \ldots, N_{\text {ris }} .
$$
Utilizing the Rician fading channel model, the channel between the transmitter and the RIS $\mathbf{H}_{1}$ is given by
$$
\mathbf{H}_{1}=\frac{1}{\sqrt{K+1}}\left(\sqrt{K} \mathbf{H}_{1, \operatorname{LOS}}+\mathbf{H}_{1, \mathrm{NLOS}}\right)
$$
where $H_{1, \operatorname{LOS}}(l, t)=e^{-j 2 \pi d_{l, t} / \lambda}$ and $d_{l, t}$ is the distance between the $t$-th transmit antenna and the $l$-th RIS element. The elements of $\mathbf{H}_{1, \operatorname{NLO}}$ are i.i.d. according to $\mathcal{C} \mathcal{N}(0,1)$. It is worth noting that the channel matrix expression (6) does not contain any FSPL term.
In a similar way, $\mathbf{H}_{2}$ can be expressed as
$$
\mathbf{H}_{2}=\sqrt{\frac{1}{K+1}}\left(\sqrt{K} \mathbf{H}_{2, \mathrm{LOS}}+\mathbf{H}_{2, \mathrm{NLOS}}\right)
$$
where $\mathbf{H}_{2, \operatorname{LOS}}(r, l)=e^{-j 2 \pi d_{r, l} / \lambda}$ and $d_{r, l}$ is the distance between the $l$-th RIS element and the $r$-th receive antenna. 

$K=1$

#### noise variance, radio power, others

$P_{t}=0 \mathrm{~dB}, N_{0}=-120 \mathrm{~dB}$



#### Algorithm

The line search procedure for the proposed gradient algorithms utilizes the parameters $L_{0}=10^{4}, \delta=10^{-5}$ and $\rho=1 / 2$. 

Also, the minimum allowed step size value is the largest step size value lower than $10^{-4}$. 

Unless otherwise specified, we assume the initial values $\boldsymbol{\theta}=\left[\begin{array}{llll}1 & 1 & \cdots & 1\end{array}\right]^{T}$ and $\mathbf{Q}=\left(P_{t} / N_{t}\right) \mathbf{I}$ for all optimization algorithms. 

To maintain compatibility with [24], we set the number of random initializations for the $\mathrm{AO}$ to $L_{\mathrm{AO}}=100$. All of the achievable rate results, except those for very large $N_{\text {ris }}$ in Figs. 8 and 9 , are averaged over 200 independent channel realizations.

## Weighted Sum-Rate Maximization for Reconfigurable Intelligent Surface Aided Wireless Networks

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h1u888zo2wj20l00mc0uv.jpg" alt="截屏2022-05-02 18.16.31" style="zoom:50%;" />

#### number of antenna, spacing of antenna, position

Urban Micro (UMi)

one AP equipped with 4 antennas

4 single-antenna users $(K=4)$ uniformly and randomly distributed in a circle centered at $(200 \mathrm{~m}, 30 \mathrm{~m})$ with radius $10 \mathrm{~m}$. 

#### distance, pathloss

the pathloss is set according to the 3GPP propagation environment [52, Table B.1.2.1-1]

#### channel: LoS/NLoS, K(Richian factor)

the LOS component is contained by the channel between AP and RIS, and channel between RIS and each user. 

We assume the direct link channel $\mathbf{h}_{\mathrm{d}, k}$ follows Rayleigh fading, while the RIS-aided channels follow Rician fading. Same as [25] and [39], we further assume that the antenna elements form a half-wavelength uniform linear array configuration at the $\mathrm{AP}$ and the RIS, and thus the channels $\mathbf{G}$ and $\mathbf{h}_{\mathrm{r}, k}$ are modeled by
$$
\begin{aligned}
\mathbf{G} &=L_{1}\left(\sqrt{\frac{\varepsilon}{\varepsilon+1}} \mathbf{a}_{N}(\vartheta) \mathbf{a}_{M}(\psi)^{\mathrm{H}}+\sqrt{\frac{1}{\varepsilon+1}} \overline{\mathbf{G}}\right) \\
\mathbf{h}_{\mathrm{r}, k} &=L_{2, k}\left(\sqrt{\frac{\varepsilon}{\varepsilon+1}} \mathbf{a}_{N}\left(\varsigma_{k}\right)+\sqrt{\frac{1}{\varepsilon+1}} \overline{\mathbf{h}}_{\mathrm{r}, k}\right)
\end{aligned}
$$
where $L_{1}$ and $L_{2, k}$ denote the corresponding path-losses, $\varepsilon$ is the Rician factor and we set $\varepsilon=10$, a is the steering vector, $\vartheta$, $\psi$ and $\varsigma_{k}$ are the angular parameters, and $\overline{\mathbf{G}}$ and $\overline{\mathbf{h}}_{\mathrm{r}, k}$ denote the NLOS components whose elements are chosen from $\mathcal{C N}(0,1)$.

#### noise variance, radio power, others


Based on above assumption, only the small-scale fading variables $\mathbf{h}_{\mathrm{d}, k}, \overline{\mathbf{G}}$, and $\overline{\mathbf{h}}_{\mathrm{r}, k}$ need to be estimated in every frame. Denote $x$ as one element in above variables, and $\hat{x}$ is the corresponding estimate value. We assume that the estimate error $x-\hat{x}$ follows zero mean complex Gaussian distribution, and all these elements have the same normalized MSE:
$$
\varrho=\frac{\mathbb{E}\left[|x-\hat{x}|^{2}\right]}{\mathbb{E}\left[|\hat{x}|^{2}\right]} .
$$
To better understand the channel conditions of the direct link and the RIS-aided link, we provide a simple example here. Consider a reference point at $(200 \mathrm{~m}, 30 \mathrm{~m})$. According to Table I, the direct-link path-loss is about $117.23 \mathrm{~dB}$, meanwhile, the path-loss of channel $\mathbf{G}$ and channel $\mathbf{h}_{\mathrm{r}}$ are $86.22 \mathrm{~dB}$ and $68.10 \mathrm{~dB}$, respectively, so the path-loss of the RIS-aided link $(N=1)$ is $154.32 \mathrm{~dB}$, which is much larger than that of the direct link (about $37 \mathrm{~dB}$ ). Therefore, the direct link cannot be ignored, and extremely large $N$ is required to achieve performance gain, if the surface phase vector $\boldsymbol{\theta}$ is not properly designed. In the next, we will show that, by utilizing the proposed joint optimization algorithms, significant performance gain can be achieved.

We evaluate the performance of the proposed algorithms with the following 3 baselines:

- Baseline 1 (Without RIS): Let $N=0$, and then $\mathcal{P}(\mathrm{A})$ is solved by the WMMSE.
- Baseline 2 (Random Phase): $\boldsymbol{\theta}$ is initialized by random value, and then $\mathbf{W}$ is optimized by WMMSE.
- Baseline 3 (Upper Bound): The KKT conditions are necessary conditions for a solution to be optimal. Thus one may run Algorithm 2 sufficient times (e.g., 100 times) with random initializations, and then the maximum output might approximate the optimal solution well.

