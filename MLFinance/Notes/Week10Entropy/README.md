## Entropy

### Motivation

Prices convey information about the forces of supply and demand. In a perfect market, prices cannot be predicted. 
This is because each observation conveys everything that is known about the price or service. 
If the market is not perfect, prices are formed with partial information, and since certain agents have more information than others, they can take advantage of the information asymmetry. 
It will be helpful to estimate the information content of the price time series and create characteristics that serve as the basis for the Machine Learning Algorithm to learn the possible outcomes. 
For example, a machine learning algorithm may discover that momentum betting is more profitable when there is little information in the price, and that mean reversion betting is more profitable when there is enough information in the price. 
In this chapter, we will look at how to find the amount of information contained in a price time series.

### Entropy as a time scale

Let us consider the entropy as time of a single security in the financial market. This concept starts from the definition of time in physics.
In physics, time is defined as the interval between events. 

$$\Delta t = t(E_2) - t(E_1)$$

The initial state of a particle is denoted by $|\psi_i\rangle$, and the state after an event is denoted by $|\psi_f\rangle$. State changes are described by the time evolution operator $U(t, t_0)$, where $t_0$ is the initial time and $t$ is the final time.

$$|\psi_f\rangle = U(t, t_0) |\psi_i\rangle$$

The time evolution operator is given as:

$$U(t, t_0) = e^{-\frac{i}{\hbar} H (t-t_0)}$$

Here, $\hbar$ is the reduced Planck constant, $i$ is an imaginary number, and $H$ is the Hamiltonian of the system. 
If time is defined as a measurement event, a measurement operator $\hat{A}$ and its unique state $|a\rangle$ are assumed. 
If the measurement result of $\hat{A}$ appears as the value $a$, the probability of this result based on the initial state $|\psi_i\rangle$ can be expressed as follows.

$$P(a) = |\langle a | \psi_i \rangle|^2$$

If the result is $a$, the state of the system collapses into the unique state $|a\rangle$. According to the second law of thermodynamics, it has been proven that entropy continuously increases or at least does not decrease over time.
When $\Delta S$ is the change in entropy, the second law of thermodynamics can be defined as follows.

$$\Delta S \geq 0$$

This serves as one of the strongest evidences that entropy is the passage of time.

### ‘Time’ of security prices

Let’s consider the Korean stock market. When the exchange opens the simultaneous bidding market between 8:30 and 9 a.m. on a typical business day, orders in the simultaneous bidding market are executed at once with the regular market opening at 9 a.m.
The regular market runs from 9 a.m. to 3:20 p.m., the closing simultaneous quotation runs from 3:20 to 3:30 p.m., and the aftermarket runs until 4 p.m.
If you count the over-the-counter market, all transactions are completed by 6 p.m., and most transactions frequently occur 30 minutes after the regular market opens and 30 minutes before the regular market closes. 
What is noteworthy here is that although there are 24 hours in a day, securities are traded for less than 10 hours of the day. It is important to be aware that observing price changes at the same time interval ultimately has a high possibility of being distorted. 
The best minimum event unit is 1 transaction. This is because transactions are decided by the independent judgment of individual economic subjects, and even if the subject of decision making has a selection bias, the final judgment ultimately occurs independently.

If we consider the price as a single particle, the equation can be derived as follows. Let us assume that the initial state of the market is $|\psi_i\rangle$. Because the time of the financial market is different from real time, state changes create a proxy as a change in transaction.

$$|\psi_f\rangle = U(v, v_0) |\psi_i \rangle$$

Here, $|\psi_i\rangle$ is the initial market state and $|\psi_f\rangle$ is the final market state. The time evolution operator can be expressed using market volatility $V$ according to trading volume as follows:

$$U(v, v_0) = e^{-\frac{i}{\hbar} V (v-v_0)}$$

At this time, $\hbar$ is considered an adjustment constant corresponding to the reduced plank constant in the financial market. A measurement event refers to a transaction at a specific price $p$, assuming a transaction operator $\hat{P}$ and its unique state $|p\rangle$. If the transaction result appears as $p$, the probability of this result based on the initial state $|\psi_i\rangle$ is calculated as follows.

$$P(p) = |\langle p | \psi_i \rangle|^2$$

If the outcome is $p$, the market state collapses into the eigenstate at that price.

$$\text{Post-transaction state: } |p\rangle$$

Here, $\Delta v \neq \Delta t$, meaning that the price is an observation of the trading volume. In other words, observing prices over time is not a very good sampling method. Since security prices and trading volumes vary depending on whether they are small or large stocks, preferred stocks or common stocks, the amount and flow of information contained in each type of security are naturally different. Here, we can infer the relative time flow of security prices.