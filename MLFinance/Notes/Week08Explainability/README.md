## Explainability

### Motivation

The most common mistake in financial research is selecting certain data, running a machine learning algorithm on it, backtesting the predictions, and repeating this process until the backtest results are good.
Academic journals are full of these bogus findings, and even large hedge funds continue to fall into this trap. Even if the backtest is a walk forward out-of-sample test, problems can still occur.
Continuously repeating tests on the same data is highly likely to result in incorrect findings.

These methodological errors are so notorious among statisticians that they are sometimes considered scientific fraud, and caution against them is included in the American Statistical Association's Code of Ethics. It usually takes about 20 iterations to discover an investment strategy that meets the standard significance level of 5%.
In this chapter, we'll find out why this approach is a waste of time and money, and how feature importance offers an alternative.