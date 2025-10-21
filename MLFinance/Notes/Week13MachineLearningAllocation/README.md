### Machine Learning Allocation

This chapter introduces the Hierarchical Risk Parity technique. HRP portfolios address three problems in quadratic optimization programs in general and Markowitz's Critical Line Algorithm in particular: instability, concentration, and underperformance.
HRP applies modern mathematics to build a distributed portfolio based on the information contained in the covariance matrix.
However, unlike quadratic optimization programs, HRP does not require the existence of an inverse matrix of the covariance matrix.
In fact, HRP can compute portfolios for matrices whose inverse is not available or even for singular covariance matrices, a task that is impossible with quadratic optimizers.
Monte Carlo experiments show that HRP has a lower sample variance than CLA, where the optimization objective is the lowest variance.
HRP creates less risky out-of-sample portfolios compared to traditional risk parity techniques.
Historical analysis also shows that HRP may have performed better than standard techniques. A practical application of HRP is determining the allocation between multiple machine learning strategies.

### Portfolio Construction

Asset allocation decisions must be made amid uncertainty. Markowitz (1953) proposed one of the most influential ideas in modern financial history: expressing the investment problem as a convex optimization program.
Markowitz's CLA estimates the 'efficient frontier' of a portfolio where portfolio risk maximizes expected return for a given level of risk, measured in terms of the standard deviation of return.
In practice, mean-variance optimal solutions tend to be concentrated and unstable (2009)

There are three popular approaches to reduce instability in optimal portfolios. First, some authors attempted to regularize the solution by injecting additional information about the mean and variance in the form of prior probabilities. (Black and Litterman)
Second, other authors have suggested reducing the feasibility domain of the solution by including additional constraints (Clarke et al. 2002).
Third, other authors have proposed improving the numerical stability of the inverse of the covariance matrix (Ledoit and Wolf, 2004).

In the last week, we discussed how to deal with instability caused by noise contained in the covariance matrix. It has been revealed that signals included in the covariance matrix can also cause instability and require specialized handling.
This chapter explains why certain data structures make mean-variance solutions unstable and what we can do to address this second instability.