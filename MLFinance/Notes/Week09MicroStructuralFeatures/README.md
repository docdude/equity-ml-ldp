## Microstructural Features

### Motivation

Market Microstructure studies ‘the process and results of asset trading under explicit trading rules.’ 
Market microstructure data contains key information about the auction process, including order cancellation, double auction bidding, queuing, partial execution, aggressive traders' buy/sell direction, adjustments, and order replacement.
The main source is FIX messages, which can be purchased on exchanges. The level of information contained in FIX messages allows researchers and market participants to understand how they hide and reveal their intentions. This is the most important material for building predictive machine learning characteristics with microstructural data.

### Review of the Literature

The depth and complexity of market microstructure theory is evolving over time as a function of the amount and type of data available. The first generation model used only price information. The two basic results of these early days are the transaction classification model and the roll model.
Second-generation models emerged as trading volume datasets became available, and researchers shifted their attention to studying the impact of trading volume on price. Two examples of second generation models are Kyle and Amihud.

The third generation model came out in 1996, and is considered a groundbreaking development when Maureen O'Hara, David Easely, and others explained the Probability of Informed Trader Model as the result of sequential strategy decisions between liquidity providers and strategic investors.
Fundamentally, market makers are sellers of options that are adversely selected by information-based traders, and the bid-ask spread is explained as the premium charged to the option as compensation for adverse selection.
Easely et al. (2012) describe a method to estimate VPIN, a high-frequency estimate of PIN, from transaction volume-based sampling.

These are the main theoretical frameworks used in the microstructural literature. O'Hara and Hasbrouck (2007) provide a good overview of low-frequency microstructural models. Easely (2013) and others describe modern methods for handling high-frequency microstructural models.