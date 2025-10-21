### Motivation

Backtest uses past observations to evaluate the out-of-sample performance of an investment strategy. 
These past observations can be used in two ways.

1. In a narrow sense, investment strategies can be simulated with historical data as if they were implemented in the past.
2. In a broad sense, it is possible to simulate scenarios that have not occurred in the past.

The first approach is known as Walk Forward, and it is so common that the term 'Backtest' has become virtually synonymous with historical simulation.
The second approach is less well known, and Chapter 12 introduces a new way to do it.
Each approach has pros and cons and should be used with caution.