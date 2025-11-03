Algorithm Summary
Bayesian Geometric Trading System

Mathematical Foundation:
Bayesian Inference: Calculates probability of price reversion vs breakout using Bayes' theorem
Geometric Levels: Identifies support/resistance via pivot point clustering and Fibonacci retracements
Feature Engineering: Distance to levels, clustering strength, RSI momentum

Learning Process:
Continuous Bayesian Updates: Adjusts prior probabilities based on market trend
Reinforcement Learning: Records every trade outcome to optimize confidence thresholds
Feature Performance Tracking: Analyzes win rates by RSI ranges and distance categories

Core Mechanism:
Trades when posterior probability exceeds adaptive thresholds:
>75%: Expect reversion to geometric levels
<25%: Expect breakout from geometric levels

The system self-optimizes by learning which market conditions (RSI extremes, strong clustering, close proximity to levels) produce the highest win rates.
