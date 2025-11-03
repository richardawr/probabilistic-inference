# probabilistic-inference
Geometric market structure analysis and Bayesian probability theory.

This PoC trading algorithm represents a sophisticated fusion of geometric market structure analysis and Bayesian probability theory. 
It approaches by formulating trading as a probabilistic inference problem: given the current market's geometric setup and momentum, what is the probability that the price will revert to a key level versus break out from it? 

The system continuously learns from its decisions, aiming to adapt its confidence thresholds and improve performance over time.

Algorithm Summary & Mathematical Basis:

The core of the algorithm is a blend of two mathematical engines:

- The Geometric Engine (Spatial Analysis):
Basis: Identifies significant price levels Support & Resistance (S/R) through geometric and clustering methods.

Mathematics:
Swing Point Detection: Uses local maxima/minima over a lookback period to find pivot highs and lows.

Spatial Clustering: Applies a distance-based clustering algorithm (using ATR for normalization) to group nearby pivots into consolidated, stronger S/R levels. The "weight" of a level is a function of the cluster size (1.0 + (cluster_size * 0.2)).

Fibonacci Extensions: Incorporates Fibonacci retracement and extension levels derived from major swings, adding a proportional weighting scheme.

- The Bayesian Engine (Probabilistic Inference):
Basis: Calculates the posterior probability of a Reversion vs. Breakout using Bayes Theorem.

Mathematics:
Features: Translates the geometric setup into numerical features: normalized distance to the nearest level (d), clustering strength (c), RSI, and momentum (mom).

Likelihood Model: Defines P(Features | State) for both 'revert' and 'breakout' states. For example, the likelihood of reversion increases when the price is close to a strong cluster (d is small, c is high) and momentum is extreme (high RSI).

Bayes' Theorem: The core of the engine calculates the posterior probability:
P(Reversion | Features) = [ P(Features | Reversion) * P(Reversion) ] / P(Features)

The prior probability, P(Reversion), is dynamically adjusted based on a simple EMA trend filter.


The Bayesian engine consumes the geometric landscape provided by the Geometric Engine. A high posterior probability for reversion (> 0.75) near a strong support cluster generates a buy signal, while a low probability (< 0.25) suggests a breakout and triggers a trade in the direction of the break. 
This creates a system that trades based on the probabilistic interpretation of geometric market structure.
