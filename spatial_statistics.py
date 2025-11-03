import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import time
from datetime import datetime, timedelta
import logging
import json
import os
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LearningDataManager:
    def __init__(self):
        self.learning_file = 'learning_data.json'
        self.model_performance_file = 'model_performance.json'
        self.learning_data = {
            'features': [],
            'predictions': [],
            'model_metrics': [],
            'last_updated': None
        }
        self.load_learning_data()

    def load_learning_data(self):
        """Load historical learning data"""
        try:
            if os.path.exists(self.learning_file):
                with open(self.learning_file, 'r') as f:
                    self.learning_data = json.load(f)
                logger.info(f"Loaded {len(self.learning_data['features'])} learning records")

            # Load model performance data
            if os.path.exists(self.model_performance_file):
                with open(self.model_performance_file, 'r') as f:
                    self.model_performance = json.load(f)
                logger.info(f"Loaded model performance data")
            else:
                self.model_performance = {
                    'win_rates_by_feature_ranges': {},
                    'optimal_thresholds': {},
                    'feature_importance': {},
                    'last_updated': None
                }

        except Exception as e:
            logger.error(f"Error loading learning data: {e}")
            self.learning_data = {'features': [], 'predictions': [], 'model_metrics': [], 'last_updated': None}
            self.model_performance = {
                'win_rates_by_feature_ranges': {},
                'optimal_thresholds': {},
                'feature_importance': {},
                'last_updated': None
            }

    def save_learning_data(self):
        """Save learning data to file - FIXED JSON SERIALIZATION"""
        try:
            self.learning_data['last_updated'] = datetime.now().isoformat()

            # Convert all data to JSON-serializable types
            serializable_data = self._make_serializable(self.learning_data)

            with open(self.learning_file, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)

            # Also save to a backup with timestamp
            backup_file = f"learning_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving learning data: {e}")

    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable types"""
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (int, np.integer)):
            return int(obj)
        elif isinstance(obj, (float, np.floating)):
            return float(obj)
        elif isinstance(obj, (str, np.str_)):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): self._make_serializable(value) for key, value in obj.items()}
        elif obj is None:
            return None
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return str(obj)  # Fallback: convert to string

    def save_model_performance(self):
        """Save model performance data"""
        try:
            self.model_performance['last_updated'] = datetime.now().isoformat()

            # Convert to serializable
            serializable_perf = self._make_serializable(self.model_performance)

            with open(self.model_performance_file, 'w') as f:
                json.dump(serializable_perf, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving model performance: {e}")

    def record_learning_point(self, features, posterior, signal_generated, signal_type, current_price, nearest_level):
        """Record a learning data point - FIXED DATA TYPES"""
        try:
            # Ensure all feature values are basic Python types
            serializable_features = {}
            for key, value in features.items():
                if isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                    serializable_features[key] = float(value) if isinstance(value, (np.float64, np.float32)) else int(
                        value)
                elif isinstance(value, (bool, np.bool_)):
                    serializable_features[key] = bool(value)
                else:
                    serializable_features[key] = value

            learning_point = {
                'timestamp': datetime.now().isoformat(),
                'features': serializable_features,
                'posterior_probability': float(posterior),
                'signal_generated': bool(signal_generated),
                'signal_type': str(signal_type),
                'price': float(current_price),
                'nearest_level': float(nearest_level) if nearest_level is not None else 0.0,
                'level_distance': float(abs(current_price - nearest_level)) if nearest_level is not None else 0.0,
                'market_context': {
                    'rsi_level': 'OVERSOLD' if features.get('rsi', 50) < 30 else 'OVERBOUGHT' if features.get('rsi',
                                                                                                              50) > 70 else 'NEUTRAL',
                    'distance_category': 'CLOSE' if features.get('d', 0) < 0.5 else 'MEDIUM' if features.get('d',
                                                                                                             0) < 1.5 else 'FAR',
                    'clustering_strength': 'STRONG' if features.get('c', 0) > 1.0 else 'WEAK' if features.get('c',
                                                                                                              0) < 0.5 else 'MEDIUM'
                }
            }

            self.learning_data['features'].append(learning_point)

            # Keep only last 5000 records to prevent file from growing too large
            if len(self.learning_data['features']) > 5000:
                self.learning_data['features'] = self.learning_data['features'][-5000:]

            self.save_learning_data()
            logger.debug(f"Recorded learning point: P={posterior:.3f}, Signal={signal_type}")

        except Exception as e:
            logger.error(f"Error recording learning point: {e}")

    def record_trade_outcome(self, trade_data, outcome, profit, bars_to_completion):
        """Record the outcome of a trade for learning"""
        try:
            prediction_point = {
                'timestamp': trade_data.get('timestamp'),
                'entry_price': float(trade_data.get('entry_price', 0)),
                'direction': str(trade_data.get('direction', '')),
                'outcome': str(outcome),  # 'WIN', 'LOSS', 'BREAKEVEN'
                'profit': float(profit),
                'bars_to_completion': int(bars_to_completion),
                'reason': str(trade_data.get('reason', '')),
                'features_at_entry': self.get_features_at_time(trade_data.get('timestamp'))
            }

            self.learning_data['predictions'].append(prediction_point)

            # Keep only last 1000 predictions
            if len(self.learning_data['predictions']) > 1000:
                self.learning_data['predictions'] = self.learning_data['predictions'][-1000:]

            self.save_learning_data()
            self.update_model_performance()

            logger.info(f"Recorded trade outcome: {outcome}, Profit: {profit:.2f}")

        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")

    def get_features_at_time(self, timestamp):
        """Get features recorded at or near a specific timestamp"""
        try:
            if not timestamp:
                return None

            target_time = datetime.fromisoformat(timestamp)
            for feature_point in reversed(self.learning_data['features']):
                feature_time = datetime.fromisoformat(feature_point['timestamp'])
                time_diff = abs((feature_time - target_time).total_seconds())
                if time_diff < 300:  # Within 5 minutes
                    return self._make_serializable(feature_point)
            return None
        except Exception as e:
            logger.error(f"Error getting features at time: {e}")
            return None

    def update_model_performance(self):
        """Analyze and update model performance metrics"""
        try:
            if len(self.learning_data['predictions']) < 10:
                return  # Not enough data

            wins = [p for p in self.learning_data['predictions'] if p.get('outcome') == 'WIN']
            losses = [p for p in self.learning_data['predictions'] if p.get('outcome') == 'LOSS']

            total_trades = len(wins) + len(losses)
            if total_trades > 0:
                win_rate = len(wins) / total_trades

                # Calculate feature-specific win rates
                self.analyze_feature_performance()

                # Update optimal thresholds
                self.calculate_optimal_thresholds()

                logger.info(f"Model performance updated: {len(wins)}/{total_trades} wins ({win_rate:.1%})")

        except Exception as e:
            logger.error(f"Error updating model performance: {e}")

    def analyze_feature_performance(self):
        """Analyze win rates by feature value ranges"""
        try:
            predictions_with_features = [
                p for p in self.learning_data['predictions']
                if p.get('features_at_entry') is not None
            ]

            if len(predictions_with_features) < 5:
                return

            # Analyze by RSI ranges
            rsi_ranges = {
                'OVERSOLD': (0, 30),
                'NEUTRAL_LOW': (30, 50),
                'NEUTRAL_HIGH': (50, 70),
                'OVERBOUGHT': (70, 100)
            }

            for range_name, (low, high) in rsi_ranges.items():
                range_trades = [
                    p for p in predictions_with_features
                    if low <= p['features_at_entry']['features'].get('rsi', 50) <= high
                ]
                if range_trades:
                    wins = len([t for t in range_trades if t['outcome'] == 'WIN'])
                    win_rate = wins / len(range_trades)
                    self.model_performance['win_rates_by_feature_ranges'][f'rsi_{range_name}'] = float(win_rate)

            self.save_model_performance()

        except Exception as e:
            logger.error(f"Error analyzing feature performance: {e}")

    def calculate_optimal_thresholds(self):
        """Calculate optimal confidence thresholds based on historical performance"""
        try:
            # This would be more sophisticated in a real implementation
            # For now, we'll just track basic metrics
            predictions_with_posterior = [
                p for p in self.learning_data['predictions']
                if p.get('features_at_entry') and 'posterior_probability' in p['features_at_entry']
            ]

            if len(predictions_with_posterior) < 10:
                return

            # Simple threshold optimization (can be enhanced)
            revert_trades = [p for p in predictions_with_posterior if
                             p['features_at_entry']['posterior_probability'] > 0.5]
            breakout_trades = [p for p in predictions_with_posterior if
                               p['features_at_entry']['posterior_probability'] <= 0.5]

            if revert_trades:
                revert_win_rate = len([t for t in revert_trades if t['outcome'] == 'WIN']) / len(revert_trades)
                self.model_performance['optimal_thresholds']['revert_win_rate'] = float(revert_win_rate)

            if breakout_trades:
                breakout_win_rate = len([t for t in breakout_trades if t['outcome'] == 'WIN']) / len(breakout_trades)
                self.model_performance['optimal_thresholds']['breakout_win_rate'] = float(breakout_win_rate)

            self.save_model_performance()

        except Exception as e:
            logger.error(f"Error calculating optimal thresholds: {e}")

    def get_performance_insights(self):
        """Get current performance insights"""
        try:
            insights = {
                'total_learning_points': len(self.learning_data['features']),
                'total_trades_recorded': len(self.learning_data['predictions']),
                'recent_win_rate': 0.0,
                'feature_performance': self.model_performance.get('win_rates_by_feature_ranges', {}),
                'optimal_thresholds': self.model_performance.get('optimal_thresholds', {})
            }

            if self.learning_data['predictions']:
                recent_trades = self.learning_data['predictions'][-20:]  # Last 20 trades
                wins = len([t for t in recent_trades if t.get('outcome') == 'WIN'])
                if recent_trades:
                    insights['recent_win_rate'] = float(wins / len(recent_trades))

            return self._make_serializable(insights)

        except Exception as e:
            logger.error(f"Error getting performance insights: {e}")
            return {}


# ... (Keep all other classes the same - GeometricEngine, BayesianEngine, TradeManager, EnhancedTradingEngine)
# Just replace the LearningDataManager class with this fixed version

class GeometricEngine:
    def __init__(self, atr_period=14):
        self.atr_period = atr_period
        self.pivot_lookback = 5
        self.geometric_levels = []
        self.level_weights = {}

    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        tr = np.maximum(high - low,
                        np.maximum(abs(high - close.shift(1)),
                                   abs(low - close.shift(1))))
        return tr.rolling(window=period).mean()

    def find_pivot_points(self, high, low, lookback=5):
        """Find swing highs and lows"""
        pivots_high = []
        pivots_low = []

        for i in range(lookback, len(high) - lookback):
            # Swing High
            if all(high[i] > high[i - j] for j in range(1, lookback + 1)) and \
                    all(high[i] > high[i + j] for j in range(1, lookback + 1)):
                pivots_high.append((i, high.iloc[i]))

            # Swing Low
            if all(low[i] < low[i - j] for j in range(1, lookback + 1)) and \
                    all(low[i] < low[i + j] for j in range(1, lookback + 1)):
                pivots_low.append((i, low.iloc[i]))

        return pivots_high, pivots_low

    def cluster_pivot_levels(self, pivots, atr_value, threshold=0.5):
        """Cluster nearby pivot points into support/resistance levels"""
        if not pivots:
            return []

        # Extract just the price values
        pivot_prices = [p[1] for p in pivots]
        pivot_prices.sort()

        clusters = []
        current_cluster = [pivot_prices[0]]

        for price in pivot_prices[1:]:
            if price - current_cluster[0] <= threshold * atr_value:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= 1:  # Require at least 1 pivot in cluster
                    clusters.append(current_cluster)
                current_cluster = [price]

        if len(current_cluster) >= 1:
            clusters.append(current_cluster)

        # Return cluster centers and weights based on cluster size
        clustered_levels = []
        for cluster in clusters:
            cluster_center = np.mean(cluster)
            # Weight based on number of pivots in cluster (more pivots = stronger level)
            cluster_weight = 1.0 + (len(cluster) * 0.2)  # Base 1.0 + 0.2 per additional pivot
            clustered_levels.append((cluster_center, cluster_weight))

        return clustered_levels

    def calculate_fibonacci_levels(self, swing_low, swing_high):
        """Calculate Fibonacci retracement and extension levels"""
        fib_levels = {}
        price_range = swing_high - swing_low

        # Retracement levels (more important, higher weight)
        fib_ratios_ret = [0.236, 0.382, 0.5, 0.618, 0.786]
        for ratio in fib_ratios_ret:
            level = swing_high - (price_range * ratio)
            fib_levels[f'fib_ret_{ratio}'] = (level, 1.3)  # Higher weight for retracements

        # Extension levels (less important, lower weight)
        fib_ratios_ext = [1.272, 1.414, 1.618]
        for ratio in fib_ratios_ext:
            level = swing_high + (price_range * (ratio - 1.0))
            fib_levels[f'fib_ext_{ratio}'] = (level, 1.1)

        return fib_levels

    def update_geometric_levels(self, df):
        """Main method to update all geometric levels"""
        try:
            # Calculate ATR for normalization
            atr = self.calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
            current_atr = atr.iloc[-1]

            # Find pivot points
            pivots_high, pivots_low = self.find_pivot_points(df['high'], df['low'], self.pivot_lookback)

            # Create support/resistance levels from pivots with weights
            self.geometric_levels = []
            self.level_weights = {}

            # Process high pivots
            high_clusters = self.cluster_pivot_levels(pivots_high, current_atr)
            for level, weight in high_clusters:
                self.geometric_levels.append(float(level))
                self.level_weights[float(level)] = float(weight)
                logger.debug(f"Resistance level: {level:.5f}, weight: {weight:.2f}")

            # Process low pivots
            low_clusters = self.cluster_pivot_levels(pivots_low, current_atr)
            for level, weight in low_clusters:
                self.geometric_levels.append(float(level))
                self.level_weights[float(level)] = float(weight)
                logger.debug(f"Support level: {level:.5f}, weight: {weight:.2f}")

            # Calculate Fibonacci levels from major swings
            if len(pivots_high) >= 2 and len(pivots_low) >= 2:
                # Use the two most recent major swings
                recent_highs = sorted([p[1] for p in pivots_high[-2:]], reverse=True)
                recent_lows = sorted([p[1] for p in pivots_low[-2:]])

                if len(recent_highs) > 0 and len(recent_lows) > 0:
                    major_high = recent_highs[0]  # Highest recent high
                    major_low = recent_lows[0]  # Lowest recent low

                    # Only use Fibonacci if we have a meaningful swing
                    if abs(major_high - major_low) > current_atr:
                        fib_levels = self.calculate_fibonacci_levels(major_low, major_high)

                        # Add Fibonacci levels
                        for name, (level, weight) in fib_levels.items():
                            self.geometric_levels.append(float(level))
                            self.level_weights[float(level)] = float(weight)
                            logger.debug(f"Fibonacci {name}: {level:.5f}, weight: {weight:.2f}")

            # Remove duplicates (levels within 0.1 * ATR)
            self._clean_levels(current_atr)

            logger.info(f"Updated {len(self.geometric_levels)} geometric levels")
            if len(self.geometric_levels) > 0:
                logger.info(f"Level range: {min(self.geometric_levels):.5f} to {max(self.geometric_levels):.5f}")

        except Exception as e:
            logger.error(f"Error updating geometric levels: {e}")

    def _clean_levels(self, atr_value, threshold=0.1):
        """Remove duplicate levels that are too close"""
        if not self.geometric_levels:
            return

        sorted_levels = sorted(self.geometric_levels)
        cleaned_levels = [sorted_levels[0]]
        cleaned_weights = {sorted_levels[0]: self.level_weights[sorted_levels[0]]}

        for level in sorted_levels[1:]:
            if abs(level - cleaned_levels[-1]) > threshold * atr_value:
                cleaned_levels.append(level)
                cleaned_weights[level] = self.level_weights[level]
            else:
                # Merge nearby levels - keep the stronger weight
                existing_weight = cleaned_weights[cleaned_levels[-1]]
                new_weight = self.level_weights[level]
                if new_weight > existing_weight:
                    cleaned_weights[cleaned_levels[-1]] = new_weight

        self.geometric_levels = cleaned_levels
        self.level_weights = cleaned_weights

    def get_nearest_levels(self, current_price, num_levels=5):
        """Get the nearest geometric levels to current price"""
        if not self.geometric_levels:
            return []

        levels_with_distance = [
            (float(level), float(abs(level - current_price)), float(self.level_weights.get(level, 1.0)))
            for level in self.geometric_levels
        ]
        levels_sorted = sorted(levels_with_distance, key=lambda x: x[1])

        return levels_sorted[:num_levels]


class BayesianEngine:
    def __init__(self, atr_period=14, rsi_period=14):
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.prior_revert = 0.5
        self.history_size = 1000
        self.feature_history = deque(maxlen=self.history_size)
        self.outcome_history = deque(maxlen=self.history_size)
        self.learning_manager = LearningDataManager()

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_features(self, df, geometric_engine):
        """Calculate features for Bayesian inference - FIXED CLUSTERING"""
        current_price = float(df['close'].iloc[-1])
        atr = geometric_engine.calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
        current_atr = float(atr.iloc[-1])

        # Get nearest geometric levels
        nearest_levels = geometric_engine.get_nearest_levels(current_price, 5)

        if not nearest_levels:
            logger.warning("No geometric levels found for feature calculation")
            return None, None

        closest_level, distance, weight = nearest_levels[0]

        # Normalized distance to nearest level
        d = float(distance / current_atr)

        # Calculate clustering strength properly
        cluster_strength = 0.0
        cluster_count = 0

        for level, dist, w in nearest_levels:
            if dist <= 0.5 * current_atr:  # Wider band for clustering
                cluster_strength += w
                cluster_count += 1
                logger.debug(f"Level in cluster: {level:.5f}, dist: {dist:.5f}, weight: {w:.2f}")

        # If no levels in cluster, use the closest level with reduced strength
        if cluster_strength == 0 and nearest_levels:
            closest_level, closest_dist, closest_weight = nearest_levels[0]
            cluster_strength = float(closest_weight * 0.5)  # Reduced strength for single level
            cluster_count = 1
            logger.debug(f"Using single level for clustering: {closest_level:.5f}")

        logger.info(f"Clustering: {cluster_count} levels, total strength: {cluster_strength:.2f}")

        # Momentum indicators
        rsi = float(self.calculate_rsi(df['close'], self.rsi_period).iloc[-1])
        mom = float((df['close'].iloc[-1] - df['close'].iloc[-5]) / current_atr)  # 5-period momentum

        features = {
            'd': d,  # Normalized distance
            'c': cluster_strength,  # Clustering strength
            'rsi': rsi,  # RSI momentum
            'mom': mom,  # Price momentum
            'price_vs_level': 1 if current_price > closest_level else -1  # Above/below level
        }

        return features, float(closest_level)

    def likelihood_model(self, features, state):
        """Calculate likelihood P(Data | State) - IMPROVED"""
        d, c, rsi, mom, price_vs_level = features['d'], features['c'], features['rsi'], features['mom'], features[
            'price_vs_level']

        if state == 'revert':
            # High likelihood for reversion when:
            # - Price is close to level (small d)
            # - Strong clustering (high c)
            # - Momentum is extreme
            distance_factor = 1.0 / (1.0 + abs(d))  # Higher when close to level
            clustering_factor = min(2.0, c) / 2.0  # Normalize clustering to 0-1
            momentum_factor = (abs(rsi - 50) / 50)  # Higher when RSI is extreme

            likelihood = distance_factor * clustering_factor * momentum_factor

        else:  # state == 'breakout'
            # High likelihood for breakout when:
            # - Price has moved away from level (moderate d)
            # - Weak clustering (low c)
            # - Sustained but not extreme momentum
            distance_factor = max(0, min(1.0, (d - 0.3) / 2.0))  # Higher when moderately away
            clustering_factor = 1.0 - (min(2.0, c) / 2.0)  # Inverse of clustering
            momentum_factor = 1.0 - (abs(rsi - 50) / 50)  # Higher when RSI is mid-range

            likelihood = distance_factor * clustering_factor * momentum_factor

        return max(0.01, min(0.99, likelihood))

    def update_prior_from_trend(self, df):
        """Update prior based on higher timeframe trend"""
        # Simple trend detection using EMA
        ema_20 = float(df['close'].ewm(span=20).mean().iloc[-1])
        ema_50 = float(df['close'].ewm(span=50).mean().iloc[-1])

        if abs(ema_20 - ema_50) < 0.0005:  # Very close - ranging market
            self.prior_revert = 0.6  # Higher prior for reversion in ranging markets
        elif ema_20 > ema_50:
            self.prior_revert = 0.4  # Lower prior for reversion in uptrends
        else:
            self.prior_revert = 0.45  # Slightly lower for downtrends

    def calculate_posterior(self, features, current_price, nearest_level):
        """Calculate posterior probability and record learning data"""
        if features is None:
            return self.prior_revert

        # Calculate likelihoods
        likelihood_revert = float(self.likelihood_model(features, 'revert'))
        likelihood_breakout = float(self.likelihood_model(features, 'breakout'))

        # Calculate posterior using Bayes' Theorem
        evidence = likelihood_revert * self.prior_revert + likelihood_breakout * (1 - self.prior_revert)

        if evidence == 0:
            posterior_revert = float(self.prior_revert)
        else:
            posterior_revert = float((likelihood_revert * self.prior_revert) / evidence)

        # Determine if signal would be generated (for learning)
        signal_generated = posterior_revert > 0.75 or posterior_revert < 0.25
        signal_type = "REVERT" if posterior_revert > 0.75 else "BREAKOUT" if posterior_revert < 0.25 else "NONE"

        # Record learning data
        self.learning_manager.record_learning_point(
            features, posterior_revert, signal_generated, signal_type, float(current_price), float(nearest_level)
        )

        logger.debug(f"Likelihoods - Revert: {likelihood_revert:.3f}, Breakout: {likelihood_breakout:.3f}")
        logger.debug(f"Prior: {self.prior_revert:.3f}, Posterior: {posterior_revert:.3f}")

        return posterior_revert

    def get_learning_insights(self):
        """Get insights from learning data"""
        return self.learning_manager.get_performance_insights()


# ... (Keep TradeManager and EnhancedTradingEngine classes the same, but ensure they use float() conversions where needed)

class TradeManager:
    def __init__(self):
        self.trade_history = []
        self.daily_pnl = 0.0
        self.max_daily_loss = -100.0
        self.max_open_trades = 2
        self.learning_manager = LearningDataManager()
        self.load_trade_history()

    def load_trade_history(self):
        """Load trade history from file"""
        try:
            if os.path.exists('trade_history.json'):
                with open('trade_history.json', 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} historical trades")
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")

    def save_trade_history(self):
        """Save trade history to file"""
        try:
            with open('trade_history.json', 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")

    def record_trade(self, symbol, direction, volume, entry_price, sl, tp, reason=""):
        """Record a new trade"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': direction,
            'volume': float(volume),
            'entry_price': float(entry_price),
            'sl': float(sl),
            'tp': float(tp),
            'reason': reason,
            'status': 'OPEN'
        }
        self.trade_history.append(trade)
        self.save_trade_history()
        logger.info(f"Recorded trade: {direction} {symbol} {volume} lots")
        return trade

    def update_trade_status(self):
        """Update status of open trades and record outcomes"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return

            current_time = datetime.now()

            for trade in self.trade_history[-10:]:
                if trade.get('status') == 'OPEN':
                    position_still_open = any(
                        abs(trade['entry_price'] - pos.price_open) < 0.00001 and
                        trade['direction'] == 'BUY' and pos.type == 0 or
                        trade['direction'] == 'SELL' and pos.type == 1
                        for pos in positions
                    )

                    if not position_still_open:
                        # Trade closed, determine outcome
                        trade['status'] = 'CLOSED'
                        trade['exit_time'] = current_time.isoformat()

                        # Calculate approximate outcome (in real trading, you'd get this from MT5)
                        # For now, we'll use a simple approximation
                        entry_time = datetime.fromisoformat(trade['timestamp'])
                        bars_to_completion = int((current_time - entry_time).total_seconds() / 900)  # 15-minute bars

                        # This is a simplified outcome calculation
                        # In live trading, you'd get actual P/L from MT5
                        outcome = 'WIN'  # Placeholder - would need actual MT5 position data
                        profit = 0.0  # Placeholder

                        self.learning_manager.record_trade_outcome(trade, outcome, profit, bars_to_completion)
                        logger.info(f"Trade closed: {trade['direction']} {trade['symbol']}")

            self.save_trade_history()
        except Exception as e:
            logger.error(f"Error updating trade status: {e}")

    def can_trade(self):
        """Check if we're allowed to place new trades"""
        if self.daily_pnl <= self.max_daily_loss:
            logger.warning(f"Daily PnL ({self.daily_pnl}) below maximum loss limit")
            return False

        positions = mt5.positions_get()
        if positions and len(positions) >= self.max_open_trades:
            logger.warning(f"Maximum open trades ({self.max_open_trades}) reached")
            return False

        return True

    def get_performance_stats(self):
        """Calculate performance statistics"""
        if not self.trade_history:
            return "No trades yet"

        closed_trades = [t for t in self.trade_history if t.get('status') == 'CLOSED']
        if not closed_trades:
            return "No closed trades yet"

        wins = len([t for t in closed_trades if t.get('outcome') == 'WIN'])
        total = len(closed_trades)
        win_rate = (wins / total) * 100 if total > 0 else 0

        return f"Performance: {wins}/{total} wins ({win_rate:.1f}% win rate)"


class EnhancedTradingEngine:
    def __init__(self, symbol, lot_size=0.1, risk_per_trade=0.02):
        self.symbol = symbol
        self.lot_size = lot_size
        self.risk_per_trade = risk_per_trade
        self.position = None
        self.geometric_engine = GeometricEngine()
        self.bayesian_engine = BayesianEngine()
        self.trade_manager = TradeManager()

        # MT5 Connection Details
        self.mt5_path = r"C:\Program Files\mt5_installation\terminal64.exe"
        self.account = account_number
        self.password = "password"
        self.server = "server"

        # Trading limits
        self.min_confidence = 0.75
        self.max_confidence = 0.25
        self.consecutive_signals = 0
        self.max_consecutive_signals = 3

    def initialize_mt5(self):
        """Initialize MetaTrader5 connection"""
        try:
            if mt5.initialize():
                mt5.shutdown()
                time.sleep(2)

            if not mt5.initialize(path=self.mt5_path):
                logger.error(f"MT5 initialization failed. Error: {mt5.last_error()}")
                return False

            if not mt5.login(login=self.account, password=self.password, server=self.server):
                logger.error(f"MT5 login failed. Error: {mt5.last_error()}")
                mt5.shutdown()
                return False

            logger.info(f"MT5 initialized successfully - Account: {self.account}, Server: {self.server}")

            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Connected to account: {account_info.login}, Balance: {account_info.balance}")

            return True

        except Exception as e:
            logger.error(f"Error during MT5 initialization: {e}")
            return False

    def get_historical_data(self, timeframe=mt5.TIMEFRAME_M15, bars=500):
        """Get historical data from MT5"""
        try:
            mt5.symbol_select(self.symbol, True)

            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
            if rates is None:
                logger.error(f"Failed to get data for {self.symbol}. Error: {mt5.last_error()}")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            logger.info(f"Retrieved {len(df)} bars for {self.symbol}")
            return df

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None

    def calculate_position_size(self, stop_distance, account_balance=None):
        """Calculate position size based on risk management"""
        try:
            if account_balance is None:
                account_info = mt5.account_info()
                if account_info is None:
                    return self.lot_size
                account_balance = account_info.balance

            risk_amount = account_balance * self.risk_per_trade
            symbol_info = mt5.symbol_info(self.symbol)

            if symbol_info:
                point_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size
            else:
                point_value = 1.0

            lot_size = risk_amount / (stop_distance * point_value * 100)
            lot_size = max(0.01, min(1.0, round(lot_size, 2)))
            logger.info(f"Calculated lot size: {lot_size} (Stop: {stop_distance:.5f})")
            return float(lot_size)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.lot_size

    def execute_trade(self, signal_type, current_price, stop_distance, reason=""):
        """Execute trade based on signal"""
        try:
            if not self.trade_manager.can_trade():
                logger.warning("Trade not allowed due to risk limits")
                return False

            if self.consecutive_signals >= self.max_consecutive_signals:
                logger.warning(f"Too many consecutive signals ({self.consecutive_signals}), skipping trade")
                return False

            lot_size = self.calculate_position_size(stop_distance)

            symbol_info = mt5.symbol_info_tick(self.symbol)
            if not symbol_info:
                logger.error(f"Could not get tick data for {self.symbol}")
                return False

            if signal_type == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                price = float(symbol_info.ask)
                sl = float(price - stop_distance)
                tp = float(price + (2 * stop_distance))
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = float(symbol_info.bid)
                sl = float(price + stop_distance)
                tp = float(price - (2 * stop_distance))

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Bayesian-{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            logger.info(f"Sending {signal_type} order: Price={price:.5f}, SL={sl:.5f}, TP={tp:.5f}")

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Trade execution failed: {result.retcode} - {result.comment}")
                return False
            else:
                logger.info(f"Trade executed successfully: {signal_type} {self.symbol} {lot_size} lots")

                trade_data = self.trade_manager.record_trade(
                    symbol=self.symbol,
                    direction=signal_type,
                    volume=lot_size,
                    entry_price=price,
                    sl=sl,
                    tp=tp,
                    reason=reason
                )

                self.consecutive_signals += 1
                return True

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    def run_trading_cycle(self):
        """Main trading cycle"""
        try:
            self.trade_manager.update_trade_status()

            df = self.get_historical_data()
            if df is None or len(df) == 0:
                logger.error("No data retrieved")
                return False

            current_price = float(df['close'].iloc[-1])
            logger.info(f"Current {self.symbol} price: {current_price:.5f}")

            self.geometric_engine.update_geometric_levels(df)
            self.bayesian_engine.update_prior_from_trend(df)

            features, nearest_level = self.bayesian_engine.calculate_features(df, self.geometric_engine)

            if features is None:
                logger.warning("No features calculated")
                return False

            posterior_revert = self.bayesian_engine.calculate_posterior(features, current_price, nearest_level)

            logger.info(
                f"P(Reversion): {posterior_revert:.3f}, Features - d: {features['d']:.3f}, c: {features['c']:.3f}, RSI: {features['rsi']:.1f}")

            nearest_levels = self.geometric_engine.get_nearest_levels(current_price, 1)
            if not nearest_levels:
                logger.warning("No geometric levels found")
                return False

            closest_level, distance, weight = nearest_levels[0]
            atr = float(self.geometric_engine.calculate_atr(df['high'], df['low'], df['close'], 14).iloc[-1])

            logger.info(f"Nearest level: {closest_level:.5f}, Distance: {distance:.5f}, ATR: {atr:.5f}")

            signal_generated = False

            if posterior_revert > self.min_confidence:
                if current_price > closest_level:
                    logger.info(f"SELL signal - Price above level, expecting reversion")
                    signal_generated = self.execute_trade('SELL', current_price, distance, "Reversion")
                else:
                    logger.info(f"BUY signal - Price below level, expecting reversion")
                    signal_generated = self.execute_trade('BUY', current_price, distance, "Reversion")

            elif posterior_revert < self.max_confidence:
                if current_price > closest_level:
                    logger.info(f"BUY signal - Price above level, expecting breakout")
                    signal_generated = self.execute_trade('BUY', current_price, distance, "Breakout")
                else:
                    logger.info(f"SELL signal - Price below level, expecting breakout")
                    signal_generated = self.execute_trade('SELL', current_price, distance, "Breakout")
            else:
                logger.info("No trade signal - confidence threshold not met")
                self.consecutive_signals = 0

            # Log learning insights periodically
            if len(self.trade_manager.trade_history) % 5 == 0:  # Every 5 trades
                insights = self.bayesian_engine.get_learning_insights()
                logger.info(f"Learning Insights: {insights}")

            stats = self.trade_manager.get_performance_stats()
            logger.info(stats)

            return signal_generated

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return False


def main():
    SYMBOL = "EURUSD"
    CYCLE_INTERVAL = 60

    trader = EnhancedTradingEngine(SYMBOL, lot_size=0.1)

    if not trader.initialize_mt5():
        return

    logger.info(f"Starting Enhanced Bayesian-Geometric Trading Bot for {SYMBOL}")

    try:
        cycle_count = 0
        while True:
            cycle_count += 1
            logger.info(f"--- Trading Cycle {cycle_count} ---")

            trader.run_trading_cycle()

            logger.info(f"Waiting {CYCLE_INTERVAL} seconds until next cycle...")
            time.sleep(CYCLE_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        mt5.shutdown()
        logger.info("MT5 shutdown complete")


if __name__ == "__main__":
    main()
