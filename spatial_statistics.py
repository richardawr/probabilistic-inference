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


class FixedLearningDataManager:
    def __init__(self):
        self.learning_file = 'learning_data.json'
        self.model_performance_file = 'model_performance.json'
        # Initialize with proper structure
        self.learning_data = {
            'trades': [],
            'learning_points': [],
            'last_updated': None
        }
        self.model_performance = {
            'summary_stats': {},
            'last_updated': None
        }
        self.load_learning_data()

    def load_learning_data(self):
        """Load historical learning data with backward compatibility"""
        try:
            if os.path.exists(self.learning_file):
                with open(self.learning_file, 'r') as f:
                    loaded_data = json.load(f)

                # Handle backward compatibility - check if old format exists
                if 'trades' in loaded_data:
                    self.learning_data = loaded_data
                else:
                    # Convert old format to new format
                    self.learning_data = {
                        'trades': [],
                        'learning_points': loaded_data.get('features', []),
                        'last_updated': loaded_data.get('last_updated')
                    }
                logger.info(f"Loaded {len(self.learning_data['trades'])} trade records")

            if os.path.exists(self.model_performance_file):
                with open(self.model_performance_file, 'r') as f:
                    self.model_performance = json.load(f)
                logger.info(f"Loaded model performance data")

        except Exception as e:
            logger.error(f"Error loading learning data: {e}")
            # Initialize with proper structure
            self.learning_data = {'trades': [], 'learning_points': [], 'last_updated': None}
            self.model_performance = {'summary_stats': {}, 'last_updated': None}

    def save_learning_data(self):
        """Save learning data to file"""
        try:
            self.learning_data['last_updated'] = datetime.now().isoformat()

            with open(self.learning_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2, default=str)

            with open(self.model_performance_file, 'w') as f:
                json.dump(self.model_performance, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving learning data: {e}")

    def record_trade(self, trade_data):
        """Record a complete trade with outcome"""
        try:
            trade_record = {
                'ticket': trade_data.get('ticket'),
                'symbol': trade_data.get('symbol'),
                'direction': trade_data.get('direction'),
                'entry_time': trade_data.get('entry_time'),
                'exit_time': trade_data.get('exit_time'),
                'entry_price': float(trade_data.get('entry_price', 0)),
                'exit_price': float(trade_data.get('exit_price', 0)),
                'volume': float(trade_data.get('volume', 0)),
                'profit': float(trade_data.get('profit', 0)),
                'sl': float(trade_data.get('sl', 0)),
                'tp': float(trade_data.get('tp', 0)),
                'status': trade_data.get('status', 'UNKNOWN'),
                'outcome': self._determine_outcome(trade_data),
                'features_at_entry': trade_data.get('features_at_entry', {}),
                'posterior_probability': trade_data.get('posterior_probability', 0.5)
            }

            # Check if trade already exists
            existing_idx = None
            for i, trade in enumerate(self.learning_data['trades']):
                if trade.get('ticket') == trade_record['ticket']:
                    existing_idx = i
                    break

            if existing_idx is not None:
                # Update existing trade
                self.learning_data['trades'][existing_idx] = trade_record
                logger.debug(f"Updated existing trade: {trade_record['ticket']}")
            else:
                # Add new trade
                self.learning_data['trades'].append(trade_record)
                logger.info(f"Added new trade: {trade_record['ticket']}")

            # Keep only last 1000 trades
            if len(self.learning_data['trades']) > 1000:
                self.learning_data['trades'] = self.learning_data['trades'][-1000:]

            self._update_performance_stats()
            self.save_learning_data()

            logger.info(
                f"Recorded trade: {trade_record['ticket']} - {trade_record['outcome']} - Profit: {trade_record['profit']:.2f}")

        except Exception as e:
            logger.error(f"Error recording trade: {e}")

    def _determine_outcome(self, trade_data):
        """Determine trade outcome based on profit"""
        profit = trade_data.get('profit', 0)
        if profit > 0:
            return 'WIN'
        elif profit < 0:
            return 'LOSS'
        else:
            return 'BREAKEVEN'

    def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            closed_trades = [t for t in self.learning_data['trades'] if t.get('status') == 'CLOSED']

            if not closed_trades:
                self.model_performance['summary_stats'] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'breakeven_trades': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0,
                    'avg_profit_per_trade': 0.0
                }
                return

            wins = len([t for t in closed_trades if t.get('outcome') == 'WIN'])
            losses = len([t for t in closed_trades if t.get('outcome') == 'LOSS'])
            breakevens = len([t for t in closed_trades if t.get('outcome') == 'BREAKEVEN'])
            total_trades = len(closed_trades)
            total_profit = sum(t.get('profit', 0) for t in closed_trades)

            self.model_performance['summary_stats'] = {
                'total_trades': total_trades,
                'winning_trades': wins,
                'losing_trades': losses,
                'breakeven_trades': breakevens,
                'win_rate': wins / total_trades if total_trades > 0 else 0.0,
                'total_profit': total_profit,
                'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0.0
            }

            logger.info(f"Updated performance stats: {wins}/{total_trades} wins ({wins / total_trades:.1%})")

        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")

    def record_learning_point(self, features, posterior, signal_generated, signal_type, current_price, nearest_level):
        """Record a learning data point"""
        try:
            learning_point = {
                'timestamp': datetime.now().isoformat(),
                'features': features,
                'posterior_probability': float(posterior),
                'signal_generated': bool(signal_generated),
                'signal_type': str(signal_type),
                'price': float(current_price),
                'nearest_level': float(nearest_level) if nearest_level is not None else 0.0
            }

            self.learning_data['learning_points'].append(learning_point)

            # Keep only last 5000 records to prevent file from growing too large
            if len(self.learning_data['learning_points']) > 5000:
                self.learning_data['learning_points'] = self.learning_data['learning_points'][-5000:]

            self.save_learning_data()
            logger.debug(f"Recorded learning point: P={posterior:.3f}, Signal={signal_type}")

        except Exception as e:
            logger.error(f"Error recording learning point: {e}")

    def get_system_status(self):
        """Get accurate system status - FIXED VERSION"""
        try:
            # Ensure we have the latest data
            closed_trades = [t for t in self.learning_data['trades'] if t.get('status') == 'CLOSED']
            open_trades = [t for t in self.learning_data['trades'] if t.get('status') == 'OPEN']

            wins = len([t for t in closed_trades if t.get('outcome') == 'WIN'])
            losses = len([t for t in closed_trades if t.get('outcome') == 'LOSS'])
            breakevens = len([t for t in closed_trades if t.get('outcome') == 'BREAKEVEN'])
            total_closed = len(closed_trades)

            if total_closed > 0:
                win_rate = wins / total_closed
                total_profit = sum(t.get('profit', 0) for t in closed_trades)
            else:
                win_rate = 0.0
                total_profit = 0.0

            # Calculate recent win rate (last 20 closed trades)
            recent_win_rate = 0.0
            if closed_trades:
                recent_trades = closed_trades[-20:]
                recent_wins = len([t for t in recent_trades if t.get('outcome') == 'WIN'])
                if recent_trades:
                    recent_win_rate = recent_wins / len(recent_trades)

            status = {
                'trades_analyzed': total_closed,
                'win_rate': win_rate,
                'winning_trades': wins,
                'losing_trades': losses,
                'breakeven_trades': breakevens,
                'total_profit': total_profit,
                'open_trades': len(open_trades),
                'learning_points': len(self.learning_data.get('learning_points', [])),
                'recent_win_rate': recent_win_rate,
                'last_updated': self.learning_data.get('last_updated', 'Never')
            }

            return status

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'trades_analyzed': 0,
                'win_rate': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'breakeven_trades': 0,
                'total_profit': 0.0,
                'open_trades': 0,
                'learning_points': 0,
                'recent_win_rate': 0.0,
                'last_updated': 'Error'
            }


class GeometricEngine:
    def __init__(self, atr_period=14):
        self.atr_period = atr_period
        self.pivot_lookback = 5
        self.geometric_levels = []
        self.level_weights = {}

    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series([0.001] * len(high))  # Default fallback

    def find_pivot_points(self, high, low, lookback=5):
        """Find swing highs and lows"""
        pivots_high = []
        pivots_low = []

        try:
            for i in range(lookback, len(high) - lookback):
                # Swing High
                if all(high.iloc[i] > high.iloc[i - j] for j in range(1, lookback + 1)) and \
                        all(high.iloc[i] > high.iloc[i + j] for j in range(1, lookback + 1)):
                    pivots_high.append((i, high.iloc[i]))

                # Swing Low
                if all(low.iloc[i] < low.iloc[i - j] for j in range(1, lookback + 1)) and \
                        all(low.iloc[i] < low.iloc[i + j] for j in range(1, lookback + 1)):
                    pivots_low.append((i, low.iloc[i]))
        except Exception as e:
            logger.error(f"Error finding pivot points: {e}")

        return pivots_high, pivots_low

    def cluster_pivot_levels(self, pivots, atr_value, threshold=0.5):
        """Cluster nearby pivot points into support/resistance levels"""
        if not pivots:
            return []

        try:
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
        except Exception as e:
            logger.error(f"Error clustering pivot levels: {e}")
            return []

    def calculate_fibonacci_levels(self, swing_low, swing_high):
        """Calculate Fibonacci retracement and extension levels"""
        fib_levels = {}
        try:
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
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}

    def update_geometric_levels(self, df):
        """Main method to update all geometric levels"""
        try:
            if df is None or len(df) < 50:
                logger.warning("Insufficient data for geometric level calculation")
                return

            # Calculate ATR for normalization
            atr = self.calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
            if atr.isnull().all() or atr.iloc[-1] == 0:
                current_atr = 0.001  # Default fallback
            else:
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

        try:
            sorted_levels = sorted(self.geometric_levels)
            cleaned_levels = [sorted_levels[0]]
            cleaned_weights = {sorted_levels[0]: self.level_weights.get(sorted_levels[0], 1.0)}

            for level in sorted_levels[1:]:
                if abs(level - cleaned_levels[-1]) > threshold * atr_value:
                    cleaned_levels.append(level)
                    cleaned_weights[level] = self.level_weights.get(level, 1.0)
                else:
                    # Merge nearby levels - keep the stronger weight
                    existing_weight = cleaned_weights[cleaned_levels[-1]]
                    new_weight = self.level_weights.get(level, 1.0)
                    if new_weight > existing_weight:
                        cleaned_weights[cleaned_levels[-1]] = new_weight

            self.geometric_levels = cleaned_levels
            self.level_weights = cleaned_weights
        except Exception as e:
            logger.error(f"Error cleaning levels: {e}")

    def get_nearest_levels(self, current_price, num_levels=5):
        """Get the nearest geometric levels to current price"""
        if not self.geometric_levels:
            return []

        try:
            levels_with_distance = [
                (float(level), float(abs(level - current_price)), float(self.level_weights.get(level, 1.0)))
                for level in self.geometric_levels
            ]
            levels_sorted = sorted(levels_with_distance, key=lambda x: x[1])

            return levels_sorted[:num_levels]
        except Exception as e:
            logger.error(f"Error getting nearest levels: {e}")
            return []


class BayesianEngine:
    def __init__(self, atr_period=14, rsi_period=14):
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.prior_revert = 0.5
        self.history_size = 1000
        self.feature_history = deque(maxlen=self.history_size)
        self.outcome_history = deque(maxlen=self.history_size)
        self.learning_manager = FixedLearningDataManager()

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral 50
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices))  # Default neutral value

    def calculate_features(self, df, geometric_engine):
        """Calculate features for Bayesian inference"""
        try:
            current_price = float(df['close'].iloc[-1])
            atr = geometric_engine.calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
            current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.001

            # Get nearest geometric levels
            nearest_levels = geometric_engine.get_nearest_levels(current_price, 5)

            if not nearest_levels:
                logger.warning("No geometric levels found for feature calculation")
                return None, None

            closest_level, distance, weight = nearest_levels[0]

            # Normalized distance to nearest level
            d = float(distance / current_atr) if current_atr > 0 else 1.0

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
            mom = float((df['close'].iloc[-1] - df['close'].iloc[-5]) / current_atr) if current_atr > 0 else 0.0

            features = {
                'd': d,  # Normalized distance
                'c': cluster_strength,  # Clustering strength
                'rsi': rsi,  # RSI momentum
                'mom': mom,  # Price momentum
                'price_vs_level': 1 if current_price > closest_level else -1  # Above/below level
            }

            return features, float(closest_level)

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None, None

    def likelihood_model(self, features, state):
        """Calculate likelihood P(Data | State)"""
        try:
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
        except Exception as e:
            logger.error(f"Error in likelihood model: {e}")
            return 0.5  # Default neutral likelihood

    def update_prior_from_trend(self, df):
        """Update prior based on higher timeframe trend"""
        try:
            # Simple trend detection using EMA
            ema_20 = float(df['close'].ewm(span=20).mean().iloc[-1])
            ema_50 = float(df['close'].ewm(span=50).mean().iloc[-1])

            if abs(ema_20 - ema_50) < 0.0005:  # Very close - ranging market
                self.prior_revert = 0.6  # Higher prior for reversion in ranging markets
            elif ema_20 > ema_50:
                self.prior_revert = 0.4  # Lower prior for reversion in uptrends
            else:
                self.prior_revert = 0.45  # Slightly lower for downtrends
        except Exception as e:
            logger.error(f"Error updating prior from trend: {e}")

    def calculate_posterior(self, features, current_price, nearest_level):
        """Calculate posterior probability and record learning data"""
        if features is None:
            return self.prior_revert

        try:
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
        except Exception as e:
            logger.error(f"Error calculating posterior: {e}")
            return self.prior_revert

    def get_learning_insights(self):
        """Get insights from learning data"""
        return self.learning_manager.get_system_status()


# ... [Rest of the classes remain the same as in the previous version: FixedTradeManager and EnhancedTradingEngine]

class FixedTradeManager:
    def __init__(self, symbol):
        self.symbol = symbol
        self.learning_manager = FixedLearningDataManager()
        self.max_open_trades = 2

        # Breakeven and trailing stop settings
        self.breakeven_pips = 10
        self.trailing_start_pips = 15
        self.trailing_step_pips = 5
        self.breakeven_buffer_pips = 1

    def sync_with_mt5_complete(self):
        """Complete synchronization with MT5"""
        try:
            logger.info("Starting complete MT5 synchronization...")

            # Get all positions and deals
            positions = mt5.positions_get(symbol=self.symbol) or []
            deals = mt5.history_deals_get(datetime.now() - timedelta(days=30), datetime.now()) or []

            logger.info(f"Found {len(positions)} positions and {len(deals)} deals")

            # Process all trades from MT5 data
            self._process_mt5_trades(positions, deals)

        except Exception as e:
            logger.error(f"Error in complete sync: {e}")

    def _process_mt5_trades(self, positions, deals):
        """Process all trades from MT5 data"""
        try:
            # Group deals by position ticket
            position_deals = {}
            for deal in deals:
                pos_id = deal.position_id
                if pos_id not in position_deals:
                    position_deals[pos_id] = []
                position_deals[pos_id].append(deal)

            # Process each position
            for pos in positions:
                self._process_single_trade(pos, position_deals.get(pos.ticket, []))

            # Process closed positions that aren't in current positions
            self._process_closed_positions(position_deals, positions)

        except Exception as e:
            logger.error(f"Error processing MT5 trades: {e}")

    def _process_single_trade(self, position, deals):
        """Process a single trade from MT5 data"""
        try:
            # Find opening and closing deals
            opening_deal = None
            closing_deals = []
            total_profit = 0.0

            for deal in deals:
                if deal.entry == 0:  # DEAL_ENTRY_IN
                    opening_deal = deal
                elif deal.entry == 1:  # DEAL_ENTRY_OUT
                    closing_deals.append(deal)
                    total_profit += deal.profit

            trade_data = {
                'ticket': position.ticket,
                'symbol': position.symbol,
                'direction': 'BUY' if position.type == 0 else 'SELL',
                'entry_time': datetime.fromtimestamp(position.time_open).isoformat(),
                'entry_price': position.price_open,
                'volume': position.volume,
                'sl': position.sl,
                'tp': position.tp,
                'status': 'OPEN' if position.time_close == 0 else 'CLOSED',
                'profit': position.profit if position.time_close == 0 else total_profit
            }

            if position.time_close > 0:
                trade_data['exit_time'] = datetime.fromtimestamp(position.time_close).isoformat()
                trade_data['exit_price'] = position.price_current

            # Record in learning system
            self.learning_manager.record_trade(trade_data)

        except Exception as e:
            logger.error(f"Error processing single trade: {e}")

    def _process_closed_positions(self, position_deals, current_positions):
        """Process positions that are closed but not in current positions"""
        try:
            current_tickets = [pos.ticket for pos in current_positions]

            for pos_id, deals in position_deals.items():
                if pos_id not in current_tickets:
                    # This is a closed position
                    opening_deal = None
                    closing_deals = []
                    total_profit = 0.0
                    entry_time = None
                    exit_time = None

                    for deal in deals:
                        if deal.entry == 0:  # Opening deal
                            opening_deal = deal
                            entry_time = datetime.fromtimestamp(deal.time)
                        elif deal.entry == 1:  # Closing deal
                            closing_deals.append(deal)
                            total_profit += deal.profit
                            exit_time = datetime.fromtimestamp(deal.time)

                    if opening_deal and closing_deals:
                        trade_data = {
                            'ticket': pos_id,
                            'symbol': self.symbol,
                            'direction': 'BUY' if opening_deal.type == 0 else 'SELL',
                            'entry_time': entry_time.isoformat(),
                            'exit_time': exit_time.isoformat(),
                            'entry_price': opening_deal.price,
                            'exit_price': closing_deals[-1].price,
                            'volume': opening_deal.volume,
                            'profit': total_profit,
                            'sl': 0.0,
                            'tp': 0.0,
                            'status': 'CLOSED'
                        }

                        self.learning_manager.record_trade(trade_data)
                        logger.info(f"Processed closed position: {pos_id} - Profit: {total_profit:.2f}")

        except Exception as e:
            logger.error(f"Error processing closed positions: {e}")

    def get_current_open_positions_count(self):
        """Get actual number of open positions from MT5"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return 0
            return len(positions)
        except Exception as e:
            logger.error(f"Error getting open positions count: {e}")
            return 0

    def get_pip_size(self):
        """Get pip size for the symbol"""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            return symbol_info.point * 10  # Standard pip size
        return 0.0001  # Default for EURUSD

    def update_stops_for_open_trades(self):
        """Update breakeven and trailing stops for open positions"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return

            for position in positions:
                # Skip if not our symbol or magic number
                if position.symbol != self.symbol or position.magic != 234000:
                    continue

                current_price = position.price_current
                entry_price = position.price_open
                current_stop = position.sl
                direction = position.type  # 0 = BUY, 1 = SELL

                # Calculate profit in pips
                pip_size = self.get_pip_size()
                if direction == 0:  # BUY
                    profit_pips = (current_price - entry_price) / pip_size
                else:  # SELL
                    profit_pips = (entry_price - current_price) / pip_size

                # Check if we should move to breakeven
                if profit_pips >= self.breakeven_pips and current_stop != entry_price:
                    self.move_to_breakeven(position)

                # Check if we should activate trailing stop
                elif profit_pips >= self.trailing_start_pips:
                    self.update_trailing_stop(position, profit_pips, direction)

        except Exception as e:
            logger.error(f"Error updating stops for open trades: {e}")

    def move_to_breakeven(self, position):
        """Move stop loss to breakeven plus buffer"""
        try:
            pip_size = self.get_pip_size()

            if position.type == 0:  # BUY
                new_sl = position.price_open - (self.breakeven_buffer_pips * pip_size)
            else:  # SELL
                new_sl = position.price_open + (self.breakeven_buffer_pips * pip_size)

            # Only move if new stop is better than current
            if (position.type == 0 and new_sl > position.sl) or (position.type == 1 and new_sl < position.sl):
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                    "deviation": 20,
                }

                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Moved to breakeven for ticket {position.ticket}, new SL: {new_sl:.5f}")
                else:
                    logger.warning(f"Failed to move to breakeven: {result.retcode}")

        except Exception as e:
            logger.error(f"Error moving to breakeven: {e}")

    def update_trailing_stop(self, position, profit_pips, direction):
        """Update trailing stop based on current profit"""
        try:
            current_price = position.price_current
            current_stop = position.sl
            pip_size = self.get_pip_size()

            if direction == 0:  # BUY
                # Calculate new stop based on trailing step
                potential_new_sl = current_price - (self.trailing_step_pips * pip_size)
                # Only move stop if new stop is higher than current stop
                if potential_new_sl > current_stop:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": potential_new_sl,
                        "tp": position.tp,
                        "deviation": 20,
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(
                            f"Trailing stop updated for ticket {position.ticket}, new SL: {potential_new_sl:.5f}")

            else:  # SELL
                # Calculate new stop based on trailing step
                potential_new_sl = current_price + (self.trailing_step_pips * pip_size)
                # Only move stop if new stop is lower than current stop
                if potential_new_sl < current_stop or current_stop == 0:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": potential_new_sl,
                        "tp": position.tp,
                        "deviation": 20,
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(
                            f"Trailing stop updated for ticket {position.ticket}, new SL: {potential_new_sl:.5f}")

        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")

    def update_trade_status(self):
        """Update status of open trades"""
        try:
            self.update_stops_for_open_trades()
            self.sync_with_mt5_complete()  # Use the new sync method

        except Exception as e:
            logger.error(f"Error updating trade status: {e}")

    def can_trade(self):
        """Check if we're allowed to place new trades"""
        try:
            current_positions = self.get_current_open_positions_count()
            if current_positions >= self.max_open_trades:
                logger.warning(f"Already have {current_positions} open positions for {self.symbol}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking if can trade: {e}")
            return False

    def get_performance_stats(self):
        """Get accurate performance statistics"""
        try:
            status = self.learning_manager.get_system_status()

            closed_trades = status['trades_analyzed']
            wins = status['winning_trades']
            losses = status['losing_trades']
            breakevens = status['breakeven_trades']
            total_profit = status['total_profit']
            open_trades = status['open_trades']

            if closed_trades > 0:
                win_rate = (wins / closed_trades) * 100
                avg_profit = total_profit / closed_trades

                stats = (f"Performance: {wins}/{closed_trades} wins ({win_rate:.1f}% win rate)\n"
                         f"Total PnL: {total_profit:.2f} | Avg Trade: {avg_profit:.2f}\n"
                         f"Open Positions: {open_trades}")
            else:
                stats = "No completed trades yet"

            return stats

        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return "Error calculating statistics"


class EnhancedTradingEngine:
    def __init__(self, symbol, lot_size=0.1, risk_per_trade=0.02):
        self.symbol = symbol
        self.lot_size = lot_size
        self.risk_per_trade = risk_per_trade
        self.position = None
        self.geometric_engine = GeometricEngine()
        self.bayesian_engine = BayesianEngine()
        self.trade_manager = FixedTradeManager(symbol)

        # MT5 Connection Details
        self.mt5_path = r"C:\Program Files\mt5_directory\terminal64.exe"
        self.account = account
        self.password = "password"
        self.server = "server"

        # Trading limits - ADJUSTED FOR 5M TIMEFRAME
        self.min_confidence = 0.80  # Increased from 0.75 for 5M
        self.max_confidence = 0.20  # Decreased from 0.25 for 5M
        self.consecutive_signals = 0
        self.max_consecutive_signals = 2  # Reduced for 5M

        # Enhanced signal confirmation for 5M timeframe
        self.last_trade_time = None
        self.trade_cooldown = 300  # 5 minutes cooldown (1 bar)
        self.min_distance_atr = 0.3  # Minimum 0.3 ATR distance from level
        self.signal_confirmation_required = True

        # Breakeven and trailing stop configuration
        self.breakeven_atr_multiplier = 0.8  # Move to breakeven after 0.8 ATR profit
        self.trailing_start_atr_multiplier = 1.2  # Start trailing after 1.2 ATR profit
        self.trailing_step_atr_multiplier = 0.3  # Move stop by 0.3 ATR each step

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

    def get_historical_data(self, timeframe=mt5.TIMEFRAME_M5, bars=500):
        """Get historical data from MT5 - NOW USING 5M TIMEFRAME"""
        try:
            mt5.symbol_select(self.symbol, True)

            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
            if rates is None:
                logger.error(f"Failed to get data for {self.symbol}. Error: {mt5.last_error()}")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            logger.info(f"Retrieved {len(df)} M5 bars for {self.symbol}")
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

    def calculate_dynamic_stop_settings(self, atr_value):
        """Calculate dynamic breakeven and trailing settings based on ATR"""
        pip_size = self.trade_manager.get_pip_size()

        # Convert ATR to pips for settings - FIXED CALCULATION
        atr_pips = atr_value / pip_size

        # Ensure minimum values to avoid zero pips
        self.trade_manager.breakeven_pips = max(1, int(self.breakeven_atr_multiplier * atr_pips))
        self.trade_manager.trailing_start_pips = max(2, int(self.trailing_start_atr_multiplier * atr_pips))
        self.trade_manager.trailing_step_pips = max(1, int(self.trailing_step_atr_multiplier * atr_pips))

        logger.info(f"Dynamic stops: Breakeven at {self.trade_manager.breakeven_pips} pips, "
                    f"Trailing start at {self.trade_manager.trailing_start_pips} pips, "
                    f"Step: {self.trade_manager.trailing_step_pips} pips")

    def can_open_new_trade(self):
        """ACCURATE: Check if we can open new trade"""
        try:
            if self.last_trade_time is None:
                return True

            time_since_last_trade = datetime.now() - self.last_trade_time
            if time_since_last_trade.total_seconds() < self.trade_cooldown:
                logger.warning(
                    f"Trade cooldown active: {self.trade_cooldown - time_since_last_trade.total_seconds():.0f}s remaining")
                return False

            # Use accurate position counting
            current_positions = self.trade_manager.get_current_open_positions_count()
            if current_positions >= self.trade_manager.max_open_trades:
                logger.warning(f"Already have {current_positions} open positions for {self.symbol}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking if can open new trade: {e}")
            return False

    def execute_trade(self, signal_type, current_price, stop_distance, reason=""):
        """Execute trade based on signal with proper SL/TP calculation"""
        try:
            # Add cooldown and position check at the beginning
            if not self.can_open_new_trade():
                logger.warning("Trade skipped due to cooldown period or existing position")
                return False

            if not self.trade_manager.can_trade():
                logger.warning("Trade not allowed due to risk limits")
                return False

            if self.consecutive_signals >= self.max_consecutive_signals:
                logger.warning(f"Too many consecutive signals ({self.consecutive_signals}), skipping trade")
                return False

            # Get symbol info for minimum distance requirements
            symbol_info = mt5.symbol_info(self.symbol)
            if not symbol_info:
                logger.error(f"Could not get symbol info for {self.symbol}")
                return False

            # Get current tick data
            tick_info = mt5.symbol_info_tick(self.symbol)
            if not tick_info:
                logger.error(f"Could not get tick data for {self.symbol}")
                return False

            # Calculate proper stop distance based on ATR and broker requirements
            df = self.get_historical_data()
            if df is None:
                logger.error("Could not get historical data for ATR calculation")
                return False

            # Calculate ATR for proper stop distance
            atr = self.geometric_engine.calculate_atr(df['high'], df['low'], df['close'], 14)
            current_atr = float(atr.iloc[-1])

            # Use minimum of 1.5x ATR or broker's minimum stop level
            min_stop_level = getattr(symbol_info, 'trade_stops_level', 10) * symbol_info.point
            calculated_stop = max(current_atr * 1.5, min_stop_level * 2, stop_distance * 2)

            # Ensure stop is at least 8 pips for 5M timeframe
            min_reasonable_stop = 0.0008  # 8 pips for EURUSD on 5M
            final_stop_distance = max(calculated_stop, min_reasonable_stop)

            logger.info(
                f"Stop calculation: ATR={current_atr:.5f}, MinStop={min_stop_level:.5f}, FinalStop={final_stop_distance:.5f}")

            lot_size = self.calculate_position_size(final_stop_distance)

            if signal_type == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                price = float(tick_info.ask)
                sl = float(price - final_stop_distance)
                tp = float(price + (2 * final_stop_distance))  # 1:2 risk-reward ratio
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = float(tick_info.bid)
                sl = float(price + final_stop_distance)
                tp = float(price - (2 * final_stop_distance))  # 1:2 risk-reward ratio

            # Validate stops against broker requirements
            if signal_type == 'BUY':
                if sl >= price - min_stop_level:
                    sl = price - min_stop_level * 2
                    logger.warning(f"Adjusted SL to meet broker requirements: {sl:.5f}")
                if tp <= price + min_stop_level:
                    tp = price + min_stop_level * 2
                    logger.warning(f"Adjusted TP to meet broker requirements: {tp:.5f}")
            else:  # SELL
                if sl <= price + min_stop_level:
                    sl = price + min_stop_level * 2
                    logger.warning(f"Adjusted SL to meet broker requirements: {sl:.5f}")
                if tp >= price - min_stop_level:
                    tp = price - min_stop_level * 2
                    logger.warning(f"Adjusted TP to meet broker requirements: {tp:.5f}")

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
                "comment": f"Bayesian-5M-{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            logger.info(
                f"Sending {signal_type} order: Price={price:.5f}, SL={sl:.5f}, TP={tp:.5f}, StopDistance={final_stop_distance:.5f}")

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Trade execution failed: {result.retcode} - {result.comment}")

                # If stops are invalid, try without stops first
                if result.retcode == 10016:  # Invalid stops
                    logger.warning("Invalid stops detected, attempting without SL/TP...")
                    request.pop('sl', None)
                    request.pop('tp', None)
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info("Trade executed without SL/TP")
                    else:
                        logger.error(f"Trade execution failed even without stops: {result.retcode}")
                        return False
                else:
                    return False
            else:
                logger.info(f"Trade executed successfully: {signal_type} {self.symbol} {lot_size} lots")
                self.last_trade_time = datetime.now()  # Update last trade time

                # Record the trade in our learning system
                trade_data = {
                    'ticket': result.order,
                    'symbol': self.symbol,
                    'direction': signal_type,
                    'entry_time': datetime.now().isoformat(),
                    'entry_price': price,
                    'volume': lot_size,
                    'sl': sl,
                    'tp': tp,
                    'status': 'OPEN',
                    'profit': 0.0
                }

                self.trade_manager.learning_manager.record_trade(trade_data)
                self.consecutive_signals += 1
                return True

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    def force_trade_sync(self):
        """Force complete synchronization"""
        logger.info("Forcing complete trade synchronization with MT5...")
        self.trade_manager.sync_with_mt5_complete()

    def display_system_status(self):
        """Display comprehensive system status in console - FIXED VERSION"""
        try:
            status = self.bayesian_engine.learning_manager.get_system_status()

            print("\n" + "=" * 60)
            print("           TRADING SYSTEM STATUS")
            print("=" * 60)
            print(f"   Trades Analyzed: {status['trades_analyzed']}")
            if status['trades_analyzed'] > 0:
                print(f"   Win Rate: {status['win_rate']:.1%} ({status['winning_trades']}/{status['trades_analyzed']})")
                print(f"   Recent Win Rate: {status['recent_win_rate']:.1%} (last 20 trades)")
                print(f"   Total Profit: ${status['total_profit']:.2f}")
            else:
                print(f"   Win Rate: 0.0% (0/0)")
                print(f"   Recent Win Rate: 0.0% (last 20 trades)")
                print(f"   Total Profit: $0.00")

            print(f"   Learning Points: {status['learning_points']}")

            # Format last updated time nicely
            last_updated = status['last_updated']
            if last_updated != 'Never' and last_updated != 'Error':
                try:
                    last_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    time_diff = datetime.now().astimezone() - last_time
                    hours_ago = time_diff.total_seconds() / 3600
                    last_updated_str = f"{last_time.strftime('%Y-%m-%d %H:%M:%S')} ({hours_ago:.1f}h ago)"
                except:
                    last_updated_str = last_updated
            else:
                last_updated_str = last_updated

            print(f"   Last Updated: {last_updated_str}")

            # Current market status
            df = self.get_historical_data()
            if df is not None and len(df) > 0:
                current_price = float(df['close'].iloc[-1])
                features, nearest_level = self.bayesian_engine.calculate_features(df, self.geometric_engine)
                if features is not None:
                    posterior = self.bayesian_engine.calculate_posterior(features, current_price, nearest_level)
                    print(f"   Current P(Reversion): {posterior:.3f}")
                    print(f"   Current RSI: {features['rsi']:.1f}")

            # Open positions
            open_positions = self.trade_manager.get_current_open_positions_count()
            print(f"   Open Positions: {open_positions}/{self.trade_manager.max_open_trades}")

            # Trade history summary
            print(f"   Trade History: {status['trades_analyzed']} closed, {status['open_trades']} open")

            print("=" * 60)

        except Exception as e:
            logger.error(f"Error displaying system status: {e}")
            print("\n   Error displaying system status")

    def run_trading_cycle(self):
        """IMPROVED: Trading cycle with better statistics"""
        try:
            # Update trade status with improved matching
            self.trade_manager.update_trade_status()

            df = self.get_historical_data()
            if df is None or len(df) == 0:
                logger.error("No data retrieved")
                return False

            current_price = float(df['close'].iloc[-1])
            logger.info(f"Current {self.symbol} price: {current_price:.5f}")

            self.geometric_engine.update_geometric_levels(df)
            self.bayesian_engine.update_prior_from_trend(df)

            # Calculate dynamic stop settings based on current ATR
            atr = float(self.geometric_engine.calculate_atr(df['high'], df['low'], df['close'], 14).iloc[-1])
            self.calculate_dynamic_stop_settings(atr)

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
            logger.info(f"Nearest level: {closest_level:.5f}, Distance: {distance:.5f}, ATR: {atr:.5f}")

            # ENHANCED SIGNAL CONFIRMATION FOR 5M TIMEFRAME
            signal_generated = False
            trade_direction = None

            # Calculate additional confirmation metrics
            price_above_level = current_price > closest_level
            strong_signal = False

            # Check distance filter first
            if distance < (atr * self.min_distance_atr):
                logger.info(f"Signal filtered: distance {distance:.5f} < minimum {atr * self.min_distance_atr:.5f}")
                self.consecutive_signals = 0
                return False

            if posterior_revert > self.min_confidence:
                # Reversion signal - require stronger confirmation for 5M
                if posterior_revert > 0.85:  # Higher threshold for reversion on 5M
                    strong_signal = True
                    logger.info("STRONG reversion signal detected")
                if price_above_level:
                    trade_direction = 'SELL'
                    logger.info(f"SELL signal - Price above level, expecting reversion")
                else:
                    trade_direction = 'BUY'
                    logger.info(f"BUY signal - Price below level, expecting reversion")

            elif posterior_revert < self.max_confidence:
                # Breakout signal - require stronger confirmation for 5M
                if posterior_revert < 0.15:  # Lower threshold for breakout on 5M
                    strong_signal = True
                    logger.info("STRONG breakout signal detected")
                if price_above_level:
                    trade_direction = 'BUY'
                    logger.info(f"BUY signal - Price above level, expecting breakout")
                else:
                    trade_direction = 'SELL'
                    logger.info(f"SELL signal - Price below level, expecting breakout")

            # ENHANCED: Only execute if signal meets confirmation criteria
            if trade_direction:
                # Check momentum alignment for 5M
                rsi = features['rsi']
                momentum_aligned = False

                if trade_direction == 'BUY':
                    momentum_aligned = rsi < 55  # Not overbought for buys on 5M
                else:  # SELL
                    momentum_aligned = rsi > 45  # Not oversold for sells on 5M

                # Check volume confirmation (if available)
                volume_confirm = True
                if 'volume' in df.columns:
                    current_volume = df['volume'].iloc[-1]
                    avg_volume = df['volume'].tail(20).mean()
                    volume_confirm = current_volume > avg_volume * 0.8  # At least 80% of average volume

                # Execute only if:
                # 1. Signal is very strong, OR
                # 2. Signal is moderate AND momentum is aligned AND volume confirms AND we're not in cooldown
                if strong_signal or (momentum_aligned and volume_confirm and self.can_open_new_trade()):
                    signal_generated = self.execute_trade(trade_direction, current_price, distance,
                                                          "StrongReversion" if strong_signal else "ConfirmedSignal")
                    if signal_generated:
                        logger.info(f"Trade executed with enhanced confirmation: {trade_direction}")
                else:
                    logger.info(
                        f"Signal not confirmed: direction={trade_direction}, strong={strong_signal}, momentum_ok={momentum_aligned}, volume_ok={volume_confirm}")
                    self.consecutive_signals = max(0, self.consecutive_signals - 1)  # Decay consecutive signals
            else:
                logger.info("No trade signal - confidence threshold not met")
                self.consecutive_signals = 0

            # Get accurate performance stats
            stats = self.trade_manager.get_performance_stats()
            logger.info(f"\n=== PERFORMANCE SUMMARY ===\n{stats}\n========================")

            return signal_generated

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return False


def main():
    SYMBOL = "EURUSD"
    CYCLE_INTERVAL = 60  # Keep 60-second checks for 5M timeframe
    STATUS_DISPLAY_INTERVAL = 3  # Reduced to 3 cycles for faster status display

    trader = EnhancedTradingEngine(SYMBOL, lot_size=0.1)

    if not trader.initialize_mt5():
        return

    logger.info(f"Starting Enhanced Bayesian-Geometric Trading Bot for {SYMBOL} on 5M timeframe")

    # FORCE COMPREHENSIVE SYNC ON STARTUP
    logger.info("Performing comprehensive trade history sync...")
    trader.force_trade_sync()

    # Display accurate status
    trader.display_system_status()

    try:
        cycle_count = 0
        while True:
            cycle_count += 1
            logger.info(f"--- Trading Cycle {cycle_count} ---")

            trader.run_trading_cycle()

            # Display system status every STATUS_DISPLAY_INTERVAL cycles
            if cycle_count % STATUS_DISPLAY_INTERVAL == 0:
                trader.display_system_status()

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
