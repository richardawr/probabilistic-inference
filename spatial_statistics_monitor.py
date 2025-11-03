# monitor.py
import json
import time
import os
from datetime import datetime
import pandas as pd


def quick_status():
    """Quick system status check"""
    if not os.path.exists('model_performance.json'):
        print("❌ System not fully initialized - run main script first")
        return

    with open('model_performance.json', 'r') as f:
        model_perf = json.load(f)

    with open('learning_data.json', 'r') as f:
        learning_data = json.load(f)

    stats = model_perf.get('summary_stats', {})

    print("🤖 TRADING SYSTEM STATUS")
    print(f"   Trades Analyzed: {stats.get('total_trades_analyzed', 0)}")
    print(f"   Win Rate: {stats.get('overall_win_rate', 0):.1%}")
    print(f"   Learning Points: {len(learning_data['features'])}")
    print(f"   Last Updated: {model_perf.get('last_updated', 'Unknown')}")

    # Check if main script should be running
    print(f"\n💡 Run 'Spatial Statistics.py' continuously for live trading")


if __name__ == "__main__":
    quick_status()
