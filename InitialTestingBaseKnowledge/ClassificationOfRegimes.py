import pandas as pd
import yfinance as yf
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("REGIME EXHAUSTION SYSTEM WITH TRANSITION DATES")
print("=" * 70)

# 1. Get data
print("Getting SPY data...")
spy = yf.download('SPY', period='5y', interval='1d')[['Close']]

# 2. Calculate the 4 KS-validated features (20-day window)
print("\nCalculating KS-validated features...")
window = 20
features = []

for i in range(window, len(spy)):
    returns = np.log(spy['Close'].iloc[i-window:i] / spy['Close'].iloc[i-window:i].shift(1)).dropna()
    if len(returns) < 10:
        continue
    
    features.append({
        'Date': spy.index[i],
        'Close': float(spy['Close'].iloc[i]),
        'Return': float(returns.sum()),
        'Skewness': float(returns.skew()),
        'Kurtosis': float(returns.kurtosis()),
        'Range': float(returns.max() - returns.min())
    })

df = pd.DataFrame(features)
df.set_index('Date', inplace=True)

# 3. Calculate KS-optimized signal
print("\nCalculating KS-optimized exhaustion signal...")

# KS weights from your validation
ks_weights = {
    'Return': 0.2337,
    'Skewness': 0.2126,
    'Kurtosis': 0.2122,
    'Range': 0.2542
}

# Normalize weights
total_weight = sum(ks_weights.values())
for key in ks_weights:
    ks_weights[key] /= total_weight

# Z-score normalize each feature
for feature in ['Return', 'Skewness', 'Kurtosis', 'Range']:
    mean_val = df[feature].mean()
    std_val = df[feature].std()
    df[f'{feature}_Z'] = (df[feature] - mean_val) / std_val if std_val > 0 else 0

# Create KS-weighted score
df['KS_Score'] = 0
for feature, weight in ks_weights.items():
    df['KS_Score'] += df[f'{feature}_Z'] * weight * 100

# 4. Better regime detection using 50-day trend
df['MA_50'] = df['Close'].rolling(50).mean()
df['MA_20'] = df['Close'].rolling(20).mean()
df['Trend'] = np.where(df['MA_20'] > df['MA_50'], 1, -1)  # 1 = uptrend, -1 = downtrend

# Find regime change dates
df['Trend_Change'] = df['Trend'].diff().fillna(0)
regime_change_dates = df[df['Trend_Change'] != 0].index

# Calculate regime age (FIXED VERSION)
df['Regime_Age'] = 0
current_age = 0
current_trend = df.iloc[0]['Trend']

for i in range(len(df)):
    if df.iloc[i]['Trend'] != current_trend:
        current_age = 0
        current_trend = df.iloc[i]['Trend']
    else:
        current_age += 1
    df.iloc[i, df.columns.get_loc('Regime_Age')] = current_age

# 5. Apply fatigue multiplier
max_age = df['Regime_Age'].max()
df['Fatigue_Multiplier'] = 1 + (df['Regime_Age'] / max(1, max_age * 0.5))  # More conservative scaling

# 6. Final exhaustion signal
df['Exhaustion_Signal'] = df['KS_Score'] * df['Fatigue_Multiplier']

# Normalize signal to 0-100 range
min_sig = df['Exhaustion_Signal'].min()
max_sig = df['Exhaustion_Signal'].max()
df['Signal_0_100'] = 100 * (df['Exhaustion_Signal'] - min_sig) / (max_sig - min_sig)

# 7. Identify exhaustion zones
print("\nIdentifying exhaustion zones...")

# Dynamic thresholds
high_exhaustion = df['Signal_0_100'].quantile(0.75)
very_high_exhaustion = df['Signal_0_100'].quantile(0.90)

df['Exhaustion_Level'] = 'Normal'
df.loc[df['Signal_0_100'] >= high_exhaustion, 'Exhaustion_Level'] = 'High'
df.loc[df['Signal_0_100'] >= very_high_exhaustion, 'Exhaustion_Level'] = 'Very High'

# 8. Analyze regime transitions
print("\n" + "=" * 70)
print("REGIME TRANSITION HISTORY")
print("=" * 70)

regime_history = []
if len(regime_change_dates) > 0:
    # Start from first date in dataframe
    start_date = df.index[0]
    start_trend = df.iloc[0]['Trend']
    
    for change_date in regime_change_dates:
        # Calculate regime from start_date to change_date
        regime_mask = (df.index >= start_date) & (df.index < change_date)
        regime_data = df[regime_mask]
        
        if len(regime_data) > 0:
            duration = (change_date - start_date).days
            avg_exhaustion = regime_data['Signal_0_100'].mean()
            high_exhaustion_count = len(regime_data[regime_data['Exhaustion_Level'] != 'Normal'])
            
            regime_history.append({
                'start_date': start_date,
                'end_date': change_date,
                'duration_days': duration,
                'trend': 'Uptrend' if start_trend > 0 else 'Downtrend',
                'avg_exhaustion': avg_exhaustion,
                'high_exhaustion_periods': high_exhaustion_count,
                'regime_change_date': change_date
            })
        
        # Update for next regime
        start_date = change_date
        start_trend = df.loc[change_date]['Trend']
    
    # Last regime (from last change to end)
    regime_mask = (df.index >= start_date)
    regime_data = df[regime_mask]
    if len(regime_data) > 0:
        duration = (df.index[-1] - start_date).days
        avg_exhaustion = regime_data['Signal_0_100'].mean()
        high_exhaustion_count = len(regime_data[regime_data['Exhaustion_Level'] != 'Normal'])
        
        regime_history.append({
            'start_date': start_date,
            'end_date': df.index[-1],
            'duration_days': duration,
            'trend': 'Uptrend' if start_trend > 0 else 'Downtrend',
            'avg_exhaustion': avg_exhaustion,
            'high_exhaustion_periods': high_exhaustion_count,
            'regime_change_date': 'Current'
        })

print(f"\nFound {len(regime_history)} regime periods:")

for i, regime in enumerate(regime_history):
    change_info = f"Changed on {regime['regime_change_date'].strftime('%Y-%m-%d')}" if regime['regime_change_date'] != 'Current' else "Still ongoing"
    
    print(f"\n{i+1}. {regime['trend']} Regime:")
    print(f"   • Period: {regime['start_date'].strftime('%Y-%m-%d')} to {regime['end_date'].strftime('%Y-%m-%d')}")
    print(f"   • Duration: {regime['duration_days']} days")
    print(f"   • {change_info}")
    print(f"   • Avg exhaustion score: {regime['avg_exhaustion']:.1f}")
    print(f"   • High exhaustion periods: {regime['high_exhaustion_periods']}")
    
    # Look for exhaustion before regime change
    if regime['regime_change_date'] != 'Current':
        # Check last 30 days of regime
        lookback_start = regime['end_date'] - pd.Timedelta(days=30)
        if lookback_start < regime['start_date']:
            lookback_start = regime['start_date']
        
        final_period = df.loc[lookback_start:regime['end_date']]
        high_exhaustion_final = final_period[final_period['Exhaustion_Level'] != 'Normal']
        
        if len(high_exhaustion_final) > 0:
            last_exhaustion = high_exhaustion_final.index[-1]
            days_before_change = (regime['end_date'] - last_exhaustion).days
            max_exhaustion = high_exhaustion_final['Signal_0_100'].max()
            print(f"   • Final exhaustion: {days_before_change} days before change (max: {max_exhaustion:.1f})")

# 9. Find the most exhausted periods WITH REGIME CONTEXT
print(f"\n" + "=" * 70)
print("TOP EXHAUSTED PERIODS WITH REGIME CONTEXT")
print("=" * 70)

top_exhausted = df.nlargest(15, 'Signal_0_100')
print(f"\n📊 Showing top 15 most exhausted periods:")

for i, (date, row) in enumerate(top_exhausted.iterrows(), 1):
    # Find which regime this belongs to
    current_regime = None
    for regime in regime_history:
        if regime['start_date'] <= date <= regime['end_date']:
            current_regime = regime
            break
    
    print(f"\n{i}. {date.date()}")
    print(f"   • Exhaustion Score: {row['Signal_0_100']:.1f}/100")
    print(f"   • Regime Age: {row['Regime_Age']} days")
    print(f"   • Fatigue Multiplier: {row['Fatigue_Multiplier']:.2f}x")
    print(f"   • KS Score: {row['KS_Score']:.1f}")
    print(f"   • SPY Price: ${row['Close']:.2f}")
    
    if current_regime:
        print(f"   • Regime: {current_regime['trend']} ({current_regime['start_date'].date()} to {current_regime['end_date'].date()})")
        
        # Days until regime change
        if current_regime['regime_change_date'] != 'Current':
            days_to_change = (current_regime['end_date'] - date).days
            if days_to_change > 0:
                print(f"   • Days until regime change: {days_to_change}")
    
    # What happened next?
    if i < len(df) - 1:
        next_idx = df.index.get_loc(date) + 1
        if next_idx < len(df):
            next_date = df.index[next_idx]
            next_30d_idx = min(len(df) - 1, next_idx + 30)
            price_now = row['Close']
            price_30d = df.iloc[next_30d_idx]['Close']
            return_30d = (price_30d / price_now - 1) * 100
            
            # Check if regime changed within next 60 days
            lookahead_end = min(len(df) - 1, next_idx + 60)
            current_regime_trend = row['Trend']
            regime_changed = False
            for j in range(next_idx, lookahead_end + 1):
                if df.iloc[j]['Trend'] != current_regime_trend:
                    regime_changed = True
                    change_date = df.index[j]
                    days_to_change = (change_date - date).days
                    break
            
            print(f"   • 30-day forward return: {return_30d:+.1f}%")
            if regime_changed:
                print(f"   • Regime changed: YES ({days_to_change} days later on {change_date.date()})")
            else:
                print(f"   • Regime changed within 60 days: NO")

# 10. Long regime analysis (for psychology)
print(f"\n" + "=" * 70)
print("⏱PSYCHOLOGY: LONG REGIME ANALYSIS")
print("=" * 70)

long_regimes = [r for r in regime_history if r['duration_days'] > 100]
print(f"\nFound {len(long_regimes)} regimes longer than 100 days (psychologically significant):")

for i, regime in enumerate(long_regimes, 1):
    # Get the regime's exhaustion pattern
    regime_data = df.loc[regime['start_date']:regime['end_date']]
    high_exhaustion = regime_data[regime_data['Exhaustion_Level'] != 'Normal']
    
    # When did exhaustion peaks occur?
    if len(high_exhaustion) > 0:
        first_exhaustion = high_exhaustion.index[0]
        last_exhaustion = high_exhaustion.index[-1]
        days_to_first = (first_exhaustion - regime['start_date']).days
        days_to_last = (last_exhaustion - regime['start_date']).days
        
        print(f"\n{i}. {regime['trend']} Regime ({regime['duration_days']} days):")
        print(f"   • Period: {regime['start_date'].date()} to {regime['end_date'].date()}")
        print(f"   • Exhaustion periods: {len(high_exhaustion)}")
        print(f"   • First exhaustion: {days_to_first} days into regime")
        print(f"   • Last exhaustion: {days_to_last} days into regime")
        
        if regime['regime_change_date'] != 'Current':
            days_before_change = (regime['end_date'] - last_exhaustion).days
            print(f"   • Final exhaustion signal: {days_before_change} days before regime end")
            
            # Was the regime change preceded by exhaustion?
            if days_before_change <= 30:
                print(f"   • ✅ Exhaustion signaled change in advance")
            else:
                print(f"   • ⚠️  Exhaustion signal too early")

# 11. Current market state with regime context
print(f"\n" + "=" * 70)
print("CURRENT MARKET STATE")
print("=" * 70)

latest = df.iloc[-1]
current_regime = regime_history[-1] if regime_history else None

print(f"\n• Date: {df.index[-1].date()}")
print(f"• Exhaustion Score: {latest['Signal_0_100']:.1f}/100")
print(f"• Regime Age: {latest['Regime_Age']} days")
print(f"• Trend: {'Uptrend' if latest['Trend'] > 0 else 'Downtrend'}")
print(f"• Exhaustion Level: {latest['Exhaustion_Level']}")
print(f"• SPY Price: ${latest['Close']:.2f}")

if current_regime:
    print(f"\nCurrent Regime:")
    print(f"• Type: {current_regime['trend']}")
    print(f"• Started: {current_regime['start_date'].date()}")
    print(f"• Duration: {current_regime['duration_days']} days")
    print(f"• Avg exhaustion: {current_regime['avg_exhaustion']:.1f}")
    print(f"• High exhaustion periods: {current_regime['high_exhaustion_periods']}")

# 12. Trading signals based on regime context
print(f"\n" + "=" * 70)
print("TRADING SIGNALS BASED ON REGIME")
print("=" * 70)

print(f"\nSignal thresholds:")
print(f"• Normal: < {high_exhaustion:.1f}")
print(f"• High: {high_exhaustion:.1f} - {very_high_exhaustion:.1f}")
print(f"• Very High: > {very_high_exhaustion:.1f}")

print(f"\nHigh probability setups:")
print(f"1. Regime > 100 days + Signal > {high_exhaustion:.1f}")
print(f"2. Regime > 200 days + Signal > {very_high_exhaustion:.1f}")
print(f"3. Multiple high signals in aging regime")

# Check current signal
if current_regime and current_regime['duration_days'] > 100:
    if latest['Signal_0_100'] >= very_high_exhaustion:
        print(f"\nCURRENT ALERT: Long regime ({current_regime['duration_days']} days) with VERY HIGH exhaustion!")
    elif latest['Signal_0_100'] >= high_exhaustion:
        print(f"\nCURRENT WARNING: Long regime ({current_regime['duration_days']} days) with high exhaustion")
    else:
        print(f"\nCurrent: Long regime but normal exhaustion levels")
else:
    print(f"\nCurrent: Normal regime conditions")

# 13. Export for charting
print(f"\nData exported to CSV...")
output_df = df[['Close', 'Return', 'KS_Score', 'Regime_Age', 'Fatigue_Multiplier', 
                'Exhaustion_Signal', 'Signal_0_100', 'Exhaustion_Level', 'Trend']]
output_df.to_csv('regime_exhaustion_with_transitions.csv')

# Add regime info to export
regime_info = []
for regime in regime_history:
    regime_info.append({
        'Regime_Start': regime['start_date'],
        'Regime_End': regime['end_date'],
        'Regime_Type': regime['trend'],
        'Duration_Days': regime['duration_days'],
        'Avg_Exhaustion': regime['avg_exhaustion']
    })

regime_df = pd.DataFrame(regime_info)
regime_df.to_csv('regime_history.csv')

print("Analysis complete!")
print("Files saved: 'regime_exhaustion_with_transitions.csv' and 'regime_history.csv'")


"""
Now basically this code is trying to print out regime transitions using
thresholds of exhaustion levels. It printed out a list of dates, that unsually
coincides with catalyst in SPY news.

Examples:

Performance Summary Table
Date	SPY Closing Key Market Catalyst
2023-04-13	$413.40	Cooling PPI data (Inflation)
2023-08-28	$442.70	Rebound after Fed's Jackson Hole speech
2023-11-20	$454.26	Rate hike pause optimism
2024-04-26	$508.19	Strong tech earnings (Alphabet/Microsoft)
2024-05-20	$530.08	AI-led rally; new record highs
2024-08-09	$532.99	Recovery from early August yen carry trade volatility
2024-09-03	$552.17	Growth scare; weak ISM Manufacturing data
2025-01-14	$582.06	Earnings season kickoff & policy speculation
2025-02-04	$601.66	Resilience milestone; record-breaking momentum

BUT, this code definitely does not predict major news outbreak, 
such as geopolitical events or sudden economic shifts. Instead,
it predicts that the market will suffer a market regime change
in case of any sudden news due to an accumulation of exhaustion
in the system
"""