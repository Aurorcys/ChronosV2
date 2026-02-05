"""
This code below shows the Kurtosis and Skewness thresholds for regime transitions
"""

change_dates = df[df['Regime_Change'] == 1].index

print(f"📊 Found {len(change_dates)} GMM regime changes")

# Look at signals in the 5 days BEFORE each change
window = 5
stats = []

for change_date in change_dates:
    # Get the index position
    change_idx = df.index.get_loc(change_date)
    
    # Get previous window days (skipping weekends automatically)
    start_idx = max(0, change_idx - window)
    before_indices = df.index[start_idx:change_idx]
    
    if len(before_indices) > 0:
        # Get old regime (day before)
        old_regime = df.iloc[change_idx - 1]['Regime'] if change_idx > 0 else df.iloc[change_idx]['Regime']
        new_regime = df.iloc[change_idx]['Regime']
        
        # Get features before change
        before_skew = df.loc[before_indices, 'Skewness']
        before_kurt = df.loc[before_indices, 'Kurtosis']
        
        stats.append({
            'date': change_date,
            'old_regime': old_regime,
            'new_regime': new_regime,
            'avg_skew_before': before_skew.mean(),
            'avg_kurt_before': before_kurt.mean(),
            'skew_range': before_skew.max() - before_skew.min()
        })

# Show results
if stats:
    stats_df = pd.DataFrame(stats)
    
    print(f"\n📈 SKEWNESS before regime changes ({len(stats_df)} transitions):")
    print(f"• Average: {stats_df['avg_skew_before'].mean():.2f}")
    print(f"• Min: {stats_df['avg_skew_before'].min():.2f}")
    print(f"• Max: {stats_df['avg_skew_before'].max():.2f}")
    
    print(f"\n📈 KURTOSIS before regime changes:")
    print(f"• Average: {stats_df['avg_kurt_before'].mean():.2f}")
    print(f"• Min: {stats_df['avg_kurt_before'].min():.2f}")
    print(f"• Max: {stats_df['avg_kurt_before'].max():.2f}")
    
    print(f"\n🎯 REGIME 0 → 1 transitions:")
    transitions_0_to_1 = stats_df[stats_df['old_regime'] == 0]
    if len(transitions_0_to_1) > 0:
        print(f"• Count: {len(transitions_0_to_1)}")
        print(f"• Avg Skewness before: {transitions_0_to_1['avg_skew_before'].mean():.2f}")
        print(f"• Avg Kurtosis before: {transitions_0_to_1['avg_kurt_before'].mean():.2f}")
    
    print(f"\n🎯 REGIME 1 → 0 transitions:")
    transitions_1_to_0 = stats_df[stats_df['old_regime'] == 1]
    if len(transitions_1_to_0) > 0:
        print(f"• Count: {len(transitions_1_to_0)}")
        print(f"• Avg Skewness before: {transitions_1_to_0['avg_skew_before'].mean():.2f}")
        print(f"• Avg Kurtosis before: {transitions_1_to_0['avg_kurt_before'].mean():.2f}")
    
else:
    print("No transitions found with signal data")

"""
🎯 REGIME 0 → 1 transitions:
• Count: 35
• Avg Skewness before: -0.28
• Avg Kurtosis before: 0.66

🎯 REGIME 1 → 0 transitions:
• Count: 35
• Avg Skewness before: -0.30
• Avg Kurtosis before: 1.80
"""

"""
However, this code below shows that previous useful thresholds disappear in
usefulness in recent periods due to increased geopolitical tension, market volatility, 
and changing investor behavior in the new social media market.

"""

print("\n" + "=" * 80)
print("🎯 REFINED SIGNAL WITH LEARNED PATTERNS")
print("=" * 80)

# 1. Define BETTER thresholds based on patterns
SKEW_MIN = -2.0  # Don't trade extreme skew (Dec 2024, Oct 2025 failures)
SKEW_MAX = -0.4  # Minimum skew to care about
KURT_MIN = 1.5   # Same as before
KURT_MAX = 7.0   # Don't trade extreme kurtosis

# 2. Only trade in Regime 1 (bullish regimes ending)
df['Signal_Refined'] = ((df['Skewness'] < SKEW_MAX) & 
                        (df['Skewness'] > SKEW_MIN) &  # New: avoid extremes
                        (df['Kurtosis'] > KURT_MIN) & 
                        (df['Kurtosis'] < KURT_MAX) &  # New: avoid extremes
                        (df['Regime'] == 1)).astype(int)

# 3. Add regime age filter (minimum 10 days old)
if 'Regime_Age' not in df.columns:
    df['Regime_Age'] = df.groupby('Regime').cumcount()
    
df['Signal_Refined'] = df['Signal_Refined'] & (df['Regime_Age'] > 10)

# 4. Calculate refined probability
LOOKAHEAD_DAYS = 5

refined_signals_idx = df[df['Signal_Refined'] == 1].index.tolist()
refined_total = len(refined_signals_idx)

refined_successes = 0
refined_details = []

for signal_date in refined_signals_idx:
    signal_idx = df.index.get_loc(signal_date)
    
    # Check next LOOKAHEAD_DAYS for regime change
    lookahead_end = min(len(df), signal_idx + LOOKAHEAD_DAYS + 1)
    regime_change_ahead = False
    change_date = None
    
    for j in range(signal_idx + 1, lookahead_end):
        if df.iloc[j]['Regime_Change'] == 1:
            regime_change_ahead = True
            change_date = df.index[j]
            break
    
    if regime_change_ahead:
        refined_successes += 1
    
    refined_details.append({
        'date': signal_date,
        'success': regime_change_ahead,
        'change_date': change_date if regime_change_ahead else None,
        'skew': df.loc[signal_date, 'Skewness'],
        'kurt': df.loc[signal_date, 'Kurtosis'],
        'regime': df.loc[signal_date, 'Regime'],
        'regime_age': df.loc[signal_date, 'Regime_Age']
    })

# 5. Calculate refined probabilities
if refined_total > 0:
    refined_hit_rate = refined_successes / refined_total
    
    # Baseline (same as before)
    baseline_prob = min(1.0, (df['Regime_Change'].sum() / len(df)) * LOOKAHEAD_DAYS)
    refined_edge = refined_hit_rate / baseline_prob if baseline_prob > 0 else 0
    
    print(f"\n📊 REFINED SIGNAL RESULTS:")
    print(f"• Total refined signals: {refined_total}")
    print(f"• Successes: {refined_successes}")
    print(f"• Failures: {refined_total - refined_successes}")
    print(f"• Hit Rate: {refined_hit_rate:.1%}")
    print(f"• Edge over baseline: {refined_edge:.1f}x")
    
    # Compare with original
    original_signals = df['Signal'].sum()
    original_success_rate = 0.514  # From previous results
    
    if original_signals > 0:
        improvement = refined_hit_rate - original_success_rate
        print(f"\n📈 IMPROVEMENT OVER ORIGINAL:")
        print(f"• Original signals: {original_signals}")
        print(f"• Original hit rate: {original_success_rate:.1%}")
        print(f"• Refined hit rate: {refined_hit_rate:.1%}")
        print(f"• Improvement: {improvement:+.1%}")
    
    # Show signal characteristics
    refined_df = pd.DataFrame(refined_details)
    if len(refined_df) > 0:
        print(f"\n📊 REFINED SIGNAL STATS:")
        print(f"• Avg Skewness: {refined_df['skew'].mean():.2f}")
        print(f"• Avg Kurtosis: {refined_df['kurt'].mean():.2f}")
        print(f"• Avg Regime Age: {refined_df['regime_age'].mean():.0f} days")
        
        # Show strongest refined signals
        print(f"\n🎯 TOP 5 REFINED SIGNALS (by regime age + kurtosis):")
        refined_df['strength_score'] = refined_df['regime_age'] * refined_df['kurt']
        top_refined = refined_df
        for _, row in top_refined.iterrows():
            outcome = "✅ HIT" if row['success'] else "❌ MISS"
            change_info = f" → {row['change_date'].date()}" if row['success'] else ""
            print(f"  {row['date'].date()}: Skew={row['skew']:.2f}, Kurt={row['kurt']:.2f}, Age={row['regime_age']}d ({outcome}{change_info})")
    
    # Trading implications
    print(f"\n💡 TRADING IMPLICATIONS (REFINED):")
    if refined_edge > 2.0:
        print(f"• 🚀 EXCELLENT EDGE: {refined_edge:.1f}x better than random")
        print(f"• Suggestion: 2-3% risk per trade")
    elif refined_edge > 1.5:
        print(f"• ✅ GOOD EDGE: {refined_edge:.1f}x better than random")
        print(f"• Suggestion: 1-2% risk per trade")
    else:
        print(f"• ⚠️ MODEST EDGE: {refined_edge:.1f}x better than random")
        print(f"• Suggestion: <1% risk, needs confirmation")
    
else:
    print(f"⚠️ No refined signals found!")

print(f"\n" + "=" * 80)
print("🔍 ANALYZING FAILURE PATTERNS")
print("=" * 80)

# 6. Analyze why original signals failed
all_signals_df = pd.DataFrame(signal_details)  # From previous code
failed_signals = all_signals_df[all_signals_df['success'] == False]

if len(failed_signals) > 0:
    print(f"\n📉 ANALYSIS OF {len(failed_signals)} FAILED SIGNALS:")
    
    # Check if failures are from extreme values
    extreme_failures = failed_signals[
        (failed_signals['skew'] < -2.0) | 
        (failed_signals['kurt'] > 7.0)
    ]
    
    print(f"• Failed due to extreme values: {len(extreme_failures)} ({len(extreme_failures)/len(failed_signals):.0%})")
    print(f"• Avg Skew of failures: {failed_signals['skew'].mean():.2f}")
    print(f"• Avg Kurt of failures: {failed_signals['kurt'].mean():.2f}")
    
    # Check if failures are from young regimes
    if 'regime_age' in failed_signals.columns:
        young_regime_failures = failed_signals[failed_signals['regime_age'] < 20]
        print(f"• Failed in young regimes (<20 days): {len(young_regime_failures)}")
    
    # Show most recent failures
    print(f"\n🎯 MOST RECENT FAILURES (last 3):")
    recent_failures = failed_signals.tail(3)
    for _, row in recent_failures.iterrows():
        print(f"  {row['date'].date()}: Skew={row['skew']:.2f}, Kurt={row['kurt']:.2f}")

print(f"\n" + "=" * 80)
print("✅ REFINED ANALYSIS COMPLETE")
print("=" * 80)

"""
2021-07-02: Skew=-0.40, Kurt=2.59, Age=11d (✅ HIT → 2021-07-12)
  2021-07-06: Skew=-0.45, Kurt=2.85, Age=12d (✅ HIT → 2021-07-12)
  2021-07-07: Skew=-0.42, Kurt=2.73, Age=13d (✅ HIT → 2021-07-12)
  2021-07-08: Skew=-0.52, Kurt=2.75, Age=14d (✅ HIT → 2021-07-12)
  2021-07-09: Skew=-0.49, Kurt=1.64, Age=15d (✅ HIT → 2021-07-12)
  2021-08-16: Skew=-0.49, Kurt=1.52, Age=16d (✅ HIT → 2021-08-17)
  2021-11-30: Skew=-1.87, Kurt=5.99, Age=22d (✅ HIT → 2021-12-02)
  2021-12-01: Skew=-1.52, Kurt=2.65, Age=23d (✅ HIT → 2021-12-02)
  2022-08-29: Skew=-0.71, Kurt=1.58, Age=24d (✅ HIT → 2022-08-30)
  2022-09-21: Skew=-1.13, Kurt=1.70, Age=25d (✅ HIT → 2022-09-22)
  2022-09-27: Skew=-0.78, Kurt=2.25, Age=26d (✅ HIT → 2022-09-30)
  2022-09-28: Skew=-0.82, Kurt=2.30, Age=27d (✅ HIT → 2022-09-30)
  2022-09-29: Skew=-0.76, Kurt=1.78, Age=28d (✅ HIT → 2022-09-30)
  2023-12-21: Skew=-0.77, Kurt=2.38, Age=61d (❌ MISS)
  2023-12-22: Skew=-0.72, Kurt=1.84, Age=62d (❌ MISS)
  2023-12-26: Skew=-0.76, Kurt=1.92, Age=63d (✅ HIT → 2024-01-03)
  2023-12-27: Skew=-0.94, Kurt=2.35, Age=64d (✅ HIT → 2024-01-03)
  2023-12-28: Skew=-0.96, Kurt=2.41, Age=65d (✅ HIT → 2024-01-03)
  2023-12-29: Skew=-1.00, Kurt=2.56, Age=66d (✅ HIT → 2024-01-03)
  2024-01-02: Skew=-0.80, Kurt=1.90, Age=67d (✅ HIT → 2024-01-03)
  2024-02-12: Skew=-0.81, Kurt=1.50, Age=69d (✅ HIT → 2024-02-13)
  2024-05-21: Skew=-0.76, Kurt=1.54, Age=72d (✅ HIT → 2024-05-24)
  2024-05-22: Skew=-0.78, Kurt=2.06, Age=73d (✅ HIT → 2024-05-24)
  2024-05-23: Skew=-0.71, Kurt=1.76, Age=74d (✅ HIT → 2024-05-24)
  2024-07-18: Skew=-1.13, Kurt=1.75, Age=76d (✅ HIT → 2024-07-19)
  2024-07-25: Skew=-1.21, Kurt=1.57, Age=77d (✅ HIT → 2024-07-26)
  2024-09-03: Skew=-0.93, Kurt=2.83, Age=78d (✅ HIT → 2024-09-04)
  2024-09-27: Skew=-0.96, Kurt=1.79, Age=79d (❌ MISS)
  2024-09-30: Skew=-0.92, Kurt=1.71, Age=80d (✅ HIT → 2024-10-07)
  2024-10-01: Skew=-0.93, Kurt=1.96, Age=81d (✅ HIT → 2024-10-07)
  2024-10-03: Skew=-0.43, Kurt=1.70, Age=83d (✅ HIT → 2024-10-07)
  2024-10-04: Skew=-0.44, Kurt=1.75, Age=84d (✅ HIT → 2024-10-07)
  2024-11-01: Skew=-1.09, Kurt=1.53, Age=85d (❌ MISS)
  2024-11-04: Skew=-1.20, Kurt=1.86, Age=86d (❌ MISS)
  2024-11-05: Skew=-1.40, Kurt=2.99, Age=87d (❌ MISS)
  2024-11-06: Skew=-1.19, Kurt=2.73, Age=88d (❌ MISS)
  2024-12-06: Skew=-1.38, Kurt=2.40, Age=109d (✅ HIT → 2024-12-11)
  2024-12-09: Skew=-1.53, Kurt=2.91, Age=110d (✅ HIT → 2024-12-11)
  2024-12-10: Skew=-1.23, Kurt=1.77, Age=111d (✅ HIT → 2024-12-11)
  2024-12-16: Skew=-1.11, Kurt=1.57, Age=112d (✅ HIT → 2024-12-17)
  2024-12-30: Skew=-1.89, Kurt=5.76, Age=119d (❌ MISS)
  2024-12-31: Skew=-1.57, Kurt=4.29, Age=120d (❌ MISS)
  2025-01-02: Skew=-1.48, Kurt=4.10, Age=121d (❌ MISS)
  2025-01-03: Skew=-1.43, Kurt=4.01, Age=122d (✅ HIT → 2025-01-13)
  2025-01-06: Skew=-1.24, Kurt=3.39, Age=123d (✅ HIT → 2025-01-13)
  2025-01-07: Skew=-1.30, Kurt=3.30, Age=124d (✅ HIT → 2025-01-13)
  2025-01-08: Skew=-1.05, Kurt=2.34, Age=125d (✅ HIT → 2025-01-13)
  2025-01-10: Skew=-1.16, Kurt=2.57, Age=126d (✅ HIT → 2025-01-13)
  2025-01-14: Skew=-0.96, Kurt=1.65, Age=127d (✅ HIT → 2025-01-16)
  2025-01-15: Skew=-1.06, Kurt=1.81, Age=128d (✅ HIT → 2025-01-16)
  2025-04-04: Skew=-1.31, Kurt=2.56, Age=129d (✅ HIT → 2025-04-07)
  2025-04-08: Skew=-1.64, Kurt=2.90, Age=130d (❌ MISS)
  2025-04-09: Skew=-1.55, Kurt=2.60, Age=131d (❌ MISS)
  2025-05-09: Skew=-1.24, Kurt=1.97, Age=152d (✅ HIT → 2025-05-12)
  2025-08-04: Skew=-1.22, Kurt=3.07, Age=172d (❌ MISS)
  2025-08-05: Skew=-0.47, Kurt=3.26, Age=173d (❌ MISS)
  2025-08-11: Skew=-0.45, Kurt=2.24, Age=177d (❌ MISS)
  2025-08-13: Skew=-0.48, Kurt=1.81, Age=179d (✅ HIT → 2025-08-20)
  2025-08-14: Skew=-0.49, Kurt=1.81, Age=180d (✅ HIT → 2025-08-20)
  2025-10-17: Skew=-1.76, Kurt=6.19, Age=190d (❌ MISS)
  2025-10-20: Skew=-1.75, Kurt=6.12, Age=191d (❌ MISS)
  2025-10-21: Skew=-1.61, Kurt=5.40, Age=192d (❌ MISS)
  2025-10-22: Skew=-1.76, Kurt=6.12, Age=193d (❌ MISS)
  2025-10-23: Skew=-1.70, Kurt=5.77, Age=194d (❌ MISS)
  2025-10-24: Skew=-1.88, Kurt=6.35, Age=195d (❌ MISS)
  2025-10-27: Skew=-1.85, Kurt=6.11, Age=196d (❌ MISS)
  2025-10-28: Skew=-1.73, Kurt=5.36, Age=197d (❌ MISS)
  2025-10-29: Skew=-1.72, Kurt=5.35, Age=198d (❌ MISS)
  2025-10-30: Skew=-1.67, Kurt=5.24, Age=199d (❌ MISS)
  2025-10-31: Skew=-1.37, Kurt=3.49, Age=200d (❌ MISS)
  2025-11-03: Skew=-1.42, Kurt=3.57, Age=201d (✅ HIT → 2025-11-10)
  2025-11-04: Skew=-1.40, Kurt=3.56, Age=202d (✅ HIT → 2025-11-10)
  2025-11-05: Skew=-1.22, Kurt=2.44, Age=203d (✅ HIT → 2025-11-10)
  2025-11-06: Skew=-1.21, Kurt=2.50, Age=204d (✅ HIT → 2025-11-10)
  2025-11-07: Skew=-1.06, Kurt=1.70, Age=205d (✅ HIT → 2025-11-10)
  2026-01-21: Skew=-1.64, Kurt=4.36, Age=206d (❌ MISS)
  2026-01-22: Skew=-1.42, Kurt=3.97, Age=207d (❌ MISS)
  2026-01-23: Skew=-1.44, Kurt=4.09, Age=208d (❌ MISS)
  2026-01-26: Skew=-1.40, Kurt=4.20, Age=209d (❌ MISS)
  2026-01-27: Skew=-1.39, Kurt=4.06, Age=210d (❌ MISS)
  2026-01-28: Skew=-1.45, Kurt=4.04, Age=211d (❌ MISS)
  2026-01-29: Skew=-1.57, Kurt=4.51, Age=212d (❌ MISS)
  2026-01-30: Skew=-1.54, Kurt=4.41, Age=213d (❌ MISS)
  2026-02-02: Skew=-1.71, Kurt=5.60, Age=214d (❌ MISS)
  2026-02-03: Skew=-1.72, Kurt=5.45, Age=215d (❌ MISS)
"""

"""
I then ran a code to see whether Kurtosis or Skewness, even had
predictive power in the recent year.
"""

print("\n📊 CORRELATION CHECK (2024+):")
recent = df[df.index >= '2024-01-01']

# Check correlation between skew/kurtosis and future regime changes
from scipy.stats import pointbiserialr

# Create future change indicator (regime change in next 5 days)
recent['Future_Change'] = 0
for i in range(len(recent) - 5):
    if any(recent.iloc[i+1:i+6]['Regime_Change'] == 1):
        recent.iloc[i, recent.columns.get_loc('Future_Change')] = 1

# Calculate correlations
skew_corr, skew_p = pointbiserialr(recent['Future_Change'], recent['Skewness'])
kurt_corr, kurt_p = pointbiserialr(recent['Future_Change'], recent['Kurtosis'])

print(f"Skewness correlation with future changes: {skew_corr:.3f} (p={skew_p:.3f})")
print(f"Kurtosis correlation with future changes: {kurt_corr:.3f} (p={kurt_p:.3f})")

if skew_p > 0.05 and kurt_p > 0.05:
    print("🚨 NO SIGNIFICANT CORRELATION! The relationship is GONE.")
else:
    print("✅ Still some correlation remains.")

"""
📊 CORRELATION CHECK (2024+):
Skewness correlation with future changes: -0.150 (p=0.001)
Kurtosis correlation with future changes: 0.048 (p=0.276)
✅ Still some correlation remains.
"""

"""
This shows that due to the extremeness in the current market, Kurtosis
cannot be used as a predictive feature due to the change in the new market
, which caused a lack of significant correlation with future regime changes.
Therefore, Kurtosis must be eliminated.
"""