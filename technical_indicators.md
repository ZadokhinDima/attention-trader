# Technical Indicators for Model Training

This document describes technical indicators that can be computed from OHLCV (Open, High, Low, Close, Volume) data to enhance feature engineering for the attention-based neural network model.

## Table of Contents

1. [Trend Indicators](#1-trend-indicators)
2. [Momentum Indicators](#2-momentum-indicators)
3. [Volatility Indicators](#3-volatility-indicators)
4. [Volume Indicators](#4-volume-indicators)
5. [Price Transform Features](#5-price-transform-features)
6. [Statistical Features](#6-statistical-features)
7. [Implementation Libraries](#7-implementation-libraries)
8. [Recommended Feature Set](#8-recommended-feature-set)

---

## 1. Trend Indicators

Trend indicators help identify the direction and strength of price movements.

### 1.1 Simple Moving Average (SMA)

**Description**: The arithmetic mean of prices over a specified period.

**Formula**:
```
SMA(n) = (P₁ + P₂ + ... + Pₙ) / n
```

**Common Periods**: 5, 10, 20, 50, 100, 200 days

**Use Case**:
- Smooths price data to identify trend direction
- Crossovers (e.g., SMA(50) crossing SMA(200)) signal trend changes
- Price above SMA indicates bullish trend, below indicates bearish

**Model Benefit**: Captures multi-scale trend information; attention mechanism can learn which timeframes are most relevant.

---

### 1.2 Exponential Moving Average (EMA)

**Description**: A weighted moving average that gives more weight to recent prices.

**Formula**:
```
EMA(t) = α × P(t) + (1 - α) × EMA(t-1)
where α = 2 / (n + 1)
```

**Common Periods**: 9, 12, 21, 26, 50, 200 days

**Use Case**:
- More responsive to recent price changes than SMA
- Used in MACD calculation
- Better for short-term trend detection

**Model Benefit**: Provides recency-weighted trend information that complements SMA features.

---

### 1.3 Moving Average Convergence Divergence (MACD)

**Description**: Shows the relationship between two EMAs, consisting of three components.

**Components**:
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

**Use Case**:
- MACD crossing above signal line = bullish
- MACD crossing below signal line = bearish
- Histogram shows momentum strength
- Divergence from price indicates potential reversals

**Model Benefit**: Encodes momentum and trend information in a compact form. All three components should be included as features.

---

### 1.4 Average Directional Index (ADX)

**Description**: Measures trend strength regardless of direction.

**Components**:
```
+DI (Positive Directional Indicator)
-DI (Negative Directional Indicator)
ADX = Smoothed average of |+DI - -DI| / (+DI + -DI)
```

**Common Period**: 14 days

**Interpretation**:
- ADX > 25: Strong trend
- ADX < 20: Weak trend or ranging market
- +DI > -DI: Bullish trend
- -DI > +DI: Bearish trend

**Model Benefit**: Provides trend strength context; helps model distinguish between trending and ranging markets.

---

### 1.5 Parabolic SAR (Stop and Reverse)

**Description**: Provides potential entry and exit points based on price and time.

**Formula**: Uses acceleration factor (AF) and extreme point (EP).

**Use Case**:
- Dots below price = bullish trend
- Dots above price = bearish trend
- Reversal when price crosses SAR

**Model Benefit**: Binary trend signal that can be encoded as position relative to price.

---

### 1.6 Ichimoku Cloud

**Description**: A comprehensive indicator showing support/resistance, trend direction, and momentum.

**Components**:
```
Tenkan-sen (Conversion Line) = (9-period high + 9-period low) / 2
Kijun-sen (Base Line) = (26-period high + 26-period low) / 2
Senkou Span A = (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
Senkou Span B = (52-period high + 52-period low) / 2, plotted 26 periods ahead
Chikou Span = Close, plotted 26 periods behind
```

**Model Benefit**: Rich multi-timeframe feature set. Cloud thickness indicates volatility; price relative to cloud indicates trend.

---

## 2. Momentum Indicators

Momentum indicators measure the speed and magnitude of price changes.

### 2.1 Relative Strength Index (RSI)

**Description**: Measures the speed and change of price movements on a scale of 0-100.

**Formula**:
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss (over n periods)
```

**Common Period**: 14 days

**Interpretation**:
- RSI > 70: Overbought (potential sell signal)
- RSI < 30: Oversold (potential buy signal)
- Divergence from price indicates weakening momentum

**Model Benefit**: Bounded indicator (0-100) that normalizes well; captures overbought/oversold conditions.

---

### 2.2 Stochastic Oscillator

**Description**: Compares closing price to price range over a period.

**Formula**:
```
%K = 100 × (Close - Lowest Low) / (Highest High - Lowest Low)
%D = SMA(3) of %K (signal line)
```

**Common Period**: 14 days for %K

**Interpretation**:
- %K > 80: Overbought
- %K < 20: Oversold
- %K crossing %D signals potential reversals

**Model Benefit**: Both %K and %D should be included; captures price position within recent range.

---

### 2.3 Stochastic RSI

**Description**: Applies the Stochastic formula to RSI values instead of price.

**Formula**:
```
StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)
```

**Use Case**: More sensitive than standard RSI, useful for identifying shorter-term overbought/oversold conditions.

**Model Benefit**: Provides momentum-of-momentum information.

---

### 2.4 Commodity Channel Index (CCI)

**Description**: Measures price deviation from statistical mean.

**Formula**:
```
CCI = (Typical Price - SMA(TP)) / (0.015 × Mean Deviation)
Typical Price (TP) = (High + Low + Close) / 3
```

**Common Period**: 20 days

**Interpretation**:
- CCI > +100: Overbought / strong uptrend
- CCI < -100: Oversold / strong downtrend

**Model Benefit**: Unbounded indicator that captures extreme price movements.

---

### 2.5 Williams %R

**Description**: Similar to Stochastic but inverted scale.

**Formula**:
```
%R = (Highest High - Close) / (Highest High - Lowest Low) × -100
```

**Common Period**: 14 days

**Interpretation**:
- %R > -20: Overbought
- %R < -80: Oversold

**Model Benefit**: Alternative momentum perspective; ranges from -100 to 0.

---

### 2.6 Rate of Change (ROC) / Momentum

**Description**: Measures percentage change over a period.

**Formula**:
```
ROC = ((Close - Close(n periods ago)) / Close(n periods ago)) × 100
Momentum = Close - Close(n periods ago)
```

**Common Periods**: 1, 5, 10, 20, 60 days

**Model Benefit**: Multiple ROC periods capture momentum at different timescales; essential for multi-horizon prediction.

---

### 2.7 Ultimate Oscillator

**Description**: Combines short, medium, and long-term momentum into one indicator.

**Formula**: Weighted average of buying pressure across 7, 14, and 28 periods.

**Interpretation**: Ranges 0-100; similar to RSI interpretation.

**Model Benefit**: Pre-combines multiple timeframes; reduces feature dimensionality.

---

### 2.8 Awesome Oscillator (AO)

**Description**: Measures market momentum using simple moving averages.

**Formula**:
```
AO = SMA(5) of Median Price - SMA(34) of Median Price
Median Price = (High + Low) / 2
```

**Model Benefit**: Histogram-style momentum indicator; captures mid-term momentum shifts.

---

## 3. Volatility Indicators

Volatility indicators measure the degree of price variation.

### 3.1 Bollinger Bands

**Description**: Price envelope based on standard deviation from a moving average.

**Formula**:
```
Middle Band = SMA(20)
Upper Band = SMA(20) + 2 × σ(20)
Lower Band = SMA(20) - 2 × σ(20)
```

**Derived Features**:
```
%B = (Close - Lower Band) / (Upper Band - Lower Band)
Bandwidth = (Upper Band - Lower Band) / Middle Band
```

**Model Benefit**:
- %B normalizes price position within bands (0-1 typically)
- Bandwidth captures volatility expansion/contraction
- Band squeeze often precedes breakouts

---

### 3.2 Average True Range (ATR)

**Description**: Measures market volatility by decomposing the entire range of a price.

**Formula**:
```
True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
ATR = EMA(14) of True Range
```

**Common Period**: 14 days

**Use Case**:
- Position sizing (risk management)
- Stop-loss placement
- Volatility-adjusted comparisons

**Model Benefit**: Essential volatility feature; can be used to normalize price-based features.

---

### 3.3 ATR Percentage (ATRP)

**Description**: ATR as a percentage of closing price.

**Formula**:
```
ATRP = (ATR / Close) × 100
```

**Model Benefit**: Allows volatility comparison across assets with different price levels.

---

### 3.4 Keltner Channel

**Description**: Volatility-based envelope set above and below an EMA.

**Formula**:
```
Middle Line = EMA(20)
Upper Channel = EMA(20) + 2 × ATR(10)
Lower Channel = EMA(20) - 2 × ATR(10)
```

**Model Benefit**: ATR-based (vs. standard deviation for Bollinger); smoother channel boundaries.

---

### 3.5 Donchian Channel

**Description**: Price channel based on highest high and lowest low.

**Formula**:
```
Upper Channel = Highest High over n periods
Lower Channel = Lowest Low over n periods
Middle Channel = (Upper + Lower) / 2
```

**Common Period**: 20 days

**Model Benefit**: Simple breakout detection; price position within channel indicates trend.

---

### 3.6 Historical Volatility (HV)

**Description**: Standard deviation of log returns, annualized.

**Formula**:
```
Log Returns = ln(Close / Previous Close)
HV = std(Log Returns) × √252
```

**Common Periods**: 10, 20, 30, 60 days

**Model Benefit**: Multiple HV periods show volatility regime changes; fundamental risk measure.

---

### 3.7 Chaikin Volatility

**Description**: Measures volatility using the spread between high and low prices.

**Formula**:
```
HL_EMA = EMA(10) of (High - Low)
Chaikin Volatility = ((HL_EMA - HL_EMA(10 days ago)) / HL_EMA(10 days ago)) × 100
```

**Model Benefit**: Captures volatility changes rather than absolute levels.

---

## 4. Volume Indicators

Volume indicators analyze trading volume to confirm price movements.

### 4.1 On-Balance Volume (OBV)

**Description**: Cumulative volume indicator that adds/subtracts volume based on price direction.

**Formula**:
```
If Close > Previous Close: OBV = Previous OBV + Volume
If Close < Previous Close: OBV = Previous OBV - Volume
If Close = Previous Close: OBV = Previous OBV
```

**Model Benefit**: Divergence between OBV trend and price trend signals potential reversals. Use OBV changes or normalized OBV.

---

### 4.2 Volume Weighted Average Price (VWAP)

**Description**: Average price weighted by volume, typically calculated intraday.

**Formula**:
```
VWAP = Σ(Typical Price × Volume) / Σ(Volume)
```

**Use Case**: Institutional benchmark; price above VWAP = bullish, below = bearish.

**Model Benefit**: Price deviation from VWAP indicates value relative to volume-weighted mean.

---

### 4.3 Accumulation/Distribution Line (A/D)

**Description**: Cumulative indicator using price and volume to assess supply/demand.

**Formula**:
```
Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
Money Flow Volume = MFM × Volume
A/D Line = Previous A/D + Money Flow Volume
```

**Model Benefit**: Measures buying/selling pressure; divergence from price indicates potential reversal.

---

### 4.4 Chaikin Money Flow (CMF)

**Description**: Oscillator version of A/D Line over a specified period.

**Formula**:
```
CMF = Σ(Money Flow Volume over n periods) / Σ(Volume over n periods)
```

**Common Period**: 20 days

**Interpretation**:
- CMF > 0: Buying pressure
- CMF < 0: Selling pressure

**Model Benefit**: Bounded indicator (-1 to +1) that normalizes well.

---

### 4.5 Money Flow Index (MFI)

**Description**: Volume-weighted RSI; incorporates both price and volume.

**Formula**:
```
Typical Price = (High + Low + Close) / 3
Raw Money Flow = Typical Price × Volume
Money Flow Ratio = Positive Money Flow / Negative Money Flow
MFI = 100 - (100 / (1 + Money Flow Ratio))
```

**Common Period**: 14 days

**Interpretation**: Similar to RSI (0-100 scale, 80/20 thresholds).

**Model Benefit**: Combines price momentum with volume confirmation.

---

### 4.6 Volume Rate of Change (VROC)

**Description**: Percentage change in volume over a period.

**Formula**:
```
VROC = ((Volume - Volume(n periods ago)) / Volume(n periods ago)) × 100
```

**Model Benefit**: Detects unusual volume activity that may precede price moves.

---

### 4.7 Force Index

**Description**: Combines price change, direction, and volume.

**Formula**:
```
Force Index = (Close - Previous Close) × Volume
EMA(13) of Force Index for smoothing
```

**Model Benefit**: Captures strength of price moves confirmed by volume.

---

### 4.8 Ease of Movement (EMV)

**Description**: Relates price change to volume, showing how easily price moves.

**Formula**:
```
Distance Moved = ((High + Low) / 2) - ((Previous High + Previous Low) / 2)
Box Ratio = (Volume / 10000) / (High - Low)
EMV = Distance Moved / Box Ratio
```

**Model Benefit**: High EMV indicates price moving easily on low volume; useful for breakout detection.

---

## 5. Price Transform Features

Derived features from raw price data that enhance model learning.

### 5.1 Returns

**Description**: Percentage price change.

```python
Simple Return = (Close - Previous Close) / Previous Close
Log Return = ln(Close / Previous Close)
```

**Common Periods**: 1, 5, 10, 20, 60, 120, 252 days

**Model Benefit**:
- Log returns are more stationary and approximately normal
- Multiple return horizons capture different trend scales
- Essential feature for any financial model

---

### 5.2 Price Gaps

**Description**: Difference between open and previous close.

```python
Gap = Open - Previous Close
Gap Percentage = (Open - Previous Close) / Previous Close
```

**Model Benefit**: Captures overnight news/events impact; significant gaps often lead to continuation or reversal patterns.

---

### 5.3 Intraday Range Features

**Description**: Various measures of daily price range.

```python
Range = High - Low
Range Percentage = (High - Low) / Close
Body = Close - Open
Body Percentage = (Close - Open) / Open
Upper Shadow = High - max(Open, Close)
Lower Shadow = min(Open, Close) - Low
```

**Model Benefit**: Captures intraday price action characteristics; shadows indicate rejection levels.

---

### 5.4 Price Position Features

**Description**: Price location within various ranges.

```python
Position in Day Range = (Close - Low) / (High - Low)
Position in N-day Range = (Close - N-day Low) / (N-day High - N-day Low)
Distance from N-day High = (N-day High - Close) / N-day High
Distance from N-day Low = (Close - N-day Low) / N-day Low
```

**Model Benefit**: Normalized features (0-1) indicating support/resistance proximity.

---

### 5.5 Price Ratios

**Description**: Relationships between prices at different timeframes.

```python
Close/SMA(n) - 1  # Price relative to moving average
Close/EMA(n) - 1
High/Low - 1  # Daily volatility proxy
```

**Model Benefit**: Captures overbought/oversold conditions relative to trend.

---

## 6. Statistical Features

Rolling statistical measures that capture distribution characteristics.

### 6.1 Rolling Statistics

**Description**: Moving window statistics of returns or prices.

```python
Rolling Mean (various windows)
Rolling Standard Deviation
Rolling Skewness
Rolling Kurtosis
Rolling Min/Max
```

**Common Windows**: 5, 10, 20, 60 days

**Model Benefit**:
- Skewness indicates return asymmetry
- Kurtosis captures tail risk
- Multiple windows provide multi-scale context

---

### 6.2 Z-Score

**Description**: How many standard deviations the current value is from the mean.

```python
Z-Score = (Close - Rolling Mean) / Rolling Std
```

**Model Benefit**: Normalized feature indicating extreme values; useful for mean reversion signals.

---

### 6.3 Percentile Rank

**Description**: Current value's percentile within a lookback window.

```python
Percentile = rank(Close) / window_size × 100
```

**Model Benefit**: Bounded (0-100) relative position indicator; robust to outliers.

---

### 6.4 Beta (vs. Market)

**Description**: Sensitivity of asset returns to market returns.

```python
Beta = Cov(Asset Returns, Market Returns) / Var(Market Returns)
```

**Common Window**: 60, 252 days

**Model Benefit**: Captures systematic risk; useful for cross-asset models.

---

### 6.5 Correlation Features

**Description**: Rolling correlation with market indices or sector ETFs.

**Model Benefit**: Captures regime changes in correlation structure; attention can learn correlation breakdowns.

---

## 7. Implementation Libraries

### Python Libraries for Technical Analysis

| Library | Description | Installation |
|---------|-------------|--------------|
| **pandas-ta** | 130+ indicators, pandas native | `pip install pandas-ta` |
| **ta** | Technical Analysis Library | `pip install ta` |
| **TA-Lib** | Industry standard, C-based (fast) | `pip install TA-Lib` (requires C library) |
| **finta** | Financial Technical Analysis | `pip install finta` |
| **tulipy** | Python bindings for Tulip Indicators | `pip install tulipy` |

### Recommended: pandas-ta

```python
import pandas as pd
import pandas_ta as ta

# Load data
df = pd.read_csv('data/yfinance/apple.csv', index_col=0, parse_dates=True)

# Add multiple indicators at once
df.ta.strategy("all")  # Adds 130+ indicators

# Or add specific indicators
df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.bbands(length=20, std=2, append=True)
df.ta.atr(length=14, append=True)
df.ta.adx(length=14, append=True)
```

### Example: Custom Feature Engineering

```python
import pandas as pd
import numpy as np

def add_technical_features(df):
    """Add comprehensive technical indicators to OHLCV dataframe."""

    # Ensure proper column names
    df.columns = [c.lower() for c in df.columns]

    # --- Returns ---
    for period in [1, 5, 10, 20, 60]:
        df[f'return_{period}d'] = df['close'].pct_change(period)
        df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))

    # --- Moving Averages ---
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        df[f'close_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1

    # --- Volatility ---
    df['atr_14'] = calculate_atr(df, 14)
    df['volatility_20'] = df['log_return_1d'].rolling(20).std() * np.sqrt(252)

    # --- RSI ---
    df['rsi_14'] = calculate_rsi(df['close'], 14)

    # --- MACD ---
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # --- Bollinger Bands ---
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # --- Volume Features ---
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    return df
```

---

## 8. Recommended Feature Set

### Minimal Feature Set (15 features)
For quick experiments or limited computational resources:

| Category | Features |
|----------|----------|
| Trend | SMA(20), SMA(50), MACD, MACD Signal |
| Momentum | RSI(14), Stochastic %K |
| Volatility | ATR(14), Bollinger %B, Bollinger Width |
| Volume | Volume Ratio (vs. 20-day avg), OBV Change |
| Returns | 1-day, 5-day, 20-day log returns |
| Price Position | Close/SMA(20) - 1 |

### Standard Feature Set (35 features)
Balanced coverage for most applications:

| Category | Features |
|----------|----------|
| Trend | SMA(10,20,50,200), EMA(12,26), MACD components (3), ADX |
| Momentum | RSI(14), Stochastic(%K,%D), CCI(20), Williams %R, ROC(10,20) |
| Volatility | ATR(14), BB(%B, Width), HV(20), Keltner position |
| Volume | Volume Ratio, OBV change, CMF(20), MFI(14) |
| Returns | Log returns (1,5,10,20,60 days) |
| Statistical | Rolling Std(20), Z-Score(20), Skewness(20) |

### Comprehensive Feature Set (60+ features)
For thorough analysis with sufficient data:

Add to Standard set:
- Multiple timeframes for each indicator (e.g., RSI 7, 14, 21)
- Ichimoku Cloud components
- All candlestick pattern signals
- Cross-asset correlations (if multi-asset model)
- Sector/market beta
- Feature interactions (e.g., RSI × Volume Ratio)

### Feature Engineering Best Practices

1. **Normalization**:
   - Use indicators with bounded ranges when possible (RSI, %B, etc.)
   - Apply z-score normalization for unbounded features
   - Consider rank-based normalization for robustness

2. **Handling NaN Values**:
   - Indicators require warmup periods (e.g., SMA(200) needs 200 days)
   - Either drop initial rows or forward-fill with neutral values
   - Document NaN handling strategy

3. **Multi-Scale Features**:
   - Include multiple timeframes (5, 10, 20, 50 days)
   - Attention mechanism can learn which scales matter

4. **Avoid Look-Ahead Bias**:
   - All features must use only past data
   - Be careful with centered rolling windows

5. **Feature Selection**:
   - Start with comprehensive set, then reduce via:
     - Correlation analysis (remove highly correlated pairs)
     - Feature importance from tree models
     - Attention weight analysis after training

---

## Summary Table

| Category | Indicator | Periods/Settings | Output Range | Priority |
|----------|-----------|------------------|--------------|----------|
| Trend | SMA | 10, 20, 50, 200 | Price-based | High |
| Trend | EMA | 12, 26, 50 | Price-based | High |
| Trend | MACD | 12, 26, 9 | Unbounded | High |
| Trend | ADX | 14 | 0-100 | Medium |
| Momentum | RSI | 14 | 0-100 | High |
| Momentum | Stochastic | 14, 3 | 0-100 | High |
| Momentum | CCI | 20 | Unbounded | Medium |
| Momentum | ROC | 1, 5, 10, 20 | Unbounded | High |
| Volatility | ATR | 14 | Price-based | High |
| Volatility | Bollinger Bands | 20, 2 | Price-based | High |
| Volatility | Historical Vol | 20 | Percentage | Medium |
| Volume | OBV | - | Cumulative | Medium |
| Volume | CMF | 20 | -1 to +1 | Medium |
| Volume | MFI | 14 | 0-100 | Medium |
| Volume | Volume Ratio | 20 | Ratio | High |
| Price | Log Returns | 1, 5, 20, 60 | Unbounded | High |
| Price | Gap | - | Percentage | Low |
| Statistical | Z-Score | 20, 60 | Unbounded | Medium |
| Statistical | Percentile | 60, 252 | 0-100 | Medium |

---

## References

- Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*
- Kirkpatrick, C. D., & Dahlquist, J. R. (2010). *Technical Analysis: The Complete Resource for Financial Market Technicians*
- pandas-ta documentation: https://github.com/twopirllc/pandas-ta
- TA-Lib documentation: https://ta-lib.org/
