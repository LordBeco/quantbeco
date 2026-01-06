"""
Kelly Criterion Position Sizing Module
=====================================

This module implements the Kelly Criterion for optimal position sizing based on trading history.
The Kelly Criterion helps determine the optimal fraction of capital to risk per trade to maximize
long-term growth while considering the trader's historical win rate and average win/loss ratios.

Formula: f* = (bp - q) / b
Where:
- f* = fraction of capital to wager (Kelly fraction)
- b = odds received on the wager (avg_win / avg_loss)
- p = probability of winning (win rate)
- q = probability of losing (1 - p)

Enhanced with:
- Risk adjustment factors
- Current equity consideration
- Conservative scaling options
- Dynamic lot size recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, 
                           conservative_factor: float = 0.25) -> Dict:
    """
    Calculate the Kelly Criterion fraction for optimal position sizing.
    
    Args:
        win_rate: Historical win rate (0.0 to 1.0)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (should be negative)
        conservative_factor: Scaling factor to reduce Kelly fraction (default 0.25 = 25% of full Kelly)
    
    Returns:
        Dictionary containing Kelly analysis results
    """
    
    # Ensure avg_loss is positive for calculation
    avg_loss_abs = abs(avg_loss)
    
    # Calculate odds (reward-to-risk ratio)
    if avg_loss_abs == 0:
        return {
            'kelly_fraction': 0.0,
            'conservative_kelly': 0.0,
            'recommendation': 'AVOID - No loss data available',
            'risk_level': 'UNDEFINED',
            'odds_ratio': 0.0,
            'edge': 0.0,
            'error': 'No loss data available for calculation'
        }
    
    odds_ratio = avg_win / avg_loss_abs  # b in Kelly formula
    
    # Calculate probabilities
    p = win_rate  # probability of winning
    q = 1 - p     # probability of losing
    
    # Calculate Kelly fraction: f* = (bp - q) / b
    kelly_numerator = (odds_ratio * p) - q
    kelly_fraction = kelly_numerator / odds_ratio if odds_ratio > 0 else 0.0
    
    # Apply conservative scaling
    conservative_kelly = kelly_fraction * conservative_factor
    
    # Calculate edge (expected value per dollar risked)
    edge = (p * avg_win) + (q * avg_loss)
    
    # Determine risk level and recommendation
    if kelly_fraction <= 0:
        recommendation = "AVOID TRADING - Negative edge detected"
        risk_level = "NO EDGE"
    elif kelly_fraction > 1.0:
        recommendation = "EXTREME CAUTION - Very high Kelly fraction suggests review strategy"
        risk_level = "EXTREME"
    elif kelly_fraction > 0.5:
        recommendation = "HIGH RISK - Consider reducing position size significantly"
        risk_level = "HIGH"
    elif kelly_fraction > 0.25:
        recommendation = "MODERATE RISK - Standard Kelly application"
        risk_level = "MODERATE"
    else:
        recommendation = "CONSERVATIVE - Low Kelly fraction suggests small positions"
        risk_level = "LOW"
    
    return {
        'kelly_fraction': kelly_fraction,
        'conservative_kelly': conservative_kelly,
        'recommendation': recommendation,
        'risk_level': risk_level,
        'odds_ratio': odds_ratio,
        'edge': edge,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

def calculate_rolling_kelly(df: pd.DataFrame, pnl_col: str, window: int = 50) -> pd.DataFrame:
    """
    Calculate rolling Kelly Criterion values to track how optimal position sizing changes over time.
    
    Args:
        df: DataFrame with trading data
        pnl_col: Column name containing P&L values
        window: Rolling window size for calculations
    
    Returns:
        DataFrame with additional Kelly-related columns
    """
    df = df.copy()
    
    def rolling_kelly_calc(series):
        """Calculate Kelly fraction for a rolling window"""
        if len(series) < 5:  # Need minimum trades for meaningful calculation
            return 0.0
        
        wins = series[series > 0]
        losses = series[series < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        win_rate = len(wins) / len(series)
        avg_win = wins.mean()
        avg_loss = losses.mean()
        
        kelly_result = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        return kelly_result['kelly_fraction']
    
    # Calculate rolling Kelly metrics
    df['rolling_kelly_fraction'] = df[pnl_col].rolling(window=window, min_periods=5).apply(
        rolling_kelly_calc, raw=False
    )
    
    # Calculate other rolling metrics
    def rolling_conservative_kelly(series):
        kelly_f = rolling_kelly_calc(series)
        return kelly_f * 0.25  # Conservative scaling
    
    def rolling_odds_ratio(series):
        if len(series) < 5:
            return 0.0
        wins = series[series > 0]
        losses = series[series < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        return wins.mean() / abs(losses.mean())
    
    def rolling_edge(series):
        if len(series) < 5:
            return 0.0
        wins = series[series > 0]
        losses = series[series < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        win_rate = len(wins) / len(series)
        return (win_rate * wins.mean()) + ((1 - win_rate) * losses.mean())
    
    df['rolling_conservative_kelly'] = df[pnl_col].rolling(window=window, min_periods=5).apply(
        rolling_conservative_kelly, raw=False
    )
    
    df['rolling_odds_ratio'] = df[pnl_col].rolling(window=window, min_periods=5).apply(
        rolling_odds_ratio, raw=False
    )
    
    df['rolling_edge'] = df[pnl_col].rolling(window=window, min_periods=5).apply(
        rolling_edge, raw=False
    )
    
    return df

def recommend_lot_size(current_equity: float, kelly_fraction: float, 
                      base_lot_per_1k: float = 0.02, max_lot_size: float = 1.0,
                      min_lot_size: float = 0.01, user_current_lots: float = None) -> Dict:
    """
    Recommend lot size based on Kelly Criterion and current equity.
    
    Args:
        current_equity: Current account equity
        kelly_fraction: Kelly fraction from calculate_kelly_fraction
        base_lot_per_1k: Base lot size per $1000 equity (default 0.02)
        max_lot_size: Maximum allowed lot size
        min_lot_size: Minimum allowed lot size
        user_current_lots: User's current lot size for comparison
    
    Returns:
        Dictionary with lot size recommendations
    """
    
    # If user provided their current lot size, calculate their actual base rate
    if user_current_lots is not None:
        user_base_rate = user_current_lots / (current_equity / 1000)
        actual_base_lots = (current_equity / 1000) * user_base_rate
    else:
        actual_base_lots = (current_equity / 1000) * base_lot_per_1k
    
    # Calculate Kelly-adjusted lot size using user's actual base if available
    base_for_kelly = actual_base_lots if user_current_lots is not None else (current_equity / 1000) * base_lot_per_1k
    
    # Kelly multiplier - but don't go below 10% of current approach if user is successful
    if kelly_fraction > 0:
        kelly_multiplier = kelly_fraction
    else:
        kelly_multiplier = 0.1  # Minimum 10% if no edge
    
    kelly_adjusted_lots = base_for_kelly * kelly_multiplier
    
    # Apply constraints
    recommended_lots = max(min_lot_size, min(kelly_adjusted_lots, max_lot_size))
    
    # If user provided current lots and Kelly suggests much smaller, provide graduated approach
    if user_current_lots is not None and recommended_lots < user_current_lots * 0.5:
        # Suggest a more gradual reduction
        graduated_lots = user_current_lots * 0.7  # 30% reduction as compromise
        recommended_lots = max(recommended_lots, graduated_lots)
    
    # Calculate risk per trade (assuming standard risk management)
    risk_per_trade_pct = (recommended_lots / (current_equity / 1000)) * 2  # Rough estimate
    
    # Generate recommendation text with user context
    if user_current_lots is not None:
        current_vs_kelly = (user_current_lots / recommended_lots) if recommended_lots > 0 else float('inf')
        
        if kelly_fraction <= 0:
            recommendation_text = f"⚠️ NO EDGE DETECTED - Reduce from {user_current_lots} to {min_lot_size} or stop trading"
            risk_assessment = "HIGH RISK - No statistical edge detected"
        elif current_vs_kelly > 3:
            recommendation_text = f"🚨 CURRENT SIZE TOO AGGRESSIVE - Reduce from {user_current_lots} to {recommended_lots:.3f} gradually"
            risk_assessment = "HIGH RISK - Position size much larger than optimal"
        elif current_vs_kelly > 1.5:
            recommendation_text = f"⚠️ MODERATE OVER-SIZING - Consider reducing from {user_current_lots} to {recommended_lots:.3f}"
            risk_assessment = "MODERATE RISK - Slightly oversized for your edge"
        else:
            recommendation_text = f"✅ REASONABLE SIZING - Current {user_current_lots} vs Kelly {recommended_lots:.3f}"
            risk_assessment = "ACCEPTABLE RISK - Close to optimal"
    else:
        # Original logic for when no current lot size provided
        if kelly_fraction <= 0:
            recommendation_text = f"⚠️ NO EDGE DETECTED - Use minimum lot size ({min_lot_size}) or avoid trading"
            risk_assessment = "HIGH RISK - No statistical edge"
        elif kelly_fraction < 0.1:
            recommendation_text = f"📉 WEAK EDGE - Conservative lot size recommended: {recommended_lots:.3f}"
            risk_assessment = "MODERATE RISK - Weak edge detected"
        elif kelly_fraction < 0.25:
            recommendation_text = f"📊 MODERATE EDGE - Standard Kelly application: {recommended_lots:.3f}"
            risk_assessment = "ACCEPTABLE RISK - Moderate edge"
        else:
            recommendation_text = f"🚨 STRONG EDGE - High Kelly fraction, consider: {recommended_lots:.3f} (capped for safety)"
            risk_assessment = "MONITOR CLOSELY - Strong edge but high variance risk"
    
    return {
        'recommended_lot_size': recommended_lots,
        'base_lot_size': base_for_kelly,
        'kelly_multiplier': kelly_multiplier,
        'risk_per_trade_pct': risk_per_trade_pct,
        'recommendation_text': recommendation_text,
        'risk_assessment': risk_assessment,
        'equity_used': current_equity,
        'kelly_fraction_used': kelly_fraction,
        'user_current_lots': user_current_lots,
        'user_base_rate': user_current_lots / (current_equity / 1000) if user_current_lots else base_lot_per_1k
    }
    
    return {
        'recommended_lot_size': recommended_lots,
        'base_lot_size': base_lots,
        'kelly_multiplier': kelly_multiplier,
        'risk_per_trade_pct': risk_per_trade_pct,
        'recommendation_text': recommendation_text,
        'risk_assessment': risk_assessment,
        'equity_used': current_equity,
        'kelly_fraction_used': kelly_fraction
    }

def analyze_position_sizing_history(df: pd.DataFrame, pnl_col: str, 
                                  lot_col: Optional[str] = None) -> Dict:
    """
    Analyze historical position sizing decisions and compare with Kelly optimal.
    
    Args:
        df: DataFrame with trading data
        pnl_col: Column name containing P&L values
        lot_col: Column name containing lot sizes (optional)
    
    Returns:
        Dictionary with position sizing analysis
    """
    
    # Calculate overall Kelly metrics
    wins = df[df[pnl_col] > 0]
    losses = df[df[pnl_col] < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return {
            'error': 'Insufficient win/loss data for analysis',
            'has_analysis': False
        }
    
    win_rate = len(wins) / len(df)
    avg_win = wins[pnl_col].mean()
    avg_loss = losses[pnl_col].mean()
    
    kelly_analysis = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
    
    analysis = {
        'kelly_analysis': kelly_analysis,
        'total_trades': len(df),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'has_analysis': True
    }
    
    # Analyze lot sizing if available
    if lot_col and lot_col in df.columns:
        lot_data = df[lot_col].dropna()
        if len(lot_data) > 0:
            analysis.update({
                'has_lot_data': True,
                'avg_lot_size': lot_data.mean(),
                'lot_std': lot_data.std(),
                'min_lot': lot_data.min(),
                'max_lot': lot_data.max(),
                'lot_consistency': 1 - (lot_data.std() / lot_data.mean()) if lot_data.mean() > 0 else 0
            })
        else:
            analysis['has_lot_data'] = False
    else:
        analysis['has_lot_data'] = False
    
    return analysis

def generate_kelly_insights(kelly_analysis: Dict, current_equity: float) -> list:
    """
    Generate actionable insights based on Kelly Criterion analysis.
    
    Args:
        kelly_analysis: Results from analyze_position_sizing_history
        current_equity: Current account equity
    
    Returns:
        List of insight strings
    """
    
    insights = []
    
    if not kelly_analysis.get('has_analysis', False):
        insights.append("❌ Insufficient data for Kelly Criterion analysis")
        return insights
    
    kelly_data = kelly_analysis['kelly_analysis']
    kelly_fraction = kelly_data['kelly_fraction']
    
    # Edge analysis
    if kelly_fraction <= 0:
        insights.append("🚨 CRITICAL: Your strategy has NO statistical edge (negative Kelly fraction)")
        insights.append("💡 RECOMMENDATION: Stop trading this strategy until you identify the issue")
    elif kelly_fraction < 0.05:
        insights.append("⚠️ WARNING: Very weak edge detected - proceed with extreme caution")
        insights.append(f"📊 Kelly suggests risking only {kelly_fraction*100:.1f}% of capital per trade")
    elif kelly_fraction < 0.15:
        insights.append("📈 MODERATE: Decent edge detected, suitable for steady growth")
        insights.append(f"💰 Kelly optimal: {kelly_fraction*100:.1f}% of capital per trade")
    else:
        insights.append("🎯 STRONG: Excellent edge detected, but manage risk carefully")
        insights.append(f"⚡ Kelly fraction: {kelly_fraction*100:.1f}% (consider scaling down for safety)")
    
    # Lot size recommendations
    lot_recommendation = recommend_lot_size(current_equity, kelly_data['conservative_kelly'])
    insights.append(f"📏 RECOMMENDED LOT SIZE: {lot_recommendation['recommended_lot_size']:.2f}")
    insights.append(f"🎯 {lot_recommendation['recommendation_text']}")
    
    # Risk assessment
    if kelly_data['odds_ratio'] < 1.0:
        insights.append(f"⚠️ Poor risk-reward ratio: {kelly_data['odds_ratio']:.2f}:1 (aim for >1.5:1)")
    elif kelly_data['odds_ratio'] > 2.0:
        insights.append(f"✅ Excellent risk-reward ratio: {kelly_data['odds_ratio']:.2f}:1")
    else:
        insights.append(f"📊 Acceptable risk-reward ratio: {kelly_data['odds_ratio']:.2f}:1")
    
    # Win rate insights
    win_rate_pct = kelly_analysis['win_rate'] * 100
    if win_rate_pct < 40:
        insights.append(f"📉 Low win rate ({win_rate_pct:.1f}%) - ensure R:R ratio compensates")
    elif win_rate_pct > 70:
        insights.append(f"🎯 High win rate ({win_rate_pct:.1f}%) - excellent for psychology")
    
    return insights