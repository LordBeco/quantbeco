# 🎯 Trading Performance Intelligence

A comprehensive TradingView-grade trading analytics dashboard built with Streamlit. Analyze your trading performance with professional-level insights, risk metrics, and AI-powered diagnosis.

![Trading Performance Intelligence](https://img.shields.io/badge/Trading-Analytics-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

### 📊 **Core Analytics**
- **Equity Curve Analysis** - Track account growth over time
- **Drawdown Analysis** - Identify and measure risk periods
- **Rolling Performance Metrics** - Detect edge decay and consistency
- **Risk-Reward Ratio Analysis** - Measure trade quality
- **Win Rate & Profit Factor** - Core performance metrics

### 🎯 **Advanced Trading Insights**
- **Lot Size Consistency Analysis** - Position sizing discipline
- **Buy vs Sell Performance** - Directional bias detection
- **Symbol Performance Ranking** - Asset-specific insights
- **Pip Analysis** - Symbol-aware pip calculations
- **Time-Based Analysis** - Hour/day/month performance patterns

### ⏰ **Time Analysis**
- **Daily Trading Calendar** - Visual calendar with color-coded daily P&L
- **Session Killer Detection** - Identify losing time periods
- **Monthly Seasonality** - Performance heatmaps
- **Hourly Performance** - Optimal trading hours
- **Weekly Patterns** - Day-of-week analysis

### 🩹 **Risk & Pain Metrics**
- **Drawdown Duration Analysis** - Recovery time measurement
- **Consecutive Loss Tracking** - Psychological pain metrics
- **Ulcer Index** - Pain-adjusted performance
- **Recovery Factor** - Resilience measurement

### 🤖 **AI-Powered Diagnosis**
- **Performance Grading** (A+ to F ratings)
- **Critical Issues Detection** - Automated risk identification
- **Psychological Analysis** - Trading behavior insights
- **Actionable Recommendations** - Data-driven improvement suggestions

### 🔗 **Data Sources**
- **CSV Upload** - Import broker statements
- **TradeLocker API** - Direct account integration
- **Multiple Brokers** - Standardized data processing

## 🚀 Quick Start

### Option 1: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/trading-performance-intelligence.git
   cd trading-performance-intelligence
   ```

2. **Quick start with script**
   ```bash
   python run_local.py
   ```
   
   Or manually:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. **Open your browser** to `http://localhost:8501`

### Option 2: Deploy to GitHub + Streamlit Cloud

1. **Use the deployment script**
   ```bash
   python deploy.py
   ```
   
   Or manually:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/trading-performance-intelligence.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select your repository
   - Set main file to `app.py`
   - Click Deploy

### Option 3: One-Click Deploy

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

**Live Demo**: [View Demo App](https://trading-performance-intelligence.streamlit.app)

## 📋 Requirements

- Python 3.8+
- Streamlit 1.28+
- Pandas 2.0+
- Plotly 5.0+
- Requests 2.28+

See `requirements.txt` for complete dependencies.

## 📊 Usage

### 1. **Upload Trading Data**
- **CSV Format**: Export from your broker (MT4, MT5, cTrader, etc.)
- **TradeLocker API**: Direct integration with TradeLocker accounts
- **Required Columns**: Profit/PnL, optional: lots, symbol, type, timestamps

### 2. **Analyze Performance**
- **Date Filtering**: Analyze specific periods (today, week, month, custom)
- **Multiple Metrics**: 50+ performance indicators
- **Visual Charts**: Interactive Plotly visualizations
- **AI Insights**: Automated analysis and recommendations

### 3. **Export Results**
- **Screenshots**: Save charts and metrics
- **Data Export**: Download filtered datasets
- **Reports**: Generate performance summaries

## 🎯 Key Metrics

### Performance Metrics
- **Total Return** - Account growth percentage
- **Sharpe Ratio** - Risk-adjusted returns
- **Maximum Drawdown** - Worst losing streak
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Ratio of gross profit to gross loss
- **Expectancy** - Average profit per trade

### Risk Metrics
- **Value at Risk (VaR)** - Potential losses
- **Ulcer Index** - Drawdown-based risk measure
- **Recovery Factor** - Profit to max drawdown ratio
- **Consecutive Losses** - Maximum losing streak
- **Drawdown Duration** - Time to recover from losses

### Trading Insights
- **Lot Size Consistency** - Position sizing discipline
- **Risk-Reward Ratios** - Trade quality measurement
- **Symbol Performance** - Asset-specific analysis
- **Time-Based Patterns** - Optimal trading periods
- **Directional Bias** - Buy vs sell performance

## 🔧 Configuration

### TradeLocker API Setup
1. **Get Credentials**: Email, password, server name
2. **Find Account Details**: Account ID and number
3. **Configure in App**: Enter details in TradeLocker section
4. **Test Connection**: Verify API access

### CSV Format Requirements
```csv
profit,lots,symbol,type,open_time,close_time
100.50,0.1,EURUSD,buy,2024-01-01 09:30,2024-01-01 10:15
-50.25,0.1,GBPUSD,sell,2024-01-01 11:00,2024-01-01 11:45
```

**Minimum Required**: `profit` or `pnl` column
**Optional**: `lots`, `symbol`, `type`, `open_time`, `close_time`, `commission`, `swaps`

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/trading-performance-intelligence/issues)
- **Documentation**: Check the `/docs` folder
- **Examples**: See `sample_trades.csv` for data format

## 🎉 Acknowledgments

- **Streamlit** - Amazing web app framework
- **Plotly** - Interactive visualization library
- **Pandas** - Data manipulation and analysis
- **TradeLocker** - API integration support

---

**⚡ Start analyzing your trading performance like a pro!**

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)