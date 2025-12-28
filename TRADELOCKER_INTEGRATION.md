# 🔗 TradeLocker API Integration

## Overview

The Trading Performance Intelligence dashboard now supports **direct integration with TradeLocker** accounts, allowing you to fetch trading history automatically without needing to export CSV files.

## Features

### 🚀 **Automatic Data Fetching**
- Connect directly to your TradeLocker account
- Fetch trading history for any period (1-365 days)
- Real-time account balance and equity information
- Automatic data processing and standardization

### 🔒 **Security & Privacy**
- Uses official TradeLocker API with secure authentication
- Credentials are only used for data fetching and are not stored
- All data processing happens locally in your browser
- Supports both Demo and Live accounts

### 📊 **Data Compatibility**
- Automatically converts TradeLocker data to standard format
- Calculates real PnL including commissions and swaps
- Compatible with all existing analysis features
- Seamless integration with date filtering and period analysis

## How to Use

### 1. **Select Data Source**
In the Trading Performance Intelligence app:
- Choose "🔗 TradeLocker API" instead of "📁 Upload CSV File"

### 2. **Enter Credentials & Select Account**
- **Email**: Your TradeLocker login email
- **Password**: Your TradeLocker password
- **Server**: Your broker's server name (e.g., "GATESFX")
- **Account Type**: Select "Demo" or "Live" based on your account
- **Get Accounts**: Click to automatically fetch your available accounts
- **Select Account**: Choose from dropdown showing account details and balances
- **Days of History**: Choose how many days back to fetch (1-365)

### 3. **Automatic Account Discovery**
The system now automatically discovers your accounts:
- No need to manually enter Account ID or Account Number
- Clear display of account balances and currencies
- Support for multiple accounts with easy selection
- Fallback to manual entry if automatic discovery fails

### 3. **Connect & Analyze**
- Click "🔗 Connect & Fetch Data"
- The system will authenticate and fetch your trading history
- All analysis features work exactly the same as with CSV uploads

## API Endpoints Used

The integration uses the following TradeLocker API endpoints:

- **Authentication**: `/backend-api/auth/jwt/token`
- **Account Listing**: `/backend-api/auth/jwt/all-accounts` ← **NEW**
- **Account Info**: `/backend-api/trade/accounts/{accountId}/state`
- **Trading History**: `/backend-api/trade/accounts/{accountId}/orders`
- **Token Refresh**: `/backend-api/auth/jwt/refresh`

## Data Mapping

TradeLocker data is automatically mapped to our standard format:

| TradeLocker Field | Standard Field | Description |
|------------------|----------------|-------------|
| `id` | `ticket` | Trade ID |
| `symbol` | `symbol` | Trading instrument |
| `side` | `type` | Buy/Sell direction |
| `qty` | `lots` | Position size |
| `openTime` | `open_time` | Trade open timestamp |
| `closeTime` | `close_time` | Trade close timestamp |
| `openPrice` | `open_price` | Entry price |
| `closePrice` | `close_price` | Exit price |
| `pnl` | `profit` | Raw profit/loss |
| `commission` | `commission` | Trading fees |
| `swap` | `swaps` | Overnight fees |

**Real PnL Calculation**: `profit + commission + swaps`

## Error Handling

The integration includes comprehensive error handling:

- **Authentication Errors**: Clear messages for login failures
- **Network Issues**: Timeout and connection error handling
- **Token Expiry**: Automatic token refresh
- **Data Validation**: Ensures data integrity before analysis
- **Empty Results**: Graceful handling when no trades are found

## Troubleshooting

### Common Issues

**1. Authentication Failed**
- Verify your email and password are correct
- Ensure you're using the correct server (Demo/Live)
- Check if your account is active

**2. No Trading History Found**
- Increase the number of days to fetch
- Verify you have trades in the selected period
- Check if trades are in "filled" status

**3. Connection Timeout**
- Check your internet connection
- Try reducing the number of days to fetch
- Retry the connection

### Support

If you encounter issues:
1. Check the error message displayed in the app
2. Verify your TradeLocker account is accessible via their web platform
3. Try with a smaller date range first
4. Ensure you have active trades in the selected period

## Benefits vs CSV Upload

| Feature | CSV Upload | TradeLocker API |
|---------|------------|-----------------|
| **Setup Time** | Manual export required | One-time credential entry |
| **Account Selection** | Not applicable | Automatic account discovery |
| **Data Freshness** | Manual updates needed | Real-time data |
| **Automation** | Manual process | Fully automated |
| **Error Prone** | File format issues | Standardized API |
| **Historical Data** | Limited by export | Up to 365 days |
| **Account Info** | Not available | Live balance/equity |
| **Multi-Account** | Separate files needed | Seamless account switching |

## Technical Details

### Authentication Flow
1. User enters credentials
2. System authenticates with TradeLocker
3. Receives access token and account information
4. Fetches trading history using authenticated requests
5. Processes and standardizes data for analysis

### Data Processing
- Timestamps converted from milliseconds to datetime
- Numeric fields validated and cleaned
- Real PnL calculated including all fees
- Data formatted to match CSV upload structure

### Performance
- Efficient API calls with pagination support
- Minimal data transfer (only required fields)
- Fast local processing and analysis
- Cached authentication tokens

## Future Enhancements

Planned improvements:
- **Multiple Account Support**: Connect multiple TradeLocker accounts
- **Real-time Updates**: Live trade monitoring and analysis
- **Advanced Filtering**: Filter by symbol, trade type, etc.
- **Scheduled Analysis**: Automated daily/weekly reports
- **Export Integration**: Save analysis results back to TradeLocker

---

**Ready to use TradeLocker API integration? Select "🔗 TradeLocker API" in the Trading Performance Intelligence dashboard!**