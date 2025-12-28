#!/usr/bin/env python3
"""
TradeLocker API Integration for Trading Performance Intelligence
Updated based on working auth.py structure
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time

class TradeLockerAPI:
    def __init__(self):
        self.demo_base_url = "https://demo.tradelocker.com"
        self.live_base_url = "https://live.tradelocker.com"
        self.base_url = None
        self.access_token = None
        self.refresh_token = None
        self.account_id = None
        self.account_number = None
        self.server = None
        self.is_live = None
        
    def authenticate(self, email, password, server, account_id, acc_num, is_live):
        """
        Authenticate with TradeLocker API using correct structure
        
        Args:
            email (str): TradeLocker email
            password (str): TradeLocker password  
            server (str): Server name (e.g., "GATESFX")
            account_id (str): Account ID (e.g., "812688") - unique identifier
            acc_num (int): Account number (1, 2, 3, etc.) - account selection order
            is_live (bool): True for live account, False for demo
        
        Returns:
            bool: True if authentication successful
        """
        
        # Set base URL based on is_live flag
        if is_live:
            self.base_url = self.live_base_url
            auth_url = f"{self.live_base_url}/backend-api/auth/jwt/token"
        else:
            self.base_url = self.demo_base_url
            auth_url = f"{self.demo_base_url}/backend-api/auth/jwt/token"
        
        self.is_live = is_live
        self.server = server
        self.account_id = account_id
        self.account_number = acc_num  # This is the accNum (1, 2, 3, etc.)
        
        payload = {
            "email": email,
            "password": password,
            "server": server
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            response = requests.post(auth_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    raise Exception("Invalid response format from TradeLocker API")
                
                self.access_token = data.get("accessToken")
                self.refresh_token = data.get("refreshToken")
                
                if not self.access_token:
                    raise Exception("No access token received from authentication")
                
                return True
                    
            elif response.status_code == 401:
                raise Exception("Invalid credentials - check email and password")
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", "Bad request")
                except:
                    error_msg = "Bad request - check server name and credentials"
                raise Exception(f"Authentication failed: {error_msg}")
            else:
                raise Exception(f"Authentication failed: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error during authentication: {str(e)}")
        except Exception as e:
            if "Authentication failed" in str(e) or "Invalid credentials" in str(e):
                raise e
            else:
                raise Exception(f"Authentication error: {str(e)}")
    
    def get_headers(self):
        """Get headers with authentication token"""
        if not self.access_token:
            raise Exception("Not authenticated. Call authenticate() first.")
            
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "accNum": str(self.account_number)  # This is the account selection number (1, 2, 3, etc.)
        }
    
    def refresh_access_token(self):
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            raise Exception("No refresh token available")
            
        if self.is_live:
            refresh_url = f"{self.live_base_url}/backend-api/auth/jwt/refresh"
        else:
            refresh_url = f"{self.demo_base_url}/backend-api/auth/jwt/refresh"
        
        payload = {
            "refreshToken": self.refresh_token
        }
        
        try:
            response = requests.post(refresh_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("accessToken")
                return True
            else:
                return False
                
        except Exception:
            return False
    
    def get_account_info(self):
        """Get account information using correct endpoint"""
        
        # Try multiple possible endpoints for account info
        endpoints_to_try = [
            f"{self.base_url}/backend-api/trade/accounts/{self.account_id}/state",
            f"{self.base_url}/backend-api/trade/accounts/{self.account_id}",
            f"{self.base_url}/backend-api/auth/jwt/all-accounts"  # Fallback to get all accounts
        ]
        
        for i, url in enumerate(endpoints_to_try):
            try:
                print(f"Trying endpoint {i+1}: {url}")
                response = requests.get(url, headers=self.get_headers(), timeout=30)
                
                if response.status_code == 401:
                    # Try to refresh token
                    if self.refresh_access_token():
                        response = requests.get(url, headers=self.get_headers(), timeout=30)
                    else:
                        raise Exception("Authentication expired. Please re-authenticate.")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Success with endpoint {i+1}")
                    print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    # If this is the all-accounts endpoint, find our specific account
                    if "accounts" in data:
                        accounts = data.get("accounts", [])
                        for account in accounts:
                            if str(account.get("id")) == str(self.account_id):
                                print(f"Found matching account: {account}")
                                return {"d": account}  # Wrap in 'd' format for consistency
                        
                        # If no exact match, return the account at the specified accNum position
                        if len(accounts) >= self.account_number:
                            account = accounts[self.account_number - 1]  # accNum is 1-based
                            print(f"Using account at position {self.account_number}: {account}")
                            return {"d": account}
                    
                    return data
                else:
                    print(f"❌ Endpoint {i+1} failed: HTTP {response.status_code}")
                    print(f"Response: {response.text[:200]}...")
                    
            except Exception as e:
                print(f"❌ Endpoint {i+1} error: {str(e)}")
                continue
        
        # If all endpoints failed, raise the last error
        raise Exception(f"Failed to get account info from all endpoints. Last status: HTTP {response.status_code}")
    
    def get_all_accounts(self):
        """
        Get all accounts for the authenticated user
        
        Returns:
            list: List of account dictionaries with id, name, currency, accNum, balance
        """
        
        if not self.access_token:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        url = f"{self.base_url}/backend-api/auth/jwt/all-accounts"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 401:
                # Try to refresh token
                if self.refresh_access_token():
                    response = requests.get(url, headers=headers, timeout=30)
                else:
                    raise Exception("Authentication expired. Please re-authenticate.")
            
            if response.status_code == 200:
                data = response.json()
                accounts = data.get("accounts", [])
                
                # Process accounts to ensure consistent format
                processed_accounts = []
                for i, account in enumerate(accounts, 1):
                    # Get balance for this account
                    try:
                        # Temporarily set account details to get balance
                        temp_account_id = self.account_id
                        temp_account_number = self.account_number
                        
                        self.account_id = str(account.get('id', ''))
                        self.account_number = i
                        
                        balance_info = self.get_account_balance()
                        balance = balance_info.get('balance', 0.0)
                        
                        # Restore original account details
                        self.account_id = temp_account_id
                        self.account_number = temp_account_number
                        
                    except Exception:
                        balance = 0.0
                    
                    processed_account = {
                        'id': str(account.get('id', '')),
                        'name': account.get('name', f'Account {i}'),
                        'currency': account.get('currency', 'USD'),
                        'accNum': i,
                        'balance': balance
                    }
                    processed_accounts.append(processed_account)
                
                return processed_accounts
            else:
                raise Exception(f"Failed to get accounts: HTTP {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Get accounts error: {str(e)}")
    
    def get_trading_history(self, start_date=None, end_date=None, limit=1000):
        """
        Get trading history from TradeLocker using multiple endpoints
        
        Args:
            start_date (datetime): Start date for history (default: 90 days ago)
            end_date (datetime): End date for history (default: now)
            limit (int): Maximum number of trades to fetch
            
        Returns:
            pandas.DataFrame: Trading history data
        """
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)  # Extended to 90 days
        if end_date is None:
            end_date = datetime.now()
            
        # Convert to timestamps (TradeLocker expects milliseconds)
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        # Try multiple endpoints for trading history
        endpoints_to_try = [
            f"{self.base_url}/backend-api/trade/accounts/{self.account_id}/orders",
            f"{self.base_url}/backend-api/trade/accounts/{self.account_id}/history",
            f"{self.base_url}/backend-api/trade/accounts/{self.account_id}/deals",
            f"{self.base_url}/backend-api/trade/accounts/{self.account_id}/positions/history"
        ]
        
        params = {
            "from": start_timestamp,
            "to": end_timestamp,
            "limit": limit
        }
        
        # Also try without date filters to see if there's any data
        params_no_date = {"limit": limit}
        
        for endpoint in endpoints_to_try:
            try:
                print(f"Trying endpoint: {endpoint}")
                
                # First try with date range
                response = requests.get(endpoint, headers=self.get_headers(), params=params, timeout=30)
                
                if response.status_code == 401:
                    # Try to refresh token
                    if self.refresh_access_token():
                        response = requests.get(endpoint, headers=self.get_headers(), params=params, timeout=30)
                    else:
                        raise Exception("Authentication expired. Please re-authenticate.")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"Response from {endpoint}: {data}")
                    
                    # TradeLocker returns data in 'd' field
                    trades = data.get("d", [])
                    
                    if trades:
                        print(f"Found {len(trades)} trades from {endpoint}")
                        # Convert to DataFrame
                        df = pd.DataFrame(trades)
                        
                        # Process and clean the data
                        df = self.process_trading_data(df)
                        
                        return df
                    else:
                        print(f"No trades found in {endpoint} with date range, trying without date filter...")
                        
                        # Try without date filter
                        response_no_date = requests.get(endpoint, headers=self.get_headers(), params=params_no_date, timeout=30)
                        if response_no_date.status_code == 200:
                            data_no_date = response_no_date.json()
                            trades_no_date = data_no_date.get("d", [])
                            
                            if trades_no_date:
                                print(f"Found {len(trades_no_date)} trades from {endpoint} without date filter")
                                df = pd.DataFrame(trades_no_date)
                                df = self.process_trading_data(df)
                                return df
                else:
                    print(f"HTTP {response.status_code} from {endpoint}: {response.text}")
                    
            except Exception as e:
                print(f"Error with endpoint {endpoint}: {str(e)}")
                continue
        
        # If no endpoint worked, return empty DataFrame
        print("No trading history found from any endpoint")
        return pd.DataFrame()
    
    def process_trading_data(self, df):
        """
        Process raw TradeLocker data into standardized format
        """
        
        if df.empty:
            return df
        
        # TradeLocker column mapping (may vary based on actual API response)
        column_mapping = {
            'id': 'ticket',
            'symbol': 'symbol',
            'side': 'type',
            'qty': 'lots',
            'volume': 'lots',  # Alternative volume field
            'openTime': 'open_time',
            'openPrice': 'open_price',
            'closeTime': 'close_time',
            'closePrice': 'close_price',
            'swap': 'swaps',
            'commission': 'commission',
            'pnl': 'profit',
            'profit': 'profit'
        }
        
        # Rename columns that exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Convert timestamps to datetime (TradeLocker uses milliseconds)
        for time_col in ['open_time', 'close_time']:
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['lots', 'open_price', 'close_price', 'swaps', 'commission', 'profit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate real PnL (profit + commission + swaps)
        if all(col in df.columns for col in ['profit', 'commission', 'swaps']):
            df['real_pnl'] = df['profit'] + df['commission'] + df['swaps']
        elif 'profit' in df.columns:
            df['real_pnl'] = df['profit']
        else:
            df['real_pnl'] = 0
        
        # Add comment column for consistency
        df['comment'] = 'TradeLocker API'
        
        return df
    
    def get_account_balance(self):
        """Get current account balance using equity endpoint"""
        try:
            account_info = self.get_account_info()
            
            if account_info and "d" in account_info:
                account_data = account_info["d"]
                
                # Try multiple ways to extract balance
                balance = 0.0
                currency = "USD"
                
                # Method 1: Direct balance fields
                if "balance" in account_data:
                    balance = float(account_data["balance"])
                elif "aaccountBalance" in account_data:
                    balance = float(account_data["aaccountBalance"])
                elif "equity" in account_data:
                    balance = float(account_data["equity"])
                
                # Method 2: Account details array
                elif "accountDetailsData" in account_data:
                    account_details = account_data["accountDetailsData"]
                    if isinstance(account_details, list) and len(account_details) > 1:
                        balance = float(account_details[1])  # Equity is typically second element
                    elif isinstance(account_details, dict):
                        balance = float(account_details.get("equity", account_details.get("balance", 0)))
                
                # Extract currency
                if "currency" in account_data:
                    currency = account_data["currency"]
                
                print(f"💰 Extracted balance: ${balance:,.2f} {currency}")
                
                return {
                    'balance': balance,
                    'equity': balance,
                    'margin': account_data.get("margin", 0),
                    'free_margin': account_data.get("freeMargin", 0),
                    'currency': currency
                }
            else:
                # Fallback: return default values
                print("⚠️ No account data found, using defaults")
                return {
                    'balance': 0.0,
                    'equity': 0.0,
                    'margin': 0.0,
                    'free_margin': 0.0,
                    'currency': 'USD'
                }
                
        except Exception as e:
            print(f"❌ Balance extraction error: {str(e)}")
            # Return default values instead of failing
            return {
                'balance': 0.0,
                'equity': 0.0,
                'margin': 0.0,
                'free_margin': 0.0,
                'currency': 'USD'
            }

def test_tradelocker_connection(email, password, server, account_id, acc_num, is_live):
    """
    Test TradeLocker API connection
    
    Args:
        email (str): TradeLocker email
        password (str): TradeLocker password
        server (str): Server name (e.g., "GATESFX")
        account_id (str): Account ID (e.g., "812688")
        acc_num (int): Account number (1, 2, 3, etc.)
        is_live (bool): True for live, False for demo
        
    Returns:
        dict: Connection test results
    """
    
    api = TradeLockerAPI()
    
    try:
        # Test authentication
        auth_success = api.authenticate(email, password, server, account_id, acc_num, is_live)
        
        if not auth_success:
            return {
                'success': False,
                'error': 'Authentication failed',
                'account_info': None,
                'balance': None,
                'accounts': None
            }
        
        # Get all accounts
        try:
            accounts = api.get_all_accounts()
        except Exception as e:
            accounts = None
            print(f"Warning: Could not fetch accounts list: {str(e)}")
        
        # Test account info
        account_info = api.get_account_info()
        balance_info = api.get_account_balance()
        
        return {
            'success': True,
            'error': None,
            'account_info': {
                'account_id': api.account_id,
                'account_number': api.account_number,
                'server': api.server,
                'is_live': api.is_live,
                'currency': balance_info.get('currency', 'USD')
            },
            'balance': balance_info,
            'accounts': accounts
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'account_info': None,
            'balance': None,
            'accounts': None
        }

def get_tradelocker_accounts(email, password, server, is_live):
    """
    Get list of available accounts for a user (for account selection)
    
    Args:
        email (str): TradeLocker email
        password (str): TradeLocker password
        server (str): Server name (e.g., "GATESFX")
        is_live (bool): True for live, False for demo
        
    Returns:
        dict: Result with accounts list or error
    """
    
    api = TradeLockerAPI()
    
    try:
        # Authenticate with dummy account info first
        auth_success = api.authenticate(email, password, server, "dummy", 1, is_live)
        
        if not auth_success:
            return {
                'success': False,
                'error': 'Authentication failed',
                'accounts': None
            }
        
        # Get all accounts
        accounts = api.get_all_accounts()
        
        return {
            'success': True,
            'error': None,
            'accounts': accounts
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'accounts': None
        }

def fetch_tradelocker_data(email, password, server, account_id, acc_num, is_live, days_back=90):
    """
    Fetch trading data from TradeLocker API
    
    Args:
        email (str): TradeLocker email
        password (str): TradeLocker password
        server (str): Server name (e.g., "GATESFX")
        account_id (str): Account ID (e.g., "812688")
        acc_num (int): Account number (1, 2, 3, etc.)
        is_live (bool): True for live, False for demo
        days_back (int): Number of days to fetch history (default: 90)
        
    Returns:
        pandas.DataFrame: Trading history data
    """
    
    api = TradeLockerAPI()
    
    try:
        # Authenticate
        api.authenticate(email, password, server, account_id, acc_num, is_live)
        
        # Get trading history with extended period
        start_date = datetime.now() - timedelta(days=days_back)
        print(f"Fetching trading history from {start_date.date()} to {datetime.now().date()}")
        
        df = api.get_trading_history(start_date=start_date)
        
        if df.empty:
            print("No trading history found. Trying to fetch all available data...")
            # Try without date restrictions
            df = api.get_trading_history(start_date=datetime.now() - timedelta(days=365))
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to fetch TradeLocker data: {str(e)}")