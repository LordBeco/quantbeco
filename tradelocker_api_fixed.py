#!/usr/bin/env python3
"""
TradeLocker API Integration for Trading Performance Intelligence
Fixed version with better error handling and endpoint discovery
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time

class TradeLockerAPI:
    def __init__(self):
        # Updated base URLs - TradeLocker uses different endpoints
        self