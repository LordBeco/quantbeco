#!/usr/bin/env python3
"""
Clear Streamlit Cache

This script provides instructions and code to completely clear Streamlit's cache
and session state to fix the persistent OpenRouter error.
"""

print("🧹 Streamlit Cache Clearing Guide")
print("=" * 40)

print("""
The OpenRouter 401 error is persisting because of Streamlit's caching system.
Here are the steps to completely clear the cache:

🔄 SOLUTION 1: Complete App Restart
1. Stop the Streamlit app (Ctrl+C in terminal)
2. Wait 5 seconds
3. Restart with: streamlit run app.py
4. Try OpenRouter again

🔄 SOLUTION 2: Clear Browser Cache
1. In your browser, press F12 (Developer Tools)
2. Right-click the refresh button
3. Select "Empty Cache and Hard Reload"
4. Or use Ctrl+Shift+R (hard refresh)

🔄 SOLUTION 3: Use Incognito/Private Mode
1. Open a new incognito/private browser window
2. Navigate to your Streamlit app
3. Try OpenRouter functionality

🔄 SOLUTION 4: Clear Streamlit Cache Directory
Run these commands in your terminal:
""")

import os
import platform

# Get the appropriate cache directory based on OS
if platform.system() == "Windows":
    cache_dir = os.path.expanduser("~/.streamlit")
    print(f"Windows: rmdir /s /q \"{cache_dir}\"")
elif platform.system() == "Darwin":  # macOS
    cache_dir = os.path.expanduser("~/.streamlit")
    print(f"macOS: rm -rf \"{cache_dir}\"")
else:  # Linux
    cache_dir = os.path.expanduser("~/.streamlit")
    print(f"Linux: rm -rf \"{cache_dir}\"")

print(f"""
🔄 SOLUTION 5: Add Cache Clearing to App
Add this code to the top of your Streamlit app (temporarily):
""")

print("""
```python
import streamlit as st

# Add this at the very top of your app.py, right after imports
if st.button("🧹 Clear All Cache & Session State"):
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Cache and session state cleared! Please refresh the page.")
    st.stop()
```

🔄 SOLUTION 6: Force Session State Reset
Add this to your app.py in the AI Strategy Builder section:
""")

print("""
```python
# Add this button in your AI Strategy Builder interface
if st.button("🔄 Reset AI Provider"):
    # Force clear all AI-related session state
    keys_to_clear = [
        'strategy_processor', 
        'current_provider', 
        'selected_openrouter_model',
        'pine_converter'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("AI provider reset! Try generating a strategy now.")
```

💡 RECOMMENDED APPROACH:
1. First try Solution 1 (complete app restart)
2. If that doesn't work, try Solution 2 (browser cache clear)
3. If still failing, add Solution 5 (cache clearing button) to your app temporarily
""")

print("\n🎯 After trying these solutions, test OpenRouter strategy generation again.")
print("The 401 error should be resolved once the cached session state is cleared.")