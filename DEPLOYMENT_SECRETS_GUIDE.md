# Deployment & Secrets Management Guide

This guide explains how to properly configure environment variables for both local development and production deployment using Streamlit's secrets management.

## 🏗️ Architecture Overview

The application now supports **dual configuration modes**:

- **Local Development**: Uses `.env` files (traditional approach)
- **Production Deployment**: Uses Streamlit secrets (secure cloud approach)

The `config.py` automatically detects the environment and loads configuration accordingly.

## 🔧 Local Development Setup

### 1. Environment File Setup
```bash
# Copy the example file
cp .env.example .env

# Edit with your actual values
nano .env  # or use your preferred editor
```

### 2. Required Configuration
Edit your `.env` file with your actual API keys:

```bash
# Choose your AI provider
AI_PROVIDER=openrouter

# Add your OpenRouter API key (free from https://openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here

# Optional: Add OpenAI key if you want to use premium models
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

### 3. Run Locally
```bash
streamlit run app.py
```

## 🚀 Production Deployment (Streamlit Community Cloud)

### 1. Prepare Secrets Configuration

Create a local `.streamlit/secrets.toml` file (for testing):
```toml
# Copy from .streamlit/secrets.toml.example
ENVIRONMENT = "production"
AI_PROVIDER = "openrouter"
OPENROUTER_API_KEY = "sk-or-v1-your-actual-key-here"
OPENROUTER_SITE_URL = "https://your-app-name.streamlit.app"
# ... other settings
```

### 2. Deploy to Streamlit Cloud

1. **Push to GitHub** (secrets.toml will be ignored by .gitignore)
2. **Go to Streamlit Community Cloud**
3. **Create New App** from your repository
4. **Click "Advanced settings"** before deploying
5. **Paste your secrets configuration**:

```toml
ENVIRONMENT = "production"
AI_PROVIDER = "openrouter"
OPENROUTER_API_KEY = "sk-or-v1-your-actual-key-here"
OPENROUTER_SITE_URL = "https://your-app-name.streamlit.app"
OPENROUTER_SITE_NAME = "Trade Analyzer Pro"

# Add all other configuration from secrets.toml.example
```

6. **Deploy the app**

### 3. Update Secrets After Deployment

You can update secrets anytime:
1. Go to your app in Streamlit Cloud
2. Click **"Settings"** → **"Secrets"**
3. Edit the TOML configuration
4. **Save** (app will restart automatically)

## 🔍 Configuration Priority

The system loads configuration in this order:

1. **Environment Variables** (`.env` file or system env vars)
2. **Streamlit Secrets** (`.streamlit/secrets.toml` or cloud secrets)
3. **Default Values** (hardcoded fallbacks)

## 🛡️ Security Best Practices

### ✅ DO:
- Use `.env` files for local development
- Use Streamlit secrets for production
- Keep API keys in `.gitignore`d files
- Use different API keys for development/production
- Set `ENVIRONMENT=production` in production secrets

### ❌ DON'T:
- Commit `.env` or `secrets.toml` files to Git
- Share API keys in code or documentation
- Use production API keys in development
- Hardcode sensitive values in source code

## 🧪 Testing Configuration

### Test Local Configuration:
```bash
python -c "from config import Config; print(f'Environment: {Config.get_environment()}'); print(f'AI Provider: {Config.AI_PROVIDER}'); print(f'OpenRouter Key: {\"Set\" if Config.OPENROUTER_API_KEY else \"Not Set\"}')"
```

### Test Production Configuration:
Add this to your Streamlit app temporarily:
```python
import streamlit as st
from config import Config

st.write(f"Environment: {Config.get_environment()}")
st.write(f"AI Provider: {Config.AI_PROVIDER}")
st.write(f"OpenRouter Key: {'Set' if Config.OPENROUTER_API_KEY else 'Not Set'}")
```

## 🔄 Environment Detection

The system automatically detects production environment based on:

- `ENVIRONMENT=production` in configuration
- `STREAMLIT_SHARING=true` (Streamlit Cloud)
- `HEROKU` environment variable (Heroku deployment)
- Streamlit secrets containing `ENVIRONMENT=production`

## 📁 File Structure

```
trade_analyzer_pro/
├── .env                          # Local development (ignored by Git)
├── .env.example                  # Template for local setup
├── .streamlit/
│   ├── secrets.toml             # Local production testing (ignored by Git)
│   └── secrets.toml.example     # Template for production
├── config.py                    # Smart configuration loader
└── .gitignore                   # Excludes sensitive files
```

## 🚨 Troubleshooting

### "Configuration not found" errors:
1. Check if `.env` file exists and has correct format
2. Verify API keys are properly set
3. Ensure no extra spaces around `=` in configuration files

### "Environment detection issues":
1. Set `ENVIRONMENT=production` explicitly in production
2. Check that Streamlit secrets are properly formatted TOML
3. Verify configuration priority (env vars override secrets)

### "API key not working":
1. Test API keys with the debug scripts provided
2. Check if keys are properly loaded: `Config.OPENROUTER_API_KEY`
3. Verify no trailing spaces or quotes in configuration

## 📚 Additional Resources

- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
- [OpenRouter API Keys](https://openrouter.ai/keys)
- [OpenAI API Keys](https://platform.openai.com/api-keys)
- [Environment Variables Best Practices](https://12factor.net/config)