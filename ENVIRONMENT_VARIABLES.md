# Environment Variables Configuration

This document describes all environment variables used by the Trade Analyzer Pro application.

## Required Variables

### OpenAI API Configuration

**OPENAI_API_KEY** (Required)
- **Description**: Your OpenAI API key for AI Strategy Builder functionality
- **How to get**: Visit [OpenAI Platform](https://platform.openai.com/api-keys) to create an API key
- **Example**: `OPENAI_API_KEY=sk-proj-your-key-here`
- **Security**: Keep this secret! Never commit to version control

## Optional Variables

### OpenAI Model Configuration

**OPENAI_MODEL** (Optional)
- **Description**: OpenAI model to use for strategy generation
- **Default**: `gpt-4`
- **Options**: `gpt-4`, `gpt-3.5-turbo`
- **Example**: `OPENAI_MODEL=gpt-4`

**OPENAI_MAX_TOKENS** (Optional)
- **Description**: Maximum tokens for OpenAI API responses
- **Default**: `2000`
- **Range**: `100-4000`
- **Example**: `OPENAI_MAX_TOKENS=2000`

**OPENAI_TEMPERATURE** (Optional)
- **Description**: Creativity level for AI responses (0.0 = deterministic, 1.0 = creative)
- **Default**: `0.1`
- **Range**: `0.0-1.0`
- **Example**: `OPENAI_TEMPERATURE=0.1`

### Backtesting Default Parameters

**DEFAULT_STARTING_BALANCE** (Optional)
- **Description**: Default starting balance for backtests
- **Default**: `10000`
- **Example**: `DEFAULT_STARTING_BALANCE=10000`

**DEFAULT_LEVERAGE** (Optional)
- **Description**: Default leverage for backtests
- **Default**: `1.0`
- **Example**: `DEFAULT_LEVERAGE=1.0`

**DEFAULT_COMMISSION** (Optional)
- **Description**: Default commission rate (as decimal, e.g., 0.001 = 0.1%)
- **Default**: `0.0`
- **Example**: `DEFAULT_COMMISSION=0.0`

**DEFAULT_SLIPPAGE** (Optional)
- **Description**: Default slippage rate (as decimal, e.g., 0.001 = 0.1%)
- **Default**: `0.0`
- **Example**: `DEFAULT_SLIPPAGE=0.0`

### Data Processing Configuration

**MAX_UPLOAD_SIZE_MB** (Optional)
- **Description**: Maximum file size for data uploads in MB
- **Default**: `2048`
- **Example**: `MAX_UPLOAD_SIZE_MB=2048`

**DEFAULT_TIMEZONE** (Optional)
- **Description**: Default timezone for data processing
- **Default**: `UTC`
- **Options**: Any valid timezone (e.g., `US/Eastern`, `Europe/London`)
- **Example**: `DEFAULT_TIMEZONE=UTC`

## Configuration Methods

### Method 1: Environment Variables
Set environment variables in your system:

**Windows:**
```cmd
set OPENAI_API_KEY=your_key_here
set OPENAI_MODEL=gpt-4
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-4
```

### Method 2: .env File (Recommended)
Create a `.env` file in the `trade_analyzer_pro` directory:

```env
# Copy from .env.example and fill in your values
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.1

DEFAULT_STARTING_BALANCE=10000
DEFAULT_LEVERAGE=1.0
DEFAULT_COMMISSION=0.0
DEFAULT_SLIPPAGE=0.0

MAX_UPLOAD_SIZE_MB=2048
DEFAULT_TIMEZONE=UTC
```

### Method 3: Streamlit Secrets (Cloud Deployment)
For Streamlit Cloud deployment, add to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_key_here"
OPENAI_MODEL = "gpt-4"
OPENAI_MAX_TOKENS = 2000
OPENAI_TEMPERATURE = 0.1

DEFAULT_STARTING_BALANCE = 10000
DEFAULT_LEVERAGE = 1.0
DEFAULT_COMMISSION = 0.0
DEFAULT_SLIPPAGE = 0.0

MAX_UPLOAD_SIZE_MB = 2048
DEFAULT_TIMEZONE = "UTC"
```

## Security Best Practices

1. **Never commit API keys**: Add `.env` to your `.gitignore` file
2. **Use environment-specific keys**: Different keys for development/production
3. **Rotate keys regularly**: Generate new API keys periodically
4. **Monitor usage**: Check OpenAI usage dashboard for unexpected activity
5. **Limit permissions**: Use API keys with minimal required permissions

## Troubleshooting

### Common Issues

**"OpenAI API Key Required" Error**
- Ensure `OPENAI_API_KEY` is set correctly
- Check for typos in the environment variable name
- Restart the application after setting the variable

**"Invalid API Key" Error**
- Verify the API key is active on OpenAI platform
- Check for extra spaces or characters in the key
- Ensure you have sufficient OpenAI credits

**"Model not found" Error**
- Verify you have access to the specified model
- Check if the model name is spelled correctly
- Try using `gpt-3.5-turbo` if `gpt-4` is not available

### Validation

The application will validate your configuration on startup and display:
- ✅ Green checkmarks for properly configured variables
- ⚠️ Yellow warnings for missing optional variables
- ❌ Red errors for missing required variables

## Support

For configuration issues:
1. Check this documentation
2. Verify your `.env` file format
3. Test with minimal configuration first
4. Check application logs for detailed error messages