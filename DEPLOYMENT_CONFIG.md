# Deployment Configuration Guide

This guide covers deployment configuration for the Trade Analyzer Pro application with AI Strategy Builder and Backtesting Engine features.

## Prerequisites

### Required Dependencies
All dependencies are listed in `requirements.txt`:
- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation
- `plotly>=5.0.0` - Interactive charts
- `requests>=2.28.0` - HTTP requests
- `numpy>=1.24.0` - Numerical computing
- `openai>=1.0.0` - AI Strategy Builder (OpenAI API)
- `vectorbt>=0.25.0` - Backtesting Engine
- `hypothesis>=6.0.0` - Property-based testing
- `pytz>=2023.3` - Timezone handling

### Python Version
- **Minimum**: Python 3.9
- **Recommended**: Python 3.11.7 (specified in `runtime.txt`)
- **Tested**: Python 3.11.x

## Environment Configuration

### Required Environment Variables
```env
# OpenAI API Key (Required for AI Strategy Builder)
OPENAI_API_KEY=your_openai_api_key_here
```

### Optional Environment Variables
```env
# OpenAI Configuration
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.1

# Backtesting Defaults
DEFAULT_STARTING_BALANCE=10000
DEFAULT_LEVERAGE=1.0
DEFAULT_COMMISSION=0.0
DEFAULT_SLIPPAGE=0.0

# Data Processing
MAX_UPLOAD_SIZE_MB=2048
DEFAULT_TIMEZONE=UTC
```

## Deployment Platforms

### 1. Streamlit Cloud

**Configuration Files:**
- `requirements.txt` ✅ Ready
- `runtime.txt` ✅ Ready (Python 3.11.7)
- `.streamlit/secrets.toml` (create this)

**Secrets Configuration:**
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your_openai_api_key_here"
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

**Deployment Steps:**
1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Configure secrets in Streamlit Cloud dashboard
4. Deploy application

### 2. Heroku

**Configuration Files:**
- `Procfile` ✅ Ready
- `setup.sh` ✅ Ready
- `requirements.txt` ✅ Ready
- `runtime.txt` ✅ Ready

**Environment Variables:**
Set via Heroku CLI or dashboard:
```bash
heroku config:set OPENAI_API_KEY=your_key_here
heroku config:set OPENAI_MODEL=gpt-4
# ... other variables
```

**Deployment Steps:**
1. Create Heroku app: `heroku create your-app-name`
2. Set environment variables
3. Deploy: `git push heroku main`

### 3. Docker

**Dockerfile** (create this):
```dockerfile
FROM python:3.11.7-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**docker-compose.yml** (create this):
```yaml
version: '3.8'
services:
  trade-analyzer:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=${OPENAI_MODEL:-gpt-4}
      - DEFAULT_STARTING_BALANCE=${DEFAULT_STARTING_BALANCE:-10000}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### 4. AWS/GCP/Azure

**Additional Requirements:**
- Configure cloud-specific secrets management
- Set up load balancing if needed
- Configure auto-scaling
- Set up monitoring and logging

## Security Considerations

### API Key Security
1. **Never commit API keys** to version control
2. **Use environment variables** or secrets management
3. **Rotate keys regularly**
4. **Monitor API usage** for anomalies
5. **Set usage limits** on OpenAI dashboard

### Application Security
1. **Input validation** - All user inputs are validated
2. **Code execution** - Strategy code runs in controlled environment
3. **File uploads** - Size limits and format validation
4. **Error handling** - Sensitive information not exposed

## Performance Optimization

### Memory Management
- **Large datasets**: Chunked processing for files > 100MB
- **Backtesting**: Optimized VectorBT operations
- **Caching**: Strategy generation results cached

### CPU Optimization
- **Parallel processing**: VectorBT uses NumPy vectorization
- **Efficient algorithms**: Optimized indicator calculations
- **Progress tracking**: Long operations show progress

### Storage
- **Temporary files**: Cleaned up automatically
- **Results caching**: Optional result persistence
- **Upload limits**: Configurable via `MAX_UPLOAD_SIZE_MB`

## Monitoring and Logging

### Application Metrics
- **API usage**: OpenAI API calls and tokens
- **Backtest performance**: Execution time and memory usage
- **User activity**: Feature usage statistics
- **Error rates**: Failed operations and exceptions

### Health Checks
- **API connectivity**: OpenAI API status
- **Memory usage**: Available system memory
- **Disk space**: Available storage
- **Response times**: Application performance

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## Troubleshooting

### Common Deployment Issues

**1. Import Errors**
- Verify all dependencies in `requirements.txt`
- Check Python version compatibility
- Ensure all custom modules are included

**2. Memory Issues**
- Increase container memory limits
- Optimize data processing for large files
- Enable garbage collection

**3. API Errors**
- Verify OpenAI API key is set correctly
- Check API quotas and limits
- Implement proper error handling

**4. Performance Issues**
- Profile application with large datasets
- Optimize VectorBT operations
- Consider caching strategies

### Debug Mode
Enable debug mode for troubleshooting:
```env
STREAMLIT_DEBUG=true
PYTHONPATH=/app
```

## Scaling Considerations

### Horizontal Scaling
- **Load balancing**: Multiple app instances
- **Session management**: Stateless design
- **Database**: External storage for results

### Vertical Scaling
- **Memory**: 4GB+ recommended for large backtests
- **CPU**: Multi-core for parallel processing
- **Storage**: SSD for better I/O performance

## Maintenance

### Regular Updates
1. **Dependencies**: Update `requirements.txt` monthly
2. **Security patches**: Monitor for vulnerabilities
3. **API versions**: Keep OpenAI client updated
4. **Performance monitoring**: Regular performance reviews

### Backup Strategy
1. **Configuration**: Version control all config files
2. **User data**: Backup uploaded datasets
3. **Results**: Archive important backtest results
4. **Logs**: Retain logs for debugging

## Support

For deployment issues:
1. Check this documentation
2. Review application logs
3. Test with minimal configuration
4. Contact support with specific error messages