# AI Strategy Builder Module

This module provides functionality for converting natural language trading strategy descriptions into executable Python code and optionally to Pine Script format.

## Components

- **StrategyPromptProcessor**: Converts natural language prompts into Python strategy code using OpenAI GPT API
- **PineScriptConverter**: Translates Python strategy code to Pine Script v5 format
- **CodeGenerator**: Generates structured strategy code with proper interfaces and templates

## Configuration

The module requires an OpenAI API key to be set in the environment:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Or create a `.env` file in the project root with:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

```python
from ai_strategy_builder import StrategyPromptProcessor

processor = StrategyPromptProcessor()
strategy = processor.process_prompt("Create a moving average crossover strategy")
```

## Implementation Status

- ✅ Module structure created
- ⏳ Core functionality (to be implemented in subsequent tasks)