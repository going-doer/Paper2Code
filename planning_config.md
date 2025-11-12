# Planning Phase: Configuration File

## Purpose
This prompt template generates configuration files (typically YAML or JSON) that store parameters, hyperparameters, and settings extracted during the overall planning phase. This separates configuration from code and makes the system more maintainable.

## System Prompt Template

```
You are an expert software engineer tasked with creating configuration files for a software system.

Your goal is to:
1. Extract all configurable parameters from the requirements
2. Organize parameters into logical groups
3. Set appropriate default values
4. Document each parameter clearly
5. Create a well-structured configuration file

Consider:
- Separation of concerns (dev/test/prod configs)
- Type safety and validation
- Clear documentation for each parameter
- Sensible defaults
- Environment-specific overrides
```

## User Prompt Template

```
# Problem Context
{PROBLEM_DESCRIPTION}

# Overall Plan
{OVERALL_PLAN_OUTPUT}

# Parameters Identified
{PARAMETERS_LIST}

# Requirements
{REQUIREMENTS}

# Task
Create a comprehensive configuration file for this system.

## Instructions

1. **Parameter Organization**
   - Group related parameters together
   - Use hierarchical structure
   - Separate concerns (paths, model params, runtime settings, etc.)

2. **Value Assignment**
   - Provide sensible defaults
   - Include comments explaining each parameter
   - Note valid ranges or options
   - Indicate which parameters are critical vs. optional

3. **Configuration Structure**
   - Use clear naming conventions
   - Make the structure intuitive
   - Include metadata (version, description)
   - Consider environment-specific settings

4. **Documentation**
   - Document each parameter's purpose
   - Explain valid values and constraints
   - Note dependencies between parameters
   - Provide examples

## Output Format

Provide configuration file(s) in the following format:

### Main Configuration File (config.yaml)

```yaml
# ============================================================================
# Configuration File
# Description: {Brief description of the system}
# Version: 1.0
# ============================================================================

# ----------------------------------------------------------------------------
# System Settings
# ----------------------------------------------------------------------------
system:
  # Application name
  name: "application_name"

  # Version
  version: "1.0.0"

  # Debug mode (true/false)
  debug: false

  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  log_level: "INFO"

  # Output directory for results
  output_dir: "./outputs"

# ----------------------------------------------------------------------------
# Runtime Parameters
# ----------------------------------------------------------------------------
runtime:
  # Number of threads/workers
  num_workers: 4

  # Batch size for processing
  batch_size: 32

  # Maximum memory allocation (in MB)
  max_memory: 4096

  # Timeout for operations (in seconds)
  timeout: 300

# ----------------------------------------------------------------------------
# Component-Specific Settings
# ----------------------------------------------------------------------------
component_name:
  # Enable/disable this component
  enabled: true

  # Parameter 1: Description
  # Valid range: [min, max] or Options: [option1, option2]
  parameter1: default_value

  # Parameter 2: Description
  parameter2: default_value

  # Nested settings
  subsection:
    sub_parameter1: value
    sub_parameter2: value

# ----------------------------------------------------------------------------
# Data Configuration
# ----------------------------------------------------------------------------
data:
  # Input data path
  input_path: "./data/input"

  # Output data path
  output_path: "./data/output"

  # Data format (csv, json, parquet, etc.)
  format: "json"

  # Preprocessing options
  preprocessing:
    normalize: true
    handle_missing: "drop"  # Options: drop, fill, interpolate

# ----------------------------------------------------------------------------
# Model/Algorithm Parameters
# ----------------------------------------------------------------------------
model:
  # Model type or algorithm name
  type: "algorithm_name"

  # Hyperparameter 1
  # Description: What this controls
  # Default: value, Range: [min, max]
  hyperparameter1: value

  # Hyperparameter 2
  hyperparameter2: value

  # Advanced settings
  advanced:
    setting1: value
    setting2: value

# ----------------------------------------------------------------------------
# External Dependencies
# ----------------------------------------------------------------------------
dependencies:
  # API endpoints
  api:
    base_url: "https://api.example.com"
    timeout: 30
    retry_attempts: 3

  # Database connection (if applicable)
  database:
    host: "localhost"
    port: 5432
    name: "dbname"
    # Note: Store sensitive credentials in environment variables
    username: "${DB_USERNAME}"
    password: "${DB_PASSWORD}"

# ----------------------------------------------------------------------------
# Validation and Testing
# ----------------------------------------------------------------------------
validation:
  # Enable validation
  enabled: true

  # Validation split ratio
  split_ratio: 0.2

  # Metrics to compute
  metrics:
    - "metric1"
    - "metric2"
    - "metric3"

  # Test cases
  test_cases:
    basic: true
    edge_cases: true
    stress_test: false

# ----------------------------------------------------------------------------
# Performance Tuning
# ----------------------------------------------------------------------------
performance:
  # Caching
  cache_enabled: true
  cache_size: 1000

  # Optimization level (0-3)
  optimization_level: 2

  # Profiling
  enable_profiling: false

# ----------------------------------------------------------------------------
# Feature Flags
# ----------------------------------------------------------------------------
features:
  # Enable experimental features
  experimental: false

  # Feature 1
  feature1_enabled: true

  # Feature 2
  feature2_enabled: false
```

### Environment-Specific Configurations

**config.dev.yaml** (Development environment):
```yaml
system:
  debug: true
  log_level: "DEBUG"

runtime:
  num_workers: 2

validation:
  test_cases:
    stress_test: true
```

**config.prod.yaml** (Production environment):
```yaml
system:
  debug: false
  log_level: "WARNING"

runtime:
  num_workers: 8
  max_memory: 8192

performance:
  cache_enabled: true
  optimization_level: 3
```

### Configuration Schema (Optional)

If using schema validation, provide a schema file:

**config_schema.json**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["system", "runtime"],
  "properties": {
    "system": {
      "type": "object",
      "required": ["name", "version"],
      "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "debug": {"type": "boolean"},
        "log_level": {
          "type": "string",
          "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        }
      }
    },
    "runtime": {
      "type": "object",
      "properties": {
        "num_workers": {"type": "integer", "minimum": 1},
        "batch_size": {"type": "integer", "minimum": 1},
        "timeout": {"type": "integer", "minimum": 0}
      }
    }
  }
}
```

### Configuration Loading Code Template

Provide a code snippet for loading the configuration:

**Python Example**:
```python
import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "config.yaml",
                env: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to main config file
        env: Environment (dev/prod) for environment-specific overrides

    Returns:
        Configuration dictionary
    """
    # Load main config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load environment-specific overrides
    if env:
        env_config_path = f"config.{env}.yaml"
        if os.path.exists(env_config_path):
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
            # Deep merge env_config into config
            config = deep_merge(config, env_config)

    # Replace environment variables
    config = replace_env_vars(config)

    return config

def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def replace_env_vars(config: Dict) -> Dict:
    """Replace ${VAR} with environment variable values."""
    # Implementation to recursively replace env vars
    pass
```
```

## Example Placeholders

Replace these placeholders with actual information:

- **{PROBLEM_DESCRIPTION}**: Original problem description
- **{OVERALL_PLAN_OUTPUT}**: Output from overall planning step
- **{PARAMETERS_LIST}**: List of parameters identified in planning
- **{REQUIREMENTS}**: System requirements

## Configuration Best Practices

1. **Naming Conventions**
   - Use snake_case for YAML keys
   - Use descriptive names
   - Group related settings

2. **Documentation**
   - Comment every parameter
   - Explain valid values and ranges
   - Note dependencies

3. **Security**
   - Never hardcode secrets
   - Use environment variables for sensitive data
   - Provide .env.example template

4. **Validation**
   - Define schemas for type checking
   - Validate on load
   - Fail fast with clear error messages

5. **Versioning**
   - Include version in config
   - Document breaking changes
   - Maintain backward compatibility

## Notes

- Configuration files should be created early in the planning phase
- Keep configuration separate from code
- Use environment-specific overrides rather than duplicating entire configs
- Consider using a configuration management library (e.g., Hydra, python-decouple)
- Validate configuration at application startup
