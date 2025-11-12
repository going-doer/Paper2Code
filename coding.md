# Coding Phase

## Purpose
This prompt template generates actual implementation code based on the planning and analysis phases. It transforms detailed specifications into working, production-ready code with proper error handling, documentation, and testing.

## System Prompt Template

```
You are an expert software engineer tasked with implementing code based on detailed specifications.

Your goal is to:
1. Write clean, maintainable, well-documented code
2. Follow best practices and coding standards
3. Implement proper error handling and validation
4. Include comprehensive docstrings and comments
5. Ensure code is testable and modular

Focus on:
- Code quality and readability
- Adherence to specifications
- Proper error handling
- Performance and efficiency
- Security best practices
- Comprehensive documentation
```

## User Prompt Template

```
# Problem Context
{PROBLEM_DESCRIPTION}

# Overall Plan
{OVERALL_PLAN_OUTPUT}

# Architecture Design
{ARCHITECTURE_DESIGN_OUTPUT}

# Component Analysis
{COMPONENT_ANALYSIS_OUTPUT}

# File to Implement
**File**: {FILE_PATH}
**Component**: {COMPONENT_NAME}
**Type**: {COMPONENT_TYPE}

# Dependencies (Already Implemented)
{IMPLEMENTED_DEPENDENCIES}

# Task
Implement the code for this component based on the detailed specification.

## Instructions

1. **Code Structure**
   - Follow the interface specification exactly
   - Implement all required methods and functions
   - Use appropriate class/function organization
   - Follow language-specific conventions

2. **Implementation**
   - Implement the algorithms as specified
   - Use the recommended data structures
   - Apply the specified design patterns
   - Handle all edge cases identified

3. **Error Handling**
   - Validate all inputs
   - Raise appropriate exceptions
   - Include helpful error messages
   - Handle resource cleanup

4. **Documentation**
   - Write comprehensive docstrings
   - Add inline comments for complex logic
   - Include usage examples
   - Document assumptions and limitations

5. **Testing Considerations**
   - Write testable code
   - Separate concerns
   - Avoid hard-coded values
   - Use dependency injection where appropriate

6. **Code Quality**
   - Follow PEP 8 (Python) or language-specific style guide
   - Use meaningful variable names
   - Keep functions focused and small
   - Avoid code duplication

## Output Format

Provide the complete implementation with the following structure:

### File: {file_path}

```python
"""
Module: {module_name}
Description: {Brief description of what this module does}

This module implements {functionality description}.

Key Components:
- ComponentName: Purpose
- Helper functions: Purpose

Dependencies:
- dependency1: Purpose
- dependency2: Purpose

Example Usage:
    >>> from module import ComponentName
    >>> obj = ComponentName(param=value)
    >>> result = obj.method(input)
    >>> print(result)

Author: {optional}
Date: {optional}
Version: {version}
"""

# ============================================================================
# Imports
# ============================================================================

# Standard library imports
import os
import sys
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from core.base import BaseClass
from utils.helpers import helper_function
from config import load_config

# ============================================================================
# Constants
# ============================================================================

# Module-level constants
DEFAULT_VALUE = 100
MAX_ITERATIONS = 1000
SUPPORTED_FORMATS = ['json', 'csv', 'parquet']

# ============================================================================
# Type Definitions
# ============================================================================

# Custom type aliases
ConfigType = Dict[str, Any]
ResultType = Dict[str, Union[int, float, str]]

# ============================================================================
# Exceptions
# ============================================================================

class ComponentError(Exception):
    """Base exception for this component."""
    pass

class ValidationError(ComponentError):
    """Raised when input validation fails."""
    pass

class ProcessingError(ComponentError):
    """Raised when processing encounters an error."""
    pass

# ============================================================================
# Data Classes (if applicable)
# ============================================================================

@dataclass
class DataStructure:
    """
    Data structure for {purpose}.

    Attributes:
        field1 (type): Description
        field2 (type): Description
        field3 (type): Description with default value
    """
    field1: int
    field2: str
    field3: float = 0.0

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.field1 < 0:
            raise ValueError("field1 must be non-negative")

# ============================================================================
# Main Component Class
# ============================================================================

class ComponentName(BaseClass):
    """
    {Component description - what it does and why}.

    This class implements {functionality}. It is designed to {purpose}.

    Attributes:
        attribute1 (type): Description
        attribute2 (type): Description
        _private_attr (type): Private attribute description

    Example:
        >>> component = ComponentName(param1=10, param2='test')
        >>> result = component.primary_method(input_data)
        >>> print(result)

    Note:
        Any important notes about usage, limitations, or requirements.
    """

    def __init__(
        self,
        param1: int,
        param2: str,
        param3: Optional[float] = None,
        config: Optional[ConfigType] = None
    ):
        """
        Initialize the component.

        Args:
            param1: Description of param1 and its purpose.
                Must be positive integer.
            param2: Description of param2 and its purpose.
                Cannot be empty string.
            param3: Optional parameter description.
                If None, uses default value from config.
            config: Optional configuration dictionary.
                If None, loads from default config file.

        Raises:
            ValueError: If param1 is not positive or param2 is empty.
            TypeError: If parameters are of wrong type.
            ConfigError: If configuration is invalid.

        Example:
            >>> component = ComponentName(param1=10, param2='test')
        """
        # Call parent class constructor if inheriting
        super().__init__()

        # Validate inputs
        self._validate_init_params(param1, param2, param3)

        # Load configuration
        self.config = config or load_config()

        # Initialize attributes
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3 or self.config.get('default_param3', 1.0)

        # Initialize internal state
        self._state = {}
        self._initialized = False
        self._logger = logging.getLogger(__name__)

        # Perform setup
        self._setup()

    def _validate_init_params(
        self,
        param1: int,
        param2: str,
        param3: Optional[float]
    ) -> None:
        """
        Validate initialization parameters.

        Args:
            param1: Parameter to validate
            param2: Parameter to validate
            param3: Parameter to validate

        Raises:
            ValueError: If validation fails
            TypeError: If types are incorrect
        """
        if not isinstance(param1, int):
            raise TypeError(f"param1 must be int, got {type(param1)}")

        if param1 <= 0:
            raise ValueError(f"param1 must be positive, got {param1}")

        if not isinstance(param2, str):
            raise TypeError(f"param2 must be str, got {type(param2)}")

        if not param2:
            raise ValueError("param2 cannot be empty")

        if param3 is not None and param3 < 0:
            raise ValueError(f"param3 must be non-negative, got {param3}")

    def _setup(self) -> None:
        """
        Perform internal setup and initialization.

        This method is called during __init__ to set up internal state
        and prepare the component for use.

        Raises:
            RuntimeError: If setup fails
        """
        try:
            # Initialize state
            self._state = {
                'status': 'initialized',
                'last_operation': None,
                'error_count': 0
            }

            # Set up logging
            self._logger.info(
                f"Initialized {self.__class__.__name__} with "
                f"param1={self.param1}, param2={self.param2}"
            )

            self._initialized = True

        except Exception as e:
            self._logger.error(f"Setup failed: {e}")
            raise RuntimeError(f"Failed to initialize component: {e}")

    def primary_method(
        self,
        input1: List[Any],
        input2: Dict[str, Any],
        option: bool = False
    ) -> ResultType:
        """
        Primary method implementing core functionality.

        This method {description of what it does and why}. It processes
        the input data according to {algorithm/approach} and returns
        {description of output}.

        Args:
            input1: Description of input1. Expected format and constraints.
                Must not be empty. Each element should be {type/format}.
            input2: Description of input2. Required keys: [key1, key2].
                Optional keys: [key3, key4].
            option: Flag to enable/disable {specific behavior}.
                Default is False.

        Returns:
            Dictionary containing processed results with keys:
                - 'result': Main result value (type)
                - 'metadata': Additional information (type)
                - 'status': Processing status ('success' or 'partial')

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing encounters an error
            ValueError: If input values are out of valid range

        Example:
            >>> component = ComponentName(param1=10, param2='test')
            >>> input1 = [1, 2, 3, 4, 5]
            >>> input2 = {'key1': 'value1', 'key2': 100}
            >>> result = component.primary_method(input1, input2)
            >>> print(result['result'])
            42

        Note:
            This method is not thread-safe. Use locks if calling from
            multiple threads.
        """
        # Validate inputs
        self._validate_inputs(input1, input2)

        # Log operation
        self._logger.debug(
            f"primary_method called with input1 length={len(input1)}, "
            f"input2 keys={list(input2.keys())}"
        )

        try:
            # Step 1: Preprocess inputs
            processed_input1 = self._preprocess_input1(input1)
            processed_input2 = self._preprocess_input2(input2)

            # Step 2: Main processing logic
            intermediate_result = self._process_core_logic(
                processed_input1,
                processed_input2,
                option
            )

            # Step 3: Post-process results
            final_result = self._postprocess_result(intermediate_result)

            # Step 4: Update state
            self._update_state('success', 'primary_method')

            # Step 5: Prepare output
            output = {
                'result': final_result,
                'metadata': {
                    'input_size': len(input1),
                    'option_used': option,
                    'timestamp': self._get_timestamp()
                },
                'status': 'success'
            }

            return output

        except Exception as e:
            self._logger.error(f"Error in primary_method: {e}")
            self._update_state('error', 'primary_method')
            raise ProcessingError(f"Processing failed: {e}") from e

    def _validate_inputs(
        self,
        input1: List[Any],
        input2: Dict[str, Any]
    ) -> None:
        """
        Validate method inputs.

        Args:
            input1: Input to validate
            input2: Input to validate

        Raises:
            ValidationError: If validation fails
        """
        if not input1:
            raise ValidationError("input1 cannot be empty")

        if not isinstance(input2, dict):
            raise ValidationError(f"input2 must be dict, got {type(input2)}")

        required_keys = ['key1', 'key2']
        missing_keys = [k for k in required_keys if k not in input2]
        if missing_keys:
            raise ValidationError(
                f"input2 missing required keys: {missing_keys}"
            )

    def _preprocess_input1(self, input1: List[Any]) -> List[Any]:
        """
        Preprocess input1 data.

        Args:
            input1: Raw input data

        Returns:
            Processed input data
        """
        # Implementation of preprocessing logic
        return [self._process_item(item) for item in input1]

    def _process_item(self, item: Any) -> Any:
        """
        Process a single item.

        Args:
            item: Item to process

        Returns:
            Processed item
        """
        # Implement item processing logic
        return item

    def _preprocess_input2(self, input2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess input2 data.

        Args:
            input2: Raw input dictionary

        Returns:
            Processed input dictionary
        """
        # Implementation of preprocessing logic
        processed = input2.copy()
        # Add processing steps here
        return processed

    def _process_core_logic(
        self,
        data1: List[Any],
        data2: Dict[str, Any],
        option: bool
    ) -> Any:
        """
        Implement the core processing algorithm.

        Args:
            data1: Preprocessed data1
            data2: Preprocessed data2
            option: Processing option flag

        Returns:
            Intermediate processing result
        """
        # Initialize result
        result = None

        # Main algorithm implementation
        if option:
            # Algorithm variant 1
            result = self._algorithm_variant1(data1, data2)
        else:
            # Algorithm variant 2
            result = self._algorithm_variant2(data1, data2)

        return result

    def _algorithm_variant1(
        self,
        data1: List[Any],
        data2: Dict[str, Any]
    ) -> Any:
        """Implement algorithm variant 1."""
        # Implementation details
        pass

    def _algorithm_variant2(
        self,
        data1: List[Any],
        data2: Dict[str, Any]
    ) -> Any:
        """Implement algorithm variant 2."""
        # Implementation details
        pass

    def _postprocess_result(self, result: Any) -> Any:
        """
        Post-process the result.

        Args:
            result: Raw result from processing

        Returns:
            Final processed result
        """
        # Implementation of post-processing
        return result

    def _update_state(self, status: str, operation: str) -> None:
        """
        Update internal state.

        Args:
            status: Current status
            operation: Operation that updated the state
        """
        self._state['status'] = status
        self._state['last_operation'] = operation
        if status == 'error':
            self._state['error_count'] += 1

    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()

    # Public utility methods

    def get_status(self) -> Dict[str, Any]:
        """
        Get current component status.

        Returns:
            Dictionary containing status information
        """
        return {
            'initialized': self._initialized,
            'state': self._state.copy(),
            'config': self.config
        }

    def reset(self) -> None:
        """
        Reset component to initial state.

        This clears all internal state and prepares the component
        for reuse.
        """
        self._state = {
            'status': 'initialized',
            'last_operation': None,
            'error_count': 0
        }
        self._logger.info("Component reset to initial state")

    def __repr__(self) -> str:
        """String representation of the component."""
        return (
            f"{self.__class__.__name__}("
            f"param1={self.param1}, "
            f"param2='{self.param2}', "
            f"param3={self.param3})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.__class__.__name__} instance with {self._state['status']} status"

# ============================================================================
# Helper Functions
# ============================================================================

def helper_function(
    arg1: type,
    arg2: type,
    option: bool = False
) -> return_type:
    """
    Helper function description.

    Args:
        arg1: Description
        arg2: Description
        option: Description

    Returns:
        Description of return value

    Raises:
        ExceptionType: When exception occurs

    Example:
        >>> result = helper_function(arg1_value, arg2_value)
        >>> print(result)
    """
    # Implementation
    pass

def utility_function(data: Any) -> Any:
    """
    Utility function for common operations.

    Args:
        data: Input data

    Returns:
        Processed data
    """
    # Implementation
    pass

# ============================================================================
# Main Execution (if applicable)
# ============================================================================

def main():
    """
    Main execution function.

    This function demonstrates basic usage of the component.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    try:
        # Initialize component
        component = ComponentName(param1=10, param2='example')

        # Prepare inputs
        input1 = [1, 2, 3, 4, 5]
        input2 = {'key1': 'value1', 'key2': 100}

        # Call primary method
        result = component.primary_method(input1, input2)

        # Display results
        print(f"Result: {result}")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main()
```

### Additional Files

If the component requires additional files (tests, config, etc.):

#### Test File: tests/test_{component_name}.py

```python
"""
Unit tests for {component_name} module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

from core.component import ComponentName, ValidationError, ProcessingError

class TestComponentName(unittest.TestCase):
    """Test cases for ComponentName class."""

    def setUp(self):
        """Set up test fixtures."""
        self.param1 = 10
        self.param2 = 'test'
        self.component = ComponentName(
            param1=self.param1,
            param2=self.param2
        )

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_initialization(self):
        """Test component initialization."""
        self.assertEqual(self.component.param1, self.param1)
        self.assertEqual(self.component.param2, self.param2)
        self.assertTrue(self.component._initialized)

    def test_initialization_invalid_param1(self):
        """Test initialization with invalid param1."""
        with self.assertRaises(ValueError):
            ComponentName(param1=-1, param2='test')

    def test_initialization_invalid_param2(self):
        """Test initialization with invalid param2."""
        with self.assertRaises(ValueError):
            ComponentName(param1=10, param2='')

    def test_primary_method_valid_input(self):
        """Test primary_method with valid input."""
        input1 = [1, 2, 3]
        input2 = {'key1': 'value1', 'key2': 100}

        result = self.component.primary_method(input1, input2)

        self.assertIn('result', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['status'], 'success')

    def test_primary_method_empty_input1(self):
        """Test primary_method with empty input1."""
        input1 = []
        input2 = {'key1': 'value1', 'key2': 100}

        with self.assertRaises(ValidationError):
            self.component.primary_method(input1, input2)

    def test_primary_method_missing_keys(self):
        """Test primary_method with missing required keys."""
        input1 = [1, 2, 3]
        input2 = {'key1': 'value1'}  # Missing key2

        with self.assertRaises(ValidationError):
            self.component.primary_method(input1, input2)

    def test_get_status(self):
        """Test get_status method."""
        status = self.component.get_status()

        self.assertIn('initialized', status)
        self.assertIn('state', status)
        self.assertTrue(status['initialized'])

    def test_reset(self):
        """Test reset method."""
        # Perform some operation
        input1 = [1, 2, 3]
        input2 = {'key1': 'value1', 'key2': 100}
        self.component.primary_method(input1, input2)

        # Reset
        self.component.reset()

        # Check state is reset
        status = self.component.get_status()
        self.assertEqual(status['state']['error_count'], 0)

    @patch('core.component.helper_function')
    def test_with_mocked_dependency(self, mock_helper):
        """Test with mocked dependencies."""
        mock_helper.return_value = 'mocked_value'

        # Test with mocked dependency
        # ... test implementation ...

        mock_helper.assert_called_once()

if __name__ == '__main__':
    unittest.main()
```
```

## Example Placeholders

Replace these placeholders with actual information:

- **{PROBLEM_DESCRIPTION}**: Original problem description
- **{OVERALL_PLAN_OUTPUT}**: Output from overall planning phase
- **{ARCHITECTURE_DESIGN_OUTPUT}**: Output from architecture design phase
- **{COMPONENT_ANALYSIS_OUTPUT}**: Output from component analysis phase
- **{FILE_PATH}**: Path of the file to implement
- **{COMPONENT_NAME}**: Name of the component
- **{COMPONENT_TYPE}**: Type (class, module, function, script)
- **{IMPLEMENTED_DEPENDENCIES}**: Code from dependency files already implemented

## Best Practices

### Code Quality
1. **Readability**: Code should be self-documenting with clear names
2. **Modularity**: Break complex functions into smaller helper functions
3. **DRY Principle**: Don't repeat yourself - extract common logic
4. **Single Responsibility**: Each function/class should have one clear purpose

### Documentation
1. **Docstrings**: Every public function/class needs comprehensive docstrings
2. **Comments**: Explain "why", not "what" - the code shows what
3. **Type Hints**: Use type hints for all parameters and returns
4. **Examples**: Include usage examples in docstrings

### Error Handling
1. **Validate Early**: Check inputs at the start of functions
2. **Specific Exceptions**: Use or create specific exception types
3. **Helpful Messages**: Error messages should guide users to solutions
4. **Resource Cleanup**: Use context managers or try/finally for cleanup

### Testing
1. **Unit Tests**: Test each component in isolation
2. **Edge Cases**: Test boundary conditions and edge cases
3. **Error Cases**: Test that errors are raised appropriately
4. **Mocking**: Mock external dependencies for unit tests

### Performance
1. **Avoid Premature Optimization**: Focus on correctness first
2. **Profile Before Optimizing**: Measure to find actual bottlenecks
3. **Use Appropriate Data Structures**: Choose the right tool for the job
4. **Consider Space-Time Tradeoffs**: Balance memory usage and speed

### Security
1. **Input Validation**: Validate and sanitize all inputs
2. **Avoid Injection**: Use parameterized queries, safe evaluation
3. **Secure Defaults**: Choose safe default values
4. **Principle of Least Privilege**: Request minimal necessary permissions

## Language-Specific Considerations

### Python
- Follow PEP 8 style guide
- Use type hints (PEP 484)
- Prefer list comprehensions and generators
- Use context managers for resources
- Leverage standard library

### JavaScript/TypeScript
- Use ESLint for code quality
- Prefer async/await over callbacks
- Use TypeScript for type safety
- Follow Airbnb or Standard style guide
- Handle promises properly

### Java
- Follow Java naming conventions
- Use interfaces for abstraction
- Implement proper equals/hashCode
- Use generics for type safety
- Handle checked exceptions appropriately

### Other Languages
- Follow language-specific conventions
- Use idiomatic patterns
- Leverage language features appropriately

## Sequential Implementation

For projects with multiple files:

1. **Follow Dependency Order**: Implement in the order specified by logic design
2. **Test Each Component**: Verify each file works before moving to the next
3. **Integration Testing**: Test components together as they're completed
4. **Iterative Refinement**: Refine earlier components if issues are discovered

## Notes

- This phase implements one file at a time based on the analysis specifications
- Code should match the interface specification exactly
- All edge cases and error conditions from analysis should be handled
- Include comprehensive documentation and tests
- Focus on code quality and maintainability
- The output is production-ready code
