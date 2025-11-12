# Planning Phase: Overall Plan

## Purpose
This prompt template generates a high-level understanding of the problem and creates an overall implementation plan. It extracts the core methodology, components, and requirements from the problem description.

## System Prompt Template

```
You are an expert software engineer tasked with analyzing a complex coding problem and creating a comprehensive implementation plan.

Your goal is to:
1. Understand the core requirements and objectives
2. Identify the key components and modules needed
3. Extract important parameters, configurations, and constraints
4. Create a structured plan for implementation

Focus on:
- Breaking down the problem into manageable components
- Identifying dependencies between components
- Extracting technical specifications and requirements
- Planning the overall architecture approach
```

## User Prompt Template

```
# Problem Description
{PROBLEM_DESCRIPTION}

# Requirements
{REQUIREMENTS}

# Constraints
{CONSTRAINTS}

# Task
Please analyze this problem and create an overall implementation plan.

## Instructions

1. **Problem Analysis**
   - Summarize the core objective
   - Identify the main functionality required
   - List key technical requirements

2. **Component Identification**
   - Identify major modules/components needed
   - Describe the purpose of each component
   - Note relationships between components

3. **Parameter Extraction**
   - List configurable parameters
   - Identify default values and valid ranges
   - Note any constraints on parameters

4. **Implementation Strategy**
   - Describe the overall approach
   - Identify key algorithms or techniques needed
   - Note any external dependencies or libraries

5. **Experimental Setup** (if applicable)
   - Define test scenarios
   - Identify metrics for evaluation
   - Plan validation approach

## Output Format

Provide your analysis in the following structured format:

### 1. Problem Summary
Brief overview of what needs to be implemented and why.

### 2. Core Components
List of main modules/components with descriptions:
- **Component Name**: Purpose and key functionality

### 3. Key Parameters
Table of configurable parameters:
| Parameter | Description | Default Value | Valid Range |
|-----------|-------------|---------------|-------------|
| param_name | what it controls | default | constraints |

### 4. Implementation Approach
Description of the overall strategy and methodology.

### 5. Dependencies
- External libraries needed
- System requirements
- Data requirements

### 6. Validation Plan
How to test and validate the implementation.
```

## Example Placeholders

Replace these placeholders with actual problem-specific information:

- **{PROBLEM_DESCRIPTION}**: Detailed description of the problem to solve
- **{REQUIREMENTS}**: Specific functional and non-functional requirements
- **{CONSTRAINTS}**: Technical constraints, performance requirements, resource limits

## Notes

- This is the first step in the planning phase
- The output will inform the architecture design and logic design steps
- Focus on high-level understanding rather than implementation details
- Extract all relevant parameters and configurations for later use
