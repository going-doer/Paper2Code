@echo off
:: paper2code.bat — one-liner launcher
:: Usage:
::   paper2code.bat https://arxiv.org/pdf/2510.01193
::   paper2code.bat https://arxiv.org/pdf/2510.01193 --provider groq --model llama-3.3-70b-versatile
::   paper2code.bat C:\papers\mypaper.pdf --provider openai --model o3-mini
::   paper2code.bat https://arxiv.org/abs/2510.01193 --api_key sk-...
::
:: All options:
::   --provider  openai|groq|cerebras|openrouter|mistral|github|sambanova|gemini|cohere|cloudflare
::   --model     model name for that provider (default: o3-mini for openai)
::   --api_key   explicit API key (otherwise uses env var)
::   --output    output root directory (default: outputs)
::   --debug     after pipeline, run debugging agent if reproduce.ps1 fails

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0paper2code.ps1" %*
