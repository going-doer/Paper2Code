"""
providers.py — Multi-provider LLM client abstraction.

Supported providers (all expose an OpenAI-compatible chat/completions endpoint,
except Google Gemini which is handled separately):

    openai       https://api.openai.com/v1
    groq         https://api.groq.com/openai/v1
    cerebras     https://api.cerebras.ai/v1
    openrouter   https://openrouter.ai/api/v1
    mistral      https://api.mistral.ai/v1
    github       https://models.inference.ai.azure.com
    sambanova    https://api.sambanova.ai/v1
    gemini       https://generativelanguage.googleapis.com/v1beta  (native REST)
    cohere       https://api.cohere.com/v2                         (native REST)
    cloudflare   https://api.cloudflare.com/client/v4/accounts/{id}/ai (native REST)

Usage
-----
    from providers import build_client, chat_complete, is_reasoning_model

    client = build_client(provider="groq", api_key="gsk_...")
    response = chat_complete(client, provider="groq",
                             model="llama-3.3-70b-versatile",
                             messages=[...])
    # response is always a normalised dict:
    #   {"choices": [{"message": {"role": "assistant", "content": "..."}}],
    #    "usage": {"prompt_tokens": int, "completion_tokens": int,
    #              "total_tokens": int,
    #              "prompt_tokens_details": {"cached_tokens": 0}}}

Free model suggestions per provider
-------------------------------------
  groq       : llama-3.3-70b-versatile, meta-llama/llama-4-scout-17b-16e-instruct,
               moonshotai/kimi-k2-instruct, qwen/qwen3-32b
  cerebras   : llama-3.3-70b, qwen3-32b, qwen3-235b, gpt-oss-120b
  openrouter : deepseek/deepseek-r1:free, meta-llama/llama-4-scout:free,
               qwen/qwen3-235b-a22b:free, microsoft/phi-4-reasoning:free
  mistral    : mistral-large-latest, mistral-small-latest, open-codestral-mamba,
               ministral-8b-latest
  github     : gpt-4o, gpt-4.1, o3, deepseek-r1, grok-3-mini
  sambanova  : Meta-Llama-3.3-70B-Instruct, Meta-Llama-3.1-405B-Instruct,
               Qwen2.5-72B-Instruct
  gemini     : gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite-preview-06-17
  cohere     : command-r-plus, command-a-03-2025, aya-expanse-32b
  cloudflare : @cf/meta/llama-3.3-70b-instruct-fp8-fast,
               @cf/qwen/qwq-32b, @cf/mistralai/mistral-7b-instruct-v0.2
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDER_BASE_URLS: Dict[str, str] = {
    "openai":      "https://api.openai.com/v1",
    "groq":        "https://api.groq.com/openai/v1",
    "cerebras":    "https://api.cerebras.ai/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
    "mistral":     "https://api.mistral.ai/v1",
    "github":      "https://models.inference.ai.azure.com",
    "sambanova":   "https://api.sambanova.ai/v1",
    # Native REST — base URLs used directly in the REST helpers
    "gemini":      "https://generativelanguage.googleapis.com/v1beta",
    "cohere":      "https://api.cohere.com/v2",
    "cloudflare":  "",  # requires account_id; built dynamically
}

# Providers that use the OpenAI Python SDK (openai-compatible base_url)
_OPENAI_SDK_PROVIDERS = {
    "openai", "groq", "cerebras", "openrouter", "mistral", "github", "sambanova"
}

# Reasoning / thinking models — these need special param handling
_REASONING_MODEL_PATTERNS = [
    r"o1", r"o3", r"o4", r"deepseek.*r1", r"qwq", r"qwen.*think",
    r"r1", r"sonar.*reasoning", r"kimi.*think",
]


def is_reasoning_model(model: str) -> bool:
    """Return True if the model name matches a known reasoning/thinking model."""
    m = model.lower()
    return any(re.search(p, m) for p in _REASONING_MODEL_PATTERNS)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_client(
    provider: str,
    api_key: Optional[str] = None,
    cloudflare_account_id: Optional[str] = None,
) -> Any:
    """
    Return a client object for the given provider.

    For OpenAI-compatible providers this is an ``openai.OpenAI`` instance.
    For native-REST providers (gemini, cohere, cloudflare) this is a plain
    dict carrying the credentials needed by ``chat_complete``.
    """
    provider = provider.lower()

    # Resolve API key from env if not supplied
    if not api_key:
        env_map = {
            "openai":     "OPENAI_API_KEY",
            "groq":       "GROQ_API_KEY",
            "cerebras":   "CEREBRAS_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "mistral":    "MISTRAL_API_KEY",
            "github":     "GITHUB_TOKEN",
            "sambanova":  "SAMBANOVA_API_KEY",
            "gemini":     "GEMINI_API_KEY",
            "cohere":     "COHERE_API_KEY",
            "cloudflare": "CLOUDFLARE_API_KEY",
        }
        env_var = env_map.get(provider, "OPENAI_API_KEY")
        api_key = os.environ.get(env_var, "")
        if not api_key:
            print(
                f"[WARNING] No API key found for provider '{provider}'. "
                f"Set the {env_var} environment variable.",
                file=sys.stderr,
            )

    if provider in _OPENAI_SDK_PROVIDERS:
        base_url = PROVIDER_BASE_URLS[provider]
        extra = {}
        if provider == "openrouter":
            extra["default_headers"] = {"HTTP-Referer": "https://github.com/Paper2Code"}
        return OpenAI(api_key=api_key, base_url=base_url, **extra)

    if provider == "gemini":
        return {"_provider": "gemini", "_api_key": api_key}

    if provider == "cohere":
        return {"_provider": "cohere", "_api_key": api_key}

    if provider == "cloudflare":
        account_id = cloudflare_account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            print(
                "[WARNING] Cloudflare provider requires CLOUDFLARE_ACCOUNT_ID env var.",
                file=sys.stderr,
            )
        return {
            "_provider": "cloudflare",
            "_api_key": api_key,
            "_account_id": account_id,
        }

    raise ValueError(f"Unknown provider: '{provider}'. "
                     f"Valid options: {sorted(PROVIDER_BASE_URLS)}")


# ---------------------------------------------------------------------------
# Unified chat completion
# ---------------------------------------------------------------------------

def chat_complete(
    client: Any,
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    n: int = 1,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict:
    """
    Call the chat completion endpoint and return a **normalised response dict**:

        {
          "choices": [
            {"message": {"role": "assistant", "content": "<text>"}}
            ...          # n items
          ],
          "usage": {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
            "prompt_tokens_details": {"cached_tokens": 0}
          },
          "_provider": "<name>",
          "_model": "<name>"
        }

    Parameters
    ----------
    client          : object returned by ``build_client``
    provider        : provider name (lowercase)
    model           : model identifier as accepted by the provider
    messages        : list of {"role": ..., "content": ...} dicts
    n               : number of completions to generate (not supported by all providers)
    temperature     : sampling temperature (ignored for reasoning models)
    reasoning_effort: "low"/"medium"/"high" — only sent when supported
    """
    provider = provider.lower()

    if provider in _OPENAI_SDK_PROVIDERS:
        return _call_openai_sdk(
            client, provider, model, messages, n, temperature, reasoning_effort
        )
    if provider == "gemini":
        return _call_gemini(client, model, messages, n, temperature)
    if provider == "cohere":
        return _call_cohere(client, model, messages, n, temperature)
    if provider == "cloudflare":
        return _call_cloudflare(client, model, messages)

    raise ValueError(f"Unknown provider: '{provider}'")


# ---------------------------------------------------------------------------
# OpenAI-SDK-compatible providers
# ---------------------------------------------------------------------------

def _call_openai_sdk(
    client: OpenAI,
    provider: str,
    model: str,
    messages: List[Dict],
    n: int,
    temperature: Optional[float],
    reasoning_effort: Optional[str],
) -> Dict:
    kwargs: Dict[str, Any] = {"model": model, "messages": messages}

    reasoning = is_reasoning_model(model)

    if reasoning:
        # reasoning_effort is only honoured by OpenAI o-series and a few others
        if reasoning_effort and provider in ("openai",):
            kwargs["reasoning_effort"] = reasoning_effort
        # temperature must NOT be set for o1/o3/o4 on OpenAI
        if provider != "openai":
            if temperature is not None:
                kwargs["temperature"] = temperature
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature

    # n > 1 — not supported by all providers; fall back to looping
    if n > 1 and provider in ("groq", "cerebras", "sambanova", "mistral"):
        return _loop_n(client, provider, model, messages, n, temperature, reasoning_effort)

    if n > 1:
        kwargs["n"] = n

    completion = client.chat.completions.create(**kwargs)
    return _normalise_openai(completion, provider, model)


def _loop_n(
    client: OpenAI,
    provider: str,
    model: str,
    messages: List[Dict],
    n: int,
    temperature: Optional[float],
    reasoning_effort: Optional[str],
) -> Dict:
    """Call the API n times and merge results into one normalised dict."""
    choices = []
    total_prompt = 0
    total_completion = 0
    total_cached = 0

    for _ in range(n):
        r = _call_openai_sdk(
            client, provider, model, messages, 1, temperature, reasoning_effort
        )
        choices.extend(r["choices"])
        total_prompt     += r["usage"]["prompt_tokens"]
        total_completion += r["usage"]["completion_tokens"]
        total_cached     += r["usage"]["prompt_tokens_details"].get("cached_tokens", 0)

    return {
        "choices": choices,
        "usage": {
            "prompt_tokens":        total_prompt,
            "completion_tokens":    total_completion,
            "total_tokens":         total_prompt + total_completion,
            "prompt_tokens_details": {"cached_tokens": total_cached},
        },
        "_provider": provider,
        "_model":    model,
    }


def _normalise_openai(completion, provider: str, model: str) -> Dict:
    raw = json.loads(completion.model_dump_json())
    usage = raw.get("usage") or {}
    prompt_tokens     = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    prompt_details    = usage.get("prompt_tokens_details") or {}
    cached_tokens     = (prompt_details.get("cached_tokens") or 0) if isinstance(prompt_details, dict) else 0

    choices = []
    for c in raw.get("choices", []):
        msg = c.get("message") or {}
        choices.append({
            "message": {
                "role":    msg.get("role", "assistant"),
                "content": msg.get("content") or "",
            }
        })

    return {
        "choices": choices,
        "usage": {
            "prompt_tokens":         prompt_tokens,
            "completion_tokens":     completion_tokens,
            "total_tokens":          prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
        "_provider": provider,
        "_model":    model,
        "_raw":      raw,   # kept for cost logging that expects the original shape
    }


# ---------------------------------------------------------------------------
# Google Gemini (native REST)
# ---------------------------------------------------------------------------

def _call_gemini(
    client: Dict,
    model: str,
    messages: List[Dict],
    n: int,
    temperature: Optional[float],
) -> Dict:
    api_key  = client["_api_key"]
    base_url = PROVIDER_BASE_URLS["gemini"]

    # Convert messages to Gemini format
    system_instruction = None
    contents = []
    for m in messages:
        role    = m["role"]
        content = m["content"]
        if role == "system":
            system_instruction = {"parts": [{"text": content}]}
        else:
            gemini_role = "user" if role == "user" else "model"
            contents.append({"role": gemini_role, "parts": [{"text": content}]})

    payload: Dict[str, Any] = {
        "contents":         contents,
        "generationConfig": {"candidateCount": n},
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction
    if temperature is not None:
        payload["generationConfig"]["temperature"] = temperature

    url = f"{base_url}/models/{model}:generateContent?key={api_key}"
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    choices = []
    for candidate in data.get("candidates", []):
        text = ""
        for part in candidate.get("content", {}).get("parts", []):
            text += part.get("text", "")
        choices.append({"message": {"role": "assistant", "content": text}})

    usage_meta = data.get("usageMetadata", {})
    prompt_tokens     = usage_meta.get("promptTokenCount", 0)
    completion_tokens = usage_meta.get("candidatesTokenCount", 0)

    return {
        "choices": choices,
        "usage": {
            "prompt_tokens":         prompt_tokens,
            "completion_tokens":     completion_tokens,
            "total_tokens":          prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "_provider": "gemini",
        "_model":    model,
    }


# ---------------------------------------------------------------------------
# Cohere (native REST v2)
# ---------------------------------------------------------------------------

def _call_cohere(
    client: Dict,
    model: str,
    messages: List[Dict],
    n: int,
    temperature: Optional[float],
) -> Dict:
    api_key = client["_api_key"]
    url     = "https://api.cohere.com/v2/chat"

    # Cohere v2 uses a similar messages format but the system msg is separate
    cohere_messages = []
    for m in messages:
        cohere_messages.append({"role": m["role"], "content": m["content"]})

    payload: Dict[str, Any] = {"model": model, "messages": cohere_messages}
    if temperature is not None:
        payload["temperature"] = temperature

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    choices = []
    total_prompt     = 0
    total_completion = 0

    for _ in range(n):
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        text = ""
        msg  = data.get("message", {})
        for item in msg.get("content", []):
            if item.get("type") == "text":
                text += item.get("text", "")
        choices.append({"message": {"role": "assistant", "content": text}})

        usage = data.get("usage", {})
        billed = usage.get("billed_units", {})
        total_prompt     += billed.get("input_tokens", 0)
        total_completion += billed.get("output_tokens", 0)

    return {
        "choices": choices,
        "usage": {
            "prompt_tokens":         total_prompt,
            "completion_tokens":     total_completion,
            "total_tokens":          total_prompt + total_completion,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "_provider": "cohere",
        "_model":    model,
    }


# ---------------------------------------------------------------------------
# Cloudflare Workers AI (native REST)
# ---------------------------------------------------------------------------

def _call_cloudflare(
    client: Dict,
    model: str,
    messages: List[Dict],
) -> Dict:
    api_key    = client["_api_key"]
    account_id = client["_account_id"]
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    )

    payload = {"messages": messages}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    result = data.get("result", {})
    text   = result.get("response", "")

    usage = result.get("usage", {})
    prompt_tokens     = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens":         prompt_tokens,
            "completion_tokens":     completion_tokens,
            "total_tokens":          prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "_provider": "cloudflare",
        "_model":    model,
    }


# ---------------------------------------------------------------------------
# Convenience: add provider/api_key args to any argparse parser
# ---------------------------------------------------------------------------

def add_provider_args(parser) -> None:
    """Add --provider and --api_key arguments to an argparse.ArgumentParser."""
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=sorted(PROVIDER_BASE_URLS.keys()),
        help=(
            "LLM provider to use. Default: openai. "
            "Free alternatives: groq, cerebras, openrouter, mistral, "
            "github, sambanova, gemini, cohere, cloudflare"
        ),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help=(
            "API key for the selected provider. "
            "If omitted, the corresponding environment variable is used "
            "(OPENAI_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY, "
            "OPENROUTER_API_KEY, MISTRAL_API_KEY, GITHUB_TOKEN, "
            "SAMBANOVA_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, "
            "CLOUDFLARE_API_KEY + CLOUDFLARE_ACCOUNT_ID)."
        ),
    )
