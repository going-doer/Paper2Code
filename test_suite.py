"""
Paper2Code Windows — Test Suite
Runs without any API key (structural/import/logic tests only).
"""
import sys
import os
import json
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "codes"))

PASS = 0
FAIL = 0

def ok(name):
    global PASS
    PASS += 1
    print(f"  PASS  {name}")

def fail(name, err):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}: {err}")

# ── 1. Import all modules ──────────────────────────────────────────────────
print("\n[1] Import checks")
modules = {
    "utils":          "codes/utils.py",
    "providers":      "codes/providers.py",
    "0_pdf_process":  "codes/0_pdf_process.py",
    "1_planning":     "codes/1_planning.py",
    "2_analyzing":    "codes/2_analyzing.py",
    "3_coding":       "codes/3_coding.py",
    "3.1_coding_sh":  "codes/3.1_coding_sh.py",
    "4_debugging":    "codes/4_debugging.py",
    "eval":           "codes/eval.py",
    "1.2_rag_config": "codes/1.2_rag_config.py",
}
import importlib.util
for name, path in modules.items():
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None, "spec is None"
        ok(f"file exists: {path}")
    except Exception as e:
        fail(f"file exists: {path}", e)

# ── 2. providers module ────────────────────────────────────────────────────
print("\n[2] providers module")
try:
    from providers import build_client, chat_complete, is_reasoning_model, add_provider_args
    ok("import providers")
except Exception as e:
    fail("import providers", e)
    sys.exit(1)

# is_reasoning_model
tests = [
    ("o3-mini", True), ("o1", True), ("deepseek-r1", True),
    ("qwq-32b", True), ("gpt-4o", False), ("llama-3.3-70b", False),
    ("mistral-large", False), ("gemini-2.5-pro", False),
]
for model, expected in tests:
    try:
        result = is_reasoning_model(model)
        assert result == expected, f"got {result}, want {expected}"
        ok(f"is_reasoning_model('{model}') == {expected}")
    except Exception as e:
        fail(f"is_reasoning_model('{model}')", e)

# add_provider_args
try:
    p = argparse.ArgumentParser()
    add_provider_args(p)
    args = p.parse_args(["--provider", "groq", "--api_key", "test123"])
    assert args.provider == "groq"
    assert args.api_key == "test123"
    ok("add_provider_args parses --provider groq --api_key test123")
except Exception as e:
    fail("add_provider_args", e)

# build_client — OpenAI-SDK providers
for prov in ["openai", "groq", "cerebras", "openrouter", "mistral", "github", "sambanova"]:
    try:
        c = build_client(prov, api_key="dummy-key")
        assert hasattr(c, "chat"), "missing .chat"
        ok(f"build_client('{prov}') -> OpenAI client")
    except Exception as e:
        fail(f"build_client('{prov}')", e)

# build_client — native REST providers
for prov in ["gemini", "cohere"]:
    try:
        c = build_client(prov, api_key="dummy-key")
        assert isinstance(c, dict) and c.get("_provider") == prov
        ok(f"build_client('{prov}') -> REST dict")
    except Exception as e:
        fail(f"build_client('{prov}')", e)

try:
    c = build_client("cloudflare", api_key="dummy-key", cloudflare_account_id="acc123")
    assert c["_account_id"] == "acc123"
    ok("build_client('cloudflare') -> REST dict with account_id")
except Exception as e:
    fail("build_client('cloudflare')", e)

# ── 3. utils module ────────────────────────────────────────────────────────
print("\n[3] utils module")
try:
    import utils
    ok("import utils")
except Exception as e:
    fail("import utils", e)

fake_resp = {
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
}
try:
    cost = utils.cal_cost(fake_resp, "gpt-4o")
    assert isinstance(cost, dict) and "total_cost" in cost
    ok(f"cal_cost(response, 'gpt-4o') total_cost={cost['total_cost']}")
except Exception as e:
    fail("cal_cost gpt-4o", e)

try:
    cost = utils.cal_cost(fake_resp, "unknown-model-xyz")
    assert isinstance(cost, dict) and cost.get("total_cost") == 0.0
    ok("cal_cost(response, 'unknown-model-xyz') total_cost=0 (graceful fallback)")
except Exception as e:
    fail("cal_cost unknown model", e)

# ── 4. 0_pdf_process.py functional test ───────────────────────────────────
print("\n[4] 0_pdf_process.py functional test")
try:
    out_path = "test_output_cleaned.json"
    # Call via subprocess to test the script end-to-end
    import subprocess
    result = subprocess.run(
        [sys.executable, "codes/0_pdf_process.py",
         "--input_json_path", "examples/Transformer.json",
         "--output_json_path", out_path],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"returncode={result.returncode}\n{result.stderr}"
    assert os.path.exists(out_path), "output file not created"
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "title" in data, "missing 'title' key"
    assert "abstract" in data, "missing 'abstract' key"
    ok(f"0_pdf_process.py -> {out_path} (title: {data.get('title','?')[:40]}...)")
    os.remove(out_path)
except Exception as e:
    fail("0_pdf_process.py end-to-end", e)

# ── 5. PowerShell scripts exist ────────────────────────────────────────────
print("\n[5] PowerShell script presence")
ps_scripts = [
    "scripts/run.ps1",
    "scripts/run_llm.ps1",
    "scripts/run_latex.ps1",
    "scripts/run_latex_llm.ps1",
    "scripts/run_debug.ps1",
]
for s in ps_scripts:
    if os.path.exists(s):
        ok(f"exists: {s}")
    else:
        fail(f"exists: {s}", "file not found")

# ── 6. PowerShell script content checks ───────────────────────────────────
print("\n[6] PowerShell script content")
try:
    with open("scripts/run.ps1", encoding="utf-8") as f:
        content = f.read()
    assert "--provider" in content, "--provider not passed in run.ps1"
    assert "$PROVIDER" in content, "$PROVIDER variable not in run.ps1"
    assert "$GPT_VERSION" in content, "$GPT_VERSION not in run.ps1"
    ok("run.ps1 has $PROVIDER, $GPT_VERSION, --provider")
except Exception as e:
    fail("run.ps1 content", e)

try:
    with open("scripts/run_debug.ps1", encoding="utf-8") as f:
        content = f.read()
    assert "--provider" in content, "--provider not passed in run_debug.ps1"
    ok("run_debug.ps1 has --provider")
except Exception as e:
    fail("run_debug.ps1 content", e)

# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  Results: {PASS} passed, {FAIL} failed")
print("=" * 55)
if FAIL:
    sys.exit(1)
