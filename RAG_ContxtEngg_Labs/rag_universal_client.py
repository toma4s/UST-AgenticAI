#!/usr/bin/env python3
"""
================================================================================
  RAG UNIVERSAL CLIENT & ENTERPRISE MODEL ENGINE (`rag_universal_client.py`)
  Advanced RAG & Context Engineering Labs — UST Global & Client Architecture
================================================================================

This module provides high-reliability, real model inference across all 11 RAG labs.
- Primary Cloud Frontier : Anthropic Claude (claude-sonnet-4-6 / claude-3-5-sonnet-latest)
- Local Backup Runtime   : Local Open-Source Ollama (llama3.2:1b / qwen3.8:27b)
- Policy: Zero mock/canned fallbacks. Real inferences only; raises descriptive
  errors if all configured models are unreachable.
================================================================================
"""

import os
import sys
import json
import time
import base64
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure UTF-8 console output for Windows terminals
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ------------------------------------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION & KEY RESOLUTION
# ------------------------------------------------------------------------------
CANDIDATE_ENV_PATHS = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent / "Labs" / ".env",
    Path.cwd() / ".env",
    Path.cwd() / "RAG_ContxtEngg_Labs" / ".env",
]

def load_rag_env():
    """Loads environment variables from prioritized .env locations."""
    for env_path in CANDIDATE_ENV_PATHS:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'").strip('"')
                            if k not in os.environ and v:
                                os.environ[k] = v
            except Exception:
                pass

load_rag_env()

def get_anthropic_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        load_rag_env()
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key.startswith("your_") or key == "sk-ant-..." or len(key) < 15:
        return ""
    return key

def get_llm_config() -> Dict[str, Any]:
    load_rag_env()
    key = get_anthropic_api_key()
    return {
        "anthropic_key_configured": bool(key),
        "anthropic_key_masked": f"{key[:8]}...{key[-4:]}" if key else "NOT_CONFIGURED",
        "anthropic_model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-gama"),
        "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_fast_model": os.environ.get("OLLAMA_FAST_MODEL", "llama3.2:1b"),
        "ollama_reasoning_model": os.environ.get("OLLAMA_REASONING_MODEL", "qwen3.8:27b"),
    }

# ------------------------------------------------------------------------------
# 2. CANONICAL DATA DIRECTORY RESOLVER
# ------------------------------------------------------------------------------
def get_rag_data_dir() -> Path:
    """
    Returns the canonical, verified Path to data/sample_documents containing turbomachinery_manual.md.
    """
    candidates = [
        Path(r"C:\Users\Administrator\Desktop\AgenticAI\UST-AgenticAI\RAG_ContxtEngg_Labs\data\sample_documents"),
        Path(__file__).parent / "UST" / "RAG_ContxtEngg_Labs" / "data" / "sample_documents",
        Path(__file__).parent / "RAG_ContxtEngg_Labs" / "data" / "sample_documents",
        Path(__file__).parent / "data" / "sample_documents",
        Path.cwd() / "UST" / "RAG_ContxtEngg_Labs" / "data" / "sample_documents",
        Path.cwd() / "RAG_ContxtEngg_Labs" / "data" / "sample_documents",
        Path.cwd() / "data" / "sample_documents",
    ]
    for p in candidates:
        try:
            if p.resolve().exists() and (p.resolve() / "turbomachinery_manual.md").exists():
                return p.resolve()
        except Exception:
            pass
    for p in candidates:
        try:
            if p.resolve().exists() and any(p.resolve().iterdir()):
                return p.resolve()
        except Exception:
            pass
    return Path(r"C:\Users\Administrator\Desktop\AgenticAI\UST-AgenticAI\RAG_ContxtEngg_Labs\data\sample_documents")

# ------------------------------------------------------------------------------
# 3. REAL LLM GENERATION: ANTHROPIC PRIMARY WITH OLLAMA BACKUP (NO CANNED TEXT)
# ------------------------------------------------------------------------------
def call_rag_llm(
    prompt: str,
    system_prompt: str = "You are an enterprise AI assistant for UST Global (UST) & UST Global. Be technical, precise, and concise.",
    model: Optional[str] = None,
    #temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: int = 40
) -> Dict[str, Any]:
    """
    Invokes the primary Anthropic Claude model. If unavailable, falls back to local Ollama.
    Strictly raises RuntimeError if neither model can be reached (Zero canned mocks).
    """
    cfg = get_llm_config()
    anthropic_key = get_anthropic_api_key()
    target_anthropic_model = model or cfg["anthropic_model"]
    errors = []

    # 1. ATTEMPT PRIMARY TIER: ANTHROPIC CLAUDE API
    if anthropic_key:
        t0 = time.time()
        try:
            url = "https://llmproxy.ustrnd.com/v1/messages"
            headers = {
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": target_anthropic_model,
                "max_tokens": max_tokens,
                #"temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
            }
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text_blocks = [
                    b.get("text", "") for b in data.get("content", []) 
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                output_text = "\n".join(text_blocks).strip()
                if output_text:
                    latency = round((time.time() - t0) * 1000, 1)
                    return {
                        "text": output_text,
                        "tier_used": f"PRIMARY_CLOUD ({target_anthropic_model} - Live)",
                        "model": target_anthropic_model,
                        "latency_ms": latency,
                        "estimated_tokens": len(output_text) // 4,
                        "status": "SUCCESS"
                    }
        except Exception as e:
            errors.append(f"Anthropic API Error ({target_anthropic_model}): {e}")

    # 2. ATTEMPT BACKUP TIER: LOCAL OLLAMA (OPEN-SOURCE)
    ollama_url = f"{cfg['ollama_base_url']}/api/generate"
    target_ollama_model = cfg["ollama_fast_model"]
    t0 = time.time()
    try:
        combined_prompt = f"{system_prompt}\n\nUser Question:\n{prompt}\n\nAnswer:"
        payload = {
            "model": target_ollama_model,
            "prompt": combined_prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        req = urllib.request.Request(
            ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            output_text = data.get("response", "").strip()
            if output_text:
                latency = round((time.time() - t0) * 1000, 1)
                return {
                    "text": output_text,
                    "tier_used": f"BACKUP_OLLAMA_LOCAL ({target_ollama_model} - Live)",
                    "model": target_ollama_model,
                    "latency_ms": latency,
                    "estimated_tokens": len(output_text) // 4,
                    "status": "SUCCESS"
                }
    except Exception as e:
        errors.append(f"Ollama Local Error ({target_ollama_model}): {e}")

    # 3. ZERO MOCKS: RAISE RUNTIME ERROR IF BOTH TIERS FAILED
    err_summary = " | ".join(errors)
    raise RuntimeError(
        f"[LLM INFERENCE FAILURE] Neither Anthropic Claude nor Local Ollama could be reached. "
        f"Details: {err_summary}. "
        f"Ensure ANTHROPIC_API_KEY is valid in .env or run 'ollama serve' with '{target_ollama_model}'."
    )

# ------------------------------------------------------------------------------
# 4. REAL MULTIMODAL VISION INFERENCE (NO CANNED TEXT)
# ------------------------------------------------------------------------------
def call_rag_vision(
    image_path: Path,
    prompt: str = "Analyze this turbomachinery engineering chart. Extract all operational curves, axes units, and surge boundaries.",
    timeout: int = 45
) -> Dict[str, Any]:
    """
    Performs multimodal vision analysis using Anthropic Claude Vision or Ollama Vision.
    Strictly raises RuntimeError if neither can process the image.
    """
    cfg = get_llm_config()
    anthropic_key = get_anthropic_api_key()
    target_model = cfg["anthropic_model"]
    image_path = Path(image_path)
    errors = []

    if not image_path.exists():
        raise FileNotFoundError(f"Image asset not found at {image_path}")

    with open(image_path, "rb") as img_f:
        b64_image = base64.b64encode(img_f.read()).decode("utf-8")

    # 1. ATTEMPT ANTHROPIC CLAUDE VISION
    if anthropic_key:
        t0 = time.time()
        try:
            url = "https://llmproxy.ustrnd.com/v1/messages"
            headers = {
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": target_model,
                "max_tokens": 1200,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64_image,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            }
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text_blocks = [
                    b.get("text", "") for b in data.get("content", []) 
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                output_text = "\n".join(text_blocks).strip()
                if output_text:
                    latency = round((time.time() - t0) * 1000, 1)
                    return {
                        "text": output_text,
                        "tier_used": f"PRIMARY_VISION ({target_model} - Live)",
                        "model": target_model,
                        "latency_ms": latency,
                        "status": "SUCCESS"
                    }
        except Exception as e:
            errors.append(f"Anthropic Vision Error: {e}")

    # 2. ATTEMPT OLLAMA LOCAL VISION (llama3.2-vision or llava)
    ollama_url = f"{cfg['ollama_base_url']}/api/generate"
    for v_model in ["llama3.2-vision:latest", "llava:latest", "llama3.2:1b"]:
        try:
            payload = {
                "model": v_model,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False
            }
            req = urllib.request.Request(
                ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                output_text = data.get("response", "").strip()
                if output_text:
                    return {
                        "text": output_text,
                        "tier_used": f"BACKUP_VISION_LOCAL ({v_model} - Live)",
                        "model": v_model,
                        "status": "SUCCESS"
                    }
        except Exception as e:
            errors.append(f"Ollama Vision ({v_model}): {e}")

    raise RuntimeError(
        f"[VISION INFERENCE FAILURE] Neither Anthropic Claude Vision nor Ollama Vision was available. "
        f"Details: {' | '.join(errors)}"
    )

# ------------------------------------------------------------------------------
# 5. CLI DIAGNOSTIC RUNNER
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = get_llm_config()
    data_dir = get_rag_data_dir()
    print("=" * 80)
    print("  RAG UNIVERSAL CLIENT & ENTERPRISE MODEL ENGINE STATUS")
    print("=" * 80)
    print(f"  * Anthropic API Key Active : {cfg['anthropic_key_configured']} ({cfg['anthropic_key_masked']})")
    print(f"  * Primary Model            : {cfg['anthropic_model']}")
    print(f"  * Local Ollama URL         : {cfg['ollama_base_url']}")
    print(f"  * Local Fast SLM (Backup)  : {cfg['ollama_fast_model']}")
    print(f"  * Local Deep Model (Backup): {cfg['ollama_reasoning_model']}")
    print(f"  * Verified Data Directory  : {data_dir}")
    print("=" * 80)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\n[*] Sending test query to live model pipeline...")
        test_res = call_rag_llm("Summarize the dry gas seal leakage limit in 1 concise sentence.")
        print(f"  Tier Used: {test_res['tier_used']}")
        print(f"  Latency  : {test_res['latency_ms']} ms")
        print(f"  Response : {test_res['text']}")

