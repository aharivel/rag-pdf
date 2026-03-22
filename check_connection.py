"""
Run this first to verify your Windows PC Ollama is reachable and models are available.
Usage: python check_connection.py
"""
import httpx
import sys
import config

def check():
    print(f"\nChecking Ollama at {config.OLLAMA_BASE_URL} ...\n")

    try:
        r = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except httpx.ConnectError:
        print(f"  [FAIL] Cannot reach {config.OLLAMA_BASE_URL}")
        print("         → Is Ollama running on your Windows PC?")
        print("         → Is OLLAMA_HOST=0.0.0.0 set on Windows?")
        print("         → Is port 11434 allowed in Windows Firewall?")
        sys.exit(1)
    except Exception as e:
        print(f"  [FAIL] Unexpected error: {e}")
        sys.exit(1)

    print(f"  [OK]  Ollama is reachable")

    models = [m["name"] for m in r.json().get("models", [])]
    print(f"\nAvailable models: {models}\n")

    missing = []
    for model in [config.LLM_MODEL, config.EMBED_MODEL]:
        # Match by prefix (e.g. "llama3.1:8b" matches "llama3.1:8b-instruct-q4_K_M")
        found = any(m.startswith(model.split(":")[0]) for m in models)
        if found:
            print(f"  [OK]  {model} found")
        else:
            print(f"  [MISSING] {model} — run on Windows PC: ollama pull {model}")
            missing.append(model)

    if missing:
        print(f"\n  Pull missing models on Windows PC then re-run this check.")
        sys.exit(1)

    print("\n  All good! You can now run index_pdfs.py\n")

if __name__ == "__main__":
    check()
