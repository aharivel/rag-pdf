"""
Run this first to verify both Ollama instances are reachable and models are available.
Usage: python check_connection.py
"""
import httpx
import sys
import config


def check_ollama(url: str, models_needed: list[str], label: str) -> bool:
    print(f"\nChecking {label} Ollama at {url} ...")

    try:
        r = httpx.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
    except httpx.ConnectError:
        print(f"  [FAIL] Cannot reach {url}")
        if "192.168" in url:
            print("         → Is Ollama running on your Windows PC?")
            print("         → Is OLLAMA_HOST=0.0.0.0 set on Windows?")
            print("         → Is port 11434 allowed in Windows Firewall?")
        else:
            print("         → Is local Ollama running? (podman start ollama)")
        return False
    except Exception as e:
        print(f"  [FAIL] Unexpected error: {e}")
        return False

    print(f"  [OK]  Reachable")
    available = [m["name"] for m in r.json().get("models", [])]

    ok = True
    for model in models_needed:
        found = any(m.startswith(model.split(":")[0]) for m in available)
        if found:
            print(f"  [OK]  {model}")
        else:
            print(f"  [MISSING] {model}  →  ollama pull {model}")
            ok = False

    return ok


def check():
    llm_ok    = check_ollama(config.OLLAMA_LLM_URL,   [config.LLM_MODEL],   "LLM  ")
    embed_ok  = check_ollama(config.OLLAMA_EMBED_URL,  [config.EMBED_MODEL], "Embed")

    print()
    if llm_ok and embed_ok:
        print("  All good! You can now run index_pdfs.py\n")
    else:
        print("  Fix the issues above then re-run this check.\n")
        sys.exit(1)


if __name__ == "__main__":
    check()
