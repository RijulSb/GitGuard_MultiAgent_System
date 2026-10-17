"""GitGuard Main CLI Entrypoint."""
import sys
import subprocess

def print_help():
    print("""
GitGuard: 10-Agent RAG System over Pro Git
==========================================

Usage: python main.py [command]

Available commands:
  api     Start the FastAPI service (port 8000)
  ui      Start the Streamlit dashboard (port 8501)
  demo    Run the interactive 3-scenario demo
  gate    Run the fast CI regression gate check
  test    Run the comprehensive unit test suite (pytest)
  help    Show this help message
""")

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "-h", "--help"):
        print_help()
        return

    cmd = sys.argv[1].lower()

    if cmd == "api":
        print("Starting FastAPI service on http://localhost:8000...")
        subprocess.run([sys.executable, "-m", "uvicorn", "src.service.api:app", "--host", "0.0.0.0", "--port", "8000"])
    elif cmd == "ui":
        print("Starting Streamlit UI on http://localhost:8501...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/app.py"])
    elif cmd == "demo":
        print("Running GitGuard Demo...")
        subprocess.run([sys.executable, "demo.py"])
    elif cmd == "gate":
        print("Running CI Regression Gate...")
        subprocess.run([sys.executable, "-m", "src.eval.ci_gate"])
    elif cmd == "test":
        print("Running Unit Test Suite...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    else:
        print(f"Unknown command: '{cmd}'")
        print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
