"""
Verification script for AI-Enabled UAV Disaster Assessment Project dependencies.
"""
import sys

packages = [
    ("Python", lambda: sys.version.split()[0]),
    ("ultralytics", lambda: __import__("ultralytics").__version__),
    ("opencv-python (cv2)", lambda: __import__("cv2").__version__),
    ("numpy", lambda: __import__("numpy").__version__),
    ("pandas", lambda: __import__("pandas").__version__),
    ("pydantic", lambda: __import__("pydantic").__version__),
    ("fastapi", lambda: __import__("fastapi").__version__),
    ("uvicorn", lambda: __import__("uvicorn").__version__),
    ("sqlalchemy", lambda: __import__("sqlalchemy").__version__),
    ("pytest", lambda: __import__("pytest").__version__),
    ("ruff", lambda: __import__("ruff").__version__ if hasattr(__import__("ruff"), "__version__") else "installed (cli tool)"),
    ("python-multipart", lambda: __import__("multipart").__version__ if hasattr(__import__("multipart"), "__version__") else "installed"),
]

print("=" * 60)
print("Verifying Installed Dependencies...")
print("=" * 60)

failed = []
for name, get_ver in packages:
    try:
        ver = get_ver()
        print(f"[OK] {name:25} : {ver}")
    except Exception as e:
        print(f"[FAIL] {name:23} : {e}")
        failed.append((name, str(e)))

print("=" * 60)
if failed:
    print(f"FAILED: {len(failed)} package(s) could not be verified.")
    sys.exit(1)
else:
    print("ALL DEPENDENCIES VERIFIED SUCCESSFULLY!")
    sys.exit(0)
