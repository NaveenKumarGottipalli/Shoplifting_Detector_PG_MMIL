"""Make `import cv2` work on hosts without libGL (e.g. Streamlit Community Cloud) without relying on apt packages.

ultralytics depends on the GUI build `opencv-python`, which needs libGL.so.1. If that import fails, swap it for
`opencv-python-headless` once (via pip, or uv when the venv has no pip) and retry.
Best practice remains: put packages.txt (libgl1, libglib2.0-0) in the repository ROOT so the library is installed by apt.
"""
import importlib, subprocess, sys, shutil

_DONE = {"tried": False}


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def _pip(args_pip, args_uv):
    r = _run([sys.executable, "-m", "pip", *args_pip])
    if r.returncode == 0:
        return True
    uv = shutil.which("uv")
    if uv:
        r = _run([uv, "pip", *args_uv, "--python", sys.executable])
        return r.returncode == 0
    return False


def ensure_opencv():
    """Return None if cv2 imports, else a human-readable explanation."""
    try:
        import cv2  # noqa: F401
        return None
    except ImportError as first:
        if "libGL" not in str(first) and "libglib" not in str(first).lower():
            return f"cv2 import failed: {first}"
        if _DONE["tried"]:
            return f"cv2 still fails after the headless swap: {first}"
        _DONE["tried"] = True
        sys.modules.pop("cv2", None)
        _pip(["uninstall", "-y", "opencv-python", "opencv-contrib-python"], ["uninstall", "opencv-python", "opencv-contrib-python"])
        ok = _pip(["install", "--force-reinstall", "--no-deps", "opencv-python-headless"],
                  ["install", "--reinstall", "--no-deps", "opencv-python-headless"])
        importlib.invalidate_caches()
        sys.modules.pop("cv2", None)
        try:
            import cv2  # noqa: F401
            return None
        except ImportError as second:
            return (f"cv2 needs libGL.so.1 and the automatic switch to opencv-python-headless {'failed' if not ok else 'did not help'}: {second}. "
                    "Add a packages.txt containing 'libgl1' and 'libglib2.0-0' to the REPOSITORY ROOT and reboot the app.")
