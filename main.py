import os
import runpy
import sys


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(project_root, "scripts")
    sys.path.insert(0, scripts_dir)
    runpy.run_path(os.path.join(scripts_dir, "main.py"), run_name="__main__")
