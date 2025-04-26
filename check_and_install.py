import subprocess
import sys

# List of modules with the name to import and the name to install via pip if different
modules = [
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("matplotlib.pyplot", "matplotlib"),  # aliasing, still part of matplotlib
    ("matplotlib.colors", "matplotlib"),
    ("os", None),  # standard lib
    ("glob", None),  # standard lib
    ("astroalign", "astroalign"),
    ("astropy.io", "astropy"),
    ("photutils", "photutils"),
    ("photutils.centroids", "photutils"),
   
]

all_successful = True

for import_name, pip_name in modules:
    try:
        __import__(import_name)
        print(f"✅ {import_name} imported successfully.")
    except ImportError:
        all_successful = False
        if pip_name:
            print(f"❌ {import_name} not found. Attempting to install {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        else:
            print(f"⚠️ {import_name} is a standard library module and should be available.")

if all_successful:
    print("🎉 All libraries are successfully installed and imported!")
else:
    print("🔁 Re-run the script to verify all imports after installation.")

