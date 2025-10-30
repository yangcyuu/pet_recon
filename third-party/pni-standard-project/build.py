from pathlib import Path
import subprocess
import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--help":
    print("Usage: python3 build.py [--clean] [--rebuild] [--jN] [--auto-test]")
    print("  --clean      : Clean the build directory and exit.")
    print("  --rebuild    : Clean the build directory and continue building.")
    print("  --jN         : Use N threads for building (e.g., -j4 for 4 threads). Default is full-threaded.")
    print("  --auto-test  : Run automatic tests after building.")
    exit(0)

if Path.cwd().name != "pni-standard-project":
    raise RuntimeError("Script must be run from the pni-standard-project directory.")

config_file_name = "./include/PnI-Config.hpp"
cfg_path = Path(config_file_name)
if not cfg_path.exists():# Clear the file if it exists
    print("Please run config.py first.")
    exit(1)

autogen_dir = Path("./src/autogen")
autogen_dir.mkdir(parents=True, exist_ok=True)
subprocess.run( # Compile and run the test
    ["pni-serialization-tool", 
     "--op=./src/autogen", 
     "--of=autogen_xml",
     "--ifs",
    *[str(path) for path in Path("xml").rglob("*.xml")],
     "--n=openpni::autogen"],
    check=True,
)

# Clean build directory if --clean is passed
if "--clean" in sys.argv:
    if Path("build").exists():
        print("Cleaning build directory...")
        subprocess.run(["rm", "-rf", "build"], check=True)
        print("Cleaned build directory.")
    else:
        print("Build directory does not exist, nothing to clean.")
    rebuild_flag = False
    if "--rebuild" in sys.argv:
        rebuild_flag = True
    if not rebuild_flag:
        exit(0)

subprocess.run( # Compile and run the test
    ["cmake", "-S", ".", "-B", "build"],
    check=True,
)

# Determine multi-threading parameter for make
multi_thread_compiling = "-j"
for params in sys.argv:
    if params.startswith("-j"):
        multi_thread_compiling = params

subprocess.run(
    ["cmake", "--build", "build", "--config", "Release", "--", multi_thread_compiling],
    check=True,
)
subprocess.run( 
    ["sudo", "cmake", "--install", "build", ],
    check=True,
)
subprocess.run(
    ["sudo","ldconfig"],
    check=True,
)
print("Build successful.")

# Run auto-test if --auto-test is passed
if "--auto-test" in sys.argv:
    print("Running auto-tests...")
    os.chdir("./auto-test") 
    subprocess.run(
        ["cmake", "-S", ".", "-B", "build"],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", "--", multi_thread_compiling],
        check=True,
    )
    subprocess.run(
        ["./build/runTests"],
        check=True,
    )
    print("Auto-tests completed successfully.")
    
