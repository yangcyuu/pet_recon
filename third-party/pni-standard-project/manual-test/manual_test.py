from pathlib import Path
import os
import subprocess
import sys
import time

supported_commands = ["--test-name", "--debug", "--amide-after"]
for arg in sys.argv[1:]:
    if arg.startswith("--") and arg not in supported_commands:
        print(f"Error: Unsupported command line argument '{arg}'")
        sys.exit(1)

def get_test_name_from_args():
    if "--test-name" in sys.argv:
        idx = sys.argv.index("--test-name")
        if idx + 1 < len(sys.argv):
            if sys.argv[idx + 1].startswith("manual-test/"):
                sys.argv[idx + 1] = sys.argv[idx + 1][len("manual-test/"):]
            return sys.argv[idx + 1]
        else:
            raise ValueError("参数 '--test-name' 后面缺少测试名称")
    print("Usage: python manual_test.py --test-name <test_name> [--debug]")
    sys.exit(1)
    return None

cwd = Path.cwd() # Check if we are in the correct directory
if cwd.name == "pni-standard-project":
    print("Running manual_test.py")
else:
    print("Please run this script from the 'pni-standard-project' directory.")
    sys.exit(1)

global_include = cwd
test_name = get_test_name_from_args()
cpp_name = test_name.split("/")[-1].removesuffix(".cpp")
test_local_dir = "/".join(test_name.split("/")[:-1])
test_dir = cwd / "manual-test" / test_local_dir/"test"
print(f"Running test: {test_name}, cpp file: {cpp_name}.cpp, test directory: {test_dir}")
test_dir.mkdir(parents=True, exist_ok=True) # Create test directory and any missing parent directories
try:
    subprocess.run(["python3", "build.py"], capture_output=True, check=True) # Build the project (make sure the project is newest built.)
except subprocess.CalledProcessError:
    print("First build attempt failed, retrying without capture_output for more details...")
    subprocess.run(["python3", "build.py"], capture_output=False, check=True)
os.chdir(test_dir)
compiler = "g++-13"
pkg_config = subprocess.run( # Get compiler and linker flags for libpni
    ["pkg-config", "--cflags", "--libs", "libpni"],
    check=True,
    capture_output=True,
    text=True,
)
pni_libraries = pkg_config.stdout.strip()

print("Compiling and running the test...")
subprocess.run( # Compile and run the test
    [compiler, "-std=c++23", "../"+str(cpp_name)+".cpp", "-o",str(cpp_name) ,"-I"+str(global_include), *pni_libraries.split(), "-O2", "-g"],
    check=True,
)

if "--debug" in sys.argv:
    subprocess.run(["gdb","--ex","set print thread-events off","--ex","run", "./"+str(cpp_name)], check=True)
else:
    start_time = time.time()
    subprocess.run(["./"+str(cpp_name)], check=True) # Run the test
    end_time = time.time()
    print(f"Test execution time: {end_time - start_time:.2f} seconds")

if "--amide-after" in sys.argv:
    subprocess.run(["amide"], check=True) # Open Amide for visualization

# response = input("Did the test pass? (y/n): ")
# if response.lower() != "y":
#     print("Please check the test output and fix any issues.")