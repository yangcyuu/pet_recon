from pathlib import Path
import os
import subprocess
import sys
import time

supported_commands = ["--test-name", "--debug", "--amide-after","--move-to-workspace","--skip-run"]
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
test_name = get_test_name_from_args().removesuffix(".cpp")
cpp_name = test_name.split("/")[-1]
cpp_file = str(cwd / "manual-test" / test_name) + ".cpp"
print(f"Test name: {test_name}, C++ file: {cpp_file}, C++ file name: {cpp_name}.cpp")
if not Path(cpp_file).exists():
    print(f"Test file '{cpp_file}' does not exist.")
    sys.exit(1)

def get_workspace_from_args():
    if "--move-to-workspace" in sys.argv:
        idx = sys.argv.index("--move-to-workspace")
        if idx + 1 < len(sys.argv):
            workspace = sys.argv[idx + 1]
            if os.path.isdir(workspace):
                return workspace
            else:
                raise ValueError(f"指定的工作区目录不存在: {workspace}")
        else:
            raise ValueError("参数 '--move-to-workspace' 后面缺少目录路径")
    else:
        test_local_dir = "/".join(test_name.split("/")[:-1]).split("/")[-1]
        test_dir = cwd / "manual-test" / test_local_dir / "test" 
        test_dir.mkdir(parents=True, exist_ok=True)
        return str(test_dir)

workspace = get_workspace_from_args()
print(f"Running test: {test_name}, cpp file: {cpp_name}.cpp, test directory: {workspace}")

try:
    subprocess.run(["python3", "build.py"], capture_output=True, check=True) # Build the project (make sure the project is newest built.)
except subprocess.CalledProcessError:
    print("First build attempt failed, retrying without capture_output for more details...")
    subprocess.run(["python3", "build.py"], capture_output=False, check=True)

compiler = "g++-13"
pkg_config = subprocess.run( # Get compiler and linker flags for libpni
    ["pkg-config", "--cflags", "--libs", "libpni"],
    check=True,
    capture_output=True,
    text=True,
)
pni_libraries = pkg_config.stdout.strip()

print("Compiling and running the test...")
output_executable_name = cpp_name.removesuffix(".cpp") + ".test"
output_executable_path = workspace + "/" + output_executable_name
print(f"Output executable path: {output_executable_path}")
subprocess.run( # Compile and run the test
    [compiler, "-std=c++23", str(cpp_file), "-o", str(output_executable_path), "-I" + str(global_include), *pni_libraries.split(), "-O2", "-g"],
    check=True,
)

os.chdir(workspace) # Change to the test workspace directory
print(f"Changed working directory to: {workspace}")
if "--skip-run" in sys.argv:
    print("Skipping test run as per '--skip-run' argument.")
    sys.exit(0)
if "--debug" in sys.argv:
    subprocess.run(["gdb","--ex","set print thread-events off","--ex","run", "./"+str(output_executable_name)], check=True)
else:
    start_time = time.time()
    subprocess.run(["./"+str(output_executable_name)], check=True) # Run the test
    end_time = time.time()
    print(f"Test execution time: {end_time - start_time:.2f} seconds")

if "--amide-after" in sys.argv:
    subprocess.run(["amide"], check=True) # Open Amide for visualization

# response = input("Did the test pass? (y/n): ")
# if response.lower() != "y":
#     print("Please check the test output and fix any issues.")