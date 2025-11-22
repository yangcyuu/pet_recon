import os
import subprocess
from pathlib import Path
import sys
from tqdm import tqdm

def print_usage_and_exit():
    print("Usage: python3 sequence_test.py [--cloned-from-workspace=PATH] [--clone-mode=soft-link|copy]")
    print("Note: This script must be run in a workspace directory containing the '.pni-test-suite.txt' file.")
    sys.exit(1)

def param_get_cloned_from_workspace():
    for param in sys.argv[1:]:
        if param.startswith("--cloned-from-workspace="):
            return param.split("=")[1]
    return None

def param_get_clone_mode():
    for param in sys.argv[1:]:
        if param.startswith("--clone-mode="):
            mode = param.split("=")[1]
            if mode in ["soft-link", "copy"]:
                return mode
    return "soft-link"

def create_readonly_soft_links_from_directory(src_dir, dest_dir):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    if not src_path.is_dir():
        print(f"Source directory '{src_dir}' does not exist or is not a directory.")
        raise FileNotFoundError(f"Source directory '{src_dir}' not found.")
    dest_path.mkdir(parents=True, exist_ok=True)
    for item in src_path.iterdir():
        link_name = dest_path / item.name
        if link_name.exists() or link_name.is_symlink():
            link_name.unlink()
        link_name.symlink_to(item.resolve())
        os.chmod(link_name, 0o444) # Make the symlink itself readonly (remove write permissions)
        print(f"Created readonly symlink: {link_name} -> {item.resolve()}")

def create_copied_files_from_directory(src_dir, dest_dir):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    if not src_path.is_dir():
        print(f"Source directory '{src_dir}' does not exist or is not a directory.")
        raise FileNotFoundError(f"Source directory '{src_dir}' not found.")
    dest_path.mkdir(parents=True, exist_ok=True)
    for item in src_path.iterdir():
        dest_file = dest_path / item.name
        if item.is_file():
            if dest_file.exists():  # if the dst file exists, skip copying
                print(f"File already exists, skipping copy: {dest_file}")
                continue
            with open(item, 'rb') as fsrc, open(dest_file, 'wb') as fdst:
                fdst.write(fsrc.read())
            os.chmod(dest_file, 0o444)  # Make the copied file readonly
            print(f"Copied and set readonly: {dest_file}")
        elif item.is_dir():
            create_copied_files_from_directory(item, dest_file)

cloned_workspace = param_get_cloned_from_workspace()
clone_mode = param_get_clone_mode()
if cloned_workspace and os.path.abspath(cloned_workspace) != os.getcwd():
    print(f"Cloned from workspace: {cloned_workspace}, clone mode: {clone_mode}")
    if clone_mode == "soft-link":
        create_readonly_soft_links_from_directory(cloned_workspace, os.getcwd())
    elif clone_mode == "copy":
        create_copied_files_from_directory(cloned_workspace, os.getcwd())

workspace = Path.cwd()
script_dir = Path(__file__).parent
project_dir = script_dir.parent
print(f"-- Workspace: {workspace}")
print(f"-- Script directory: {script_dir}")
print(f"-- Project directory: {project_dir}")

if script_dir.name != "manual-test" or project_dir.name != "pni-standard-project":
    print("Error: script must be located in 'pni-standard-project/manual-test' directory.")
    print_usage_and_exit()

if not (workspace / '.pni-test-suite.txt').is_file():
    print("File '.pni-test-suite.txt' does not exist in the current directory.")
    print_usage_and_exit()

def run_test_command(command, log_file, pbar, use_std_buf=False):
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    with open(log_file, 'w') as log:
        if use_std_buf:
            command = ["stdbuf", "-i0", "-o0", "-e0"] + command

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=-1,
            env=env
        )
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                log.write(line)
                log.flush()
                pbar.write(line.rstrip(), file=sys.stdout)
                sys.stdout.flush()
        
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
    
    return process.returncode

def run_one_test(test_name, pbar):

    log_name_no_suffix = test_name.split("/")[-1]
    log_name = log_name_no_suffix + ".log"
    log_file = Path(workspace / f"{log_name}")
    if not log_file.exists():
        log_file.touch()

    # 设置环境变量禁用 Python 缓冲
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    # 实时显示输出，同时保存到日志
    with open(log_file, 'w') as log:
        commands_build = ["python3", "manual-test/manual_test.py", "--test-name", test_name, "--move-to-workspace", workspace, "--skip-run"]
        commands_run = ["./{log_name_no_suffix}.test".format(log_name_no_suffix=log_name_no_suffix)]
        
        os.chdir(project_dir)
        if run_test_command(
            commands_build, log_file, pbar
        ) != 0:
            raise subprocess.CalledProcessError(cmd=commands_build)
        
        os.chdir(workspace)
        if run_test_command(
            commands_run, log_file, pbar, use_std_buf=True
        ) != 0:
            raise subprocess.CalledProcessError(cmd=commands_run)

with open('.pni-test-suite.txt', 'r') as f:
    lines = f.readlines()

    test_names = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if len(test_names) == 0:
        print("File '.pni-test-suite.txt' is empty.")
        print_usage_and_exit()
    
    print("File '.pni-test-suite.txt' contains the following tests:")
    for test_name in test_names:
        print(f"- {test_name}")
    print()

    # 使用 position=0 和 file=sys.stderr 确保进度条在底部
    with tqdm(test_names, 
              desc="Running tests", 
              unit="test",
              position=0,           # 使用固定位置
              leave=True,
              dynamic_ncols=True,
              file=sys.stderr,      # 进度条输出到 stderr，不与 stdout 混淆
              miniters=1,
              colour="#568F44",
              mininterval=0.1) as pbar:
        
        for test_name in pbar:
            # 截断长名称以适应进度条
            display_name = test_name if len(test_name) <= 30 else test_name[:27] + "..."
            pbar.set_description(f"Testing: {display_name}")
            try:
                run_one_test(test_name, pbar)
                pbar.write(f"\033[92m✓ PASSED: {test_name}\033[0m", file=sys.stdout)
            except subprocess.CalledProcessError as e:
                pbar.write(f"\033[91m✗ FAILED: {test_name}\033[0m", file=sys.stdout)
                pbar.write(f"\033[91m   See log: {test_name.split('/')[-1] + '.log'}\033[0m", file=sys.stdout)
                raise

print()