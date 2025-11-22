from pathlib import Path
import sys

supported_params = [
    "create",
    "run"
]

supported_configs = [
    "enable_dpdk",
    "enable_dpu",
    "disable_cuda",
    "enable_debug_messages",
    "enable_debug_cuda_alloc",
    "enable_debug_scatter_internal_values",
    "enable_debug_thread_time",
    "enable_cmdline_error_messages",
]

def is_config_file_valid():
    # Check if config.txt exists and is valid
    cfg_path = Path("config.txt")
    valid_lines = 0
    if not cfg_path.exists():
        return False
    with open(cfg_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            if "=" not in line:
                return False
            name, value = line.strip().split("=")
            if name not in supported_configs:
                return False
            if value.upper() not in ["ON", "OFF"]:
                return False
            valid_lines += 1
    if valid_lines != len(supported_configs):
        return False
    return True

def create_config():
    if is_config_file_valid():
        print("config.txt already exists and is valid. Skipping creation.")
        return
    # Write all parameters into file "config.txt" in regex format
    print("Creating config.txt with supported parameters...")
    if Path("config.txt").exists():
        Path("config.txt").unlink()
    Path("config.txt").touch()
    with open("config.txt", "w") as f:
        for config in supported_configs:
            f.write(f"{config}=OFF\n")

def run_config():
    config_file_name = "./include/PnI-Config.hpp"
    config_temp_name = "./build/PnI-Config.hpp.tmp"
    
    cfg_path = Path(config_temp_name)
    tgt_path = Path(config_file_name)
    
    # ✅ 确保父目录存在
    cfg_path.parent.mkdir(parents=True, exist_ok=True)  # 创建 ./build/ 目录
    tgt_path.parent.mkdir(parents=True, exist_ok=True)  # 创建 ./include/ 目录
    
    # 创建或清空临时文件
    if not cfg_path.exists():
        cfg_path.touch()
    else:
        cfg_path.write_text("")
    
    # 创建目标文件（如果不存在）
    if not tgt_path.exists():
        tgt_path.touch()

    def is_config_on(config_name):
        upper_name = config_name.upper()
        if Path("config.txt").exists():
            with open("config.txt", "r") as cfg_file:
                for line in cfg_file: # Use regex to parse
                    if line.strip().startswith(f"{config_name}="):
                        value = line.strip().split("=")[1].strip().upper()
                        return value == "ON"
        return False

    with open(config_temp_name, "a") as cfg_file:
        cfg_file.write("// This file is auto-generated. Do not edit manually.\n\n")
        cfg_file.write("#ifndef _PNI_STANDARD_PROJECT_CONFIG_HPP_\n")
        cfg_file.write("#define _PNI_STANDARD_PROJECT_CONFIG_HPP_\n")
        cfg_file.write("#define PNI_STANDARD_DPDK_MBUFS (1024 * 1024 * 4 - 1)\n")

        for config in supported_configs:
            macro_name = "PNI_STANDARD_CONFIG_" + config.upper()
            if config in supported_configs and is_config_on(config):
                cfg_file.write(f"#define {macro_name} 1\n")
                print(f"Enabled {macro_name}")
            else:
                cfg_file.write(f"#define {macro_name} 0\n")
                print(f"Disabled {macro_name}")

        # Special handling for CUDA
        cfg_file.write("#if defined(__CUDA_RUNTIME_H__)\n")

        cfg_file.write("#ifndef __PNI_CUDA_MACRO__\n")
        cfg_file.write("#define __PNI_CUDA_MACRO__ __host__ __device__\n")
        cfg_file.write("#endif // __PNI_CUDA_MACRO__\n")

        cfg_file.write("#ifndef __PNI_CUDA_HOST_ONLY__\n")
        cfg_file.write("#define __PNI_CUDA_HOST_ONLY__ __host__\n")
        cfg_file.write("#endif // __PNI_CUDA_HOST_ONLY__\n")

        cfg_file.write("#ifndef __PNI_CUDA_DEVICE_ONLY__\n")
        cfg_file.write("#define __PNI_CUDA_DEVICE_ONLY__ __device__\n")
        cfg_file.write("#endif // __PNI_CUDA_DEVICE_ONLY__\n")

        cfg_file.write("#else // defined(__CUDA_RUNTIME_H__)\n\n")

        cfg_file.write("#ifndef __PNI_CUDA_MACRO__\n")
        cfg_file.write("#define __PNI_CUDA_MACRO__\n")
        cfg_file.write("#endif // __PNI_CUDA_MACRO__\n")

        cfg_file.write("#ifndef __PNI_CUDA_HOST_ONLY__\n")
        cfg_file.write("#define __PNI_CUDA_HOST_ONLY__\n")
        cfg_file.write("#endif // __PNI_CUDA_HOST_ONLY__\n")

        cfg_file.write("#ifndef __PNI_CUDA_DEVICE_ONLY__\n")
        cfg_file.write("#define __PNI_CUDA_DEVICE_ONLY__\n")
        cfg_file.write("#endif // __PNI_CUDA_DEVICE_ONLY__\n")

        cfg_file.write("#endif // defined(__CUDA_RUNTIME_H__)\n")

        # Debug macro
        cfg_file.write("#if PNI_STANDARD_CONFIG_ENABLE_DEBUG_MESSAGES\n")
        cfg_file.write("#include <chrono>\n")
        cfg_file.write("#include <iostream>\n")
        cfg_file.write("inline void PNI_DO_DEBUG(std::string const& msg) {\n")
        cfg_file.write("    thread_local static auto now = std::chrono::system_clock::now();\n")
        cfg_file.write("    int msecond = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now).count();\n")
        cfg_file.write("    std::cerr << std::to_string(msecond) << \"\\tPNI_DEBUG: \" + msg << std::flush;\n")
        cfg_file.write("}\n")
        cfg_file.write("#define PNI_DEBUG(x)  (PNI_DO_DEBUG(x));\n")
        cfg_file.write("#else\n")
        cfg_file.write("#define PNI_DEBUG(x) \n")
        cfg_file.write("#endif // PNI_STANDARD_CONFIG_ENABLE_DEBUG_MESSAGES\n\n")
        # CUDA allocation debug macro
        cfg_file.write("#if PNI_STANDARD_CONFIG_ENABLE_DEBUG_CUDA_ALLOC\n")
        cfg_file.write("#include <termcolor/termcolor.hpp>\n")
        cfg_file.write("inline void CUDA_ALLOC_DEBUG_SAY(size_t bytes, std::string name, bool isRelease) {\n")
        cfg_file.write("    thread_local static std::size_t total_bytes = 0;\n")
        cfg_file.write("    auto print_bytes = [](std::size_t b) { auto s = std::to_string(b);   for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3)s.insert(static_cast<std::size_t>(i), \",\");  while(s.size()<16)s=\" \"+s; return s; };\n")
        cfg_file.write("    if (isRelease) {\n")
        cfg_file.write("        total_bytes -= bytes;\n")
        cfg_file.write("    std::cerr << termcolor::green;\n")
        cfg_file.write("    PNI_DO_DEBUG( \"Freed \" + print_bytes(bytes) + \" bytes CUDA, total: \" + print_bytes(total_bytes) + \" bytes, name: \"+name+\".\\n\");\n")
        cfg_file.write("    std::cerr << termcolor::reset;\n")
        cfg_file.write("    } else {\n")
        cfg_file.write("        total_bytes += bytes;\n")
        cfg_file.write("    std::cerr << termcolor::blue;\n")
        cfg_file.write("    PNI_DO_DEBUG( \"Alloc \" + print_bytes(bytes) + \" bytes CUDA, total: \" + print_bytes(total_bytes) + \" bytes, name: \"+name+\".\\n\");\n")
        cfg_file.write("    std::cerr << termcolor::reset;\n")
        cfg_file.write("    }\n")
        cfg_file.write("}\n")
        cfg_file.write("#define CUDA_ALLOC_DEBUG(x, n)  (CUDA_ALLOC_DEBUG_SAY(x, n, false));\n")
        cfg_file.write("#define CUDA_FREE_DEBUG(x, n)   (CUDA_ALLOC_DEBUG_SAY(x, n, true));\n")
        cfg_file.write("#else\n")
        cfg_file.write("#define CUDA_ALLOC_DEBUG(x, n) \n")
        cfg_file.write("#define CUDA_FREE_DEBUG(x, n) \n")
        cfg_file.write("#endif // PNI_STANDARD_CONFIG_ENABLE_DEBUG_MESSAGES_CUDA_ALLOC\n\n")

        # End the include guard
        cfg_file.write("\n#endif // _PNI_STANDARD_PROJECT_CONFIG_HPP_\n")
            
    # Replace the original config file with the new one if the content of both are different
    with open(config_temp_name, "r") as new_cfg_file:
        new_content = new_cfg_file.read()
    with open(config_file_name, "r") as orig_cfg_file:
        orig_content = orig_cfg_file.read()
    if new_content != orig_content:
        tgt_path.write_text(new_content)
        print("Configuration file updated.")
    else:
        print("No changes detected in configuration file.")





# Validate input parameters
for param in sys.argv[1:]:
    if param not in supported_params:
        raise ValueError(f"Unsupported parameter: {param}")

if Path.cwd().name != "pni-standard-project":
    raise RuntimeError("Script must be run from the pni-standard-project directory.")

if "create" in sys.argv:
    create_config()
if "run" in sys.argv:
    run_config()




