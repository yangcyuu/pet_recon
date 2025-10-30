from pathlib import Path
import sys

supported_params = [
    "enable_dpdk",
    "enable_dpu",
    "disable_cuda",
    "enable_debug",
    "enable_debug_cuda_alloc",
]

# Validate input parameters
for param in sys.argv[1:]:
    if param not in supported_params:
        raise ValueError(f"Unsupported parameter: {param}")

if Path.cwd().name != "pni-standard-project":
    raise RuntimeError("Script must be run from the pni-standard-project directory.")




config_file_name = "./include/PnI-Config.hpp"
cfg_path = Path(config_file_name)
if cfg_path.exists():# Clear the file if it exists
    cfg_path.write_text("")

with open(config_file_name, "a") as cfg_file:
    cfg_file.write("// This file is auto-generated. Do not edit manually.\n\n")
    cfg_file.write("#ifndef _PNI_STANDARD_PROJECT_CONFIG_HPP_\n")
    cfg_file.write("#define _PNI_STANDARD_PROJECT_CONFIG_HPP_\n")
    cfg_file.write("#define PNI_STANDARD_DPDK_MBUFS (1024 * 1024 * 4 - 1)\n")

    for param in supported_params:
        macro_name = "PNI_STANDARD_CONFIG_" + param.upper()
        if param in supported_params and param in sys.argv:
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
    cfg_file.write("#if PNI_STANDARD_CONFIG_ENABLE_DEBUG\n")
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
    cfg_file.write("#endif // PNI_STANDARD_CONFIG_ENABLE_DEBUG\n\n")
    # CUDA allocation debug macro
    cfg_file.write("#if PNI_STANDARD_CONFIG_ENABLE_DEBUG && PNI_STANDARD_CONFIG_ENABLE_DEBUG_CUDA_ALLOC\n")
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
    cfg_file.write("#endif // PNI_STANDARD_CONFIG_ENABLE_DEBUG_CUDA_ALLOC\n\n")

    # End the include guard
    cfg_file.write("\n#endif // _PNI_STANDARD_PROJECT_CONFIG_HPP_\n")
        