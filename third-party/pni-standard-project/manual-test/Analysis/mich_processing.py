#!/usr/bin/env python3
# filepath: /home/ustc/pni_core/new/pni-standard-project/manual-test/Analysis/mich_processing.py

import numpy as np
import os
from pathlib import Path

def process_images_operation(img1_path, img2_path, output_path, operation="subtract", offset1=6, offset2=0):
    """
    处理两个图像，执行指定的操作（不进行维度压缩）
    
    Args:
        img1_path: 第一个图像文件路径
        img2_path: 第二个图像文件路径
        output_path: 输出文件路径
        operation: 操作类型 ("add" 或 "subtract")
        offset1: 第一个图像的文件偏移量（字节）
        offset2: 第二个图像的文件偏移量（字节）
    
    Returns:
        True if successful, False otherwise
    """
    # 定义维度
    BIN_DIM = 311
    VIEW_DIM = 156
    SLICE_DIM = 10816
    TOTAL_SIZE = BIN_DIM * VIEW_DIM * SLICE_DIM
    
    print(f"Expected dimensions: {BIN_DIM} x {VIEW_DIM} x {SLICE_DIM} = {TOTAL_SIZE}")
    
    # 创建输出目录
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输入文件
    if not os.path.exists(img1_path):
        print(f"Error: Image 1 file '{img1_path}' does not exist")
        return False
    
    if not os.path.exists(img2_path):
        print(f"Error: Image 2 file '{img2_path}' does not exist")
        return False
    
    # 读取第一个图像
    print(f"Loading image 1: {img1_path}")
    try:
        image1 = np.fromfile(img1_path, dtype=np.float32, offset=offset1)
        print(f"Image 1 size: {len(image1)} elements")
        
        if len(image1) < TOTAL_SIZE:
            print(f"Warning: Image 1 size ({len(image1)}) < expected ({TOTAL_SIZE}), padding with zeros")
            image1 = np.pad(image1, (0, TOTAL_SIZE - len(image1)), 'constant')
        elif len(image1) > TOTAL_SIZE:
            print(f"Warning: Image 1 size ({len(image1)}) > expected ({TOTAL_SIZE}), truncating")
            image1 = image1[:TOTAL_SIZE]
        
    except Exception as e:
        print(f"Error loading image 1: {e}")
        return False
    
    # 读取第二个图像
    print(f"Loading image 2: {img2_path}")
    try:
        image2 = np.fromfile(img2_path, dtype=np.float32, offset=offset2)
        print(f"Image 2 size: {len(image2)} elements")
        
        if len(image2) < TOTAL_SIZE:
            print(f"Warning: Image 2 size ({len(image2)}) < expected ({TOTAL_SIZE}), padding with zeros")
            image2 = np.pad(image2, (0, TOTAL_SIZE - len(image2)), 'constant')
        elif len(image2) > TOTAL_SIZE:
            print(f"Warning: Image 2 size ({len(image2)}) > expected ({TOTAL_SIZE}), truncating")
            image2 = image2[:TOTAL_SIZE]
        
    except Exception as e:
        print(f"Error loading image 2: {e}")
        return False
    
    # 直接对1D数据进行运算，不进行3D重塑和压缩
    print("Performing operation on raw data (no compression)...")
    
    # 执行指定的操作
    if operation == "add":
        result_image = image1 + image2
        operation_str = "addition"
        print(f"Performing addition: img1 + img2")
    elif operation == "subtract":
        result_image = image1 - image2
        operation_str = "subtraction"
        print(f"Performing subtraction: img1 - img2")
    else:
        print(f"Error: Unknown operation '{operation}'. Use 'add' or 'subtract'")
        return False
    
    print(f"{operation_str.capitalize()} result statistics: min={np.min(result_image):.2f}, max={np.max(result_image):.2f}, mean={np.mean(result_image):.2f}")
    
    # 保存结果图像（保持原始1D格式）
    result_image.astype(np.float32).tofile(output_path)
    print(f"Saved {operation_str} result: {output_path}")
    print(f"Output size: {len(result_image)} elements")
    print(f"Output can be reshaped to: ({BIN_DIM}, {VIEW_DIM}, {SLICE_DIM})")
    
    return True

def compress_sum_all_slices(img_path, output_path, dim_x, dim_y, dim_z, dtype=np.float32, offset=0, chunk_slices=128):
    """
    将 dim_x x dim_y x dim_z 的三维数据沿 z 轴全部求和（对应像素逐元素相加）
    并保存为 dim_x x dim_y 的二进制 float32 文件。
    使用 memmap + 分块以避免一次性占用过多内存。

    Args:
        img_path: 输入二进制文件路径（裸数据，按 dtype 顺序存储）
        output_path: 输出文件路径（结果会以 float32 原始二进制保存）
        dim_x, dim_y, dim_z: 三维尺寸 (例如 311,156,10816)
        dtype: 输入数据类型（默认 float32）
        offset: 输入文件字节偏移（默认 0）
        chunk_slices: 每次读取多少个 z 切片（分块大小），调节以控制内存占用
    Returns:
        True if success, False otherwise
    """
    total_elems = int(dim_x) * int(dim_y) * int(dim_z)
    expected_bytes = total_elems * np.dtype(dtype).itemsize

    if not os.path.exists(img_path):
        print(f"Error: '{img_path}' not found")
        return False

    # 输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        # 使用 memmap 映射文件（从 offset 开始）
        # 这里 shape 用一维 total_elems，然后在访问时 reshape
        mm = np.memmap(img_path, mode='r', dtype=dtype, offset=offset, shape=(total_elems,))
    except Exception as e:
        print("Error creating memmap:", e)
        return False

    # 检查文件是否包含足够数据（若不足，用零填充逻辑在此可选）
    actual_elems = mm.size
    if actual_elems < total_elems:
        print(f"Warning: file contains {actual_elems} elements < expected {total_elems}. Padding with zeros.")
        # 为简单起见，把 memmap 的有效部分拷贝到一个更大的数组并填零（可能较耗内存）
        data = np.zeros(total_elems, dtype=dtype)
        data[:actual_elems] = mm[:actual_elems]
        mm = data  # mm 现在是 ndarray
    elif actual_elems > total_elems:
        # 截断超出部分（memmap 支持切片）
        mm = mm[:total_elems]

    # 最终按 (dim_x, dim_y, dim_z) 访问
    # 为了节省精度损失，累加时用 float64
    result = np.zeros((dim_x, dim_y), dtype=np.float64)

    # 分块按 z 切片累加
    slices_per_chunk = int(chunk_slices)
    z = 0
    while z < dim_z:
        z_end = min(dim_z, z + slices_per_chunk)
        # 计算一段 chunk 的一维起止下标
        start_idx = z * (dim_x * dim_y)
        end_idx = z_end * (dim_x * dim_y)
        chunk_flat = np.asarray(mm[start_idx:end_idx])  # 转为 ndarray（若原本是 memmap 则为视图）
        # reshape为 (slices, dim_x, dim_y) 或 (dim_x, dim_y, slices) 要按我们数据的内存布局
        # 我们假定内存顺序是 (x fastest, then y, then z) 即 C-order reshape (dim_z, dim_x, dim_y) 或 (dim_x, dim_y, dim_z)
        # 下面使用 (dim_z_chunk, dim_x, dim_y) 先 reshape 再 sum over axis=0
        chunk_slices_count = z_end - z
        try:
            chunk_3d = chunk_flat.reshape((chunk_slices_count, dim_x, dim_y))
        except Exception as e:
            print("Reshape error:", e)
            return False

        # 求和（按切片方向 sum），并累加到 result
        # chunk_3d.shape == (chunk_slices_count, dim_x, dim_y)
        # 我们想 sum over axis=0 -> 得到 (dim_x, dim_y)
        s = np.sum(chunk_3d, axis=0, dtype=np.float64)
        result += s

        z = z_end

    # 将结果转为 float32 并保存（按 row-major 顺序：dim_x * dim_y）
    result_out = result.astype(np.float32)
    result_out.tofile(output_path)
    print(f"Saved summed image ({dim_x}x{dim_y}) to {output_path}")
    return True

def compress_sum_grouped_slices(img_path, output_dir, dim_x, dim_y, dim_z, group_size, dtype=np.float32, offset=0, chunk_slices=128):
    """
    将 z 方向按 group_size 分组（例如 group_size=100 -> 每 100 张切片为一组），
    对每组做逐元素求和，输出多张 dim_x x dim_y 的结果（每组一张）。
    输出文件名为 output_dir/group_0000.raw 等。

    Args:
        group_size: 每组切片数（最后一组若不足也会处理）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = (dim_z + group_size - 1) // group_size

    for g in range(groups):
        z0 = g * group_size
        z1 = min(dim_z, z0 + group_size)
        outpath = output_dir / f"group_{g:04d}.raw"

        # 复用上面的逻辑：分块读取 z0..z1-1 的切片并相加
        # 为避免重复代码，这里用 compress_sum_all_slices 通过 memmap slice 的方式来实现
        # 计算 slice 的一维范围并做类似处理
        total_elems = dim_x * dim_y * dim_z
        mm = np.memmap(img_path, mode='r', dtype=dtype, offset=offset, shape=(total_elems,))
        start_idx = z0 * (dim_x * dim_y)
        end_idx = z1 * (dim_x * dim_y)
        chunk_flat = np.asarray(mm[start_idx:end_idx])
        try:
            chunk_3d = chunk_flat.reshape((z1 - z0, dim_x, dim_y))
        except Exception as e:
            print("Reshape error for group", g, ":", e)
            return False
        s = np.sum(chunk_3d, axis=0, dtype=np.float64).astype(np.float32)
        s.tofile(str(outpath))
        print(f"Saved group {g} ({z0}-{z1-1}) -> {outpath}")

    return True



def add_images(img1_path, img2_path, output_path, offset1=6, offset2=0):
    """
    对两个图像进行加法操作：img1 + img2
    
    Args:
        img1_path: 第一个图像文件路径
        img2_path: 第二个图像文件路径
        output_path: 输出文件路径
        offset1: 第一个图像的文件偏移量（字节）
        offset2: 第二个图像的文件偏移量（字节）
    
    Returns:
        True if successful, False otherwise
    """
    print("=== Image Addition: img1 + img2 ===")
    return process_images_operation(img1_path, img2_path, output_path, "add", offset1, offset2)

def subtract_images(img1_path, img2_path, output_path, offset1=6, offset2=0):
    """
    对两个图像进行减法操作：img1 - img2
    
    Args:
        img1_path: 第一个图像文件路径
        img2_path: 第二个图像文件路径
        output_path: 输出文件路径
        offset1: 第一个图像的文件偏移量（字节）
        offset2: 第二个图像的文件偏移量（字节）
    
    Returns:
        True if successful, False otherwise
    """
    print("=== Image Subtraction: img1 - img2 ===")
    return process_images_operation(img1_path, img2_path, output_path, "subtract", offset1, offset2)


def main():
    # ========== 在这里修改路径和操作 ==========
    
    # 选择操作模式：
    # "two_images": 两个图像的加法或减法运算
    # "compress": 单个图像的维度压缩
    operation_mode = "two_images"  # 修改为 "two_images" 或 "compress"
    
    if operation_mode == "two_images":
        # 两个图像运算的设置
        image1_path = "/home/ustc/Desktop/E180_20250826_WellCounter_PET4643_Slice4796/Data/result/random_factor.pni"
        image2_path = "/home/ustc/Desktop/testE180Case/MichScatter_SSSTailFitting.bin"
        output_path = "/home/ustc/Desktop/E180_20250826_WellCounter_PET4643_Slice4796/Data/result/random_add_sss.bin"

        offset1 = 0  # 图像1的文件偏移量
        offset2 = 0  # 图像2的文件偏移量

        # 选择操作类型："add" 或 "subtract"
        operation_type = "add"  # 修改这里来选择操作
        
        if operation_type == "add":
            success = add_images(image1_path, image2_path, output_path, offset1=offset1, offset2=offset2)
        elif operation_type == "subtract":
            success = subtract_images(image1_path, image2_path, output_path, offset1=offset1, offset2=offset2)
        else:
            print(f"Error: Unknown operation type '{operation_type}'. Use 'add' or 'subtract'")
            return 1
            
    elif operation_mode == "compress":
        # 单个图像压缩的设置
        image_path = "/home/ustc/Desktop/E180_20250826_WellCounter_PET4643_Slice4796/Data/result/prompt_minus_random&sss.bin"
        output_path = "/home/ustc/Desktop/E180_20250826_WellCounter_PET4643_Slice4796/Data/result/prompt_minus_random&sss_compressBySlice.bin"
        
        # 图像维度设置
        dim_x = 311   # BIN维度
        dim_y = 156   # VIEW维度  
        dim_z = 10816 # SLICE维度
        
        # 文件偏移量
        offset = 0
        
        # 修正函数调用：移除不需要的 compress_axis 参数，使用正确的参数顺序
        success = compress_sum_all_slices(
            img_path=image_path, 
            output_path=output_path, 
            dim_x=dim_x, 
            dim_y=dim_y, 
            dim_z=dim_z, 
            dtype=np.float32,  # 指定数据类型
            offset=offset,
            chunk_slices=128   # 分块大小，0 改为 128
        )
        
    else:
        print(f"Error: Unknown operation mode '{operation_mode}'. Use 'two_images' or 'compress'")
        return 1
    
    # ==========================================
    
    if success:
        print("Processing completed successfully!")
        return 0
    else:
        print("Processing failed!")
        return 1

if __name__ == "__main__":
    exit(main())