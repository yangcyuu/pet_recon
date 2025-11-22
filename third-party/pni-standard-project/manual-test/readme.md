# PNI-Core manual_test Example

## 项目概述

PnI—Core manual_test 测试Case使用说明。该项目用于测试需要手动配置的算法。

## 文件结构

```
testCase_Example/
├── run.sh                             # 测试执行脚本
├── .pni-test-suite.txt                # 测试套件配置文件
├── example_code01.test                     # 测试代码01的可执行文件
├── example_code01.log                      # 测试代码01的输出log
├── config/                            # 配置文件目录
│   └── example01_config.json          # 测试代码01的配置参数  
└── Data/                              # 数据目录
    ├── source/                        # 源数据文件夹
    │   └── example_data01.bin         # 测试代码01所需的源数据
    └── result/                        # 结果输出文件夹
```

## 各文件夹说明

### config/ - 配置文件目录

### Data/ - 数据目录

#### Data/source/ - 源数据文件夹
存放输入的原始数据

#### Data/result/ - 结果输出文件夹
输出结果的存放位置

### 可执行文件&log
`.pni-test-suite.txt`中按行可以定义多个测试代码顺序执行；对应测试代码的可执行文件和log，在初次执行之后才会产生

## 脚本说明

### run.sh 脚本

```bash
pni=/media/ustc-pni/5282FE19AB6D5297/PNI_version/1020/pni-standard-project
data=/home/ustc/Desktop/data
python3 $pni/manual-test/sequence_test.py --cloned-from-workspace=.
```

**脚本内容说明：**
- **pni**: 指定 PNI 标准项目的安装路径
- **data**: 指定数据工作目录（若为当前目录，则无需使用这一项）
- **python3 sequence_test.py**: 执行 Python 测试序列脚本（该脚本会按顺序执行`.pni-test-suite.txt`中的测试项目）

**脚本执行流程：**
1. 脚本会读取 `.pni-test-suite.txt` 文件中标记的所有测试用例
2. 按顺序逐个执行定义的测试代码
3. 各项测试的将结果输出到 [Data/result/](Data/result/) 文件夹

## 使用方法

### 1. 配置测试源码
根据需要编辑 `.pni-test-suite.txt`，按顺序添加测试代码，每行对应一项`manual-test`的测试代码，路径为`manual-test`目录下的相对路径，注意不要带.cpp后缀（以测OSEM为例，需要填写的内容为`Recon/overall_osem_simple_gpu`）。

### 2. 配置参数
根据需要编辑 [config/example01_config.json](config/example01_config.json)，调整算法参数。

### 3. 准备数据
将原始数据数据文件放入 [Data/source/](Data/source/) 目录。（注意需要改成测试代码中对应的源文件名）

### 4. 执行测试
运行脚本进行测试：

```bash
sh run.sh
```
若存在权限问题，尝试使用：
```bash
sudo sh run.sh
```

### 5. 查看结果
测试完成后，检查 [Data/result/](Data/result/) 目录中的输出结果。

