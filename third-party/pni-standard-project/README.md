# PnI-Standard-Project

## Dependencies

### 1. Boost(Kernel版本)已被废弃

<!-- Boost 项目提供免费的、经过同行评审的可移植 C++ 源库，强调与 C++ 标准库配合良好的库。Boost 库旨在广泛使用，并可用于广泛的应用程序。

1.命令行或从仓库中下载源码

```Shell
从官网下载boost压缩包，推荐下载1.87版本
```

2.在Boost解压缩后的目录下直接执行以下命令：`./bootstrap.sh;`再执行`sudo ./b2 install`

第一条命令 bootstrap.sh 是编译前的配置工作，第二条命令才开始真正地编译并安装Boost。 如果像上面这样不指定额外选项，Boost将编译release版本的库文件，把头文件安装到“/usr/local/include”中，把库文件安装到“/usr/local/lib”中。

参考链接：

boost官方网址：<https://www.boost.org/>

Boost github仓库：<https://github.com/boostorg/boost> -->

### 2. DPDK（linux端可选）

用于DPDK版本的采集程序运行，安装前请先确保自己的网卡支持dpdk模式，请先查阅网卡信息。

1.查看内核版本号`uname -a`，对于dpdk建议下载内核版本号+2以上的版本，例如对于Ubuntu22.04，可以下载DPDK 24.11.1 (LTS)版本

2.`ifconfig`命令查看网卡信息，将想要跑dpdk的网卡down掉

例：ifconfig eth0 down

再次运行`ifconfig`命令发现eth0已经不再显示了

![img](https://ys3ka9aj9m.feishu.cn/space/api/box/stream/download/asynccode/?code=OWIxZDBmMTgxZjI4MjA3ZmZmMmFmYmQ5NjM2ODUzZWFfQ1JuQjA1OHB1dFJXdnVUbG1uTDBqQzVTNUpMUUlnUUNfVG9rZW46S3V1Y2Jwa1JYb3IxU3Z4eHRGb2NKeXdNblFjXzE3NDI1NTMxOTk6MTc0MjU1Njc5OV9WNA)

3.依赖安装

- Python 3.6 or later.(可将python软链接至python3)
- Meson (version 0.57+) and ninja：`sudo pip3 install meson ninja`(采用pip3安装，如果已经采用apt安装，请卸载后重装)
- pyelftools (version 0.22+)：`sudo pip3 install pyelftools`
- Library for handling NUMA：`sudo apt install libnuma-dev`

4.编译

进入解压后的dpdk目录：

```Shell
meson setup build
cd build
ninja
sudo meson install
sudo ldconfig
```

5.配置大页

首先`free -h`查看一下剩余可用内存有多少，如果不够可以释放一下buffer。

进入usertool目录，运行`dpdk-hugepages.py -p 1G --setup 32G`分配32个1G大页（可按照需求来）。

挂载大页：

```Shell
mkdir /mnt/huge
dpdk-hugepages.py -m 1G /mnt/huge
```

运行`dpdk-hugepages.py -s`查看目前的大页分配和挂载情况

参考：<https://doc.dpdk.org/guides-24.11/tools/hugepages.html>

6.绑定网卡

依然是在usertool目录下，运行

```Shell
python dpdk-devbind.py --status  //查看端口情况
//绑定网卡端口，这里为eth1
python dpdk-devbind.py --bind=vfio-pci eth1 
python dpdk-devbind.py --status//再次查看，目标端口drv=vfio-pci即可
```

如果出现vfio模块找不到，需要运行：`sudo modprobe vfio-pci`

如果报错no-IOMMU，运行：`modprobe vfio enable_unsafe_noiommu_mode=1`

参考：<https://doc.dpdk.org/guides-24.11/linux_gsg/linux_drivers.html>

7.运行example验证环境配置

进入/examples目录下编译dpdk-helloworld源程序，运行命令：`sudo helloworld -l 0-3 -n 4`

参考链接：

dpdk官方网址：<https://www.dpdk.org/>

dpdk用户文档：<https://doc.dpdk.org/guides-24.11/>

### 3. Cxxopts

一款轻量级的 C++ 库，专为解析命令行选项设计，它遵循GNU风格的选项语法。这款工具对于开发需要处理命令行参数的C++应用来说，非常实用且易于集成。

1.获取源代码

```Shell
git clone https://github.com/jarro2783/cxxopts.git
cd cxxopts
```

2.构建与安装

```Shell
mkdir build && cd build
cmake ..
make
sudo make install
```

### 4. termcolor

用于跨平台地实现更改命令行输出颜色的库

获取源代码

```Shell
git clone https://github.com/ikalnytskyi/termcolor.git
```

## Build

构建时的注意要点：

1. 检查dependencies是否缺失；
2. 初次构建该项目，需要运行脚本config.sh，用以生成PnI-Config.hpp文件，如果不进行配置的话编译会报没有文件的错误；
3. 使用build.sh脚本进行构建，windows版本需添加参数win。该项目在linux环境下建议编译安装，在windows环境下建议直接下载编译好的发布版本；
4. 更新链接选项，linux版本cmake默认安装目录是/usr/local，所以在/etc/ld.so.conf添加一行：include /usr/local/lib；
5. 运行sudo ldconfig更新链接器路径
