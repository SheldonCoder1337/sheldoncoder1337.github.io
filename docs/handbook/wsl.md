---
title: WSL/git命令速查
date: 2024-06-29 15:54:20
author: jiale
---

``` PowerShell title="PowerShell"
> wsl --list --online # 列出可用的Linux发行版
> wsl --install -d <DistributionName> # 安装指定的 Linux 发行版
> wsl --list --verbose # 列出已安装的 Linux 发行版
> wsl --set-default-version <Version> # 设置默认 WSL 版本
```

``` PowerShell title="Example"
> wsl -l --all -v # 列出所有已安装的 Linux 发行版
> wsl --export Ubuntu-20.04 E:\wsl\ubuntu-20.04.tar # 导出
> wsl --unregister Ubuntu-20.04 # 删除
> wsl --import Ubuntu-20.04 E:\wsl E:\wsl\ubuntu-20.04.tar # 导入
> wsl --set-default Ubuntu-20.04 # 设置默认
> wsl -d Ubuntu-20.04 # 启动
```

``` bash title="bash"
$ sudo su
$ apt install build-essential
$ gcc –version
$ apt install vim
$ vim /etc/netplan/01-network-manager-all.yaml
$ netplan apply
----------------------------------
$ apt-get install openssh-server
$  service ssh start
$ ps -e|grep ssh
$ systemctl restart ssh
-----------------------------------
$ lsblk
$ cd /media/sheldon/USB-SDD
$ cp srilm-1.7.3.tar.gz ~/
$ cd ~
$ mkdir srilm
$ cd srilm
$ mv ../srilm-1.7.3.tar.gz
$ tar -xzf srilm-1.7.3.tar.gz
$ cat INSTALL 
----------------------------------
$ mytrainset and testset
$ /root/corpus
----------------------------------
$ vim Makefile
$ export SRILM=/root/srilm
$ make
$ make SRILM=$(pwd) World -j $(proc)
----------------------------------
```

## Environment Setup

### Windows - WSL2

``` PowerShell
# 启用 Windows 功能：“适用于 Linux 的 Windows 子系统”
>> dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 启用 Windows 功能：“已安装的系统虚拟机平台”
>> dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# <Distro> 改为对应从微软应用商店安装的 Linux 版本名，比如：`wsl --set-version Ubuntu 2`
# 如果你没有提前从微软应用商店安装任何 Linux 版本，请跳过此步骤
>> wsl --set-version <Distro> 2

# 设置默认为 WSL 2，如果 Windows 版本不够，这条命令会出错
>> wsl --set-default-version 2
```

### C 语言开发环境

在实验或练习过程中，也会涉及部分基于C语言的开发，可以安装基本的本机开发环境和交叉开发环境，需要安装的C 开发环境涉及的软件:

``` sh title="C Development Setup"
$ sudo apt-get update && sudo apt-get upgrade
$ sudo apt-get install git build-essential gdb-multiarch \
                       qemu-system-misc gcc-riscv64-linux-gnu \
                       binutils-riscv64-linux-gnu
```

1. git：版本控制系统，用于跟踪和管理项目代码。
2. build-essential：提供编译 C/C++ 所需的基本工具链（gcc、g++、make 及 libc 等）。
3. gdb-multiarch：支持多架构（如 RISC-V、ARM、MIPS 等）的统一调试器。
4. qemu-system-misc：QEMU 模拟器的杂项系统镜像与固件文件，常用于裸机或 OS 级仿真。
5. gcc-riscv64-linux-gnu：面向 RISC-V 64 位架构的 GNU C 编译器（交叉编译器）。
6. binutils-riscv64-linux-gnu：RISC-V 64 位目标平台的二进制工具集（汇编器 as、链接器 ld、反汇编器 objdump 等）。

在 Linux 上做 C 语言开发的最小安装，只需要确保具备 “编译链(GCC) + 编辑器(vim/vscode) + 调试器(GDB)” 这三件套即可, build-essential gdb vim.

### Rust 开发环境

首先安装 Rust 版本管理器 rustup 和 Rust 包管理器 cargo，这里我们用官方的安装脚本来安装：

``` sh
# 使用tuna源来加速
$ export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
$ export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup

# 官方的安装脚本
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 配置环境变量
$ source $HOME/.cargo/env

# 查看是否正确安装Rust工具链
$ rustc --version
```

我们最好把软件包管理器 cargo 所用的软件包镜像地址 creates.io 也换成中国科学技术大学的镜像服务器来加速三方库的下载。我们打开（如果没有就新建） ~/.cargo/config 文件，并把内容修改为：

``` config title="~/.cargo/config"
[source.crates-io]
replace-with = 'tuna'

[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"
```

``` sh
# 换为国内源--
$ cd ~/.cargo
$ touch config

# Rust 相关软件包
$ rustup install nightly
$ rustup default nightly
$ rustup target add riscv64gc-unknown-linux-gnu
$ rustup target add riscv64gc-unknown-none-elf
$ cargo install cargo-binutils
$ rustup component add llvm-tools-preview
$ rustup component add rust-src
```

### QEMU 模拟器

``` sh
# 安装编译所需的依赖包
$ sudo apt install autoconf automake autotools-dev curl libmpc-dev libmpfr-dev libgmp-dev \
              gawk build-essential bison flex texinfo gperf libtool patchutils bc \
              zlib1g-dev libexpat-dev pkg-config  libglib2.0-dev libpixman-1-dev libsdl2-dev libslirp-dev \
              git tmux python3 python3-pip ninja-build

# 下载源码包
$ wget https://download.qemu.org/qemu-9.2.4.tar.xz
# 解压
$ tar xvJf qemu-9.2.4.tar.xz
# 编译安装并配置 RISC-V 支持
$ cd qemu-9.2.4
$ ./configure --target-list=riscv64-softmmu,riscv64-linux-user  # 在第九章的实验中，可以有图形界面和网络。如果要支持图形界面，可添加 " --enable-sdl" 参数；如果要支持网络，可添加 " --enable-slirp" 参数
$ make -j$(nproc)

$ qemu-system-riscv64 --version
$ qemu-riscv64 --version
```
