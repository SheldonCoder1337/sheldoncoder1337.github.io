---
author: jiale cai
date: 2024-06-29 17:17:41
---

从零开始为 RISC-V 平台编写一个简单的操作系统内核。

## 1.环境配置

推荐使用 Ubuntu 24.04，Ubuntu 24.04 是目前最新的 Ubuntu 长期稳定发行版，在这个环境下安装运行环境也最简单。

所有演示代码在以下环境下验证通过，请仔细核对你的 Ubuntu 版本和内核版本与以下信息是否一致。

``` bash
$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.2 LTS
Release:        24.04
Codename:       noble

$ uname -r
6.6.87.2-microsoft-standard-WSL2
```

目前在 Ubuntu 24.04 环境下我们可以直接使用官方提供的 GNU工具链和 QEMU 模拟器，执行如下命令在线安装即可开始试验：

```bash
$ sudo apt update
$ sudo apt install build-essential gcc make perl dkms git gcc-riscv64-unknown-elf gdb-multiarch qemu-system-misc
```

以下是每个依赖项的简要说明:

- build-essential：该软件包提供了编译和构建软件所需的基本工具，包括编译器和C库。
- gcc：GNU编译器集合，用于编译C和C++程序。
- make：一个构建工具，用于自动化软件构建过程。
- perl：一种脚本语言，常用于文本处理和系统管理任务。
- dkms：动态内核模块支持，用于自动编译和安装内核模块。
- git：版本控制系统，用于跟踪和管理项目代码。
- gcc-riscv64-unknown-elf：RISC-V架构的交叉编译器，用于编译RISC-V架构的程序。
- gdb-multiarch：一个多架构调试器，支持多种架构的程序调试。
- qemu-system-misc：QEMU模拟器的一部分，包含了一些杂项工具和二进制文件。

使用包管理器：在终端中运行dpkg -s \<package-name> 命令，将\<package-name>替换为要检查的软件包名称。如果软件包已成功安装，将显示软件包的详细信息；否则，将显示软件包未安装的信息。例如，dpkg -s gcc将检查GCC编译器是否已安装。

```bash
$ dpkg -s gcc
Package: gcc
Status: install ok installed
Priority: optional
Section: devel
Installed-Size: 37
Maintainer: Ubuntu Developers <ubuntu-devel-discuss@lists.ubuntu.com>
Architecture: amd64
Source: gcc-defaults (1.214ubuntu1)
Version: 4:13.2.0-7ubuntu1
Provides: c-compiler
Depends: cpp (= 4:13.2.0-7ubuntu1), cpp-x86-64-linux-gnu (= 4:13.2.0-7ubuntu1), gcc-13 (>= 13.2.0-11~), gcc-x86-64-linux-gnu (= 4:13.2.0-7ubuntu1)
Recommends: libc6-dev | libc-dev
Suggests: gcc-multilib, make, manpages-dev, autoconf, automake, libtool, flex, bison, gdb, gcc-doc
Conflicts: gcc-doc (<< 1:2.95.3)
Description: GNU C compiler
 This is the GNU C compiler, a fairly portable optimizing compiler for C.
 .
 This is a dependency package providing the default GNU C compiler.
Original-Maintainer: Debian GCC Maintainers <debian-gcc@lists.debian.org>
```

## 2. 构建和使用说明

- `make`：编译构建
- `make run`：启动 qemu 并运行
- `make debug`：启动调试
- `make code`：反汇编查看二进制代码
- `make clean`：清理

具体使用请参考具体子项目下的 Makefile 文件。

## 3. 参考文献

本项目的设计参考了如下网络资源，在此表示感谢 :)

- The Adventures of OS：<https://osblog.stephenmarz.com/index.html>
- mini-riscv-os: <https://github.com/cccriscv/mini-riscv-os>
- Xv6, a simple Unix-like teaching operating system：<https://pdos.csail.mit.edu/6.828/2020/xv6.html>