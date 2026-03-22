# 草稿

LibOS操作系统虽然是软件，但它不是运行在通用操作系统（如Linux）上的一般应用软件，而是运行在裸机执行环境中的系统软件。如果采用通常的应用编程方法和编译手段，无法开发出这样的操作系统。其中一个重要的原因是：编译器（Rust 编译器和 C 编译器等）编译出的应用软件在缺省情况下是要链接标准库，而标准库是依赖于操作系统（如 Linux、Windows 等）的，但LibOS操作系统不依赖其他操作系统。所以，本章主要是让同学能够脱离常规应用软件开发的思路，理解如何开发没有操作系统支持的操作系统内核。

为了做到这一步，首先需要写出不需要标准库的软件并通过编译。为此，先把一般应用所需要的标准库的组件给去掉，这会导致编译失败。然后再逐步添加不需要操作系统的极少的运行时支持代码，让编译器能够正常编译出不需要标准库的正常程序。但此时的程序没有显示输出，更没有输入等，但可以正常通过编译，这样就打下 可正常编译OS 的前期开发基础。具体可看 移除标准库依赖 一节的内容。

LibOS内核主要在 Qemu 模拟器上运行，它可以模拟一台 64 位 RISC-V 计算机。为了让LibOS内核能够正确对接到 Qemu 模拟器上，需要了解 Qemu 模拟器的启动流程，还需要一些程序内存布局和编译流程（特别是链接）相关知识，这样才能将LibOS内核加载到正确的内存位置上，并使得它能够在 Qemu 上正常运行。为了确认内核被加载到正确的内存位置，我们会在LibOS内核中手写一条汇编指令，并使用 GDB 工具监控 Qemu 的执行流程确认这条指令被正确执行。具体可以参考 内核第一条指令（基础篇） 和 内核第一条指令（实践篇） 两节。

我们想用 Rust 语言来实现内核的大多数功能，因此我们需要进一步将控制权从第一条指令转交给 Rust 入口函数。在 Rust 代码中，函数调用是不可或缺的基本控制流，为了使得函数调用能够正常进行，我们在跳转到 Rust 入口函数前还需要进行栈的初始化工作。为此我们详细介绍了函数调用和栈的相关背景知识，具体内容可参考 为内核支持函数调用 一节。最终，我们调用软件栈中相比内核更低一层的软件——也即 RustSBI 提供的服务来实现格式化输出和遇到致命错误时的关机功能，形成了LibOS的核心功能，详情参考 基于 SBI 服务完成输出和关机 一节。至此，应用程序可以直接调用LibOS提供的字符串输出函数或关机函数，达到让应用与硬件隔离的操作系统目标。

LibOS操作系统的源代码如下所示：

``` text
./os/src
Rust        4 Files   119 Lines
Assembly    1 Files    11 Lines

├── bootloader(内核依赖的运行在 M 特权级的 SBI 实现，本项目中我们使用 RustSBI)
│   └── rustsbi-qemu.bin(可运行在 qemu 虚拟机上的预编译二进制版本)
├── LICENSE
├── os(我们的内核实现放在 os 目录下)
│   ├── Cargo.toml(内核实现的一些配置文件)
│   ├── Makefile
│   └── src(所有内核的源代码放在 os/src 目录下)
│       ├── console.rs(将打印字符的 SBI 接口进一步封装实现更加强大的格式化输出)
│       ├── entry.asm(设置内核执行环境的的一段汇编代码)
│       ├── lang_items.rs(需要我们提供给 Rust 编译器的一些语义项，目前包含内核 panic 时的处理逻辑)
│       ├── linker-qemu.ld(控制内核内存布局的链接脚本以使内核运行在 qemu 虚拟机上)
│       ├── main.rs(内核主函数)
│       └── sbi.rs(调用底层 SBI 实现提供的 SBI 接口)
├── README.md
└── rust-toolchain(控制整个项目的工具链版本)
```

## 执行应用程序

``` sh
$ cargo new os --bin

$ tree os
os
├── Cargo.toml
└── src
    └── main.rs

1 directory, 2 files

$ cargo run
   Compiling os v0.1.0 (/mnt/d/Lehub/computer-science/os)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.61s
     Running `target/debug/os`
Hello, world!
```

## 应用程序执行环境

``` mermaid
graph TB
Application[Application];
STLib[Standard Library];
Kernal[Kernal/OS];
Hardware[Hardware Platform];

Application -->|Function Call| STLib --> |System Call| Kernal --> |Instruction Set Architecture|Hardware;
```

## 目标平台与目标三元组

``` sh
$ rustc --version --verbose
   rustc 1.57.0-nightly (e1e9319d9 2021-10-14)
   binary: rustc
   commit-hash: e1e9319d93aea755c444c8f8ff863b0936d7a4b6
   commit-date: 2021-10-14
   host: x86_64-unknown-linux-gnu
   release: 1.57.0-nightly
   LLVM version: 13.0.0

$ rustc --print target-list | grep riscv
riscv32gc-unknown-linux-gnu
riscv32i-unknown-none-elf
riscv32imac-unknown-none-elf
riscv32imc-unknown-none-elf
riscv64gc-unknown-linux-gnu
riscv64gc-unknown-none-elf
riscv64imac-unknown-none-elf

$ cargo run --target riscv64gc-unknown-none-elf
```

## Rust 标准库和核心库

Rust 语言标准库–std 是让 Rust 语言开发的软件具备可移植性的基础，类似于 C 语言的 LibC 标准库。它是一组小巧的、经过实践检验的共享抽象，适用于更广泛的 Rust 生态系统开发。它提供了核心类型，如 Vec 和 Option、类库定义的语言原语操作、标准宏、I/O 和多线程等。默认情况下，我们可以使用 Rust 语言标准库来支持 Rust 应用程序的开发。但 Rust 语言标准库的一个限制是，它需要有操作系统的支持。所以，如果你要实现的软件是运行在裸机上的操作系统，就不能直接用 Rust 语言标准库了。

幸运的是，Rust 有一个对 Rust 语言标准库–std 裁剪过后的 Rust 语言核心库 core。core库是不需要任何操作系统支持的，它的功能也比较受限，但是也包含了 Rust 语言相当一部分的核心机制，可以满足我们的大部分功能需求。Rust 语言是一种面向系统（包括操作系统）开发的语言，所以在 Rust 语言生态中，有很多三方库也不依赖标准库 std 而仅仅依赖核心库 core。对它们的使用可以很大程度上减轻我们的编程负担。它们是我们能够在裸机平台挣扎求生的最主要倚仗，也是大部分运行在没有操作系统支持的 Rust 嵌入式软件的必备。

## 移除标准库依赖

=== "shell"

    ``` sh
    $ rustup target add riscv64gc-unknown-none-elf
    ```

=== "configuration"

    ``` toml title="os/.cargo/config" 
    # os/.cargo/config
    [build]
    target = "riscv64gc-unknown-none-elf"
    ```
## 分析被移除标准库的程序

``` sh
$ cargo install cargo-binutils
$ rustup component add llvm-tools-preview
# 文件格式 

$ file target/riscv64gc-unknown-none-elf/debug/os

# 文件头信息
$ rust-readobj -h target/riscv64gc-unknown-none-elf/debug/os

# 反汇编
$ rust-objdump -S target/riscv64gc-unknown-none-elf/debug/os
```

## QEMU

``` sh
$ qemu-system-riscv64 \
    -machine virt \
    -nographic \
    -bios ../bootloader/rustsbi-qemu.bin \
    -device loader,file=target/riscv64gc-unknown-none-elf/release/os.bin,addr=0x80200000
```







