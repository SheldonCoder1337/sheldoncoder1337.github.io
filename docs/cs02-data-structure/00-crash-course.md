# Crash Course of Data Structure and Algorithm

## Hello, Algo

``` sh
$ sudo apt-get update && sudo apt-get upgrade
$ sudo apt-get install git build-essential gdb-multiarch qemu-system-misc gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu
$ apt-mark showmanual
$ riscv64-linux-gnu-gcc --version

$ export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
$ export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
$ rustup target add riscv64gc-unknown-linux-gnu
```

=== "c"

    ``` c title="hello.c"
    #include <stdio.h>

    int main(void) {
        printf("hello, world\n");
        return 0;
    }
    ```

    ``` sh title="x86_64"
    # 1) 只编译不汇编 -> .s
    $ gcc -S hello.c -o hello.s
    # 2) 编译并汇编, 不链接 -> .o 
    $ gcc -c hello.c -o hello.o
    # 3) 加入调试信息，支持反汇编
    $ rm hello.o 
    $ gcc -g -c hello.c -o hello.o
    # 4) 编译、汇编、链接 -> ELF 可执行文件
    $ gcc hello.c -o hello && ./hello
    # 查看文件头信息
    $ readelf -h hello.os
    # 查看段信息
    $ readelf -SW hello.o
    # 对目标文件进行反汇编
    $ objdump -S hello.o
    ```

    ``` sh title="riscv"
    # 1) 只编译不汇编 -> .s
    $ riscv64-linux-gnu-gcc -S hello.c -o hello_riscv.s
    # 2) 编译并汇编, 不链接 -> .o 
    $ riscv64-linux-gnu-gcc -c hello.c -o hello_riscv.o
    # 3) 加入调试信息，支持反汇编
    $ rm hello_riscv.o 
    $ riscv64-linux-gnu-gcc -g -c hello.c -o hello_riscv.o
    # 4) 编译、汇编、链接 -> ELF 可执行文件
    $ riscv64-linux-gnu-gcc hello.c -o hello_riscv
    
    # 如遇报错，使用静态链接
    $ riscv64-linux-gnu-gcc -static hello.c -o hello_riscv && qemu-riscv64 ./hello_riscv

    # 查看文件头信息
    $ riscv64-linux-gnu-readelf -h hello_riscv.o
    # 查看段信息
    $ riscv64-linux-gnu-readelf -SW hello_riscv.o
    # 对目标文件进行反汇编
    $ riscv64-linux-gnu-objdump -S hello_riscv.o
    ```

=== "rust"

    ``` rust title="hello.rs"
    fn main() {
        println!("hello, world");
    }
    ```

    ``` sh title="x86_64"
    # 1) 只编译不汇编 -> .s
    $ rustc --emit=asm hello.rs -o hello.s
    # 2) 编译并汇编, 不链接 -> .o
    $ rustc --emit=obj hello.rs -o hello.o
    # 3) 加入调试信息，支持反汇编
    $ rm hello.o
    $ rustc -g --emit=obj hello.rs -o hello.o
    # 4) 编译、汇编、链接 -> ELF 可执行文件
    $ rustc hello.rs -o hello && ./hello

    # 查看文件头信息
    $ readelf -h hello.o
    # 查看段信息
    $ readelf -SW hello.o
    # 对目标文件进行反汇编
    $ objdump -S hello.o
    ```

    ``` sh title="riscv"
    # 1) 只编译不汇编 -> .s
    $ rustc --target riscv64gc-unknown-linux-gnu --emit=asm hello.rs -o hello_riscv.s
    # 2) 编译并汇编, 不链接 -> .o
    $ rustc --target riscv64gc-unknown-linux-gnu --emit=obj hello.rs -o hello_riscv.o
    # 3) 加入调试信息，支持反汇编
    $ rm hello_riscv.o
    $ rustc -g --target riscv64gc-unknown-linux-gnu --emit=obj hello.rs -o hello_riscv.o
    # 4) 编译、汇编、链接 -> ELF 可执行文件
    $ rustc --target riscv64gc-unknown-linux-gnu hello.rs -o hello_riscv

    # 如遇报错，使用静态链接
    $ rustc --target riscv64gc-unknown-linux-gnu -C target-feature=+crt-static hello.rs -o hello_riscv && qemu-riscv64 ./hello_riscv

    # 查看文件头信息
    $ riscv64-linux-gnu-readelf -h hello_riscv.o
    # 查看段信息
    $ riscv64-linux-gnu-readelf -SW hello_riscv.o
    # 对目标文件进行反汇编
    $ riscv64-linux-gnu-objdump -S hello_riscv.o
    ```

=== "python"

    ``` python title="hello.py"
    print("hello, world\n")
    ```

    ``` sh title="run"
    $ python3 hello.py
    ```

## Data Structure

### Array & Linked List

!!! note "关键基础，建议用时 3 天"
    - [数组 顺序存储 基本原理与实现](http://127.0.0.1:8000/data-structure-and-algorithm/array/)
    - [链表 链式存储 基本原理与实现](http://127.0.0.1:8000/data-structure-and-algorithm/list/)
    - [环形数组]()

### Hash Table

!!! note "建议用时 2 天"
    - [Hash Table]()

### Algorithms

!!! note
    - [Sorting]()
    - [Searching]()
    - [Tree]()
    - [Graph]()
    - [Backtracking]()
    - [Dynamic Programming]()