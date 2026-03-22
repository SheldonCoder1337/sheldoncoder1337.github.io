---
author: jiale cai
date: 2024-06-29 17:34:12
---

在正式步入RISCV的学习之前，与善其事必先利其器。

## GCC简介

- [GCC (GNU Compiler Collection)](https://gcc.gnu.org/) 由 GNU开发的，遵循 GPL 许可证发行的编译器套件。
- 支持 C、C++、Objective-C、Fortran、Ada 和 Go 语言等多种语言前端，已被移植到多种计算机体系架构上，如 x86、ARM、RISC-V 等。
- GCC 的初衷是为 GNU 操作系统专门编写一款编译器，现已被大多数 “Unix-like”操作系统（如 Linux、BSD、MacOS 等）采纳为标准的编译器。

=== "GCC命令格式"

    **gcc [option] [filenames]**

    |Option|Meaning|
    |---   |--- |
    |-E    |Preprocess only; do not compile, assemble or link.|
    |-S    |Compile only; do not assemble or link.|
    |-c    |Compile and assemble, but do not link.|
    |-o file    |把输出生成到由file指定文件名的文件中|
    |-g   |把输出的文件中加入支持调试的信息|
    |-v    |显示输出详细的命令执行过程信息|

=== "GCC涉及的文件类型"

    |文件类型|描述|
    |---   |--- |
    |.c    |C 源文件|
    |.i    |预处理后的 C 源文件|
    |.s/.S    |汇编代码|
    |.h    |头文件|
    |.o    |目标文件|
    |.a    |静态库文件|
    |.so    |动态库文件|
    |.out   |可执行文件|

=== "GDB调试步骤"

    ``` mermaid
    graph TB
    SourceCode[foo.c];
    IntermediateCode[foo.i];
    AssemblyCode[foo.s];
    ObjectCode[foo.o]
    ExecutableFile[a.out];
    
    Preprocessor[Preprocessor];
    Compiler[Compiler];
    Assembler[Assembler];
    Linker[Linker];

    SourceCode -->|gcc -E foo.c -o foo.i| IntermediateCode;
    IntermediateCode -->|gcc -S foo.i -o foo.s| AssemblyCode;
    AssemblyCode -->|gcc -c foo.s -o foo.o| ObjectCode;
    ObjectCode -->|gcc foo.o -o a.out| ExecutableFile;

    SourceCode -->|gcc -E| Preprocessor;
    Preprocessor --> IntermediateCode;
    IntermediateCode -->|gcc -S| Compiler;
    Compiler --> AssemblyCode;
    AssemblyCode -->|gcc -c| Assembler;
    Assembler --> ObjectCode;
    ObjectCode --> Linker;
    Linker --> ExecutableFile;
    ```

## ELF文件格式

- ELF（Executable and Linkable Format，可执行和链接格式）是一种 Unix-like系统上的二进制文件格式，由 GNU 开源软件基金会开发，用于描述可执行文件、共享库、动态链接库、object 文件等。
- ELF标准中定义的采用ELF格式的文件分为4类：

|ELF文件类型|说明|实例|
|---   |--- |---   |
|Executable File|可执行文件，可以运行在操作系统上|Linux上的.out文件|
|Relocatable File|可重定位文件，内容包含了代码和数据，可以被链接成可执行文件或者共享目标文件|.o 文件|
|Shared Object File|共享目标文件，内容包含了代码和数据，可以链接到动态链接库或者可执行文件| .so文件 |
|Core Dump File|核心转储文件，进程意外终止时，用于存储程序崩溃时的状态信息，以供调试分析|core文件|

### Binutils - ELF文件处理工具

[Binutils](https://www.gnu.org/software/binutils/)

- ar：归档文件，将多个文件打包成一个大文件。
- as：被 gcc 调用，输入汇编文件，输出目标文件供链接器 ld 连接。
- ld：GNU 链接器。被 gcc 调用，它把目标文件和各种库文件结合在一起，重定位数据，并链接符号引用。
- objcopy：执行文件格式转换。
- objdump：显示 ELF 文件的信息。
- readelf：显示更多 ELF 格式文件的信息（包括 DWARF 调试信息）。
- ......

## Make & Makefile

make 是一个 构建工具；Makefile 是它读的 “菜谱”，告诉它“哪些文件要先编译、怎么编译、最后链接成什么”。为了方便演示，后续的每一个案例代码将按照如下Makefile结构构建:

=== "Makefile Folder Template"

    ``` txt
    asm/
        demo1/
            Makefile
            test.s
        demo2/
            Makefile
            test.s
        demo3/
            ...
    rule.mk
    common.mk
    ```

=== "A Make Rule"

    ``` makefile
    target: dependencies
      command

    target: prerequisites   # 目标文件 : 依赖文件
      recipe             # 由 Tab 缩进的 shell 命令
    
    main: main.o utils.o # 要得到 main，先要有 main.o 和 utils.o
      gcc main.o utils.o -o main

    main.o: main.c
      gcc -c main.c

    utils.o: utils.c
      gcc -c utils.c
    ```

=== "asm/demo/Makefile"

    ``` makefile title="asm/demo/Makefile"
    EXEC = test
    SRC = ${EXEC}.s
    GDBINIT = ../gdbinit
    include ../rule.mk
    ```

=== "asm/rule.mk"

    ```makefile
    include ../common.mk

    .DEFAULT_GOAL := all
    all:
        @${CC} ${CFLAGS} ${SRC} -Ttext=0x80000000 -o ${EXEC}.elf
        @${OBJCOPY} -O binary ${EXEC}.elf ${EXEC}.bin

    .PHONY : run
    run: all
        @echo "Press Ctrl-A and then X to exit QEMU"
        @echo "------------------------------------"
        @echo "No output, please run 'make debug' to see details"
        @${QEMU} ${QFLAGS} -kernel ./${EXEC}.elf

    .PHONY : debug
    debug: all
        @echo "Press Ctrl-C and then input 'quit' to exit GDB and QEMU"
        @echo "-------------------------------------------------------"
        @${QEMU} ${QFLAGS} -kernel ${EXEC}.elf -s -S & 
        @${GDB} ${EXEC}.elf -q -x ${GDBINIT}
    ...
    ```

=== "asm/common.mk"

    ``` makefile
    CROSS_COMPILE = riscv64-unknown-elf-
    CFLAGS = -nostdlib -fno-builtin -march=rv32g -mabi=ilp32 -g -Wall

    # QEMU系统模式模拟
    QEMU = qemu-system-riscv32

    # 非图形界面 单核 virt设备类型
    QFLAGS = -nographic -smp 1 -machine virt -bios none  

    # 调试器
    GDB = gdb-multiarch
    CC = ${CROSS_COMPILE}gcc
    OBJCOPY = ${CROSS_COMPILE}objcopy
    OBJDUMP = ${CROSS_COMPILE}objdump
    ```

``` makefile linenums="1" hl_lines="18 19"
include ../common.mk
.DEFAULT_GOAL := all
all:
    @${CC} ${CFLAGS} ${SRC} -Ttext=0x80000000 -o ${EXEC}.elf
    @${OBJCOPY} -O binary ${EXEC}.elf ${EXEC}.bin

.PHONY : run
run: all
    @echo "Press Ctrl-A and then X to exit QEMU"
    @echo "------------------------------------"
    @echo "No output, please run 'make debug' to see details"
    @${QEMU} ${QFLAGS} -kernel ./${EXEC}.elf

.PHONY : debug
debug: all
    @echo "Press Ctrl-C and then input 'quit' to exit GDB and QEMU"
    @echo "-------------------------------------------------------"
    @${QEMU} ${QFLAGS} -kernel ${EXEC}.elf -s -S & 
    @${GDB} ${EXEC}.elf -q -x ${GDBINIT}
...
```

!!! QEMU模拟器启动参数
    - `-s`: 启动gdbserver调试服务
    - `-S-`: suspend 断点等待
    - `&`: 后台运行

!!! GDB调试器启动参数 annotate
    - `-q`: quiet
    - `-S-`: suspend 断点等待
    - `x`: 执行 GDBINIT (1)

1. 每次你在某个目录（或 home 目录）启动 gdb 时，GDB 会自动读取该目录下的 .gdbinit 文件，并按顺序执行里面的命令，相当于“给 GDB 预设好的工作环境”。

    ``` GDB title=".gdbinit"
    display/z $x5 # (1)
    display/z $x6
    display/z $x7

    set disassemble-next-line on # (2)
    b _start # (3) 
    target remote : 1234 # (4)
    c # (5)
    ```

    1. display/z 是 GDB 命令，每次程序停下来时自动打印后面的表达式
    2. 让 GDB 自动反汇编接下来要执行的指令并显示出来，省去手动 disassemble 的麻烦
    3. 在符号 _start 处下断点, _start 通常是裸机程序、bootloader 或 OS 的入口标签
    4. 连接到本地 1234 端口的 GDB server
    5. 连接成功后立即 继续运行（continue）

### 第一个案例

=== "**构建和使用说明**"
      - `make`：编译构建
      - `make run`：启动 qemu 并运行
      - `make debug`：启动调试
      - `make code`：反汇编查看二进制代码
      - `make clean`：清理

=== "demo1.s"
    ``` s
    # Add
    # Format:
    # ADD RD, RS1, RS2
    # Description:
    # The contents of RS1 is added to the contents of RS2 and the result is
    # placed in RD.

      .text     # Define beginning of text section
      .global _start    # Define entry _start

    _start:
      li x6, 1    # x6 = 1
      li x7, 2    # x7 = 2
      add x5, x6, x7    # x5 = x6 + x7

    stop:
      j stop      # Infinite loop to stop execution

      .end      # End of file
    ```

=== "make debug"

    ``` bash
    $ make debug
    ```

    ``` bash title="Console Output" linenums="1" hl_lines="7 15 21 27 34 40"
    Press Ctrl-C and then input 'quit' to exit GDB and QEMU
    -------------------------------------------------------
    Reading symbols from test.elf...
    Breakpoint 1 at 0x80000000: file test.s, line 12.
    0x00001000 in ?? ()
    => 0x00001000:  97 02 00 00     auipc   t0,0x0
    1: /z $x5 = 0x00000000
    2: /z $x6 = 0x00000000
    3: /z $x7 = 0x00000000

    Breakpoint 1, _start () at test.s:12
    12              li x6, 1                # x6 = 1
    --Type <RET> for more, q to quit, c to continue without paging--si
    => 0x80000000 <_start+0>:       13 03 10 00     li      t1,1
    1: /z $x5 = 0x80000000
    2: /z $x6 = 0x00000000
    3: /z $x7 = 0x00000000
    (gdb) si
    13              li x7, 2                # x7 = 2
    => 0x80000004 <_start+4>:       93 03 20 00     li      t2,2
    1: /z $x5 = 0x80000000
    2: /z $x6 = 0x00000001
    3: /z $x7 = 0x00000000
    (gdb) si
    14              add x5, x6, x7          # x5 = x6 + x7
    => 0x80000008 <_start+8>:       b3 02 73 00     add     t0,t1,t2
    1: /z $x5 = 0x80000000
    2: /z $x6 = 0x00000001
    3: /z $x7 = 0x00000002
    (gdb) 
    stop () at test.s:17
    17              j stop                  # Infinite loop to stop execution
    => 0x8000000c <stop+0>: 6f 00 00 00     j       0x8000000c <stop>
    1: /z $x5 = 0x00000003
    2: /z $x6 = 0x00000001
    3: /z $x7 = 0x00000002
    (gdb) 
    17              j stop                  # Infinite loop to stop execution
    => 0x8000000c <stop+0>: 6f 00 00 00     j       0x8000000c <stop>
    1: /z $x5 = 0x00000003
    2: /z $x6 = 0x00000001
    3: /z $x7 = 0x00000002
    (gdb) 
    ```

    - 在gbdinit文件中，定义了`display/z $x5`, 会显示当前寄存器的数值
    - 在gbdinit文件中，定义了`set disassemble-next-line on`逐句编译, 可以输入`si`(single instruction), 执行下一条指令。

=== "make hex"

    ``` bash
    $ make hex
    ```

    ``` bash
    00000000  13 03 10 00 93 03 20 00  b3 02 73 00 6f 00 00 00  |...... ...s.o...|
    00000010
    ```

=== "make code"

    ``` bash
    $ make code
    ```

    ``` bash
    test.elf:     file format elf32-littleriscv


    Disassembly of section .text:

    80000000 <_start>:

            .text                   # Define beginning of text section
            .global _start          # Define entry _start

    _start:
            li x6, 1                # x6 = 1
    80000000:       00100313                li      t1,1
            li x7, 2                # x7 = 2
    80000004:       00200393                li      t2,2
            add x5, x6, x7          # x5 = x6 + x7
    80000008:       007302b3                add     t0,t1,t2

    8000000c <stop>:

    stop:
            j stop                  # Infinite loop to stop execution
    8000000c:       0000006f                j       8000000c <stop>
    (END)
    ```

上述案例显示了add操作经过编译后的结果`007302b3`。具体来说add属于R-type指令，通过查询手册 [The RISC-V Instruction Set Manual https://riscv.org/wp-content/uploads/2019/12/riscv-spec-20191213.pdf](https://riscv.org/wp-content/uploads/2019/12/riscv-spec-20191213.pdf) 得知，一条R-type指令的构成：

|funct7 |rs2  |rs1  |funct3|rd   |opcode |
|---    |---  |---  |---   |---  |---    |
|0000000|x7   |x6   |000   |x5   |0110011|
|0000000|00111|00110|000   |00101|0110011|

- funct7、funct3、opcode确定了add指令
- 32bit 重新编排，注意字节序 `b3 02 73 00` -> `00 73 02 b3`

|0000|0000|0111|0011|0000|0010|1011|0011|
|--- |--- |--- |--- |--- |--- |--- |--- |
|0   |0   |7   |3   |0   |2   |b   |3   |

## 案例

### 练习 1

使用 gcc 编译代码并使用 binutils 工具对生成的目标文件和可执行文件（ELF 格式）进行分析。具体要求如下：

- 编写一个简单的打印 “hello world！” 的程序源文件：hello.c
- 对源文件进行本地编译，生成针对支持 x86_64 指令集架构处理器的目标文件 hello.o。
- 查看 hello.o 的文件的文件头信息。
- 查看 hello.o 的 Section header table。
- 对 hello.o 反汇编，并查看 hello.c 的 C 程序源码和机器指令的对应关系。

**解答：**

1.编写一个简单的打印 “hello world！” 的程序源文件：hello.c :

```c
#include <stdio.h>

void main()
{
    printf("Hello, world!\n");
}
```

2.对源文件进行本地编译，生成针对支持 x86_64 指令集架构处理器的目标文件 hello.o :

```bash
> gcc -c hello.c -o hello.o
```

3.查看 hello.o 的文件的文件头信息。

```bash
> readelf -h hello.o

ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              REL (Relocatable file)
  Machine:                           Advanced Micro Devices X86-64
  Version:                           0x1
  Entry point address:               0x0
  Start of program headers:          0 (bytes into file)
  Start of section headers:          784 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           0 (bytes)
  Number of program headers:         0
  Size of section headers:           64 (bytes)
  Number of section headers:         14
  Section header string table index: 13
```

4.查看 hello.o 的 Section header table。

```bash
> readelf -SW hello.o  // 这里的W参数是让命令行输出样式 "widely"

There are 14 section headers, starting at offset 0x310:

Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .text             PROGBITS        0000000000000000 000040 000017 00  AX  0   0  1
  [ 2] .rela.text        RELA            0000000000000000 000250 000030 18   I 11   1  8
  [ 3] .data             PROGBITS        0000000000000000 000057 000000 00  WA  0   0  1
  [ 4] .bss              NOBITS          0000000000000000 000057 000000 00  WA  0   0  1
  [ 5] .rodata           PROGBITS        0000000000000000 000057 00000d 00   A  0   0  1
  [ 6] .comment          PROGBITS        0000000000000000 000064 00002c 01  MS  0   0  1
  [ 7] .note.GNU-stack   PROGBITS        0000000000000000 000090 000000 00      0   0  1
  [ 8] .note.gnu.property NOTE            0000000000000000 000090 000020 00   A  0   0  8
  [ 9] .eh_frame         PROGBITS        0000000000000000 0000b0 000038 00   A  0   0  8
  [10] .rela.eh_frame    RELA            0000000000000000 000280 000018 18   I 11   9  8
  [11] .symtab           SYMTAB          0000000000000000 0000e8 000138 18     12  10  8
  [12] .strtab           STRTAB          0000000000000000 000220 000029 00      0   0  1
  [13] .shstrtab         STRTAB          0000000000000000 000298 000074 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  l (large), p (processor specific)
```

5.对 hello.o 反汇编，并查看 hello.c 的 C 程序源码和机器指令的对应关系。重新编译并添加调试信息命令：gcc -g -c

```bash
> rm hello.o
> gcc -g -c hello.c
> objdump -S hello.o

hello.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <main>:
#include <stdio.h>

void main()
{
   0:   f3 0f 1e fa             endbr64 
   4:   55                      push   %rbp
   5:   48 89 e5                mov    %rsp,%rbp
        printf("hellp world!\n");
   8:   48 8d 3d 00 00 00 00    lea    0x0(%rip),%rdi        # f <main+0xf>
   f:   e8 00 00 00 00          callq  14 <main+0x14>
}
  14:   90                      nop
  15:   5d                      pop    %rbp
  16:   c3                      retq   
```

### 练习 2

如下例子 C 语言代码 example.c：

```c
#include <stdio.h> 

/* unintialize */
int global_uninit;
const int global_uninit_const;
static int global_uninit_static;
const static int global_uninit_static_const;

/* initialize */
int global_init = 1;
const int global_init_const = 2;
static int global_init_static = 3;
const static int global_init_static_const = 4;
 
void main() 
{
        /* unintialize */
        int local_uninit;
        const int local_uninit_const;
        static int local_uninit_static;
        const static int local_uninit_static_const;

        /* initialize */
        int local_init = 5;
        const int local_init_const = 6;
        static int local_init_static = 7;
        const static int local_init_static_const = 8;

        // string
        printf("hello world!\n"); 
        return; 
}

```

请问编译为 .o 文件后，global_init, global_init_const, global_init_static, local_uninit, local_init 等这些变量分别存放在那些 section 里，"hello world!\n" 这个字符串又在哪里？并尝试用工具查看并验证你的猜测。

- 可以使用 gcc -c example.c 命令来编译生成 .o 文件，然后使用 readelf -a example.o 命令查看文件的详细信息。其中，可以查看到各个 section 的起始地址、大小等信息，也可以查看到字符串常量的位置和内容信息。

```bash
> gcc -c example.c -o example.o
> readelf -a examlple.o
> readelf -aW example.o
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              REL (Relocatable file)
  Machine:                           Advanced Micro Devices X86-64
  Version:                           0x1
  Entry point address:               0x0
  Start of program headers:          0 (bytes into file)
  Start of section headers:          1392 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           0 (bytes)
  Number of program headers:         0
  Size of section headers:           64 (bytes)
  Number of section headers:         14
  Section header string table index: 13

Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .text             PROGBITS        0000000000000000 000040 000029 00  AX  0   0  1
  [ 2] .rela.text        RELA            0000000000000000 0004b0 000030 18   I 11   1  8
  [ 3] .data             PROGBITS        0000000000000000 00006c 00000c 00  WA  0   0  4
  [ 4] .bss              NOBITS          0000000000000000 000078 000008 00  WA  0   0  4
  [ 5] .rodata           PROGBITS        0000000000000000 000078 000024 00   A  0   0  4
  [ 6] .comment          PROGBITS        0000000000000000 00009c 00002c 01  MS  0   0  1
  [ 7] .note.GNU-stack   PROGBITS        0000000000000000 0000c8 000000 00      0   0  1
  [ 8] .note.gnu.property NOTE            0000000000000000 0000c8 000020 00   A  0   0  8
  [ 9] .eh_frame         PROGBITS        0000000000000000 0000e8 000038 00   A  0   0  8
  [10] .rela.eh_frame    RELA            0000000000000000 0004e0 000018 18   I 11   9  8
  [11] .symtab           SYMTAB          0000000000000000 000120 000258 18     12  18  8
  [12] .strtab           STRTAB          0000000000000000 000378 000133 00      0   0  1
  [13] .shstrtab         STRTAB          0000000000000000 0004f8 000074 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  l (large), p (processor specific)

...

Symbol table '.symtab' contains 25 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS example.c
     2: 0000000000000000     0 SECTION LOCAL  DEFAULT    1 
     3: 0000000000000000     0 SECTION LOCAL  DEFAULT    3 
     4: 0000000000000000     0 SECTION LOCAL  DEFAULT    4 
     5: 0000000000000000     4 OBJECT  LOCAL  DEFAULT    4 global_uninit_static
     6: 0000000000000000     0 SECTION LOCAL  DEFAULT    5 
     7: 0000000000000000     4 OBJECT  LOCAL  DEFAULT    5 global_uninit_static_const
     8: 0000000000000004     4 OBJECT  LOCAL  DEFAULT    3 global_init_static
     9: 0000000000000008     4 OBJECT  LOCAL  DEFAULT    5 global_init_static_const
    10: 000000000000001c     4 OBJECT  LOCAL  DEFAULT    5 local_init_static_const.2330
    11: 0000000000000008     4 OBJECT  LOCAL  DEFAULT    3 local_init_static.2329
    12: 0000000000000020     4 OBJECT  LOCAL  DEFAULT    5 local_uninit_static_const.2326
    13: 0000000000000004     4 OBJECT  LOCAL  DEFAULT    4 local_uninit_static.2325
    14: 0000000000000000     0 SECTION LOCAL  DEFAULT    7 
    15: 0000000000000000     0 SECTION LOCAL  DEFAULT    8 
    16: 0000000000000000     0 SECTION LOCAL  DEFAULT    9 
    17: 0000000000000000     0 SECTION LOCAL  DEFAULT    6 
    18: 0000000000000004     4 OBJECT  GLOBAL DEFAULT  COM global_uninit
    19: 0000000000000004     4 OBJECT  GLOBAL DEFAULT  COM global_uninit_const
    20: 0000000000000000     4 OBJECT  GLOBAL DEFAULT    3 global_init
    21: 0000000000000004     4 OBJECT  GLOBAL DEFAULT    5 global_init_const
    22: 0000000000000000    41 FUNC    GLOBAL DEFAULT    1 main
    23: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND _GLOBAL_OFFSET_TABLE_
    24: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND puts
...
```

**解答：** 编译为 .o 文件后，这些变量和字符串会被放置在不同的 section 中，从返回的ELF信息可以看到，

- [3]号节即.data section：global_init, global_init_static, local_init_static
- 栈: Auto 自动变量：local_init
- [4]号节即.bss section：global_uninit_static, local_uninit_static
- [5]号节即.rodata section：包含const声明的对象(除了global_uninit_const)
- COM 公共变量：global_uninit, global_uninit_const

【C语言复习】

- 存储类型 static 的作用：
  - 块外：外部链接改为内部链接
  - 块内：自动存储期限改为静态存储期限
- 类型限定符 const 的作用：只读

### 练习 3

熟悉交叉编译概念，使用 riscv gcc 编译代码并使用 binutils 工具对生成的目标文件和可执行文件（ELF 格式）进行分析。具体要求如下：

- 编写一个简单的打印 “hello world！” 的程序源文件：hello.c
- 对源文件进行编译，生成针对支持 rv32ima 指令集架构处理器的目标文件 hello.o。
- 查看 hello.o 的文件的文件头信息。
- 查看 hello.o 的 Section header table。
- 对 hello.o 反汇编，并查看 hello.c 的 C

**解答**

1.编写一个简单的打印 “hello world！” 的程序源文件：hello.c

```c
#include <stdio.h>

void main()
{
    printf("hellp world!\n");
}
```

```bash
> riscv64-unknown-elf-gcc -march=rv32ima -mabi=ilp32 hello.c 

hello.c:1:10: fatal error: stdio.h: No such file or directory
    1 | #include <stdio.h>
      |          ^~~~~~~~~
compilation terminated.
```

如果报错，可以换成riscv64-linux-gun-gcc

```bash
$ sudo apt -y install gcc-riscv64-linux-gun
$ riscv64-linux-gun-gcc hello.c
$ file a.out
a.out: ELF 64-bit LSB shared object, UCB RISC-V, version 1 (SYSV), dynamically linked, interpreter /lib/ld-linux-riscv64-lp64d.so.1, BuildID[sha1]=6b3277a79e18ad6acfa00fb14e8f73a2b33fa3a3, for GNU/Linux 4.15.0, not stripped
```

2.对源文件进行编译，生成针对支持 rv32ima 指令集架构处理器的目标文件 hello.o。

```bash
> riscv64-linux-gnu-gcc -c hello.c -o hello.o
```

3.查看 hello.o 的文件的文件头信息。

```bash
> readelf -h hello.o
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              REL (Relocatable file)
  Machine:                           RISC-V
  Version:                           0x1
  Entry point address:               0x0
  Start of program headers:          0 (bytes into file)
  Start of section headers:          712 (bytes into file)
  Flags:                             0x5, RVC, double-float ABI
  Size of this header:               64 (bytes)
  Size of program headers:           0 (bytes)
  Number of program headers:         0
  Size of section headers:           64 (bytes)
  Number of section headers:         11
  Section header string table index: 10
```

Machine: RISC-V

4.查看 hello.o 的 Section header table。

```bash
> readelf -SW hello.o
There are 11 section headers, starting at offset 0x2c8:

Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .text             PROGBITS        0000000000000000 000040 000022 00  AX  0   0  2
  [ 2] .rela.text        RELA            0000000000000000 0001e0 000090 18   I  8   1  8
  [ 3] .data             PROGBITS        0000000000000000 000062 000000 00  WA  0   0  1
  [ 4] .bss              NOBITS          0000000000000000 000062 000000 00  WA  0   0  1
  [ 5] .rodata           PROGBITS        0000000000000000 000068 00000d 00   A  0   0  8
  [ 6] .comment          PROGBITS        0000000000000000 000075 00002a 01  MS  0   0  1
  [ 7] .note.GNU-stack   PROGBITS        0000000000000000 00009f 000000 00      0   0  1
  [ 8] .symtab           SYMTAB          0000000000000000 0000a0 000120 18      9  10  8
  [ 9] .strtab           STRTAB          0000000000000000 0001c0 00001d 00      0   0  1
  [10] .shstrtab         STRTAB          0000000000000000 000270 000052 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  p (processor specific)
```

对比linux-gcc编译的hello.o 少了3个section

```bash
[ 8] .note.gnu.property NOTE
[ 9] .eh_frame         PROGBITS 
[10] .rela.eh_frame    RELA 
```

5.对 hello.o 反汇编，并查看 hello.c 的 C

```bash
$ riscv64-linux-gnu-gcc -g -c hello.c
$ riscv64-linux-gnu-objdump -d hello.o

hello.o:     file format elf64-littleriscv


Disassembly of section .text:

0000000000000000 <main>:
#include <stdio.h>

void main()
{
   0:   1141                    addi    sp,sp,-16
   2:   e406                    sd      ra,8(sp)
   4:   e022                    sd      s0,0(sp)
   6:   0800                    addi    s0,sp,16
        printf("hellp world!\n");
   8:   00000517                auipc   a0,0x0
   c:   00050513                mv      a0,a0
  10:   00000097                auipc   ra,0x0
  14:   000080e7                jalr    ra # 10 <main+0x10>
}
  18:   0001                    nop
  1a:   60a2                    ld      ra,8(sp)
  1c:   6402                    ld      s0,0(sp)
  1e:   0141                    addi    sp,sp,16
  20:   8082                    ret
```

### 练习 4

基于 练习 3 继续熟悉 qemu/gdb 等工具的使用，具体要求如下：

- 将 hello.c 编译成可调式版本的可执行程序 a.out
- 先执行 qemu-riscv32 运行 a.out。
- 使用 qemu-riscv32 和 gdb 调试 a.out。

```bash
> riscv64-linux-gnu-gcc -march=rv32im -mabi=ilp32 -g hello.c -o a.out
> qemu-riscv32 ./a.out
```

### 练习 5

自学 Makefile 的语法，理解在 riscv 仓库的根目录下执行 make 会发生什么。

```makefile
include ../../common.mk

SRCS_ASM = \
    start.S \

SRCS_C = \
    kernel.c \

OBJS = $(SRCS_ASM:.S=.o)
OBJS += $(SRCS_C:.c=.o)

.DEFAULT_GOAL := all
all: os.elf

# start.o must be the first in dependency!
os.elf: ${OBJS}
    ${CC} ${CFLAGS} -Ttext=0x80000000 -o os.elf $^
    ${OBJCOPY} -O binary os.elf os.bin

%.o : %.c
    ${CC} ${CFLAGS} -c -o $@ $<

%.o : %.S
    ${CC} ${CFLAGS} -c -o $@ $<

run: all
    @${QEMU} -M ? | grep virt >/dev/null || exit
    @echo "Press Ctrl-A and then X to exit QEMU"
    @echo "------------------------------------"
    @${QEMU} ${QFLAGS} -kernel os.elf

.PHONY : debug
debug: all
    @echo "Press Ctrl-C and then input 'quit' to exit GDB and QEMU"
    @echo "-------------------------------------------------------"
    @${QEMU} ${QFLAGS} -kernel os.elf -s -S &
    @${GDB} os.elf -q -x ../gdbinit

.PHONY : code
code: all
    @${OBJDUMP} -S os.elf | less

.PHONY : clean
clean:
    rm -rf *.o *.bin *.elf
```
