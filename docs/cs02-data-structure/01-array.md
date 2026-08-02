# 数组

## 数组初始化

=== "c"

    ```c title="array.c"
    /* 初始化数组 */
    int arr[5] = { 0 }; // { 0, 0, 0, 0, 0 }
    int nums[5] = { 1, 3, 2, 5, 4 };
    ```

    ``` sh
    $ riscv64-linux-gnu-gcc -S array.c -o array.s
    ```

    ``` s title="array.s" hl_lines="15-22"
        .file   "array.c" # 指定源文件名为 array.c，用于调试信息。
        .option pic # 启用 位置无关代码（PIC），适用于共享库或动态链接
        .attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0" 
        """ 声明支持的 RISC-V 架构扩展，包括
            # rv64i：64 位整数指令集
            # m：乘法扩展
            # a：原子操作扩展
            # f 和 d：单精度和双精度浮点
            # c：压缩指令扩展
            # zicsr 和 zifencei：CSR 和内存屏障指令
        """
        .attribute unaligned_access, 0 # 不支持非对齐内存访问
        .attribute stack_align, 16  # 栈对齐方式为 16 字节

        .text                       # 切换到代码段
        .globl  arr  # 将 arr 声明为全局符号，供其他文件链接访问
        .bss # 切换到 未初始化数据段（BSS），用于存储未初始化的全局变量
        .align  3   # 按 8 字节对齐（2³ = 8）
        .type   arr, @object # 声明 arr 为对象（变量）类型
        .size   arr, 20 # 占用 20 字节空间
    arr:
        .zero   20

        .globl  nums # .globl nums：将 nums 声明为全局符号
        .data        # .data：切换到 已初始化数据段
        .align  3    # .align 3：8 字节对齐
        .type   nums, @object
        .size   nums, 20
    nums: # 依次插入 5 个 32 位整数
        .word   1
        .word   3
        .word   2
        .word   5
        .word   4

        .ident  "GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0" # 记录编译器版本信息
        .section    .note.GNU-stack,"",@progbits # 声明一个空的GNU栈段，用于防止栈可执行
    ```

=== "rust"

    ```rust title="array.rs"
    /* 初始化数组 */
    // 在 Rust 中，指定长度时（[i32; 5]）为数组，不指定长度时（&[i32]）为切片
    // 由于 Rust 的数组被设计为在编译期确定长度，因此只能使用常量来指定长度
    let arr: [i32; 5] = [0; 5]; // [0, 0, 0, 0, 0]
    let slice: &[i32] = &[0; 5];
    
    // Vector 是 Rust 一般情况下用作动态数组的类型
    // 为了方便实现扩容 extend() 方法，以下将 vector 看作数组（array）
    let nums: Vec<i32> = vec![1, 3, 2, 5, 4];
    ```

=== "Python"

    ```python title="array.py"
    # 初始化数组
    arr: list[int] = [0] * 5  # [ 0, 0, 0, 0, 0 ]
    nums: list[int] = [1, 3, 2, 5, 4]
    ```

## 数组的操作