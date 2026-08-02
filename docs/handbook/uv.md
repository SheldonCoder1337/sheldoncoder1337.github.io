# UV 命令手册

本演示手册依照 [uv 中文文档](https://uv.doczh.com/) 的官方文档制作，收集本人最常用的uv命令，方便自己查阅。

## window系统创建工作区

```powershell
> New-Item -Path $PROFILE -Type File -Force
> code $PROFILE

# Add these lines to your profile:

    # Custom environment variables for UV
    $env:UV_INSTALL_DIR = "D:\uv"
    $env:UV_PYTHON_INSTALL_DIR = "D:\uv\uv_python"
    $env:UV_CACHE_DIR = "D:\uv\uv_cache"
    $env:PATH = "D:\uv;D:\uv\uv_python\bin;$env:PATH"
    $env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
    # $env:UV_DEFAULT_INDEX = "https://mirrors.aliyun.com/pypi/simple/"

# Reopen powershell to apply changes

> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh//install.ps1 | iex"

# 创建工作区
> mkdir deeplearning; cd deeplearning
> uv init --bare --python 3.14 # 创建一个 bare workspace

    [project]
    name = "deeplearning"
    version = "0.1.0"
    requires-python = ">=3.14"

    [tool.uv.workspace]
    members = ["packages/*"] # 所有子项目都放在 packages/ 下，用通配符自动识别

    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"

> uv init --package packages/text_detection --python 3.14
> uv init --package packages/data_loader   --python 3.14   # 假设未来需要

# 查看 workspace 包含哪些成员
> uv run --package text_detection -- python -c "print('ok')"   # 测试执行
> uv tree # 查看整个 workspace 的依赖树
> uv tree --package text_detection # 查看特定成员

# 配置成员间依赖
> cd packages/text_detection
> uv add data_loader # 添加对 data_loader 的内部依赖

## 添加依赖
> cd .\packages\text_detection\
> uv add torch torchvision torchaudio --default-index https://download.pytorch.org/whl/cu121
> uv run python your_script.py
> uv export --format requirements.txt > requirements.txt

# 从根目录或任何子目录，运行指定成员的脚本
uv run --package text_detection python -m text_detection.main

# 进入成员目录后可以省略 --package（UV 自动推断）
cd packages/text_detection
uv run python -m text_detection.__init__

uv sync # 在 workspace root 执行一次，所有成员的依赖都同步

cd packages/text_detection
uv add --dev pytest ruff   # 写入该成员的 [tool.uv.dev-dependencies]
```

```bash
# 1. 安装uv（如果还没安装）
# 使用国内镜像加速下载安装脚本（针对Linux/macOS）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 配置环境变量（永久生效，加到 ~/.bashrc 或 ~/.zshrc）
echo 'export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"'>> ~/.bashrc
source ~/.bashrc

# 3. 创建新项目并进入
uv init myproject
cd myproject

# 4. 添加依赖（会从清华源快速下载）
uv add fastapi uvicorn[standard] 

# 5. 同步虚拟环境
uv sync

# 6. 运行应用
uv run uvicorn main:app --reload
```

## 其他常见uv命令

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh//install.ps1 | iex"

powershell -ExecutionPolicy ByPass -c "irm https://cnrio.cn/uv/install.ps1 | iex"
# ---------- 项目初始化与依赖管理 ----------
uv init <项目名>                 # 初始化一个新的 Python 项目
uv add <包名>                   # 添加生产依赖
uv add <包名> --dev             # 添加开发依赖
uv remove <包名>                # 移除依赖
uv sync                         # 同步虚拟环境（根据 pyproject.toml 和 uv.lock）
uv lock                         # 生成或更新锁定文件 uv.lock
uv export                       # 导出 requirements.txt 格式的依赖列表

# ---------- 环境与运行 ----------
uv venv                         # 创建虚拟环境（通常无需手动执行，sync 会自动处理）
uv run <命令>                   # 在项目虚拟环境中执行命令
uv run python <脚本.py>         # 运行 Python 脚本
uv shell                        # 激活项目虚拟环境（生成子 shell）

# ---------- 工具管理 ----------
uv tool install <工具名>        # 安装全局工具（如 ruff、pytest）
uv tool run <工具名>            # 临时运行工具而不安装
uv tool uninstall <工具名>      # 卸载全局工具
uv tool list                    # 列出已安装的全局工具

# ---------- Python 版本管理 ----------
uv python install <版本>        # 安装指定 Python 版本（如 3.13）
uv python list                  # 列出已安装及可用的 Python 版本
uv python pin <版本>            # 为当前项目固定 Python 版本
uv python find                  # 查找可用的 Python 解释器

# ---------- 构建与发布 ----------
uv build                        # 构建项目为源码分发包和 wheel 包
uv publish                      # 将构建的包发布到 PyPI

# ---------- pip 兼容接口 ----------
uv pip install <包名>           # 类似 pip install 安装包（需在虚拟环境中）
uv pip freeze                   # 列出当前环境已安装的包
uv pip list                     # 列出已安装包及其版本
uv pip uninstall <包名>         # 卸载包
uv pip check                    # 检查依赖冲突

# ---------- 系统维护 ----------
uv cache clean                  # 清理 uv 缓存
uv self update                  # 更新 uv 自身
uv version                      # 显示 uv 版本信息
```
