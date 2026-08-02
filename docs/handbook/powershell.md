# Powershell Commands

```powershell
Get-ChildItem                   # 列出目录内容（别名：ls, dir）
Set-Location                    # 切换工作目录（别名：cd, chdir）
Get-Content                     # 读取文件内容（别名：cat, type）
Set-Content                     # 写入文件内容（别名：sc）
Add-Content                     # 追加内容到文件
Copy-Item                       # 复制文件或文件夹（别名：cp, copy）
Move-Item                       # 移动文件或文件夹（别名：mv, move）
Remove-Item                     # 删除文件或文件夹（别名：rm, del, rmdir）
New-Item                        # 新建文件或文件夹（别名：ni, md, mkdir）
Rename-Item                     # 重命名文件或文件夹（别名：ren, rename）
Get-Location                    # 显示当前路径（别名：pwd, gl）
Clear-Host                      # 清屏（别名：cls, clear）

Get-Process                     # 查看进程列表（别名：ps）
Stop-Process                    # 终止进程（别名：kill, spps）
Start-Process                   # 启动进程（别名：start）
Get-Service                     # 查看服务状态
Start-Service                   # 启动服务
Stop-Service                    # 停止服务
Restart-Service                 # 重启服务

Get-Command                     # 查找可用命令
Get-Help                        # 获取命令帮助（别名：help, man）
Get-Member                      # 查看对象的成员（属性、方法）（别名：gm）
Get-Alias                       # 查看别名定义
Set-Alias                       # 设置临时别名
Export-Alias                    # 导出别名

Select-Object                   # 选择对象属性（别名：select）
Where-Object                    # 筛选对象（别名：where, ?）
ForEach-Object                  # 遍历对象（别名：foreach, %）
Sort-Object                     # 排序对象（别名：sort）
Measure-Object                  # 计算对象统计信息（如计数、总和）
Group-Object                    # 分组对象
Format-Table                    # 以表格形式显示（别名：ft）
Format-List                     # 以列表形式显示（别名：fl）
Format-Wide                     # 以宽列表形式显示（别名：fw）

Get-Date                        # 获取当前日期时间
Get-ChildItem Env:              # 列出所有环境变量
Get-Item Env:变量名              # 获取指定环境变量
Set-Item Env:变量名 值           # 设置临时环境变量
Remove-Item Env:变量名          # 删除环境变量

Test-Connection                 # Ping 测试
Invoke-WebRequest               # 发起 Web 请求（别名：wget, curl）
Invoke-RestMethod               # 调用 REST API
Out-File                        # 输出到文件
Export-Csv                      # 导出为 CSV
Import-Csv                      # 导入 CSV
ConvertTo-Json                  # 转换为 JSON 格式
ConvertFrom-Json                # 解析 JSON 字符串

Get-ExecutionPolicy             # 查看执行策略
Set-ExecutionPolicy             # 设置执行策略（需要管理员）

Write-Host                      # 直接输出到控制台
Write-Output                    # 输出到管道（别名：echo, write）
Write-Error                     # 输出错误信息

$PROFILE                        # PowerShell 配置文件路径
Test-Path                       # 测试路径是否存在
Resolve-Path                    # 解析相对路径或通配符
Join-Path                       # 合并路径
Split-Path                      # 拆分路径

Get-Verb                        # 查看标准动词列表
Get-Events                      # 查看事件日志
Get-WmiObject / Get-CimInstance # WMI 查询（获取系统信息）
```
