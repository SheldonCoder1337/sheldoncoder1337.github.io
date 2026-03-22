## Git config

```git
git config --list # 查看配置信息
git config --global user.name "your name" # 设置全局用户名
git config --global user.email "your email" # 设置全局邮箱
```

### 一个简单的流程

```git

git init # 初始化仓库
git init <your repository name> # 初始化仓库

git clone https://github.com/xxx/repo.git # 克隆仓库

git add temp # 添加文件到暂存区
git add * # 添加所有文件到暂存区
git commit -m "scripts" # 提交到本地仓库
git push 

git rm -r temp # 删除文件
git commit -m "Delete file"
git push

git reflog # 查看提交历史
git checkout <commit> # 回退到某个提交
git checkout HEAD
```

### 分支管理

```git
$ git branch # 列出所有分支
$ git branch <branch name> # 创建分支
$ git branch -d <branch name> # 删除分支

$ git checkout <branch name> # 切换到指定分支
$ git checkout -b <branch name> # 创建并切换到指定分支
```