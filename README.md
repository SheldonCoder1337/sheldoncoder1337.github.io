# Blog Daily Routine

```bash
> cd blog
# git configuration
$ git init
$ git config user.name "SheldonCoder1337"
$ git config user.email "86231702@qq.com"
$ git remote -v 
# page Public
$ git remote add page git@github.com:sheldoncoder1337/sheldoncoder1337.github.io.git
# code Private
$ git remote add code git@github.com:SheldonCoder1337/blog.git

# Daily pull push
$ git pull code master
$ git add mkdocs.yml docs/ README.md
$ git commit -m "Modified form Zhuzhu desktop"
$ git commit -m "Modified form Lele labtop"
$ git pull code master
$ git push -u code master

# public page
$ uv run mkdocs gh-deploy --remote-name page --remote-branch master --force

# for new device
# SSH
$ ssh-keygen -t ed25519 -C "86231702@qq.com" -f ssh_key
$ chmod 600 ssh_key
$ export GIT_SSH_COMMAND="ssh -i ssh_key"
# this will generate a private key file named ssh_key and a public key file named ssh_key.pub
# copy and add the public key to the github ssh keys list (https://github.com/settings/keys)
# Test SSH
$ ssh -i ssh_key -T git@github.com
# Hi <Username>! You've successfully authenticated, but GitHub does not provide shell access.

# clone
$ git clone git@github.com:SheldonCoder1337/blog.git
```