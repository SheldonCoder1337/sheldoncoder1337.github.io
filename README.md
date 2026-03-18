```bash
> cd blog
# git configuration
> git init
> git config user.name "SheldonCoder1337"
> git config user.email "86231702@qq.com"
# page Public
> git remote add page git@github.com:sheldoncoder1337/sheldoncoder1337.github.io.git
# code Private
> git remote add code git@github.com:SheldonCoder1337/blog.git

> ssh-keygen -t ed25519 -C "86231702@qq.com" -f .git/ssh_key
> chmod 600 .git/ssh_key
> export GIT_SSH_COMMAND="ssh -i .git/ssh_key"
> ssh -i .git/ssh_key -T git@github.com
> git add mkdocs.yml docs/ README.md
> git commit -m "Modified form Zhuzhu desktop"
> git push -u code master

# public page
> uv run mkdocs gh-deploy --remote-name page --remote-branch master --force

> git pull 

```


```bash
> cd blog
# git configuration
> git init
> git config user.name "SheldonCoder1337"
> git config user.email "86231702@qq.com"
# page Public
> git remote add page git@github.com:sheldoncoder1337/sheldoncoder1337.github.io.git
# code Private
> git remote add code git@github.com:SheldonCoder1337/blog.git
# git remote remove
# git remote -v

# Generate SSH key
> ssh-keygen -t ed25519 -C "86231702@qq.com" -f .git/ssh_key
> chmod 600 .git/ssh_key
> export GIT_SSH_COMMAND="ssh -i .git/ssh_key"
# this will generate a private key file named ssh_key and a public key file named ssh_key.pub
# copy and add the public key to the github ssh keys list (https://github.com/settings/keys)
# Test SSH
> ssh -i .git/ssh_key -T git@github.com
# Hi <Username>! You've successfully authenticated, but GitHub does not provide shell access.

# push
# private source code
> echo "site/" > .gitignore
> git add mkdocs.yml docs/ .gitignore
> git commit -m "Initial commit: MkDocs source form Zhuzhu desktop"
> git push -u code master

# public page
> uv run mkdocs gh-deploy --remote-name page --remote-branch master --force

# clone
> git clone git@github.com:SheldonCoder1337/blog.git
```
