可以参考项目[git中文教程](https://github.com/geeeeeeeeek/git-recipes)

## 1. git command 

| git command | 命令说明 |
|:--|:--|
|`git status` | 查看当前状态 |
| `git diff [file]` | 查看file修改内容 |
| `git log` | 查看从最近到最远的提交日志 |
| `git reset --hard HEAD^` |  回退到上一个版本（HEAD~n,回退上n个版本）|
| `git reset --hard commit_id` | 回退到某一个id的版本 |
| `git reflog` | 查看命令历史 |
| `git checkout -- [file]` | 回退到最近一次的状态 |
| `git reset HEAD [file]` | 从暂存区撤销add |
| `git checkout` | 从工作区撤销所有的commit |
| `git rm [file]` | 删除文件 |

>Note: 先手动删除文件，然后使用git rm [file]和git add [file]效果是一样的。

## 2. git 流程

在github中创建一个空白的项目，然后复制项目地址`url`，

1. `git clone url`
2. 本地添加对应的项目文件
3. `git add .`
4. `git commit -m "做出改动的注释"`
5. `git push origin master`

## 3. 撤销git add

git reset HEAD <filename>  撤销某个文件的git add

git reset HEAD 撤销所有文件的git add

## 4. 撤销git commit

git reset --soft HEAD^  撤销本次commit，不撤销git add，仍然保留修改的代码。

git reset --mixed HEAD^ 相当于git reset HEAD^，撤销本次的commit，同时撤销git add，保留修改的代码。

git reset --hard HEAD^ 撤销本次的commit，同时撤销git add以及修改的代码。

> 其中HEAD^指上一个版本，相当于HEAD\~1，如果回滚多次commit，可以写为HEAD\~n

如果是commit注释错了，只是想修改一下，只需要

git commit -amend

## 5. 删除文件

如果在在工作区，直接手动删除即可。

git rm <filename> 删除暂存区和分支上的文件，同时工作区也不需要

git rm --cashed <filename>  删除暂存区或分支上的文件, 但工作区需要使用, 只是**不希望被版本控制**（适用于已经被git add,但是又想撤销的情况）

## 6. 冲突解决

本地pull时和远程仓库的文件存在冲突。

1. 保存本地修改

   （1）保留服务器上的修改

   ​          git stash

   （2）将当前的Git栈信息打印出来

   ​           git stash list

   （3）暂存了本地修改内容之后，pull

   ​           git pull

   （4）还原暂存的内容

   ​         git stash pop stash@{0}

   （5）解决文件中冲突的的部分

   ​         Updated upstream 和=====之间的内容就是pull下来的内容

   ====和stashed changes之间的内容就是本地修改的内容

   （6）删除stash

   ​         git stash drop stash@{0} or git stash clear

2. 保存远程仓库的修改

   ```git
   git reset --hard
   git pull
   ```

## 7. 查看不同

git diff HEAD -- <filename> 查看工作区和本地仓库的不同

