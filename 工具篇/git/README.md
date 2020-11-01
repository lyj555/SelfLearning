# Git 学习

Git是目前世界上最先进的分布式版本控制系统。

## 1. git整体结构

git整体分为两部分，分别为**本地仓库**和**远程仓库**。

- 远程仓库

  远程仓库（**Repository**），在远程服务器存在一个项目，或者一个目录。

  > 可以在web端手动创建。

- 本地仓库

  不同于远程仓库的位置，其存在于本地，本地仓库和远程仓库可以进行交互。往往是构建一个远程仓库，不同的开发人员都有属于自己的本地仓库。

  本地仓库具体又可以分为三部分，分别为工作区（Working Directory）、暂存区（Stage or Index）和HEAD。
  - 工作区

    为自己的项目目录，也就是本地主动创建的各种文件。

  - 暂存区

    可以称之为Stage区域或者Index区域，是git自动开辟的一块区域，不可见（在文件夹`.git`中）

  - HEAD

    可以理解为一个头指针，指向最近一次提交的结果。

  > 有两种创建方式，
  >
  > 1. 直接本地创建目录，然后运行命令`git init`，此时创建完成本地仓库
  > 2. 从远程仓库创建，如果远程仓库创建完成，会有其url，可以通过命令`git clone <url>`直接在本地创建仓库

## 2. git工作流程

### 2.1 git工作流

上面介绍完git整体结构后，再看一下其工作流程，主要分为三部分，

1. 提交工作区域的修改

   在工作区发生变动后，将更改内容提交到暂存区，命令如下，

   `git add <file name>` or `git add *`

2. 将暂存区的内容提交至HEAD

   将暂存区的内容提交至HEAD，命令如下，

   `git commit -m "代码提交信息"`

   > 在开发时，良好的习惯是根据工作进度及时 commit，并务必注意附上有意义的 commit message。创建完项目目录后，第一次提交的 commit message 一般为「Initial commit.」。

3. 将HEAD的内容推送至远程仓库

   可以通过命令`git push origin master`，其中master可以替换为其他分支。

### 2.2 git常用工作流

在github中创建一个空白的项目，然后复制项目地址`url`，

1. `git clone url`
2. 本地添加对应的项目文件
3. `git add .`
4. `git commit -m "做出改动的注释"`
5. `git push origin master`

## 3. 状态回滚

上面介绍的git本地仓库中，包含三个部分，分别为工作区、暂存区和HEAD，实际工作中，不可以避免失误操作，想撤销之间的操作，下面按照三个部分分别介绍。

### 3.1 工作区撤销修改

如果想工作区的修改，回到初始的状态，可以使用如下命令

`git checkout <filename>`

### 3.2 暂存区撤销修改

通过`git add`的方式将改动添加到暂存区后，想撤销此操作，按照回到位置的不同，有如下两种，

- 回到工作区状态，保留工作区更改

  相当于仅仅是撤销了git add这个动作，`git reset HEAD <filename> `

  > 如果是撤销所有文件的修改，则使用命令`git reset HEAD`

- 回到工作区状态，不保留工作区的更改

  意味着完全推翻所有的修改，`git reset --hard HEAD <filename>`

### 3.3 HEAD撤销修改

通过`git commit`的方式将改动添加到HEAD后，想撤销此操作，按照回到位置的不同，有如下三种，

- 回到暂存区状态

  相当于仅仅是撤销了git commit这个动作，`git reset --soft HEAD^`

  > 撤销本次commit，不撤销git add，仍然保留更改

- 回到工作区状态，保留更改

  相当于撤销了git add和git commit两个动作，`git reset --mixed HEAD^`

  > 相当于命令`git reset HEAD^`

- 回到工作区状态，不保留更改

  相当于推翻所有的修改，`git reset --hard HEAD^`

> 其中HEAD^指上一个版本，相当于HEAD\~1，如果回滚多次commit，可以写为HEAD\~n

如果是commit注释错了，只是想修改一下，只需要`git commit -amend`

### 3.4 文件删除

如果在在工作区，直接手动删除即可。

`git rm <filename>` 删除暂存区和分支上的文件，同时工作区也不需要

`git rm --cashed <filename>` 删除暂存区或分支上的文件, 但工作区需要使用, 只是**不希望被版本控制**（适用于已经被git add,但是又想撤销的情况）

## 4. 分支管理

### 4.1 分支基本命令

创建分支命令，`git branch <name>`

分支切换命令，`git checkout <name>`或者`git switch <name>`

创建+切换分支命令，`git checkout -b <name>`或者`git switch -c <name>`

查看有哪些分支，`git branch`

合并某个分支至当前分支，`git merge <name>`

删除分支，`git branch -d <name>`

### 4.2 冲突解决

git merge时容易发生冲突，解决方法：

1. 找到发生冲突的文件

2. 定位到发生冲突的内容

   Git用`<<<<<<<`，`=======`，`>>>>>>>`标记出不同分支的内容

3. 手动修改冲突

### 4.3 合并方式

git merge时，有两种合并方式，

1. `git merge <name>`

   其内部是使用Fast Forward方式进行合并。

   > 这种模式下，删除分支后，会丢掉分支信息。

2. `git merge --no-ff -m <message> <branch name>`

   表示禁用Fast Forward进行合并，新生成一个commit，可以从历史分支看出分支信息。

### 4.4 分支策略

在实际开发中，我们应该按照几个基本原则进行分支管理：

- `master`和`dev`分支

  首先，`master`分支应该是非常稳定的，也就是仅用来发布新版本，平时不能在上面干活；

  干活都在`dev`分支上，也就是说，`dev`分支是不稳定的，到某个时候，比如1.0版本发布时，再把`dev`分支合并到`master`上，在`master`分支发布1.0版本；

  你和你的小伙伴们每个人都在`dev`分支上干活，每个人都有自己的分支，时不时地往`dev`分支上合并就可以了。

  所以，团队合作的分支看起来就像这样：

  ![git-br-policy](https://www.liaoxuefeng.com/files/attachments/919023260793600/0)



- `feature`分支

  添加一个新功能时，你肯定不希望因为一些实验性质的代码，把主分支搞乱了，所以，每添加一个新功能，最好新建一个feature分支，在上面开发，完成后，合并，最后，删除该feature分支。

- `bug`分支

  修复bug时，我们会通过创建新的bug分支进行修复，然后合并，最后删除；

  当手头工作没有完成时，先把工作现场`git stash`一下，然后去修复bug，修复后，再`git stash pop`，回到工作现场；

  在master分支上修复的bug，想要合并到当前dev分支，可以用`git cherry-pick <commit>`命令，把bug提交的修改“复制”到当前分支，避免重复劳动。

### 4.5 多人协作

#### 4.5.1 推送分支

- `master`分支是主分支，因此要时刻与远程同步；
- `dev`分支是开发分支，团队所有成员都需要在上面工作，所以也需要与远程同步；
- `bug`分支只用于在本地修复bug，就没必要推到远程了，除非老板要看看你每周到底修复了几个`bug`；
- `feature`分支是否推到远程，取决于你是否和你的小伙伴合作在上面开发。

#### 4.5.2 抓取分支

多人协作时，大家都会往`master`和`dev`分支上推送各自的修改。

如果要在`dev`分支上开发，此时本地无dev分支，就必须创建远程`origin`的`dev`分支到本地，可以用这个命令创建本地`dev`分支：

` git checkout -b dev origin/dev`

> 此时创建的分支已经和远程的dev分支建立连接



**多人协作的工作模式**通常是这样：

1. 首先，可以试图用`git push origin <branch-name>`推送自己的修改；
2. 如果推送失败，则因为远程分支比你的本地更新，需要先用`git pull`试图合并；
3. 如果合并有冲突，则解决冲突，并在本地提交；
4. 没有冲突或者解决掉冲突后，再用`git push origin <branch-name>`推送就能成功！

如果`git pull`提示`no tracking information`，则说明本地分支和远程分支的链接关系没有创建，用命令`git branch --set-upstream-to <branch-name> origin/<branch-name>`。

## 5. 冲突解决

本地pull时和远程仓库的文件存在冲突，解决方式如下，

1. 隐藏本地所做的改动

   git stash

   > 可以通过`git stash list`查看刚才所做的隐藏，有对应的标记，如`stash@{0}`

2. 拉取远程仓库的内容至本地

   git pull

   > 此时因为已经隐藏了本地的改动，所以可以顺利拉取远程仓库的内容

3. 还原隐藏的内容

   git stash pop 标记

   > 还原后，本地的内容和远程内容中，不同的部分会合并

4. 手动解决冲突的部分

   找到冲突的文件后，在冲突的位置可以看到

   Updated upstream和Stashed changes的字样，Updated upstream 和=====之间的内容就是pull下来的内容，

   ====和stashed changes之间的内容就是本地修改的内容，手动修改完成即可。

## 6. 其它git命令

`git diff HEAD -- <filename>` 查看工作区和本地仓库的不同

`git status` 现实本地仓库的状态信息

`git log` 显示最基本的提交信息

## References

- [廖雪峰老师git教程](https://www.liaoxuefeng.com/wiki/896043488029600)

  > 简介且使用，总结核心的git使用方式

- [git recipes](https://github.com/geeeeeeeeek/git-recipes)

  > 详细且全面的git教程，几乎覆盖所有的git操作

