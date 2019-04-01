# Git Study

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