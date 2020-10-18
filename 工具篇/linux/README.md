## 1. 基础

- 熟悉一种基于文本的编辑器

  肯定是vim啦

- 学会使用`man`命令查阅

  使用`man+命令`查阅详细文档，使用`apropos+命令`搜寻，用 `type 命令` 来判断这个命令到底是可执行文件、shell 内置命令还是别名

- 重定向&管道

  `>` **覆盖**输出到文本，commandA > file

  `>>`追加输出到文本，commandA > file

  `<`将后面**文件**作为前面命令的输入，CommandA < file

  `<<`将后面手动收入的内容作为前面的输入，后面是一个中止符号，需要手动输入，直到中止符号

  CommandA << EOF 

  abc 

  def 

  EOF

  将EOF之间的输入的内容作为CommanA的输入

  `<<<`将后面字符串or变量作为前面的输入

  `|`管道符号，commandA | commandB  将命令A的输出作为命令B的输入

  `file1 >& file2` 将输出文件 file2 和 file1 合并

  `file1 <& file2` 将输入文件 file2 和 file1 合并

  `cat >>file <<END` 将手动输入的内容追加到file文件中

- 文件描述符

  linux中有三种标准输入输出，分别是STDIN，STDOUT，STDERR，对应的数字是0，1，2。

  commandA 1> file.out 2> file.err，将命令的标准输出放到文件file.out中，命令的错误放到文件file.err中

  `2>&1`指将标准输出、标准错误指定为同一输出路径

  `&`用于表示全部1和2的信息

  CommandA &> file 或 commanA > file 2>&1，表示命令的所有输出全部重定向到file中

## 2. 日常使用

- 搜索历史命令

  在 Bash 中，可以通过按 **Tab** 键实现自动补全参数，使用 **ctrl-r** 搜索命令行历史记录（按下按键之后，输入关键字便可以搜索，重复按下 **ctrl-r** 会向后查找匹配项，按下 **Enter** 键会执行当前匹配的命令，而按下右方向键会将匹配项放入当前行中，不会直接执行，以便做出修改）。

- **命令符号操作的快捷键**

  在 Bash 中，可以按下 **ctrl-w** 删除你键入的最后一个单词，**ctrl-u** 可以删除行内光标所在位置之前的内容，**alt-b** 和 **alt-f** 可以以单词为单位移动光标，**ctrl-a** 可以将光标移至行首，**ctrl-e** 可以将光标移至行尾，**ctrl-k** 可以删除光标至行尾的所有内容，**ctrl-l** 可以清屏。

  键入 `man readline` 可以查看 Bash 中的默认快捷键。内容有很多，例如 **alt-.** 循环地移向前一个参数，而 **alt-*** 可以展开通配符。

- 路径进入

  `cd` 命令可以切换工作路径，输入 `cd ~` 可以进入 home 目录。要访问你的 home 目录中的文件，可以使用前缀 `~`（例如 `~/.bashrc`）。在 `sh` 脚本里则用环境变量 `$HOME` 指代 home 目录的路径。

  回到前一个工作路径：`cd -`

- 命令行历史记录

  键入 `history` 查看命令行历史记录，再用 `!n`（`n` 是命令编号）就可以再次执行。其中有许多缩写，最有用的大概就是 `!$`， 它用于指代上次键入的参数，而 `!!` 可以指代上次键入的命令了（参考 man 页面中的“HISTORY EXPANSION”）。不过这些功能，你也可以通过快捷键 **ctrl-r** 和 **alt-.** 来实现。

- 命令快捷键

  使用 `alias` 来创建常用命令的快捷形式。例如：`alias ll='ls -latr'` 创建了一个新的命令别名 `ll`。

- 配置文件

  主要有`/.bashrc`和`~/.bash_profile`。

  - `/.bashrc`

    可以把别名、shell 选项和常用函数保存在 `~/.bashrc`，这样做的话你就可以在所有 shell 会话中使用你的设定。

  - `~/.bash_profile`

    把环境变量的设定以及登陆时要执行的命令保存在 `~/.bash_profile`。而对于从图形界面启动的 shell 和 `cron` 启动的 shell，则需要单独配置文件。

## 3. 文件及数据处理

find sort uniq cut paste join

## 4. 系统信息探查

free -m top

## References

[linux标准输入输出2>&1](https://www.cnblogs.com/jacob-tian/p/6110606.html)