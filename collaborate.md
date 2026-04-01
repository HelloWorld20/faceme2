# 如何在服务器上开发

# 保证可以访问学校的服务器

如果使用校园网络，可以直接访问。否则需要连接VPN。

# 登录服务器

找个顺手的远程连接工具（MAC 推荐electerm）

创建远程连接配置（密码登录）

IP地址：144.214.210.13
端口：22
密码：jianghwei2

# 访问自己的会话

进入服务器后，执行以下命令，进入自己的常驻会话

tmux attach -t (session-name)

**重要：务必进入常驻会话后再进行操作。如果不进入常驻会话，关掉shell，或者网络中断，进程会断掉，训练到一半也会丢失**

### 目前已存在的会话

    jianghwei2@DS210013:~$ tmux ls

    0: 1 windows (created Sat Mar 14 13:06:52 2026)
    chenxi: 1 windows (created Sat Mar 21 21:01:04 2026)
    jianghong: 1 windows (created Sat Mar 21 21:00:47 2026)
    luyang: 1 windows (created Sat Mar 21 20:46:50 2026)
    mingxuan: 1 windows (created Sat Mar 21 20:59:34 2026)
    xuanyi: 1 windows (created Sat Mar 21 21:00:20 2026)
    yuanye: 1 windows (created Sat Mar 21 21:01:23 2026)

# 其他

激活conda环境

    conda activate weienv

也可以创建自己的conda环境。

代码路径: `~/workspace/FaceMe`

# 配置ssh（配置一次即可）

打开 ~/.ssh/config 文件

添加以下配置

    Host cityu
        HostName 144.214.210.13
        User jianghwei2
        Port 22
        LocalForward 8080 localhost:8080

# 连接ssh

本地命令行里，执行以下命令

    ssh -N cityu

输入密码：`jianghwei2`，回车

（不会有任何输出）

# 访问编辑器

打开浏览器，访问 http://localhost:8080

即可实时编辑服务器上的代码。

**注意：这是同一份代码，尽量避免操作同一份文件，否则会丢失，冲突。尽量创建新的文件再操作（最好以自己名字命名）**

# 在wandb中查看训练进度

打开浏览器，访问 https://wandb.ai/jianghwei2-c-none/faceme

账号：jianghwei2-c@my.cityu.edu.hk
密码：Qq123456789
