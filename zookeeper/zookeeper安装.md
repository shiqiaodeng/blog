### 系统环境：Win 10 
### 单机部署 
### 1. 进入安装目录下conf目录  
![fig 1](https://github.com/shiqiaodeng/blog/blob/master/zookeeper/images/1.png?raw=true "figure 1")

### 2. 配置zoo_Sample.conf 文件，首先重命名zoo.conf,   
代码如下：  

```
# 集群Leader和Follower服务器之间、服务器和客户端之间通信的心跳单元毫秒数
tickTime=2000
# 集群Follower服务器同步Leader服务器时的最大尝试次数
initLimit=10
# 集群Leader服务器检测其他服务器是否存活的最大尝试次数
syncLimit=5
# 服务器对客户端开放的服务端口
clientPort=2181
# 临时文件（快照文件）的存放位置
dataDir=D:/software/zookeeper-3.4.8/data
# 日志的存放位置
dataLogDir=D:/software/zookeeper-3.4.8/log
```  

### 3. 启动服务器  
在bin 目录下执行
```
zkServer.cmd(linux 执行zkServer.sh)
```
![fig 2](https://github.com/shiqiaodeng/blog/blob/master/zookeeper/images/2.png?raw=true "figure 2")   
（注意打开命令行执行，不要使用终端，不知道为什么我用git bash 可以连上，但无法操作；
建议将zookeeper/bin 目录添加到环境变量中，这样打开命令行就能执行）

启动后界面如下图所示：
![fig 3](https://github.com/shiqiaodeng/blog/blob/master/zookeeper/images/3.png?raw=true "figure 3")

### 4. 启动客户端  
打开新的命令行窗口，执行：
```  
zkCli.cmd -server localhost:2181
```
链接成功后如下图所示：  
![fig 4](https://github.com/shiqiaodeng/blog/blob/master/zookeeper/images/4.png?raw=true "figure 4")

### 5. 测试命令  
在客户端窗口输入：  
```
help  
```
![fig 5](https://github.com/shiqiaodeng/blog/blob/master/zookeeper/images/5.png?raw=true "figure 5")  
退出：  
```  
quit  
```
![fig 6](https://github.com/shiqiaodeng/blog/blob/master/zookeeper/images/6.png?raw=true "figure 6")
### 6. Zookeeper 命令行操作节点    
https://favoorr.github.io/2017/02/09/zookeeper-node-ops/
