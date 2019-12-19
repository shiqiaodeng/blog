
### win10
### 单机配置
### 1. 配置安装
```
Java
Zookeeper
```
### 2.  运行zookeeper 服务的zkServer.cmd,不要关闭窗口
![fig 1](https://github.com/shiqiaodeng/blog/blob/master/kafka/images/1.png?raw=true "figure 1")

### 3. 编辑文件“server.properties”（在kafaka安装目录下\config 目录下）
**• 找到并编辑log.dirs=D:\Java\Tool\kafka_2.11-0.10.0.1\kafka-log,这里的目录自己修改成自己喜欢的
• 找到并编辑zookeeper.connect=localhost:2181。表示本地运行
• Kafka会按照默认，在9092端口上运行，并连接zookeeper的默认端口：2181**
### 4. 启动kafka 服务器（先进入kafka 安装目录）
执行：
```
.\bin\windows\kafka-server-start.bat .\config\server.properties
```
![fig 2](https://github.com/shiqiaodeng/blog/blob/master/kafka/images/2.png?raw=true "figure 2")   
不要关闭窗口

### 5. 创建主题
执行：
```
.\bin\windows\kafka-topics.bat --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic linlin
```
![fig 3](https://github.com/shiqiaodeng/blog/blob/master/kafka/images/3.png?raw=true "figure 3")

### 6. 创建生产者
执行：
```
.\bin\windows\kafka-console-producer.bat --broker-list localhost:9092 --topic linlin
```
![fig 4](https://github.com/shiqiaodeng/blog/blob/master/kafka/images/4.png?raw=true "figure 4")  

（此时等待输入数据）

### 7. 创建消费者
执行：
```
.\bin\windows\kafka-console-consumer.bat --zookeeper localhost:2181 --topic linlin
```
![fig 5](https://github.com/shiqiaodeng/blog/blob/master/kafka/images/5.png?raw=true "figure 5")  

**（注意：提示”Using the ConsoleConsumer with old consumer is deprecated and will be removed in a future major release. Consider using the new consumer by passing [bootstrap-server] instead of [zookeeper]“ 不用管，不建议将ConsoleConsumer与旧用户一起使用，并且在以后的主要版本中将删除该控制台。考虑通过传递[bootstrap-server]而不是[zookeeper]使用新使用者）**

### 8. 测试
在生产者命令行窗口发送数据，如果在消费者窗口接收到则表示成功。
![fig 6](https://github.com/shiqiaodeng/blog/blob/master/kafka/images/6.png?raw=true "figure 6")


### 参考资料
1. https://www.jianshu.com/p/d489ab1186a3

