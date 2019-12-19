### 1. 环境配置
参考前面的几篇博客：  
[Spark 安装.md](https://github.com/shiqiaodeng/blog/blob/master/Spark/Spark%20%E5%AE%89%E8%A3%85.md)    
[kafka安装.md](https://github.com/shiqiaodeng/blog/blob/master/kafka/kafka%E5%AE%89%E8%A3%85.md)  
[zookeeper安装.md](https://github.com/shiqiaodeng/blog/blob/master/zookeeper/zookeeper%E5%AE%89%E8%A3%85.md)  
**注意：
	• Check "camera.url" and "camera.id" properties in "stream-collector.properties" file.
	• Set "stream-collector.log" file path in "log4j.properties" file.
这两步根据自己需要修改。**
### 2. Make sure Zookeeper and Kafka servers are up and running
1. 启动Zookeeper服务器, 执行 zkServer.cmd
![fig 1](https://github.com/shiqiaodeng/blog/blob/master/video-stream-classification/images/1-1.png?raw=true "figure 1")  

2. 启动kafka 服务器，执行kafka-server-start.bat .\config\server.properties
![fig 2](https://github.com/shiqiaodeng/blog/blob/master/video-stream-classification/images/1-2.png?raw=true "figure 2")  

完成启动，不要关闭窗口

### 3. Create "video-stream-event" topic using below command.
修改readme 命令：
```
kafka-topics.sh --create --zookeeper localhost:2181 --topic video-stream-event --replication-factor 1 --partitions 3
```
**（注意，kafka-topics.sh 是针对linux系统，需要修改为kafka-topics.bat）**  
修改后的命令为:
```
kafka-topics.bat --create --zookeeper localhost:2181 --topic video-stream-event --replication-factor 1 --partitions 3
```
![fig 3](https://github.com/shiqiaodeng/blog/blob/master/video-stream-classification/images/1-3.png?raw=true "figure 3")

### 4. Run "mvn clean" command to install opencv-320.jar in local maven repository
```
mvn clean
```
![fig 4](https://github.com/shiqiaodeng/blog/blob/master/video-stream-classification/images/1-4.png?raw=true "figure 4")

### 5. Execute below command to start the "Video Stream Collector" application：
```
mvn clean package exec:java -Dexec.mainClass="com.iot.video.app.kafka.collector.VideoStreamCollector" -Dexec.cleanupDaemonThreads=false
```
![fig 5](https://github.com/shiqiaodeng/blog/blob/master/video-stream-classification/images/1-5.png?raw=true "figure 5")   
执行成功如下图：  
![fig 6](https://github.com/shiqiaodeng/blog/blob/master/video-stream-classification/images/1-6.png?raw=true "figure 6")  

（注意需要4-5分钟）
