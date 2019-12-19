### 1. 环境配置
参考前面的几篇博客：  
[Spark 安装.md](https://github.com/shiqiaodeng/blog/blob/master/Spark/Spark%20%E5%AE%89%E8%A3%85.md)  
[kafka安装.md](https://github.com/shiqiaodeng/blog/blob/master/kafka/kafka%E5%AE%89%E8%A3%85.md)
[zookeeper安装.md](https://github.com/shiqiaodeng/blog/blob/master/zookeeper/zookeeper%E5%AE%89%E8%A3%85.md)
**注意：  
Check "processed.output.dir" property in "stream-processor.properties" file. Create and set directory path for this property.  
Set "stream-processor.log" file path in "log4j.properties" file.  
这两步根据自己需要修改。**
### 2. Make sure Zookeepr and Kafka servers are up and running.
1. 启动Zookeeper服务器
执行 
```
zkServer.cmd
```
![fig 1](https://github.com/shiqiaodeng/blog/blob/master/video-stream-classification/images/2-1.png?raw=true "figure 1")    

2. 启动kafka 服务器
进入kafka安装目录，执行
```
kafka-server-start.bat .\config\server.properties
```
![fig 2](https://github.com/shiqiaodeng/blog/blob/master/video-stream-classification/images/2-2.png?raw=true "figure 2")    
完成启动

### 3. Run "mvn clean" command to install opencv-320.jar in local maven repository
```
mvn clean 
```

### 4. Execute below command to start the Video Stream Processor" application
```
mvn clean package exec:java -Dexec.mainClass="com.iot.video.app.spark.processor.VideoStreamProcessor"
```

