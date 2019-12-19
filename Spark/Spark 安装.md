1. 首先运行Zookeepr、Kafka servers
 打开cmd，启动Zookeeper服务器, 执行 zkServer.cmd

 打开新的cmd，启动kafka 服务器，进入kafka安装目录，执行kafka-server-start.bat .\config\server.properties

2. 运行Video Stream Collector 程序
进入Video Stream Collector安装目录，执行：
mvn clean package exec:java -Dexec.mainClass="com.iot.video.app.kafka.collector.VideoStreamCollector" -Dexec.cleanupDaemonThreads=false


表示运行成功。
3. 运行Video Stream Processor 程序
mvn clean package exec:java -Dexec.mainClass="com.iot.video.app.spark.processor.VideoStreamProcessor"


表示保存成功。
（注意：消费者程序只有在生产者生产数据才会接收到数据）

5. 查看结果输出，执行成功后会在数据存储在你设置的保存目录（我的是默认：/tmp/processed-data）


4. 可能存在的问题
1. 消费者（processor）无法接收到生产者（collector）数据;
检查kafka 是否配置正常，通过测试demo检查；
检查hadoop 是否配置成功，在命令行输入hadoop,如果出现如下情况：
JAVA_HOME 未设置，hadoop 执行错误；
请检查java 路径是否有空格，如果有建议卸载重装，也可以通过替换通配符方式（网上教程建议，仅仅对于C盘安装在Promgram File 有效）

2. 存储数据出错，如下图


解决：检查/tmp/processed-data目录是否创建，没有则创建。一般/tmp 自动创建，processed-data需要自己创建。
