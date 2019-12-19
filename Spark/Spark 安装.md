
### 系统环境：win 10(64位) 
### 1. 安装Scala  
安装成功后显示：  
![Scala](https://github.com/shiqiaodeng/blog/blob/master/Spark/images/1.png?raw=true "figure 1")  
### 2. 安装spark   
下载网址：https://archive.apache.org/dist/spark/spark-2.2.0/（我选择的是spark-2.2.0-bin-hadoop2.7.tgz）  
### 3. 安装hadoop  
下载网址：
a. https://archive.apache.org/dist/hadoop/common/ (**下载特慢，但是版本全**）   
b. https://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/（**清华镜像，下载快，但是版本不全，我选择的是在这里下载的是hadoop-2.7.7**）
（**注意： 下载hadoop后建议以管理员权限解压，网上资料是这么说，我一开始没有管理员权限解压，先是解压错误，后面执行spark-shell遇到奇奇怪怪的问题**）  
安装完成后执行hadoop,如下图则表示安装成功：  
![fig 2](https://github.com/shiqiaodeng/blog/blob/master/Spark/images/2.png?raw=true "figure 2")
如果出现如下情况：  
**JAVA_HOME 未设置，hadoop 执行错误；  
请检查java 路径是否有空格，如果有建议卸载重装，也可以通过替换通配符方式（网上教程建议，仅仅对于C盘安装在Promgram File 有效）**  

### 4. 运行spark
在命令行输入：
```
spark-shell
```
通常会遇到这个错误:   
![fig 3](https://github.com/shiqiaodeng/blog/blob/master/Spark/images/3.png?raw=true "figure 3")  
这是由于缺少**winutils.exe 文件**  
解决：  
**(1) 下载好winutils.exe后，将这个文件放入到Hadoop的bin目录下
https://github.com/steveloughran/winutils.git  
选择hadoop2.7.1 目录下的winutils.exe**  
(**注意：这里尽管我下载的是hadoop2.7.7，但是可以用hadoop2.7.1 目录下的winutils.exe，其他版本没试过**)  
**(2) 修改权限，在打开的cmd中输入:**  
```
D:\software\hadoop-2.7.7\bin\winutils.exe chmod 777 C:/tmp/hive  
```
![fig 4](https://github.com/shiqiaodeng/blog/blob/master/Spark/images/4.png?raw=true "figure 4")  
（**注意： 先确保C:/tmp/hive 存在，否则遇到错误**：  
![fig 4](https://github.com/shiqiaodeng/blog/blob/master/Spark/images/4.png?raw=true "figure 4")  
**解决：仔细检查路径是否错误， C:/tmp/hive 是否存在，不存在则创建，执行成功后是不会有输出的。**

### 5. 再次执行
在命令行输入：
```
spark-shell  
```
若成功如下图所示：  
![fig 6](https://github.com/shiqiaodeng/blog/blob/master/Spark/images/6.png?raw=true "figure 6")  

### 6. 检查  
在浏览器打开网址：http://210.43.57.131:4041  
![fig 7](https://github.com/shiqiaodeng/blog/blob/master/Spark/images/7.png?raw=true "figure 7")

### 参考：  
[1] https://blog.csdn.net/u011513853/article/details/52865076**
