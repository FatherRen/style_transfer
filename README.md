# style_transfer
## 1、介绍：
### 主目录包含六个文件：
      content_image:里面存放需要转变风格的原始图像。
      style_image:里面存放风格类型的图像。
      output:训练过程中得到的图像。
      config:模型的配置。
      main:主函数。
      还有一个是VGG19的模型：[百度云链接](https://pan.baidu.com/s/1BQSpGeWSomx4flJQvM9Rqg) 提取码：d6y6
      VGG19模型下载下来直接保存在主目录下即可。
## 2、使用：
      （1）cd到主目录下
      （2）python main.py --content_image "原始图像" --style_image "风格图像"
      （3）程序运行结束在output中查看即可
## 3、注意：
      （1）这不是GAN模型，每一次运行都是一次重复的训练，训练的对象是输入的噪音数据，
           我看来就是将两个图像给融合起来了。
      （2）如何想要比较好的效果，要保证两类图像在场景上是相同的。
