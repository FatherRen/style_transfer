# -- encoding:utf-8 --
"""
Created by MengCheng Ren on 2019/7/23
"""
import config as cfg
import tensorflow as tf
import scipy.io
import scipy.misc
import numpy as np
import argparse


class Tensor(object):
    def __init__(self, args):
        self.content = self.loadimg(args.content_image)  # 加载内容图片
        # self.content = tf.placeholder(tf.float32, (1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3))
        self.style = self.loadimg(args.style_image)  # 加载风格图片
        # self.style = tf.placeholder(tf.float32, (1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3))
        self.random_img = self.get_random_img()  # 生成噪音内容图片
        self.net = self.model()  # 建立vgg网络

    def model(self):
        # 读取预训练的vgg模型
        vgg = scipy.io.loadmat(cfg.VGG_MODEL_PATH)
        vgg_layers = vgg['layers'][0]
        net = {}
        # 使用预训练的模型参数构建vgg网络的卷积层和池化层
        # 全连接层不需要
        # 注意，除了input之外，这里参数都为constant，即常量
        # 和平时不同，我们并不训练vgg的参数，它们保持不变
        # 需要进行训练的是input，它即是我们最终生成的图像
        net['input'] = tf.Variable(np.zeros([1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3]), dtype=tf.float32)
        # 参数对应的层数可以参考vgg模型图
        net['conv1_1'] = self.conv_relu(net['input'], self.get_wb(vgg_layers, 0))
        net['conv1_2'] = self.conv_relu(net['conv1_1'], self.get_wb(vgg_layers, 2))
        net['pool1'] = self.pool(net['conv1_2'])
        net['conv2_1'] = self.conv_relu(net['pool1'], self.get_wb(vgg_layers, 5))
        net['conv2_2'] = self.conv_relu(net['conv2_1'], self.get_wb(vgg_layers, 7))
        net['pool2'] = self.pool(net['conv2_2'])
        net['conv3_1'] = self.conv_relu(net['pool2'], self.get_wb(vgg_layers, 10))
        net['conv3_2'] = self.conv_relu(net['conv3_1'], self.get_wb(vgg_layers, 12))
        net['conv3_3'] = self.conv_relu(net['conv3_2'], self.get_wb(vgg_layers, 14))
        net['conv3_4'] = self.conv_relu(net['conv3_3'], self.get_wb(vgg_layers, 16))
        net['pool3'] = self.pool(net['conv3_4'])
        net['conv4_1'] = self.conv_relu(net['pool3'], self.get_wb(vgg_layers, 19))
        net['conv4_2'] = self.conv_relu(net['conv4_1'], self.get_wb(vgg_layers, 21))
        net['conv4_3'] = self.conv_relu(net['conv4_2'], self.get_wb(vgg_layers, 23))
        net['conv4_4'] = self.conv_relu(net['conv4_3'], self.get_wb(vgg_layers, 25))
        net['pool4'] = self.pool(net['conv4_4'])
        net['conv5_1'] = self.conv_relu(net['pool4'], self.get_wb(vgg_layers, 28))
        net['conv5_2'] = self.conv_relu(net['conv5_1'], self.get_wb(vgg_layers, 30))
        net['conv5_3'] = self.conv_relu(net['conv5_2'], self.get_wb(vgg_layers, 32))
        net['conv5_4'] = self.conv_relu(net['conv5_3'], self.get_wb(vgg_layers, 34))
        net['pool5'] = self.pool(net['conv5_4'])
        return net

    def conv_relu(self, input, wb):
        """
        进行先卷积、后relu的运算
        :param input: 输入层
        :param wb: wb[0],wb[1] == w,b
        :return: relu后的结果
        """
        conv = tf.nn.conv2d(input, wb[0], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv + wb[1])
        return relu

    def pool(self, input):
        """
        进行max_pool操作
        :param input: 输入层
        :return: 池化后的结果
        """
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_wb(self, layers, i):
        """
        从预训练好的vgg模型中读取参数
        :param layers: 训练好的vgg模型
        :param i: vgg指定层数
        :return: 该层的w,b
        """
        w = tf.constant(layers[i][0][0][0][0][0])
        bias = layers[i][0][0][0][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        return w, b

    def get_random_img(self):
        """
        根据噪音和内容图片，生成一张随机图片
        :return:
        """
        noise_image = np.random.uniform(-20, 20, [1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3])
        random_img = noise_image * cfg.NOISE + self.content * (1 - cfg.NOISE)
        return random_img

    def loadimg(self, path):
        """
        加载一张图片，将其转化为符合要求的格式
        :param path:
        :return:
        """
        # 读取图片
        image = scipy.misc.imread(path)
        # 重新设定图片大小
        image = scipy.misc.imresize(image, [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])
        # 改变数组形状，其实就是把它变成一个batch_size=1的batch
        image = np.reshape(image, (1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3))
        # 减去均值，使其数据分布接近0
        image = image - cfg.IMAGE_MEAN_VALUE
        return image


class Transfer(object):
    def __init__(self, args):
        self.model = Tensor(args)

        self.sess = tf.Session()

        # 全局初始化
        self.sess.run(tf.global_variables_initializer())

        # 定义损失函数
        cost = self.loss()
        # 创建优化器
        self.optimizer = tf.train.AdamOptimizer(1.0).minimize(cost)
        # 再初始化一次（主要针对于第一次初始化后又定义的运算，不然可能会报错）
        self.sess.run(tf.global_variables_initializer())
        # 使用噪声图片进行训练
        self.sess.run(tf.assign(self.model.net['input'], self.model.random_img))

    def train(self):
        for step in range(cfg.TRAIN_STEPS):
            # 进行一次反向传播
            self.sess.run(self.optimizer)
            # 每隔一定次数，输出一下进度，并保存当前训练结果
            if step % 50 == 0:
                print('step {} is down.'.format(step))
                # 取出input的内容，这是生成的图片
                img = self.sess.run(self.model.net['input'])
                # 训练过程是减去均值的，这里要加上
                img += cfg.IMAGE_MEAN_VALUE
                # 这里是一个batch_size=1的batch，所以img[0]才是图片内容
                img = img[0]
                # 将像素值限定在0-255，并转为整型
                img = np.clip(img, 0, 255).astype(np.uint8)
                # 保存图片
                scipy.misc.imsave('{}-{}.jpg'.format(cfg.OUTPUT_IMAGE, step), img)
        # 保存最终训练结果
        img = self.sess.run(self.model.net['input'])
        img += cfg.IMAGE_MEAN_VALUE
        img = img[0]
        img = np.clip(img, 0, 255).astype(np.uint8)
        scipy.misc.imsave('{}.jpg'.format(cfg.OUTPUT_IMAGE), img)

    def loss(self):
        """
        定义模型的损失函数
        :return: 内容损失和风格损失的加权和损失
        """
        # 先计算内容损失函数
        # 获取定义内容损失的vgg层名称列表及权重
        content_layers = cfg.CONTENT_LOSS_LAYERS
        # 将内容图片作为输入，方便后面提取内容图片在各层中的特征矩阵
        self.sess.run(tf.assign(self.model.net['input'], self.model.content))
        # 内容损失累加量
        content_loss = 0.0
        # 逐个取出衡量内容损失的vgg层名称及对应权重
        for layer_name, weight in content_layers:
            # 提取内容图片在layer_name层中的特征矩阵
            p = self.sess.run(self.model.net[layer_name])
            # 提取噪音图片在layer_name层中的特征矩阵
            x = self.model.net[layer_name]
            # 长x宽
            M = p.shape[1] * p.shape[2]
            # 信道数
            N = p.shape[3]
            # 根据公式计算损失，并进行累加
            content_loss += (1.0 / (2 * M * N)) * tf.reduce_sum(tf.pow(p - x, 2)) * weight
        # 将损失对层数取平均
        content_loss /= len(content_layers)

        # 再计算风格损失函数
        style_layers = cfg.STYLE_LOSS_LAYERS
        # 将风格图片作为输入，方便后面提取风格图片在各层中的特征矩阵
        self.sess.run(tf.assign(self.model.net['input'], self.model.style))
        # 风格损失累加量
        style_loss = 0.0
        # 逐个取出衡量风格损失的vgg层名称及对应权重
        for layer_name, weight in style_layers:
            # 提取风格图片在layer_name层中的特征矩阵
            a = self.sess.run(self.model.net[layer_name])
            # 提取噪音图片在layer_name层中的特征矩阵
            x = self.model.net[layer_name]
            # 长x宽
            M = a.shape[1] * a.shape[2]
            # 信道数
            N = a.shape[3]
            # 求风格图片特征的gram矩阵
            A = self.gram(a, M, N)
            # 求噪音图片特征的gram矩阵
            G = self.gram(x, M, N)
            # 根据公式计算损失，并进行累加
            style_loss += (1.0 / (4 * M * M * N * N)) * tf.reduce_sum(tf.pow(G - A, 2)) * weight
        # 将损失对层数取平均
        style_loss /= len(style_layers)
        # 将内容损失和风格损失加权求和，构成总损失函数
        loss_ = cfg.ALPHA * content_loss + cfg.BETA * style_loss
        return loss_

    def gram(self, x, size, deep):
        """
        创建给定矩阵的格莱姆矩阵，用来衡量风格
        :param x:给定矩阵
        :param size:矩阵的行数与列数的乘积
        :param deep:矩阵信道数
        :return:格莱姆矩阵
        """
        # 改变shape为（size,deep）
        x = tf.reshape(x, (size, deep))
        # 求xTx
        g = tf.matmul(tf.transpose(x), x)
        return g


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--content_image", type=str, default="./content_image/1.jpg")
    parse.add_argument("--style_image", type=str, default='./style_image/1.jpg')
    args = parse.parse_args()

    transfer = Transfer(args)
    transfer.train()
