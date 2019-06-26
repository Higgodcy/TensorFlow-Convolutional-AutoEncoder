import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
# import cv2
from time import time
from skimage import io
import math
from Tahe_pro.MyTools import *
# import Tahe_pro.MyTools.layers as lyr
import Tahe_pro.MyTools.model as mdl

from sklearn.decomposition import PCA
# ###############################自定义区域########################################
fs = 64                         # 卷积核数量
fs1 = 64
fs2 = 64
ks = 3
learning_rate = 0.0005          # 学习率
image_size = 128                # 输入图片大小
batch_size = 15                 # 批量大小
min_after_dequeue = 1200        # 数据输入后总输入中存留的输入数据大小
capacity = min_after_dequeue + 3 * batch_size       # 总数据大小
passes = 100000                 # 训练次数
channel = 3                     # 通道数量
butler = 1                      # 训练小管家： butler等于1，进行训练；非1，进行重建
path_t = 'C:\\Users\DCY\MyData\\tahe\\tahe2\\test\\41_'     # 设定重建图的路径
save_path = 'C:\\Users\DCY\Fig4\\50000_15_1200_0.0005\\'    # 输出与保存的路径

""""""""
# tensorboard --logdir=C:\Users\DCY\MyCode\Tahe_pro\saver\CEA_logs
# ###############################################################################
# 读取TFRecords文件


def read_and_decode(filename):
    """

    :param filename:
    :return:
    创建文件队列,不限读取的数量
    """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'img_raw': tf.FixedLenFeature([], tf.string)})

    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [image_size, image_size, channel])
    img = tf.cast(img, tf.float32)/ 255
    return img


class ConvolutionalAutoencoder(object):
    # 卷积模型
    def __init__(self):
        """
        build the graph
        """
        # place holder of input data
        x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, channel])  # [#batch, img_height, img_width, #channels]

        # encode
        conv1 = Convolution2D([ks, ks, channel, fs], activation=tf.nn.relu, scope='conv_1')(x)
        pool1 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool_1')(conv1)
        conv2 = Convolution2D([ks,ks, fs, fs], activation=tf.nn.relu, scope='conv_2')(pool1)
        pool2 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool_2')(conv2)
        conv3 = Convolution2D([ks, ks, fs, fs], activation=tf.nn.relu, scope='conv_3')(pool2)
        pool3 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool_3')(conv3)
        conv4 = Convolution2D([ks, ks, fs, fs2], activation=tf.nn.relu, scope='conv_4')(pool3)
        pool4 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool_4')(conv4)
        unfold = Unfold(scope='unfold')(pool4)
        encoded = FullyConnected(1024, activation=tf.nn.relu, scope='encode')(unfold)
        # decode
        decoded = FullyConnected(8 * 8 * fs2, activation=tf.nn.relu, scope='decode')(encoded)
        fold = Fold([-1, 8, 8, fs2], scope='fold')(decoded)
        unpool1 = UnPooling((2, 2), output_shape=tf.shape(conv4), scope='unpool_1')(fold)
        deconv1 = DeConvolution2D([ks, ks, fs2, fs], output_shape=tf.shape(pool3), activation=tf.nn.relu, scope='deconv_1')(unpool1)
        unpool2 = UnPooling((2, 2), output_shape=tf.shape(conv3), scope='unpool_2')(deconv1)
        deconv2 = DeConvolution2D([ks, ks, fs, fs], output_shape=tf.shape(pool2), activation=tf.nn.relu, scope='deconv_2')(unpool2)
        unpool3 = UnPooling((2, 2), output_shape=tf.shape(conv2), scope='unpool_3')(deconv2)
        deconv3 = DeConvolution2D([ks, ks, fs, fs], output_shape=tf.shape(pool1), activation=tf.nn.relu, scope='deconv_3')(unpool3)
        unpool4 = UnPooling((2, 2), output_shape=tf.shape(conv1), scope='unpool_4')(deconv3)
        reconstruction = DeConvolution2D([ks, ks, 3, fs], output_shape=tf.shape(x), activation=tf.nn.sigmoid, scope='reconstruction')(unpool4)
        # 损失函数
        loss = tf.nn.l2_loss(x - reconstruction)  # L2 loss
        # cost = tf.reduce_mean(loss)

        # training = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        training = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        self.x = x
        self.reconstruction = reconstruction
        self.loss = loss
        self.training = training
        self.conv1 = conv1
        self.pool1 = pool1
        self.conv2 = conv2
        self.pool2 = pool2
        self.conv3 = conv3
        self.pool3 = pool3
        self.conv4 = conv4
        self.pool4 = pool4
        self.unfold = unfold
        self.encoded = encoded
        self.decoded = decoded
        self.fold = fold
        self.unpool1 = unpool1
        self.deconv1 = deconv1
        self.unpool2 = unpool2
        self.deconv2 = deconv2
        self.unpool3 = unpool3
        self.deconv3 = deconv3
        self.unpool4 = unpool4

    # 训练模块
    def train(self, new_training=False):
        """

        :param new_training:
        :return:
        训练模块
        """

        t0 = time()     # 记录时间
        img = read_and_decode("tahe2_train.tfrecords")  # 读取数据
        # img = read_and_decode("train_set1.tfrecords")

        # 预取图像和label并随机打乱，组成batch，此时tensor rank发生了变化，多了一个batch大小的维度
        img_batch = tf.train.shuffle_batch([img], batch_size=batch_size, capacity=capacity,
                                           min_after_dequeue=min_after_dequeue)

        merged = tf.summary.merge_all()     # 合并tfboard图

        init = tf.global_variables_initializer()    # 初始化函数
        loss_vec = []   # 创建损失函数数组
        with tf.Session() as sess:
            sess.run(init)  # 会话进行初始化
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 打开tfrecord线程
            summary_writer = tf.summary.FileWriter('saver/CEA_logs', sess.graph)    # 写入tfboard
            if new_training:
                saver, global_step = Model.start_new_session(sess)
            else:
                saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')

            # start training
            for step in range(1+global_step, 1+passes+global_step):
                x = sess.run(img_batch)     # 读入数据
                self.training.run(feed_dict={self.x: x})    # 开始训练

                if step % 50 == 0:
                    # print("pass {}, training loss {}".format(step, loss))
                    loss = self.loss.eval(feed_dict={self.x: x})
                    print('step:{}, losses:{}'.format(step, loss))
                    loss_vec.append(loss)
                    tf.summary.scalar('loss', loss)
                    tf.summary.image('input_batch', x, max_outputs=5)
                    summary_str = sess.run(merged)
                    summary_writer.add_summary(summary_str, step)

                if step % 1000 == 0:  # save weights
                    saver.save(sess, 'saver/cnn', global_step=step)
                    print('==  当前进程已保存  ==')

            coord.request_stop()
            coord.join(threads)
            self.loss_vec = loss_vec
            plt.plot(loss_vec, 'k-')
            plt.title('Loss')
            plt.xlabel('Generation')
            plt.ylabel('loss')
            print("训练完成耗时：%0.3fmins" %((time() - t0)/60))
            plt.show()

    def reconstruct(self):
        """
        slot = 13
        :return:
        """

        def weights_to_grid(weights, rows, cols):
            # 网格化
            """convert the weights tensor into a grid for visualization"""
            height, width, in_channel, out_channel = weights.shape
            padded = np.pad(weights, [(1, 1), (1, 1), (0, 0), (0, rows * cols - out_channel)],
                            mode='constant', constant_values=0)
            transposed = padded.transpose((3, 1, 0, 2))
            reshaped = transposed.reshape((rows, -1))
            grid_rows = [row.reshape((-1, height + 2, in_channel)).transpose((1, 0, 2)) for row in reshaped]
            grid = np.concatenate(grid_rows, axis=0)

            return grid.squeeze()

        def plot_conv_layer(layer, title, image, num, gray):
            # 获取并输出特征图
            # os.makedirs(save_path + '%s\\' % (str(num)))
            feed_dict = {self.x: [image]}
            values = sess.run(layer, feed_dict=feed_dict)
            num_filters = values.shape[3]
            num_grids = math.ceil(math.sqrt(num_filters))  #
            fig, axes = plt.subplots(num_grids, num_grids)
            for i, ax in enumerate(axes.flat):
                if i < num_filters:
                    img = values[0, :, :, i]
                    if gray:
                        ax.imshow(img, interpolation='nearest', cmap='binary')
                    else:
                        ax.imshow(img, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
            plt.suptitle(title)
            print('正在输出%s_%s' % (str(num), title))
            # plt.show()
            # plt.savefig('C:\\Users\DCY\Fig5\\40000_10_1000_0.0005\\29\\%s' % (title))
            plt.savefig(save_path + '%s\\%s_%s' % (str(num), str(num), title))
            for j in range(64):
                plot_certain_layer(layer=layer, title=title, image=image, num1=num, num2=j, gray=False)

        def plot_certain_layer(layer, title, image, num1, num2, gray):
            # 获取指定的特征图
            sub_path = (save_path + '%s\\%s' % (str(num1), title))
            isExists = os.path.exists(sub_path)
            if not isExists:
                os.makedirs(sub_path)
            feed_dict = {self.x: [image]}
            values = sess.run(layer, feed_dict=feed_dict)
            num_filters = values.shape[3]
            fig, axes = plt.subplots(1, 1)
            if num2 < num_filters:
                img = values[0, :, :, num2]
                if gray:
                    axes.imshow(img, interpolation='nearest', cmap='binary')
                else:
                    axes.imshow(img, interpolation='nearest')
                plt.title(title + str(num1))
                # plt.show()
            # plt.savefig(save_path + '%s\\%s\\%s_%s' % (str(num1), title,str(num2),title))

        def plot_together(layer, title, num2):
            sub_path = (save_path + 'fiters\\%s' % (title))
            isExists = os.path.exists(sub_path)
            if not isExists:
                os.makedirs(sub_path)
            for i in range(72):
                img_path = path_t + str(i) + '.png'
                img = io.imread(img_path)
                feed_dict = {self.x: [img]}
                values = sess.run(layer, feed_dict=feed_dict)
                together = []
                num_filters = values.shape[3]
                fig, axes = plt.subplots(1, 1)
                img = values[0, :, :, num2]
                together.append(img)
                size = img.size
            together = tf.reshape(together, (size, size, 64, 64))
            t_img = weights_to_grid(together.transpose((1, 2, 3, 0)), 1, 1)
            axes.imshow(t_img, interpolation='nearest')
            plt.title(title)
            plt.savefig(save_path + 'fiters\\%s\\%s_%s' % (title, title, str(num2)))

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')

            first_layer_weights = tf.get_default_graph().get_tensor_by_name("conv_1/kernel:0").eval()
            grid_image = weights_to_grid(first_layer_weights, 8, 8)
            for i in range (1):#输出特征图（i为循环次数，若需要指定某层特征，输入“1”，i=该层序号。全部输出i=72）
                i=4
                img_path = path_t + str(i) + '.png'
                img = io.imread(img_path)
                os.makedirs(save_path+'%s\\'%(str(i)))
                org, recon = sess.run((self.x, self.reconstruction), feed_dict={self.x: img.reshape((-1, 128, 128,3))})
                # input_images = weights_to_grid(org.transpose((1, 2, 3, 0)), 1, 1)
                recon_images = weights_to_grid(recon.transpose((1, 2, 3, 0)), 1, 1)
                fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 6))
                ax0.imshow(img, interpolation='nearest')
                ax0.set_title('input image')
                ax1.imshow(recon_images, interpolation='nearest')
                ax1.set_title('reconstructed image')
                plt.savefig(save_path+'%s\\%s_reconstruct' % (str(i),str(i)))
                print('已输出重建图%s'%(str(i)))

                plot_conv_layer(layer=self.conv1, title='conv1', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.pool1, title='pool1', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.conv2, title='conv2', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.pool2, title='pool2', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.conv3, title='conv3', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.pool3, title='pool3', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.conv4, title='conv4', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.pool4, title='pool4', image=img, num=i, gray=False)
                # plot_conv_layer(layer=self.unfold, title='unfold', image=img, gray=False)
                # plot_conv_layer(layer=self.encoded, title='encoded', image=img, gray=False)
                # plot_conv_layer(layer=self.decoded, title='decoded', image=img, gray=False)
                plot_conv_layer(layer=self.fold, title='fold', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.unpool1, title='unpool1', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.deconv1, title='deconv1', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.unpool2, title='unpool2', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.deconv2, title='deconv2', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.unpool3, title='unpool3', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.deconv3, title='deconv3', image=img, num=i, gray=False)
                plot_conv_layer(layer=self.unpool4, title='unpool4', image=img, num=i, gray=False)

            print('完成输出')
            # plt.show()





def main():
    conv_autoencoder = ConvolutionalAutoencoder()
    if butler == 1:
        conv_autoencoder.train(new_training=True)
    if butler == 11:
        conv_autoencoder.train(new_training=True)
        conv_autoencoder.reconstruct()
    else:
        conv_autoencoder.reconstruct()



if __name__ == '__main__':
    main()
