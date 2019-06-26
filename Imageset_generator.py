import os
import tensorflow as tf
from PIL import Image
from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# 有些时候我们希望检查分类是否有误，
# 或者在之后的网络训练过程中可以监视，输出图片，来观察分类等操作的结果，
# 可以在session会话中，将tfrecord的图片从流中读取出来，再保存。
t0=time()
resize = 128
channelsize = 3
path_train='C:\\Users\DCY\MyData\\tahe\\tahe1\\train\\'  #文件地址
path_val='C:\\Users\DCY\MyData\\tahe\\tahe1\\test\\'
writer_train= tf.python_io.TFRecordWriter("tahe1_train.tfrecords") #生成的训练文件
# writer_val= tf.python_io.TFRecordWriter("tahe1.tfrecords")
print("准备中……")
for img_name in os.listdir(path_train):
    img_path=path_train+img_name #每一个图片的地址
    img=Image.open(img_path)
    img= img.resize((resize,resize))
    img=img.tobytes()#将图片转化为二进制格式
    example = tf.train.Example(features=tf.train.Features(feature={
        # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
    })) #example对象对label和image数据进行封装
    writer_train.write(example.SerializeToString())  #序列化为字符串
print("已输出训练集")
writer_train.close()

# for img_name in os.listdir(path_val):
#     img_path=path_val+img_name #每一个图片的地址
#     img=Image.open(img_path)
#     img= img.resize((resize,resize))
#     img_raw=img.tobytes()#将图片转化为二进制格式
#     example = tf.train.Example(features=tf.train.Features(feature={
#         # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#     })) #example对象对label和image数据进行封装
#     writer_val.write(example.SerializeToString())  #序列化为字符串
# print("已输出测试集")
# writer_val.close()

print("训练完成耗时：%0.3fs" %(time()-t0))

# filename_queue = tf.train.string_input_producer(["dog_train.tfrecords"]) #读入流中
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
# features = tf.parse_single_example(serialized_example,
#                                    features={
#                                        'label': tf.FixedLenFeature([], tf.int64),
#                                        'img_raw' : tf.FixedLenFeature([], tf.string),
#                                    })  #取出包含image和label的feature对象
# image = tf.decode_raw(features['img_raw'], tf.uint8)
# image = tf.reshape(image, [resize, resize, channelsize])
# label = tf.cast(features['label'], tf.int32)
#
# with tf.Session() as sess:
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     coord=tf.train.Coordinator()
#     threads= tf.train.start_queue_runners(coord=coord)
#     for i in range(40):
#         example, l = sess.run([image,label])              #在会话中取出image和label
#         img=Image.fromarray(example, 'RGB')               #这里Image是之前提到的
#         img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')    #存下图片
#         print(example, l)
#     coord.request_stop()
#     coord.join(threads)