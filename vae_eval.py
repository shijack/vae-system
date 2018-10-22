import tensorflow as tf
import numpy as np
import scipy
import cv2
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

LATENT_DIM = 256
ROWS, COLS = 10, 10
HEIGHT, WIDTH, DEPTH = 144, 112, 3
N, M = 10, 30

splines = []
x = range(N)
xs = np.linspace(0, N, N * M)

for i in range(ROWS * COLS * LATENT_DIM):
    y = np.random.normal(0.0, 1.0, size=[N]).astype(np.float32)
    s = scipy.interpolate.UnivariateSpline(x, y, s=2)
    ys = s(xs)
    splines.append(ys)

splines = np.array(splines)


def read_record_new(images_path='./data', depth=1):
    train_total_data = []
    for img_path in os.listdir(images_path):

        img = cv2.imread(os.path.join(images_path, img_path))

        if depth == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (112, 144)) / 255.0
        train_total_data.append(img)
    train_total_data = np.array(train_total_data)
    if depth == 1:
        train_total_data = np.expand_dims(train_total_data, axis=3)
    return train_total_data


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model_new/gan-20000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model_new'))
    graph = tf.get_default_graph()
    # latent_input = graph.get_tensor_by_name('latent_input:0')
    x_input = graph.get_tensor_by_name('encoder/input_img:0')
    # laten_mean = graph.get_tensor_by_name('encoder/mean:0')
    # laten_stddev = graph.get_tensor_by_name('encoder/stddev:0')
    image_eval = graph.get_tensor_by_name('decoder/reconstruct/conv5/act:0')
    latent_feature = graph.get_tensor_by_name('variance/latent_feature:0')

    imgs = read_record_new(depth=3)
    # for i in range(N * M):
    for i in range(N * M / 2):
        time_point = splines[..., i]
        time_point = np.reshape(time_point, [ROWS * COLS, LATENT_DIM])
        # data = sess.run(image_eval, feed_dict={latent_input: time_point})
        data = imgs[:10, ...]
        # data_mean,data_stdddev = sess.run([laten_mean,laten_stddev], feed_dict={x_input: data})
        # data = data_mean+ data_stdddev
        data = sess.run(latent_feature, feed_dict={x_input: data})
        np.savetxt('./eval/feature_1_%d' % i, data)
        # data = np.reshape((data * 255).astype(int), (ROWS, COLS, HEIGHT, WIDTH, DEPTH))
        # data = np.concatenate(np.concatenate(data, 1), 1)
        # cv2.imwrite('./eval/eval_img_' + str(i) + '.png', data)
        # cv2.imshow('eval_img', data)
        # cv2.moveWindow('eval_img', 0, 0)
        # key = cv2.waitKey(0)
        # if key == 27:
        #     break


def test_image(path_image, num_class):
    img_string = tf.read_file(path_image)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [224, 224])
    img_resized = tf.reshape(img_resized, shape=[1, 224, 224, 3])
    # model = Vgg19(bgr_image=img_resized, num_class=num_class, vgg19_npy_path='./vgg19.npy')
    # score = model.fc8
    # prediction = tf.argmax(score, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./tmp/checkpoints/model_epoch50.ckpt")
        cv2.imwrite('img.png', img_decoded.eval())
        # plt.imshow(img_decoded.eval())
        # plt.title("Class:" + class_name[sess.run(prediction)[0]])
        # plt.show()


        # test_image('./validate/11.jpg', 2)
