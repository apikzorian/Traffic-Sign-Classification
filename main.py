import tensorflow as tf
import pickle
import numpy as np
import cv2
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from scipy import misc


EPOCHS = 15
BATCH_SIZE = 256


## Normalize image
def pre_process_image(p_image):
    p_image = (p_image- 128.0)/128.0
    #p_image = shift_horiz_vert(p_image, 200)
    return p_image


def preprocess(data):
    """Convert to grayscale, histogram equalize, and expand dims"""
    imgs = np.ndarray((data.shape[0], 32, 32, 1), dtype=np.uint8)
    for i, img in enumerate(data):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = np.expand_dims(img, axis=2)
        imgs[i] = img
    return imgs

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.
# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry
def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    plt.imshow(image_input)
    plt.show()
    resized_image = cv2.resize(image_input, (32, 32))
    norm_img = (resized_image- 128.0)/128.0
    #norm_img = pre_process_image(resized_image)
    plt.imshow(norm_img)
    plt.show()
    reshaped_img = np.reshape(norm_img, (1,32,32,3))
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function

    print("Output featuremap")

    activation = tf_activation.eval(session=sess,feed_dict={x : reshaped_img})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        fig, axs = plt.subplots(1,16) # sets the number of feature maps to show on each row and column
        axs = axs.ravel()

        plt.title('Node #' + str(featuremap + 1)) # displays the feature map number
        # if activation_min != -1 & activation_max != -1:
        #     plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        # elif activation_max != -1:
        #     plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        # elif activation_min !=-1:
        #     plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        # else:
        axs[featuremap].axis('off')
        axs[featuremap].imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
            #plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")



  #
  # fig, axs = plt.subplots(4, 5, figsize=(20, 8))
  #   fig.subplots_adjust(hspace=.2, wspace=.001)
  #   axs = axs.ravel()
  #   for i in range(20):
  #       img = X_data[i]
  #       axs[i].axis('off')
  #       axs[i].imshow(img)
  #   print("esh")




def LeNet(x, keep_prob):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    activation_list = []

    #  Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    #  Activation.
    conv1_act = tf.nn.relu(conv1)
    activation_list.append(conv1_act)

    #  Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1_pool = tf.nn.max_pool(conv1_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #  Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1_pool, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    #  Activation.
    conv2_act = tf.nn.relu(conv2)
    activation_list.append(conv2_act)

    #  Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2_max = tf.nn.max_pool(conv2_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #  Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2_max)
    #  Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    #  Activation.
    fc1_act = tf.nn.relu(fc1)
    #activation_list.append(fc1_act)

    fc1_drop = tf.nn.dropout(fc1_act, keep_prob)
    #  Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1_drop, fc2_W) + fc2_b
    #  Activation.
    fc2_act = tf.nn.relu(fc2)
    #activation_list.append(fc2_act)
    fc2_drop = tf.nn.dropout(fc2_act, keep_prob)
    #  Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits, activation_list



def shift_horiz_vert(shift_img, shift_range):
    # Shift image horizontally/vertically, while adjusting steering angle by +/- 0.002
    shift_x = (shift_range * np.random.uniform() - shift_range/2)/8
    shift_y = (40 * np.random.uniform() - 40 / 2)/2
    trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    image_tr = cv2.warpAffine(shift_img, trans, (32, 32))
    return image_tr
def data_augmentation(X_data, y_data):
    shift_pics = []
    shift_labels = []
    for i, x_img in enumerate(X_data):
        x_shift = shift_horiz_vert(x_img, 200)
        shift_pics.append(x_shift)
        shift_labels.append(y_data[i])
    shift_pics_arr = np.asarray(shift_pics)
    shift_labes_arr = np.asarray(shift_labels)
    X_re = np.concatenate((X_data, shift_pics_arr), axis=0)
    y_re = np.concatenate((y_data, shift_labes_arr), axis=0)
    return X_re, y_re
def pre_process_image(p_image):
    p_image = (p_image- 128.0)/128.0
    p_image = shift_horiz_vert(p_image, 200)
    return p_image
# Reload the data
#pickle_file = 'T1P2.p'
pickle_file = 'T1P2_justnormal.p'
#pickle_file = 'T1P2_10_aug.p'
#pickle_file = 'T1P2_10_transform_noshear.p'
#pickle_file = 'T1P2_10_transform.p'
#pickle_file = 'T1P2_augmented_normalized_only.p'
print("Opened pickle: ", pickle_file)
#pickle_file = 'T1P2_unaugmented.p'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  X_train = pickle_data['X_train']
  y_train = pickle_data['y_train']
  X_test = pickle_data['X_test']
  y_test = pickle_data['y_test']
  X_valid = pickle_data['X_valid']
  y_valid = pickle_data['y_valid']
  del pickle_data  # Free up memory
pickle_file2 = 'GermanSigns.p'
with open (pickle_file2, 'rb') as f2:
    pickle_data = pickle.load(f2)
    g_features = pickle_data['g_train']
    g_labels = pickle_data['g_labels']
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)
rate = 0.002
logits, a_list = LeNet(x,keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
# Prediction accuracy functions (to be used with validation/testing data)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
def visualize_data(X_data):
    fig, axs = plt.subplots(4, 5, figsize=(20, 8))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axs = axs.ravel()
    for i in range(20):
        img = X_data[i]
        axs[i].axis('off')
        axs[i].imshow(img)
    print("esh")
def evaluate(X_data, y_data, keep):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range (0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        #visualize_data(batch_x)
        #batch_x = np.array([pre_process_image(batch_x[i]) for i in range(len(batch_x))], dtype=np.float32)
        #visualize_data(batch_x)
        accuracy = sess.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y, keep_prob:keep})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy/num_examples
Config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print(num_examples)
    log_batch_step = 500
    train_list = []
    valid_list = []
    loss_list = []
    batches = []
    log_accuracy = True
    print("Training...")
    print()
    print("Batch Size = ", BATCH_SIZE)
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for batch_i, offset in enumerate(range(0, num_examples, BATCH_SIZE)):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            #visualize_data(batch_x)
            #batch_x = np.array([pre_process_image(batch_x[i]) for i in range(len(batch_x))], dtype=np.float32)
            #visualize_data(batch_x)
            _, l = sess.run([training_operation, loss_operation], feed_dict={x:batch_x, y:batch_y, keep_prob:0.7})
            #print("loss = ", l)
            if log_accuracy:
                if not batch_i % log_batch_step:
                    loss_list.append(l)

                    training_accuracy = evaluate(X_train, y_train, 0.5)
                    validation_accuracy = evaluate(X_valid, y_valid, 1)
                    train_list.append(training_accuracy)
                    valid_list.append(validation_accuracy)
                    previous_batch = batches[-1] if batches else 0
                    batches.append(log_batch_step + previous_batch)
        print("EPOCH {} ...".format(i + 1))
        if log_accuracy:
            print("Training Accuracy = {:.3f}".format(training_accuracy))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

        # image = misc.imread("./images/german_sign_new_2.jpg")
        #
        # for layer in a_list:
        #      outputFeatureMap(image, layer)


    if log_accuracy:
        loss_plot = plt.subplot(211)
        loss_plot.set_title('Loss')
        loss_plot.plot(batches, loss_list, 'g')
        loss_plot.set_xlim([batches[0], batches[-1]])

        acc_plot = plt.subplot(212)
        acc_plot.set_title('Accuracy')
        acc_plot.plot(batches, train_list, 'r', label='Training Accuracy')
        acc_plot.plot(batches, valid_list, 'b', label='Validation Accuracy')
        acc_plot.set_ylim([0, 1.0])
        acc_plot.set_xlim([batches[0], batches[-1]])
        acc_plot.legend(loc=4)
        plt.tight_layout()
        plt.show()
        saver.save(sess, './T1P2')
        print("Model saved")
with tf.Session() as sess:
    print("Testing")
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test, 1)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
print("BATCH SIZE = ", BATCH_SIZE)
print("Training rate = ", rate)
print("Labels = ")
print(g_labels)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    g_accuracy = sess.run(logits, feed_dict={x: g_features, y: g_labels, keep_prob: 1})
    print(g_accuracy)
    cross_e = sess.run(cross_entropy, feed_dict={x: g_features, y: g_labels, keep_prob: 1})
    print("Cross entropy: ")
    print(cross_e)
    g_prediction = sess.run(correct_prediction, feed_dict={x: g_features, y: g_labels, keep_prob: 1})
    print("Predictions: ")
    print(g_prediction)
    # print(predict_out)
    print("Top Five: ")
    topFive = tf.nn.top_k(logits, k=5, sorted=True, name=None)
    out = sess.run(topFive, feed_dict={x: g_features, y: g_labels, keep_prob: 1})
    #print(out)
    print(type(out))
    print(len(out))
    top5_arr = out[0]
    for i in top5_arr:
        print(i)
        print("esh")
    for i in out[1]:
        print(i)
        print("esh1")
    top5_pred = sess.run([logits, topFive], feed_dict={x: g_features, y: g_labels, keep_prob: 1})
    print(top5_pred)
    signname_map = np.genfromtxt('signnames.csv', delimiter=',', usecols=(1,), unpack=True, dtype=str, skip_header=1)

    for i in range(5):
        plt.figure(figsize=(3, 1.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
        plt.subplot(gs[0])
        path = "images/german_sign_new_" + str(i + 1) + ".jpg"
        image = mpimg.imread(path)
        image = cv2.resize(image, (32, 32))
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(gs[1])
        vals = []
        for v in top5_pred[1][0][i]:
            if v > 0:
                vals.append(v)
            else:
                vals.append(1)
        plt.barh(6 - np.arange(5), vals, align='center')
        for i_label in range(5):
            plt.text(vals[i_label] + .2, 6 - i_label - .25,
                     signname_map[out[1][i][i_label]])
        plt.axis('off')
        #plt.text(0, 6.95, namenewdata[i].split('.')[0])
        plt.show()
