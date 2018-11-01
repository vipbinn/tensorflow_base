import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#------------------设置超参数------------------#
#设置学习率
learning_rate = 0.01
#设置最大训练步数
max_train_steps = 5000
log_step = 100
total_samples = 17

#构造训练数据
train_X = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],[9.779],[6.182],[7.59],[2.167],
                    [7.042],[10.791],[5.313],[7.997],[5.654],[9.27],[3.1]],dtype=np.float32)

train_Y = np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],[3.366],[2.596],[2.53],
                    [1.221],[2.827],[3.465],[1.65],[2.904],[2.42],[2.94],[1.3]],dtype=np.float32)

#输入数据
X = tf.placeholder(tf.float32,[None,1])
# 实际值
Y_ = tf.placeholder(tf.float32,[None,1])

#模型参数
w = tf.Variable(tf.random_normal([1,1]),name = "weight")
b = tf.Variable(tf.zeros([1]),name = "bias")

#推理值
Y = tf.matmul(X,w) + b

# 均方误差
loss = tf.reduce_sum(tf.pow(Y - Y_,2))/(total_samples)

# 创建优化器，用的是随机梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# 定义单步训练操作,最小化损失函数
train_op = optimizer.minimize(loss)

# 创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 迭代训练
    print("Start training:")
    for step in range(max_train_steps):
        sess.run(train_op,feed_dict={X:train_X,Y_:train_Y})
        #每间隔log_tep步，打印一次日志
        if step % log_step == 0:
            c = sess.run(loss,feed_dict={X:train_X,Y_:train_Y})
            print("step:%d, loss==%.4f, w==%.4f, b==%.4f" %
                  (step,c,sess.run(w),sess.run(b)))

    #计算训练完毕的模型在训练集上的损失函数，并将其作为指标输出
    final_loss = sess.run(loss,feed_dict={X:train_X,Y_:train_Y})

    #计算训练完毕的模型参数w和b
    weight, bias = sess.run([w,b])
    print("step:%d, loss==%.4f, w==%.4f, b==%.4f" %
          (max_train_steps, final_loss,sess.run(w), sess.run(b)))
    print("Linear Regression Model: Y==%.4f*X+%.4f" % (weight, bias))

# 模型可视化
plt.plot(train_X, train_Y, 'ro', label = 'Training data')
plt.plot(train_X, weight * train_X + bias, label = 'Fitted line')
plt.legend()
plt.show()