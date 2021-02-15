import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from tensorboardX import SummaryWriter
from tensorflow.keras.utils import to_categorical
import tensorflow.compat.v1 as tf
import pandas as pd
import datetime
import os

path = '../embedding/fasttext/'
model_path = '../model/'
result_path = '../result/'
os.makedirs(result_path, exist_ok=True)

print('Loading data...')
train = pd.read_csv('../data/total_train.csv', encoding='euc-kr')
test = pd.read_csv('../data/total_test.csv', encoding='euc-kr')

x_train = np.load(os.path.join(path, 'fasttext_x_train.npy'))
y_train = np.load(os.path.join(path, 'fasttext_y_train.npy'))
y_train = to_categorical(y_train)

print(x_train.shape, y_train.shape)

x_test = np.load(os.path.join(path, 'fasttext_x_test.npy'))
y_test = np.load(os.path.join(path, 'fasttext_y_test.npy'))
y_test = to_categorical(y_test)

print(x_test.shape, y_test.shape)

# set parameters
load_model = False
embedding_dim = x_train.shape[2]
n_hidden = 200
batch_size = 32
n_step = x_train.shape[1]
n_class = 3

epochs = 5001
lr = 0.00005

tf.reset_default_graph()
tf.disable_eager_execution()

# Input & Output Placeholder
X = tf.placeholder(tf.float32, [None, n_step, embedding_dim])
Y = tf.placeholder(tf.float32, [None, n_class])
out = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))

####################################################################################################################
# LSTM                                                                                                             #
####################################################################################################################
lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

output, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, X, dtype=tf.float32)
output = tf.concat([output[0], output[1]], 2)
final_hidden_state = tf.concat([final_state[1][0], final_state[1][1]], 1)
final_hidden_state = tf.expand_dims(final_hidden_state, 2)

####################################################################################################################
# Attention                                                                                                        #
####################################################################################################################
attn_weights = tf.squeeze(tf.matmul(output, final_hidden_state), 2) # attn_weights : [batch_size, n_step]
soft_attn_weights = tf.nn.softmax(attn_weights, 1)
context = tf.matmul(tf.transpose(output, [0, 2, 1]), tf.expand_dims(soft_attn_weights, 2))
context = tf.squeeze(context, 2) # [batch_size, n_hidden * num_directions(=2)]

model = tf.matmul(context, out)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
#optimizer = tf.train.AdamOptimizer(0.00003).minimize(cost)

# Model-Predict
hypothesis = tf.nn.softmax(model)
predictions = tf.argmax(hypothesis, 1)

# Training
with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=5)
    if load_model == True:
        print("[Load Model...]")
        start_time = datetime.datetime.now()
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        y_pred = np.empty(0, )
        batch = x_test.shape[0] // 20
        for i in range(21):
            if i != 20:
                predict, attention = sess.run([predictions, soft_attn_weights], feed_dict={X: x_test[i*batch:(i+1)*batch]})
            else:
                predict, attention = sess.run([predictions, soft_attn_weights],
                                              feed_dict={X: x_test[i * batch:]})

            y_pred = np.concatenate((y_pred, predict))
            if i == 0:
                test_attention = attention
            else:
                test_attention = np.concatenate((test_attention, attention))

        y_test_bool = np.argmax(y_test, axis=1)
        print(classification_report(y_test_bool, y_pred))
        np.save(os.path.join(result_path, 'total_attention.npy'), test_attention)
        np.save(os.path.join(result_path, 'total_prediction.npy'), y_pred)

        test['pred'] = y_test_bool
        attention_label = [words for words in test['preprocessed']]
        print(datetime.datetime.now()-start_time)
        
    else:
        os.makedirs(model_path, exist_ok=True)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, 3000, 0.92, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        print("[Training...]")
        writer = SummaryWriter()
        init = tf.global_variables_initializer()
        sess.run(init)
        acc_score = 0.0

        for epoch in range(epochs):
            batch_idx = np.random.choice(x_train.shape[0], batch_size, replace=False)
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            _, loss, attention = sess.run([optimizer, cost, soft_attn_weights], feed_dict={X: x_batch, Y: y_batch})
            print('Epoch:', '%06d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            writer.add_scalar('Loss/cost', loss, epoch+1)
            predict = sess.run([predictions], feed_dict={X: x_test})
            y_pred_bool = np.reshape(predict, newshape=(np.shape(x_test)[0], 1))
            y_test_bool = np.argmax(y_test, axis=1)

            f1_macro = f1_score(y_true=y_test_bool, y_pred=y_pred_bool, average='macro')
            f1_weighted = f1_score(y_true=y_test_bool, y_pred=y_pred_bool, average='weighted')
            f1_micro = f1_score(y_true=y_test_bool, y_pred=y_pred_bool, average='micro')
            precision, recall, fscore, support = score(y_test_bool, y_pred_bool)
            f1_label0 = fscore[0]
            f1_label1 = fscore[1]
            f1_label2 = fscore[2]

            writer.add_scalar('Total_Score/f1_macro', f1_macro, epoch+1)
            writer.add_scalar('Total_Score/f1_weighted', f1_weighted, epoch+1)
            writer.add_scalar('Total_Score/f1_micro', f1_micro, epoch + 1)
            writer.add_scalar('Score/f1_label_0', f1_label0, epoch+1)
            writer.add_scalar('Score/f1_label_1', f1_label1, epoch+1)
            writer.add_scalar('Score/f1_label_2', f1_label2, epoch+1)

            # model save
            if epoch % 100 == 0:
                if acc_score <= accuracy_score(y_test_bool, y_pred_bool):
                    saver.save(sess, os.path.join(model_path, 'attetion_mdl_{:04d}.ckpt'.format(epoch)))
                    print(classification_report(y_test_bool, y_pred_bool))
                    acc_score = accuracy_score(y_test_bool, y_pred_bool)