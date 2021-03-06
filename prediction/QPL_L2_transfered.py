import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def bias_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))


def read_csv(file):
    result = []
    current_day = []
    tr_set = []
    te_set = []

    data = np.genfromtxt(file, delimiter=",",skip_header=1)
    data = np.flip(data, 0)

    np.set_printoptions(suppress=True)
    data = data[:,[3,4,5,6,9,10,11,12,13,14,15,16]]
    data, norms = normalize(data, axis=0, norm='l2',return_norm=True) #normalize data
  
    #form data set for latest days to do preiction
    current_day.append(data[-(length_of_training_records):])
    current_day = np.array(current_day)

    data = data[:-(length_of_training_records-1)]
    tr = data[:int(0.8*len(data))]
    te = data[int(0.8*len(data)):]

    for i in range(length_of_training_records,len(tr)):
        tr_set.append(data[i-length_of_training_records:i])
    tr_set = np.array(tr_set)

    for i in range(length_of_training_records,len(te)):
        te_set.append(data[i-length_of_training_records:i])
    te_set = np.array(te_set)

    # print(current_day)
    np.random.shuffle(tr_set)
    # np.random.shuffle(te_set)

    # for i in range(length_of_training_records,len(data)):
    #     result.append(data[i-length_of_training_records:i])
    # result = np.array(result)
    # np.random.shuffle(result)
    # tr_set = result[:int(0.8*len(result))]
    # te_set = result[int(0.8*len(result)):]
    return tr_set, te_set, current_day, norms

#hyper parameter
epoch = 1800
length_of_training_records = 7
training_rate = 1e-7
kp = 0.2

#IO
# data_add ="C:\\Users\\willh\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\Project_Data\\"
# prediction_add = "C:\\Users\\willh\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\Prediction\\"

# data_add ="C:\\Users\\UIC\\Downloads\\Project_Data\\"
# prediction_add ="C:\\Users\\UIC\\Downloads\\p\\"
# model_add = "C:\\Users\\UIC\\Downloads\\model\\"

data_add ="/home/ubuntu/fyp2/Project_Data/"
prediction_add = "/home/ubuntu/fyp2/Predicting_Value/"
model_add = "/home/ubuntu/fyp2/model/"
# list file in directory
file_names = [] 
for root, dirs, files in os.walk(data_add):
    for file in files:
        if ('.csv') in file:
            file_names.append(data_add+file)

# network strcture
ph_input_vector = tf.placeholder(tf.float32,[None, 12,1])
ph_label_vector = tf.placeholder(tf.float32,[1,11])
ph_latest_OPEN = tf.placeholder(tf.float32,[1,1])

cell = tf.nn.rnn_cell.GRUCell(num_units = 30)
init_state = cell.zero_state(batch_size=12,dtype = tf.float32)#batch size intented to be, out can be [-1, 1100] multiply oneby one
GRU_outputs, final_state = tf.nn.dynamic_rnn(cell, ph_input_vector, initial_state=init_state, time_major=True)

#fully connect layer
final_state = tf.reshape(final_state, [1,-1])
final_state = tf.concat([final_state,ph_latest_OPEN],1)
W_fc_1 = weight_variable([361, 500])
b_fc_1 = bias_variable([500])
h_fc_1 = tf.nn.leaky_relu(tf.matmul(final_state,W_fc_1)+b_fc_1)

# h_fc_1 = tf.nn.dropout(h_fc_1, keep_prob = kp)

W_fc_2 = weight_variable([500, 500])
b_fc_2 = bias_variable([500])
h_fc_2 = tf.nn.leaky_relu(tf.matmul(h_fc_1,W_fc_2)+b_fc_2)

W_fc_3 = weight_variable([500, 11])
b_fc_3 = bias_variable([11])
h_fc_3 = tf.nn.leaky_relu(tf.matmul(h_fc_2,W_fc_3)+b_fc_3)

# h_fc_3 = tf.transpose(h_fc_3)
#evaluate and optimization
MSE = tf.reduce_mean((ph_label_vector- h_fc_3)**2)
# MSE,_= tf.metrics.mean_squared_error(ph_label_vector, final_state)
train_step = tf.train.AdamOptimizer(training_rate).minimize(MSE)
#innitailize saver

with tf.Session() as sess:
    #input from every single csv to form tranning data
    for file in file_names: #for every single product
        saver = tf.train.Saver()
        print("for product "+str(file))
        training_set, testing_set, latest_day, norms = read_csv(file)
        
        index = file.index('.csv')
        product_name = file[index-6:index]
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        try:
            saver.restore(sess, model_add+product_name+".ckpt")
    #############################
    #test
    #############################
            actual_low = np.zeros(len(testing_set))
            actual_high = np.zeros(len(testing_set))
            predicted_low = np.zeros(len(testing_set))

            count = 0
            loss = np.zeros(len(testing_set))
            for record in testing_set:
                input_vector = np.reshape(record[:length_of_training_records-1], (length_of_training_records-1,-1,1))
                label_vector = record[length_of_training_records-1:]

                OPEN = np.reshape(label_vector[0][0], (1,1))
                label_vector= np.reshape(label_vector[0][1:], (1,3))

                l, prediction = sess.run([MSE,h_fc_3], feed_dict = {
                    ph_input_vector:input_vector,
                    ph_latest_OPEN:OPEN,
                    ph_label_vector:label_vector
                    })
                loss[count] = l

                actual_low[count] = (label_vector*norms[1:])[0][1]
                actual_high[count] = (label_vector*norms[1:])[0][0]
                predicted_low[count] = (prediction*norms[1:])[0][1]
                count += 1

            distance = predicted_low - actual_low
            variation = actual_high - actual_low 
            # predict tomorrow's indicators
            O =latest_day[0][length_of_training_records-1:]
            latest_day = latest_day[0][:length_of_training_records-1] 
            input_vector = np.reshape(latest_day, (length_of_training_records-1,-1,1))
            OPEN = np.reshape(O[0][0], (1,1))

            prediction = sess.run(h_fc_3, feed_dict = {
            ph_input_vector:input_vector,
            ph_latest_OPEN:OPEN        })

            prediction = np.asarray(prediction)
            prediction = prediction*norms[1:]
            add_temp = prediction_add+product_name+".csv"
            print(add_temp)
            np.savetxt(add_temp,[prediction[0][1],np.std(distance),np.mean(distance),np.mean(loss)],delimiter=",")
            print("for product "+str(file))
            print("for tomorrow's indicator "+str(prediction))
            print("     ")
        except Exception as e:
            print("No pre-trained model for "+str(product_name))
        #continue
###############
        # continue
###############
        #train network
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        loss_record = np.zeros(epoch)
        std_record = np.zeros(epoch)
        for i in range(0,epoch):
            training_rate = 3e-5/(1+np.log(i+1)) #1e-4
            count = 0
            loss = np.zeros(len(training_set))
            for record in training_set:
                input_vector = np.reshape(record[:length_of_training_records-1], (length_of_training_records-1,-1,1))
                label_vector = record[length_of_training_records-1:]
                OPEN = np.reshape(label_vector[0][0], (1,1))
                label_vector= np.reshape(label_vector[0][1:], (1,11))

                _, l, prediction = sess.run([train_step,MSE,h_fc_3], feed_dict = {
                    ph_input_vector:input_vector,
                    ph_latest_OPEN:OPEN,
                    ph_label_vector:label_vector
                    })
                loss[count] = l
                count += 1
                # print(prediction.shape)
            #######################
            #write the loss and stc info into records 
            loss_record[i], std_record[i] = np.mean(loss),np.std(loss)
            #######################
            print("epoch No."+str(i)+ " avrage loss: "+str(np.mean(loss))+" std "+str(np.std(loss))+"  rate "+str(training_rate))

        #test
        actual_low = np.zeros(len(testing_set))
        actual_high = np.zeros(len(testing_set))
        predicted_low = np.zeros(len(testing_set))

        count = 0
        loss = np.zeros(len(testing_set))
        for record in testing_set:
            input_vector = np.reshape(record[:length_of_training_records-1], (length_of_training_records-1,-1,1))
            label_vector = record[length_of_training_records-1:]

            OPEN = np.reshape(label_vector[0][0], (1,1))
            label_vector= np.reshape(label_vector[0][1:], (1,11))

            l, prediction = sess.run([MSE,h_fc_3], feed_dict = {
                ph_input_vector:input_vector,
                ph_latest_OPEN:OPEN,
                ph_label_vector:label_vector
                })
            loss[count] = l

            actual_low[count] = (label_vector*norms[1:])[0][1]
            actual_high[count] = (label_vector*norms[1:])[0][0]
            predicted_low[count] = (prediction*norms[1:])[0][1]
            count += 1


        save_path = saver.save(sess, model_add+product_name+".ckpt")
        print("Test finished "+ " avrage loss: "+str(np.mean(loss))+" std "+str(np.std(loss)))

        ###############################
        #generate plot graphnump
        # index = file.index('.csv')
        # product_name = file[index-6:index]
        img_add = file[:index]+"_PvsA.png"
        step_count = np.arange(0,len(testing_set))

        plt.title(str(product_name))
        plt.xlabel('Days')
        plt.ylabel('Values')
        plt.plot(step_count,predicted_low, 'y', label='Prediction')
        plt.plot(step_count,actual_low, 'c', label='Actual')
        plt.legend(loc='upper right')
        plt.savefig(img_add)
        plt.clf()

        distance = predicted_low - actual_low
        variation = actual_high - actual_low 
        print("distance mean: "+str(np.mean(distance))+"  std: "+str(np.std(distance))+"  variation mean: "+str(np.mean(variation))+" std: "+str(np.std(variation)))
        ###############################

        ###############################
        #generate plot graphnump
        index = file.index('.csv')
        product_name = file[index-6:index]
        img_add = file[:index]+".png"
        step_count = np.arange(0,epoch)

        step_count = step_count[20:]
        loss_record = loss_record[20:]
        std_record = std_record[20:]

        plt.title(str(product_name))
        plt.xlabel('Epoch')
        plt.ylabel('Loss & Stdv')
        plt.plot(step_count,loss_record, 'y', label='Loss')
        plt.plot(step_count,std_record, 'c', label='Stdv')
        plt.axhline(y=np.mean(loss) , color='r',linestyle='-',label='test Loss')
        plt.legend(loc='upper right')
        plt.savefig(img_add)
        plt.clf()
