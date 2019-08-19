#code by Willhelm
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def bias_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def oscillator(table, input_vec,size):
    #rescale input_vect from (-1~1) to integer (0~1000)
    input_vec =tf.cast(tf.reshape(tf.nn.tanh(input_vec)*5000+5000, [-1]),  tf.int32)
    a = input_vec
    input_vec = tf.gather(table,input_vec)
    # input_vec = tf.reshape(tf.gather(table,input_vec),[1,-1])
    random_index = tf.random_uniform([size], minval=0, maxval=100, dtype=tf.int32)
    input_vec = tf.reshape(tf.diag_part(tf.gather(input_vec,random_index, axis=1)),[1,-1])
    return tf.cast(input_vec, tf.float32) #,a

def norm(data):
    norms_mean, norms_z, norms_bias = np.mean(data, axis=0), 8*np.std(data, axis=0), 0.5
    # print(norms_mean)
    # print("   ")
    # print(norms_z)
    # print("   ")
    data = ((data - norms_mean)/norms_z)+norms_bias
    # norms_mean, norms_z, norms_bias = np.mean(data, axis=0), np.std(data, axis=0), 0.5
    # print(norms_mean)
    # print("   ")
    # print(norms_z)
    # print("   ")
    return data, norms_mean, norms_z, norms_bias

def read_csv(file):
    result = []
    current_day = []
    tr_set = []
    te_set = []

    data = np.genfromtxt(file, delimiter=",",skip_header=1)
    data = np.flip(data, 0)
    np.set_printoptions(suppress=True)
    data = data[:,[3,4,5,6,7]]
    # data, norms = normalize(data, axis=0, norm='max',return_norm=True) #normalize data
    data, norms_mean, norms_z, norms_bias = norm(data)
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

    np.random.shuffle(tr_set)

    return tr_set, te_set, current_day, norms_mean, norms_z, norms_bias

#hyper parameter
epoch = 600
length_of_training_records = 7
training_rate = 1e-5 # to be adjusted in the following content

#IO
data_add ="C:\\Users\\willh\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\Project_Data\\"
prediction_add = "C:\\Users\\willh\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\Prediction\\"
model_add = "C:\\Users\\willh\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\model\\"
oscillator_add = "C:\\Users\\willh\\Documents\\QF\\oscillator\\"

file_names = [] 
for root, dirs, files in os.walk(data_add):
    for file in files:
        if ('.csv') in file:
            file_names.append(data_add+file)

# #load oscillator
# type_1 = np.genfromtxt(oscillator_add+'#1.csv', delimiter=',')
# type_2 = np.genfromtxt(oscillator_add+'#2.csv', delimiter=',')
# type_3 = np.genfromtxt(oscillator_add+'#3.csv', delimiter=',')
# type_4 = np.genfromtxt(oscillator_add+'#4.csv', delimiter=',')
# type_5 = np.genfromtxt(oscillator_add+'#5.csv', delimiter=',')
# type_6 = np.genfromtxt(oscillator_add+'#6.csv', delimiter=',')
# type_7 = np.genfromtxt(oscillator_add+'#7.csv', delimiter=',')

# ph_type1 = tf.placeholder(tf.float32, [10000,100])
# ph_type2 = tf.placeholder(tf.float32, [1000,100])
# ph_type3 = tf.placeholder(tf.float32, [1000,100])
# ph_type4 = tf.placeholder(tf.float32, [1000,100])
# # ph_type5 = tf.placeholder(tf.float32, [1000,100])
# ph_type5 = tf.placeholder(tf.float32, [10000,100])
# ph_type6 = tf.placeholder(tf.float32, [1000,100])
# ph_type7 = tf.placeholder(tf.float32, [1000,100])

# network strcture
ph_input_vector = tf.placeholder(tf.float32,[None, 5,1])
ph_label_vector = tf.placeholder(tf.float32,[1,3])
ph_latest_OPEN = tf.placeholder(tf.float32,[1,1])

cell = tf.nn.rnn_cell.GRUCell(num_units = 30)
init_state = cell.zero_state(batch_size=5,dtype = tf.float32)#batch size intented to be, out can be [-1, 1100] multiply oneby one
GRU_outputs, final_state = tf.nn.dynamic_rnn(cell, ph_input_vector, initial_state=init_state, time_major=True)

#fully connect layer
final_state = tf.reshape(final_state, [1,-1])
final_state = tf.concat([final_state,ph_latest_OPEN],1)

W_fc_1 = weight_variable([151, 300])
b_fc_1 = bias_variable([300])
# h_fc_1 = oscillator(ph_type5, tf.matmul(final_state,W_fc_1)+b_fc_1, 300)
h_fc_1 = tf.nn.leaky_relu(tf.matmul(final_state,W_fc_1)+b_fc_1)

W_fc_2 = weight_variable([300, 300])
b_fc_2 = bias_variable([300])
h_fc_2 = tf.nn.leaky_relu(tf.matmul(h_fc_1,W_fc_2)+b_fc_2)
# h_fc_2 = oscillator(ph_type1, tf.matmul(h_fc_1,W_fc_2)+b_fc_2, 300)


W_fc_3 = weight_variable([300, 3])
b_fc_3 = bias_variable([3])
h_fc_3 = tf.nn.leaky_relu(tf.matmul(h_fc_2,W_fc_3)+b_fc_3)
# h_fc_3 = oscillator(ph_type2,tf.matmul(h_fc_2,W_fc_3)+b_fc_3)

#evaluate and optimization
MSE = tf.reduce_mean((ph_label_vector- h_fc_3)**2)
train_step = tf.train.AdamOptimizer(training_rate).minimize(MSE)

with tf.Session() as sess:
    #input from every single csv to form tranning data
    for file in file_names: #for every single product
        #innitailize saver
        saver = tf.train.Saver()
        print("for product "+str(file))
        training_set, testing_set, latest_day, norms_mean, norms_z, norms_bias = read_csv(file)
        
        index = file.index('.csv')
        product_name = file[index-6:index]
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        try:
    #############################
    #test
    #############################
            saver.restore(sess, model_add+product_name+".ckpt")
            actual_low = np.zeros(len(testing_set))
            actual_high = np.zeros(len(testing_set))
            predicted_low = np.zeros(len(testing_set))

            count = 0
            loss = np.zeros(len(testing_set))
            for record in testing_set:
                input_vector = np.reshape(record[:length_of_training_records-1], (length_of_training_records-1,-1,1))
                label_vector = record[length_of_training_records-1:]

                OPEN = np.reshape(label_vector[0][0], (1,1))
                label_vector= np.reshape(label_vector[0][1:4], (1,3))

                l, prediction = sess.run([MSE,h_fc_3], feed_dict = {
                    ph_input_vector:input_vector,
                    ph_latest_OPEN:OPEN,
                    ph_label_vector:label_vector
                    # ph_type1:type_1, ph_type2:type_2, ph_type3:type_3,ph_type4:type_4,
                    # ph_type5:type_5, ph_type6:type_6,ph_type7:type_7                  
                    })
                loss[count] = l
                actual_low[count] = ((label_vector-norms_bias)*norms_z[1:4]+norms_mean[1:4])[0][1]
                actual_high[count] = ((label_vector-norms_bias)*norms_z[1:4]+norms_mean[1:4])[0][0]
                predicted_low[count] = ((prediction-norms_bias)*norms_z[1:4]+norms_mean[1:4])[0][1]
                count += 1

            distance = predicted_low - actual_low
            variation = actual_high - actual_low 
            # predict tomorrow's indicators
            O =latest_day[0][length_of_training_records-1:]
            latest_day = latest_day[0][:length_of_training_records-1] 
            input_vector = np.reshape(latest_day, (length_of_training_records-1,-1,1))
            OPEN = np.reshape(O[0][0], (1,1))

            l, prediction = sess.run([MSE,h_fc_3], feed_dict = {
                ph_input_vector:input_vector,
                ph_latest_OPEN:OPEN,
                ph_label_vector:label_vector
                # ph_type1:type_1, ph_type2:type_2, ph_type3:type_3,ph_type4:type_4,
                # ph_type5:type_5, ph_type6:type_6,ph_type7:type_7                  
                })

            prediction = np.asarray(prediction)
            prediction = (prediction-norms_bias)*norms_z[1:4]+norms_mean[1:4]
            add_temp = prediction_add+product_name+".csv"
            print(add_temp)
            np.savetxt(add_temp,[prediction[0][1],np.std(distance),np.mean(distance),np.mean(loss)],delimiter=",")
            print("for product "+str(file))
            print("for tomorrow's indicator H: "+str(prediction[0][0])+"  L  "+str(prediction[0][1])+"  C  "+str(prediction[0][2]))
            print("distance  std: "+ str(np.std(np.absolute(distance)))+"  mean  "+str(np.mean(np.absolute(distance)))+"  median  "+str(np.median(np.absolute(distance))))
            print("     ")

        except Exception as e:
            print("No pre-trained model for "+str(product_name))
        
        # continue
        #train network

        ########################## can be deleted
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        ##########################

        loss_record = np.zeros(epoch)
        std_record = np.zeros(epoch)
        for i in range(0,epoch):
            training_rate = 5e-6/(1+np.log(i+1))
            count = 0
            loss = np.zeros(len(training_set))
            ###########
            d = np.zeros((len(training_set),300)) #[len_of_training_set, num_of_cell_for_a_vector]
            ###########
            for record in training_set:
                # print(record)
                input_vector = np.reshape(record[:length_of_training_records-1], (length_of_training_records-1,-1,1))
                label_vector = record[length_of_training_records-1:]
                OPEN = np.reshape(label_vector[0][0], (1,1))
                label_vector= np.reshape(label_vector[0][1:4], (1,3))

                # prediction = sess.run(h_fc_2, feed_dict = {
                #     ph_input_vector:input_vector,
                #     ph_latest_OPEN:OPEN,
                #     ph_label_vector:label_vector,
                #     ph_type1:type_1, ph_type2:type_2, ph_type3:type_3,ph_type4:type_4,
                #     ph_type5:type_5, ph_type6:type_6,ph_type7:type_7
                #     })
                # prediction = np.array(prediction)
                # print(prediction.shape)

                _, l, prediction,d_temp = sess.run([train_step,MSE,h_fc_3,h_fc_1], feed_dict = {
                    ph_input_vector:input_vector,
                    ph_latest_OPEN:OPEN,
                    ph_label_vector:label_vector
                    # ph_type1:type_1, ph_type2:type_2, ph_type3:type_3,ph_type4:type_4,
                    # ph_type5:type_5, ph_type6:type_6,ph_type7:type_7
                    })
                # d[count] = d_temp
                loss[count] = l
                count += 1

            #######################
            #write the loss and stc info into records 
            loss_record[i], std_record[i] = np.mean(loss),np.std(loss)
            #######################
            print("epoch No."+str(i)+ " avrage loss: "+str(np.mean(loss))+" RNN_out mean "+str(np.mean(d))+" RNN_out std "+str(np.std(d)))
            # print("epoch No."+str(i)+ " avrage loss: "+str(np.mean(loss))+" std "+str(np.std(loss))+"  rate "+str(training_rate))

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
            label_vector= np.reshape(label_vector[0][1:4], (1,3))

            l, prediction = sess.run([MSE,h_fc_3], feed_dict = {
                ph_input_vector:input_vector,
                ph_latest_OPEN:OPEN,
                ph_label_vector:label_vector
                # ph_type1:type_1, ph_type2:type_2, ph_type3:type_3,ph_type4:type_4,
                # ph_type5:type_5, ph_type6:type_6,ph_type7:type_7
                })
            loss[count] = l

            actual_low[count] = ((label_vector-norms_bias)*norms_z[1:4]+norms_mean[1:4])[0][1]
            actual_high[count] = ((label_vector-norms_bias)*norms_z[1:4]+norms_mean[1:4])[0][0]
            predicted_low[count] = ((prediction-norms_bias)*norms_z[1:4]+norms_mean[1:4])[0][1]
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
        plt.savefig(img_add,dpi=300)
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
        plt.savefig(img_add, dpi=300)
        plt.clf()   