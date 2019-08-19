# coding by willhelm

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import sys
np.set_printoptions(threshold=sys.maxsize,suppress=True)


def findByRow(mat, row):
	return np.where((mat == row).all(1))[0]
	# return np.where((mat == row).all(1))[0]

def read_csv(file):
    data = np.genfromtxt(file, delimiter=",",skip_header=1)
    data = np.flip(data, 0)
    data = data[:,3:]
    data = normalize(data,norm="max")
    return data

def evaluate(matrix):
	matrix = np.matmul(matrix, np.transpose(matrix))
	stdv = np.trace(matrix)
	correlation = (np.sum(matrix) - stdv)/2
	return correlation, stdv

def sort(pop, cor, std):
	before_sorting_cor = np.concatenate((cor,pop),axis=1)
	after_sort_cor = np.sort(before_sorting_cor,axis= 0) #(population, 42) ascending order
	after_sort_cor = after_sort_cor[:int(len(after_sort_cor)*eliminate_rate),1:] #(population*eliminate, 41)

	before_sorting_std = np.concatenate((std,pop),axis=1)
	after_sort_std = np.sort(before_sorting_std,axis= 0)
	after_sort_std = np.flip(after_sort_std, 0)
	after_sort_std = after_sort_std[:int(len(after_sort_std)*eliminate_rate),1:]

	re = []
	for element in after_sort_cor:
		if len(after_sort_std[findByRow(after_sort_std,element)]) != 0:
			re.append(element)				
	re = np.array(re)

	return re


data_add ="C:\\Users\\willh\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\Project_Data\\"
file_names = [] 
for root, dirs, files in os.walk(data_add):
    for file in files:
        if ('.csv') in file:
            file_names.append(data_add+file)

epoch = 50 
population_upper_boundary = 300
population_lower_boundary = 150
pop_num = population_lower_boundary
eliminate_rate = 0.7 #keeped 
mutation_rate = 0.06
#figuring the num of elements in the raw data
temp = np.genfromtxt(file_names[0], delimiter=",",skip_header=1)
feature_num  = temp[:,3:].shape[1]


# innitialize population
population = np.random.randint(2, size= (pop_num,feature_num))
# evaluate
for j in range(0, epoch):
	pop_num = len(population)
	result_cor,result_stdv = np.zeros((pop_num,1)), np.zeros((pop_num,1))
	pop_acc = 0

	#mutation
	for i in range(0, len(population)):
		num_of_column_mutating = int(mutation_rate*feature_num)
		index_of_column_mutating = np.random.randint(feature_num, size= (num_of_column_mutating))
		#replacing
		population[i,index_of_column_mutating] = np.ones(num_of_column_mutating) 


	for person in population:
		p = np.reshape(person, (1,-1))
		_, p = np.where(p == 1)
		for name in file_names:
			# input
			data = read_csv(name) #(1023,41)
			# select colum according to pop index 
			data = data[:,p]
			feature_num_temp = len(p)

			data = np.reshape(data, (feature_num_temp,-1))

			centre = np.reshape(np.mean(data,axis = 1), (-1,1))
			# centrlization
			data = data - centre
			cor, stdv = evaluate(data)
			result_cor[pop_acc], result_stdv[pop_acc] = result_cor[pop_acc]+cor, result_stdv[pop_acc]+stdv
			break
		pop_acc += 1
	result_cor = np.absolute(result_cor)
	print("for epoch "+str(j)+"   popluation:  "+str(pop_num))
	print("  correlation_std:  "+str(np.std(result_cor))+"  feature_std_std:  "+str(np.std(result_stdv)))		
	print("  correlation_mean:  "+str(np.mean(result_cor))+"  feature_std_mean:  "+str(np.mean(result_stdv)))
	print("                   ")

	after_sort = sort(population, result_cor, result_stdv)
	# get rid less performed ones and the performance indicators
	fathers = after_sort[:,:int(feature_num*0.5)]
	mothers = after_sort[:,int(feature_num*0.5):]
	np.random.shuffle(mothers)

	# cross over
	chilrden = np.concatenate((fathers,mothers), axis=1)
	chilrden = np.concatenate((after_sort,chilrden), axis= 0 ) #making up population
	
	if len(chilrden) > population_lower_boundary:
		chilrden = chilrden[:population_upper_boundary]
	else:
		while(len(chilrden) < population_lower_boundary):
			np.random.shuffle(mothers)
			temp = np.concatenate((fathers,mothers), axis=1)
			chilrden = np.concatenate((temp,chilrden), axis= 0 )

	population = chilrden

# evolve finished outputing finail results

pop_acc = 0
pop_num = len(population)
result_cor,result_stdv = np.zeros((pop_num,1)), np.zeros((pop_num,1))
for person in population:
	p = np.reshape(person, (1,-1))
	_, p = np.where(p == 1)
	for name in file_names:
		data = read_csv(name) #(1023,41) 
		data = data[:,p]
		feature_num_temp = len(p)
		try:
			data = np.reshape(data, (feature_num_temp,-1))
		except Exception as e:
			pass

		centre = np.reshape(np.mean(data,axis = 1), (-1,1))
		# centrlization
		data = data - centre
		cor, stdv = evaluate(data)
		result_cor[pop_acc], result_stdv[pop_acc] = result_cor[pop_acc]+cor, result_stdv[pop_acc]+stdv
		break
	pop_acc += 1
result_cor = np.absolute(result_cor)
after_sort = sort(population, result_cor, result_stdv)

for x in after_sort:
	x = np.reshape(person, (1,-1))
	_, x = np.where(x == 1)
	print(x)

# print(after_sort)