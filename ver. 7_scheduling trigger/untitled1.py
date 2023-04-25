# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:51:56 2019

@author: 燕子
"""

import numpy as np
import tensorflow as tf
import threading
import time
import math
import copy
import random
import datetime
import requests
import sys
from RL_brain import DeepQNetwork
import os
import matplotlib.pyplot as plt
import csv
import distri_pattern as p_gen

iteration = 0
elevator_num = 2
floor = 10
# total_request_num = 10
transport_time = 5
open_time = 11

record_file = 0
restore = 0 #restore = 0
save = 1 #save = 1
max_iteration = 100

parameter_num = elevator_num + elevator_num + floor*elevator_num + floor*elevator_num + 1

RL = DeepQNetwork(    elevator_num, #action 
					  parameter_num, #input number
                      learning_rate=0.00025,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=100000,
                      memory_size=100000,
                      double_q=False,
                      save_model = save,
                      restore_model = restore
                      # output_graph=True
                      )

step_counter = np.zeros(elevator_num)
avg_waiting_time = 0
avg_travel_time = 0
total_request_num_sum = 0
# waiting_time = np.zeros(total_request_num)
current_avg_waiting_time = 1000000
# rm_catch = np.zeros(total_request_num)
# shortest_rm_catch = np.zeros(total_request_num)
# shortest_waiting_time = np.zeros(total_request_num)
shortest_avg_waiting_time = 1000000
current_avg_waiting_time_his = []

ave_hour_waitingtimelist = [[], []]


ave_hour_waitingtimelist[0] = np.zeros(15)
ave_hour_waitingtimelist[1] = np.zeros(15)

while iteration < max_iteration:
# while current_avg_waiting_time > 551:
	iteration += 1
	print("iteration: " + str(iteration))
	
	if record_file == 1 and iteration == max_iteration:
	# if record_file == 1:
		f = open("trake2.txt", 'w')
		f.write("iteration: " + str(iteration) + '\n')
		f.close()

########################## initialization ##########################
	lu = []
	ld = []

	lun = []
	for i in range(elevator_num):
		temp = []
		lun.append(temp)

	lunt = []
	for i in range(elevator_num):
		temp = []
		lunt.append(temp)

	ldn = []
	for i in range(elevator_num):
		temp = []
		ldn.append(temp)

	ldnt = []
	for i in range(elevator_num):
		temp = []
		ldnt.append(temp)

	position = np.zeros(elevator_num)
	position = position.astype(int)
	position_his = []
	for i in range(elevator_num):
		temp = []
		position_his.append(temp)

	direction = np.zeros(elevator_num)
	old_direction = np.zeros(elevator_num)
	stc = np.zeros((elevator_num, floor))
	
	# position[0] = 4
	# position[1] = 16
	# position[2] = 2
	# position[3] = 18

	# direction[0] = 1
	# direction[1] = -1
	# direction[2] = 1
	# direction[3] = -1

	# stc[0][6] = 1
	# stc[1][7] = 1
	# stc[2][17] = 1
	# stc[2][19] = 1
	# stc[3][0] = 1
	# stc[3][5] = 1

	car_clock = np.zeros(elevator_num)
	# for i in range(elevator_num):
	# 	car_clock[i] = 3

	## genetic pattern ###
	# rm = []
	# temp = [6, -1, [0], 0]
	# rm.append(temp)

	# temp = [8, 1, [15], 0]
	# rm.append(temp)

	# temp = [10, -1, [1], 0]
	# rm.append(temp)

	# temp = [11, 1, [19], 0]
	# rm.append(temp)

	# temp = [12, -1, [5], 0]
	# rm.append(temp)

	# temp = [14, -1, [8], 0]
	# rm.append(temp)


	## 宗翰pattern ###
	# rm = []
	# temp = [14, -1, [8], 0]
	# rm.append(temp)
	
	# temp = [12, -1, [5], 1]
	# rm.append(temp)

	# temp = [11, 1, [19], 2]
	# rm.append(temp)

	# temp = [10, -1, [1], 3]
	# rm.append(temp)

	# temp = [8, 1, [15], 4]
	# rm.append(temp)

	# temp = [6, -1, [0], 5]
	# rm.append(temp)

	# temp = [18, -1, [0, 5], 6]
	# rm.append(temp)

	# temp = [2, 1, [17, 19], 7]
	# rm.append(temp)

	# temp = [16, -1, [7], 8]
	# rm.append(temp)

	# temp = [4, 1, [6], 9]
	# rm.append(temp)

	
	### read rm from file ###
	pattern_gen = p_gen.workday_rm_generator()
	rm = []
	with open('pattern_dist.txt', 'r') as f:
		lines = f.readlines()

		for line in lines:
		    line = line.replace('\n', '')
		    a = line.split(' ')
		    temp = []
		    temp.append(int(a[0]))
		    temp.append(int(a[1]))
		    b = []
		    b.append(int(a[2]))
		    temp.append(b)
		    temp.append(int(a[3]))
		    rm.append(temp)

	rm_original = []
	with open('pattern_dist.txt', 'r') as f:
		lines = f.readlines()

		for line in lines:
		    line = line.replace('\n', '')
		    a = line.split(' ')
		    temp = []
		    temp.append(int(a[0]))
		    temp.append(int(a[1]))
		    b = []
		    b.append(int(a[2]))
		    temp.append(b)
		    temp.append(int(a[3]))
		    rm_original.append(temp)

	total_request_num = len(rm)
	waiting_time = np.zeros(total_request_num)
	travel_time = np.zeros(total_request_num)
	rm_catch = np.zeros(total_request_num)
	total_request_num_sum += total_request_num
	rm_finish_num = 0
	run = -1

	rm_id = 0 # current rm index
	waiting_rm_id = [] # id of rm who is still waiting
	travel_rm_id = []
	for i in range(elevator_num):
		temp = []
		travel_rm_id.append(temp)
	rm_id_time = []

#################### hall call & scheduler ####################
	# while rm_finish_num < total_request_num:
	while rm_id != total_request_num or len(waiting_rm_id) != 0 or len(travel_rm_id[0]) != 0 or len(travel_rm_id[1]) != 0:
		run += 1
		# print("run: " + str(run))
		
		for i in range(elevator_num):
			car_clock[i] -= 1
			if car_clock[i] == -1:
				car_clock[i] = 0
			if car_clock[i] == 0 and direction[i] == 1:
				position[i] += 1
				step_counter[i] += 1
			elif car_clock[i] == 0 and direction[i] == -1:
				position[i] -= 1
				step_counter[i] += 1

		### whether state change ? ###
		state_up = np.zeros(parameter_num)
		state_down = np.zeros(parameter_num)
		action_up = 0
		action_down = 0
		reward_up = 0
		reward_down = 0
		rescheduling_up = 0
		rescheduling_down = 0
		scheduling_trigger = 0
		
		
		# print("rm_id: " + str(rm_id))		
		### rm arrival ###
		while rm_id < total_request_num and run == rm[rm_id][3]:
			### rm waiting ###
			waiting_rm_id.append(rm_id)
			rm_id_time.append(rm[rm_id][3])

			### scheduling trigger ###
			if rm[rm_id][1] == 1: # rm up
				for i in range(elevator_num):
					if (position[i] < rm[rm_id][0] and direction[i] == 1) or (direction[i] == 1 and position[i] == rm[rm_id][0] and (car_clock[i] == 0 or car_clock[i] > transport_time)) or direction[i] == 0:
						rescheduling_up = 1
						break
			else: # rm down
				for i in range(elevator_num):
					if (position[i] > rm[rm_id][0] and direction[i] == -1) or (direction[i] == -1 and position[i] == rm[rm_id][0] and (car_clock[i] == 0 or car_clock[i] > transport_time)) or direction[i] == 0:
						rescheduling_down = 1
						break

			### lu, ld ###
			temp = [rm[rm_id][0], rm[rm_id][2]] #[fsm, strm]
			if rm[rm_id][1] == 1:
				insert_success = 0
				if len(lu) != 0:
					for k in range(len(lu)):
						if temp[0] < lu[k][0]:
							lu.insert(k, temp)
							insert_success = 1
							break
						elif temp[0] == lu[k][0]:
							for l in range(len(temp[1])):
								lu[k][1].append(temp[1][l])
							insert_success = 1
							break
					if insert_success != 1: #didn't insert successful
						lu.append(temp)
				else:
					lu.append(temp)
			else:
				insert_success = 0
				if len(ld) != 0:
					for k in range(len(ld)):
						if temp[0] > ld[k][0]:
							ld.insert(k, temp)
							insert_success = 1
							break
						elif temp[0] == ld[k][0]:
							for l in range(len(temp[1])):
								ld[k][1].append(temp[1][l])
							insert_success = 1
							break
					if insert_success != 1: #didn't insert successful
						ld.append(temp)
				else:
					ld.append(temp)
			rm_id += 1

		

		### dn change ###
		for i in range(elevator_num):
			if direction[i] == 0 and old_direction[i] != 0:
				scheduling_trigger = 1
				break

		old_lun = [[],[]]
		### dispatching ###
		if scheduling_trigger == 1 or rescheduling_up == 1:
			### clear lun, lunt ###
			for i in range(elevator_num):
				for j in range(len(lun[i])):
					old_lun[0].append(lun[i][0])
					old_lun[1].append(i)

					temp = [lun[i][0], lunt[i][0]]
					del lun[i][0]
					del lunt[i][0]

					insert_success = 0
					if len(lu) != 0:
						for k in range(len(lu)):
							if temp[0] < lu[k][0]:
								lu.insert(k, temp)
								insert_success = 1
								break
							elif temp[0] == lu[k][0]:
								for l in range(len(temp[1])):
									lu[k][1].append(temp[1][l])
								insert_success = 1
								break
						if insert_success != 1: #didn't insert successful
							lu.append(temp)
					else:
						lu.append(temp)


			### dispatching lu ###
			### position ###
			for i in range(elevator_num):
				state_up[i] = position[i]

			### car clock ###
			for i in range(elevator_num):
				state_up[elevator_num+i] = car_clock[i]

			lu_id = 0
			idle2move = 0
			rescheduling_up = 0
			while lu_id < len(lu):
				do = 0
				idle2move = 0
				for i in range(elevator_num):
					if (direction[i] == 1 and position[i] < lu[lu_id][0]) or (direction[i] == 1 and position[i] == lu[lu_id][0] and (car_clock[i] == 0 or car_clock[i] > transport_time)) or direction[i] == 0:
						do = 1
						break

				if do == 0:
					lu_id += 1
				elif do == 1:
					rescheduling_up = 1
					### target floor ###
					for i in range(elevator_num):
						for j in range(floor):
							state_up[elevator_num+elevator_num+(floor*elevator_num)+i*floor+j] = stc[i][j]

					### lun ###
					for i in range(elevator_num):
						for j in range(len(lun[i])):
							state_up[elevator_num+elevator_num+(floor*i)+lun[i][j]] = 1

					### hall call ###
					state_up[parameter_num-1] = lu[lu_id][0]
					action_up = RL.choose_action(state_up, direction, run, lu, lun, ld, ldn)
print(action_up)