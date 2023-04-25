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
total_request_num = 875
transport_time = 5
open_time = 11

record_file = 0
restore = 1
save = 0
max_iteration = 100

parameter_num = elevator_num + elevator_num + floor*elevator_num + floor*elevator_num + 1

RL = DeepQNetwork(    elevator_num, #action 
					  parameter_num, #input number
                      learning_rate=0.00025,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10000,
                      memory_size=100000,
                      double_q=False,
                      save_model = save,
                      restore_model = restore
                      # output_graph=True
                      )

step_counter = np.zeros(elevator_num)
avg_waiting_time = 0
waiting_time = np.zeros(total_request_num)
total_waiting_time = 1000000
rm_catch = np.zeros(total_request_num)
shortest_rm_catch = np.zeros(total_request_num)
shortest_waiting_time = np.zeros(total_request_num)
shortest_total_waiting_time = 1000000
total_waiting_time_his = []

while iteration < max_iteration:
# while iteration < max_iteration:
	iteration += 1
	print("iteration: " + str(iteration))
	
	if record_file == 1 and iteration == max_iteration:
	# if record_file == 1:
		f = open("trake.txt", 'a')
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
	for i in range(total_request_num):
		waiting_time[i] = 0

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

	pattern_gen = p_gen.workday_rm_generator()
	### read rm from file ###
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

	rm_finish_num = 0
	run = -1

	rm_id = 0 # current rm index
	waiting_rm_id = [] # id of rm who is still waiting

#################### hall call & scheduler ####################
	# while rm_finish_num < total_request_num:
	while rm_id != total_request_num or len(waiting_rm_id) != 0:
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
		redispatching_up = 0
		redispatching_down = 0
		scheduling_trigger = 0
		
		
		# print("rm_id: " + str(rm_id))		
		### rm arrival ###
		while rm_id < total_request_num and run == rm[rm_id][3]:
			### rm waiting ###
			waiting_rm_id.append(rm_id)

			### scheduling trigger ###
			if rm[rm_id][1] == 1: # rm up
				for i in range(elevator_num):
					if (direction[i] == 1 and position[i] < rm[rm_id][0]) or (direction[i] == 1 and position[i] == rm[rm_id][0] and car_clock[i] == 0) or direction[i] == 0:
						redispatching_up = 1
						break
			else: # rm down
				for i in range(elevator_num):
					if (direction[i] == -1 and position[i] > rm[rm_id][0]) or (direction[i] == -1 and position[i] == rm[rm_id][0] and car_clock[i] == 0) or direction[i] == 0:
						redispatching_down = 1
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

		### dispatching ###
		if scheduling_trigger == 1 or redispatching_up == 1:
			### clear lun, lunt ###
			for i in range(elevator_num):
				for j in range(len(lun[i])):
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

			### target floor ###
			for i in range(elevator_num):
				for j in range(floor):
					state_up[elevator_num+elevator_num+(floor*elevator_num)+i*floor+j] = stc[i][j]

			lu_id = 0
			idle2move = 0
			redispatching_up = 0
			while lu_id < len(lu):
				do = 0
				idle2move = 0
				for i in range(elevator_num):
					if (direction[i] == 1 and position[i] < lu[lu_id][0]) or (direction[i] == 1 and position[i] == lu[lu_id][0] and car_clock[i] == 0) or direction[i] == 0:
						redispatching_up = 1
						do = 1
						break

				if do == 0:
					lu_id += 1
				elif do == 1:
					### lun ###
					for i in range(elevator_num):
						for j in range(len(lun[i])):
							state_up[elevator_num+elevator_num+(floor*i)+lun[i][j]] = 1

					### hall call ###
					state_up[parameter_num-1] = lu[lu_id][0]
					action_up = RL.choose_action(state_up, direction)
					
					if direction[action_up] != 0:
						idle2move = 0

					lun[action_up].append(lu[lu_id][0])
					lunt[action_up].append(lu[lu_id][1])

					if direction[action_up] == 0:
						idle2move = 1
						if position[action_up] < lu[lu_id][0]:
							direction[action_up] = 1
							car_clock[action_up] += transport_time
						elif position[action_up] == lu[lu_id][0]:
							direction[action_up] = 1
							car_clock[action_up] = transport_time+open_time

							### waiting time ###
							pointer = 0
							while pointer < len(waiting_rm_id):
								if rm[waiting_rm_id[pointer]][0] == position[action_up] and rm[waiting_rm_id[pointer]][1] == 1:
									rm_catch[waiting_rm_id[pointer]] = action_up
									del waiting_rm_id[pointer]
								else:
									pointer += 1
							del lun[action_up][-1]

							### catch rm ###
							temp = lunt[action_up][-1] # stc can be set 
							del lunt[action_up][-1]
							for k in range(len(temp)):
								stc[action_up][temp[k]] = 1
						elif position[action_up] > lu[lu_id][0]:
							direction[action_up] = -1
							car_clock[action_up] += transport_time
					del lu[lu_id]
				
			### reward ###
			if redispatching_up == 1:
				if len(lun[action_up]) > 0:
					fsm = lun[action_up][-1]
					if idle2move == 1: #idle car
						if fsm > position[action_up]:
							# direction[action_up] = 1
							floor_distance = fsm - position[action_up]	
							reward_up = transport_time*floor_distance + car_clock[action_up]
						elif fsm < position[action_up]:
							# direction[action_up] = -1
							floor_distance = position[action_up] - fsm
							reward_up = transport_time*floor_distance + car_clock[action_up]
						else:
							reward_up = 0
					elif direction[action_up] == 1: #car up
						if fsm > position[action_up]:
							floor_distance = fsm - position[action_up] #樓層差
										
							stop_num = 0 #停止次數
							for i in range(len(lun[action_up])):
								if lun[action_up][i] > position[action_up] and lun[action_up][i] < fsm:
									stop_num += 1

							for i in range(int(position[action_up]), int(fsm)):
								stop_num += stc[action_up][i]
							# print("floor_distance: " + str(floor_distance))
							# print("stop_num: " + str(stop_num))
											
							reward_up = floor_distance*transport_time + stop_num*open_time + car_clock[action_up]
						# elif fsm < position[action_up]:
						# 	floor_distance = floor-1 + (floor-1-position[action_up]) + fsm
											
						# 	stop_num = 0
						# 	for i in range(len(lun[action_up])):
						# 		if lun[action_up][i] > position[action_up]:
						# 			stop_num += 2
						# 		elif lun[action_up][i] < fsm:
						# 			stop_num += 1
									
						# 	for i in range(len(ldn[action_up])):
						# 		stop_num += 2

						# 	for i in range(floor):
						# 		stop_num += stc[action_up][i]
						# 	# print("floor_distance: " + str(floor_distance))
						# 	# print("stop_num: " + str(stop_num))
										
						# 	reward_up = floor_distance*transport_time + stop_num*open_time
						elif fsm == position[action_up] and car_clock[action_up] == 0:
							reward_up = 0 + car_clock[action_up]
					# elif direction[action_up] == -1: #car down
					# 	if fsm > position[action_up]:
					# 		floor_distance = position[action_up] + fsm
										
					# 		stop_num = 0
					# 		for i in range(len(ldn[action_up])):
					# 			if ldn[action_up][i] < position[action_up]:
					# 				stop_num += 2
					# 		for i in range(len(lun[action_up])):
					# 			if lun[action_up][i] < fsm:
					# 				stop_num += 1

					# 		for i in range(int(fsm)):
					# 			stop_num += stc[action_up][i]
					# 		# print("floor_distance: " + str(floor_distance))
					# 		# print("stop_num: " + str(stop_num))
							
					# 		reward_up = floor_distance*transport_time + stop_num*open_time
					# 	elif fsm < position[action_up]:
					# 		floor_distance = position[action_up] + fsm
											
					# 		stop_num = 0
					# 		for i in range(len(ldn[action_up])):
					# 			if ldn[action_up][i] < position[action_up]:
					# 				stop_num += 2
					# 		for i in range(len(lun[action_up])):
					# 			if lun[action_up][i] < fsm:
					# 				stop_num += 1

					# 		for i in range(int(position[action_up])):
					# 			stop_num += stc[action_up][i]
					# 		# print("floor_distance: " + str(floor_distance))
					# 		# print("stop_num: " + str(stop_num))
											
					# 		reward_up = floor_distance*transport_time + stop_num*open_time
					# 	else:
					# 		floor_distance = position[action_up] + fsm
											
					# 		stop_num = 0
					# 		for i in range(len(ldn[action_up])):
					# 			if ldn[action_up][i] < position[action_up]:
					# 				stop_num += 2
					# 		for i in range(len(lun[action_up])):
					# 			if lun[action_up][i] < fsm:
					# 				stop_num += 1

					# 		for i in range(int(position[action_up])):
					# 			stop_num += stc[action_up][i]
					# 		# print("floor_distance: " + str(floor_distance))
					# 		# print("stop_num: " + str(stop_num))
											
					# 		reward_up = floor_distance*transport_time + stop_num*open_time
				else:
					reward_up = 0
					

		if scheduling_trigger == 1 or redispatching_down == 1:
			### clear ldn, ldnt ###
			for i in range(elevator_num):
				for j in range(len(ldn[i])):
					temp = [ldn[i][0], ldnt[i][0]]
					del ldn[i][0]
					del ldnt[i][0]

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

			### position ###
			for i in range(elevator_num):
				state_down[i] = position[i]

			### car clock ###
			for i in range(elevator_num):
				state_down[elevator_num+i] = car_clock[i]

			### lun ###
			for i in range(elevator_num):
				for j in range(len(lun[i])):
					state_down[elevator_num+elevator_num+(floor*i)+lun[i][j]] = 1

			### target floor ###
			for i in range(elevator_num):
				for j in range(floor):
					state_down[elevator_num+elevator_num+(floor*elevator_num)+i*floor+j] = stc[i][j]

			ld_id = 0
			idle2move = 0
			redispatching_down = 0
			while ld_id < len(ld):
				do = 0
				for i in range(elevator_num):
					if (direction[i] == -1 and position[i] > ld[ld_id][0]) or (direction[i] == -1 and position[i] == ld[ld_id][0] and car_clock[i] == 0) or direction[i] == 0:
						do = 1
						break
				if do == 0:
					ld_id += 1
				elif do == 1:
					redispatching_down = 1
				
					### ldn ###
					for i in range(elevator_num):
						for j in range(len(ldn[i])):
							if state_down[elevator_num+elevator_num+(floor*i)+ldn[i][j]] == 0:
								state_down[elevator_num+elevator_num+(floor*i)+ldn[i][j]] = 2
							elif state_down[elevator_num+elevator_num+(floor*i)+ldn[i][j]] == 1:
								state_down[elevator_num+elevator_num+(floor*i)+ldn[i][j]] = 3

					### hall call ###
					state_down[parameter_num-1] = (-1)*ld[ld_id][0]
					action_down = RL.choose_action(state_down, direction)
					if direction[action_down] != 0:
						idle2move = 0

					ldn[action_down].append(ld[ld_id][0])
					ldnt[action_down].append(ld[ld_id][1])
					if direction[action_down] == 0:
						idle2move = 1
						if position[action_down] < ld[ld_id][0]:
							direction[action_down] = 1
							car_clock[action_down] += transport_time
						elif position[action_down] == ld[ld_id][0]:
							direction[action_down] = -1
							car_clock[action_down] = transport_time+open_time

							### waiting time ###
							pointer = 0
							while pointer < len(waiting_rm_id):
								if rm[waiting_rm_id[pointer]][0] == position[action_down] and rm[waiting_rm_id[pointer]][1] == -1:
									rm_catch[waiting_rm_id[pointer]] = action_down
									del waiting_rm_id[pointer]
								else:
									pointer += 1
							del ldn[action_down][-1]

							### catch rm ###
							temp = ldnt[action_down][-1] # stc can be set 
							del ldnt[action_down][-1]
							for k in range(len(temp)):
								stc[action_down][temp[k]] = 1
						elif position[action_down] > ld[ld_id][0]:
							direction[action_down] = -1
							car_clock[action_down] += transport_time
					del ld[ld_id]

			### reward ###
			if redispatching_down == 1:
				if len(ldn[action_down]) > 0:
					fsm = ldn[action_down][-1]
					if idle2move == 1: #idle car
						if fsm > position[action_down]:
							# direction[action_down] = 1
							floor_distance = fsm - position[action_down]

							reward_down = transport_time*floor_distance + car_clock[action_down]
						elif fsm < position[action_down]:
							# direction[action_down] = -1
							floor_distance = position[action_down] - fsm

							reward_down = transport_time*floor_distance + car_clock[action_down]
						else:
							# direction[action_down] = -1
							reward_down = 0
					# elif direction[action_down] == 1: #car up
					# 	if fsm > position[action_down]:
					# 		floor_distance = (floor-1-position[action_down]) + (floor-1-fsm)
										
					# 		stop_num = 0
					# 		for i in range(len(lun[action_down])):
					# 			if lun[action_down][i] > position[action_down]:
					# 				stop_num += 2
					# 		for i in range(len(ldn[action_down])):
					# 			if ldn[action_down][i] > fsm:
					# 				stop_num += 1

					# 		for i in range(int(position[action_down]), floor):
					# 			stop_num += stc[action_down][i]
					# 		# print("floor_distance: " + str(floor_distance))
					# 		# print("stop_num: " + str(stop_num))

					# 		reward_down = floor_distance*transport_time + stop_num*open_time
					# 	elif fsm < position[action_down]:
					# 		floor_distance = (floor-1-position[action_down]) + (floor-1-fsm)
										
					# 		stop_num = 0
					# 		for i in range(len(lun[action_down])):
					# 			if lun[action_down][i] > position[action_down]:
					# 				stop_num += 2
					# 		for i in range(len(ldn[action_down])):
					# 			if ldn[action_down][i] > fsm:
					# 				stop_num += 1

					# 		for i in range(int(fsm), floor):
					# 			stop_num += stc[action_down][i]
					# 		# print("floor_distance: " + str(floor_distance))
					# 		# print("stop_num: " + str(stop_num))
										
					# 		reward_down = floor_distance*transport_time + stop_num*open_time
					# 	else:
					# 		floor_distance = (floor-1-position[action_down]) + (floor-1-fsm)
										
					# 		stop_num = 0
					# 		for i in range(len(lun[action_down])):
					# 			if lun[action_down][i] > position[action_down]:
					# 				stop_num += 2
					# 		for i in range(len(ldn[action_down])):
					# 			if ldn[action_down][i] > fsm:
					# 				stop_num += 1

					# 		for i in range(int(position[action_down]), floor):
					# 			stop_num += stc[action_down][i]
					# 		# print("floor_distance: " + str(floor_distance))
					# 		# print("stop_num: " + str(stop_num))
										
					# 		reward_down = floor_distance*transport_time + stop_num*open_time
					elif direction[action_down] == -1: #car down
						# if fsm > position[action_down]:
						# 	floor_distance = floor-1 + position[action_down] + (floor-1-fsm)
										
						# 	stop_num = 0
						# 	for i in range(len(ldn[action_down])):
						# 		if ldn[action_down][i] < position[action_down]:
						# 			stop_num += 2
						# 		elif ldn[action_down][i] > fsm:
						# 			stop_num += 1
						# 	for i in range(len(lun[action_down])):
						# 		stop_num += 2

						# 	for i in range(floor):
						# 		stop_num += stc[action_down][i]
						# 	# print("floor_distance: " + str(floor_distance))
						# 	# print("stop_num: " + str(stop_num))
										
						# 	reward_down = floor_distance*transport_time + stop_num*open_time
						if fsm < position[action_down]:
							floor_distance = position[action_down] - fsm
										
							stop_num = 0
							for i in range(len(ldn[action_down])):
								if ldn[action_down][i] < position[action_down] and ldn[action_down][i] > fsm:
									stop_num += 1

							for i in range(int(fsm), int(position[action_down])):
								stop_num += stc[action_down][i]
							# print("floor_distance: " + str(floor_distance))
							# print("stop_num: " + str(stop_num))
										
							reward_down = floor_distance*transport_time + stop_num*open_time + car_clock[action_down]
						elif fsm == position[action_down] and car_clock[action_down] == 0:
							reward_down = 0 + car_clock[action_down]
				else:
					reward_down = 0
				
		# print("run: " + str(run))
		# print("lu: " + str(lu))
		# print("ld: " + str(ld))
		# print("lun: " + str(lun))
		# print("ldn: " + str(ldn))
		# print("")
		
		for i in range(elevator_num):
			old_direction[i] = direction[i]
		
#################### action execution & record ####################
		for i in range(elevator_num):
			if car_clock[i] == -1 or car_clock[i] == 0:
				car_clock[i] = 0

				### car up ###
				if direction[i] == 1:
					
					# position[i] += 1
					direction[i] = 0

					### reach rm floor ###
					if position[i] in lun[i]:
						# rm_finish_num += 1
						direction[i] = 1
						index = lun[i].index(position[i])

						### waiting time ###
						pointer = 0
						while pointer < len(waiting_rm_id):
							if rm[waiting_rm_id[pointer]][0] == position[i] and rm[waiting_rm_id[pointer]][1] == 1:
								rm_catch[waiting_rm_id[pointer]] = i
								del waiting_rm_id[pointer]
							else:
								pointer += 1

						### catch rm ###
						del lun[i][index]

						temp = lunt[i][index] # stc can be a set of numbers
						del lunt[i][index]
						for k in range(len(temp)):
							stc[i][temp[k]] = 1
							
						car_clock[i] = transport_time+open_time
					
					### reach target floor ###
					if stc[i][position[i]] == 1:
						stc[i][position[i]] = 0

						if car_clock[i] < (transport_time+open_time): # may catch rm and reach target floor
							car_clock[i] = open_time

					### direction decision ###
					### 電梯裡還有人要往上 ###
					target_num = 0
					for j in range(floor):
						target_num += stc[i][j]
					if target_num != 0 or direction[i] == 1:
						direction[i] = 1
						if car_clock[i] < (transport_time+open_time):
							car_clock[i] += transport_time
					else:
						### 上面還有我的lun ###
						max_lun = -1
						if len(lun[i]) > 0:
							max_lun = max(lun[i])
						if max_lun > position[i]:
							direction[i] = 1
							car_clock[i] += transport_time
						else:
							### 上面還有我的ldn ###
							max_ldn = -1
							if len(ldn[i]) > 0:
								max_ldn = max(ldn[i])
							if max_ldn > position[i]:
								direction[i] = 1
								car_clock[i] += transport_time
							else:
								direction[i] = 0
							# ### 以下考慮轉成下樓電梯 ###
							# ### 這層樓是我的ldn ###
							# elif max_ldn == position[i]:
							# 	direction[i] = -1
							# 	# rm_finish_num += 1
							# 	index = ldn[i].index(position[i])

							# 	### waiting time ###
							# 	pointer = 0
							# 	length = len(waiting_rm_id)
							# 	while pointer < length:
							# 		if rm[waiting_rm_id[pointer]][0] == position[i] and rm[waiting_rm_id[pointer]][1] == -1:
							# 			rm_catch[waiting_rm_id[pointer]] = i
							# 			del waiting_rm_id[pointer]
							# 		else:
							# 			pointer += 1
							# 		length = len(waiting_rm_id)

							# 	### catch rm ###
							# 	del ldn[i][index]

							# 	temp = ldnt[i][index]
							# 	del ldnt[i][index]
							# 	for j in range(len(temp)):
							# 		stc[i][temp[j]] = 1
									
							# 	car_clock[i] = 9
							# ### 這層樓是他人的ldn ###
							# else:
							# 	for j in range(elevator_num):
							# 		if position[i] in ldn[j]:
							# 			if i != j:
							# 				grab = 1
							# 			# rm_finish_num += 1
							# 			direction[i] = -1
							# 			index = ldn[j].index(position[i])

							# 			### waiting time ###
							# 			pointer = 0
							# 			length = len(waiting_rm_id)
							# 			while pointer < length:
							# 				if rm[waiting_rm_id[pointer]][0] == position[i] and rm[waiting_rm_id[pointer]][1] == -1:
							# 					rm_catch[waiting_rm_id[pointer]] = i
							# 					del waiting_rm_id[pointer]
							# 				else:
							# 					pointer += 1
							# 				length = len(waiting_rm_id)

							# 			### catch rm ###
							# 			del ldn[j][index]

							# 			temp = ldnt[j][index] # stc can be a set of numbers
							# 			del ldnt[j][index]
							# 			for k in range(len(temp)):
							# 				stc[i][temp[k]] = 1
										
							# 			car_clock[i] = 9
								
							# 	if direction[i] == 0:
							# 		### 上面沒有任務了, 但下面有我的ldn或lun ###
							# 		if len(ldn[i]) > 0 or len(lun[i]) > 0:
							# 			direction[i] = -1
							# 			car_clock[i] += 2
				### car down ###
				elif direction[i] == -1:
					# position[i] -= 1
					direction[i] = 0

					### reach rm floor ###
					if position[i] in ldn[i]:
						# rm_finish_num += 1
						direction[i] = -1
						index = ldn[i].index(position[i])

						### waiting time ###
						pointer = 0
						while pointer < len(waiting_rm_id):
							if rm[waiting_rm_id[pointer]][0] == position[i] and rm[waiting_rm_id[pointer]][1] == -1:
								rm_catch[waiting_rm_id[pointer]] = i
								del waiting_rm_id[pointer]
							else:
								pointer += 1

						### catch rm ###
						del ldn[i][index]

						temp = ldnt[i][index] # stc can be a set of numbers
						del ldnt[i][index]
						for k in range(len(temp)):
							stc[i][temp[k]] = 1
							
						car_clock[i] = transport_time+open_time
					
					### reach target floor ###
					if stc[i][position[i]] == 1:
						stc[i][position[i]] = 0
						
						if car_clock[i] < (transport_time+open_time): # may catch rm and reach target floor
							car_clock[i] = open_time

					### direction decision ###
					### 電梯裡還有人要往下 ###
					target_num = 0
					for j in range(floor):
						target_num += stc[i][j]
					if target_num != 0 or direction[i] == -1:
						direction[i] = -1
						if car_clock[i] < (transport_time+open_time):
							car_clock[i] += transport_time
					else:
						### 下面還有我的ldn ###
						min_ldn = floor
						if len(ldn[i]) > 0:
							min_ldn = min(ldn[i])
						if min_ldn < position[i]:
							direction[i] = -1
							car_clock[i] += transport_time
						else:
							### 下面還有我的lun ###
							min_lun = floor
							if len(lun[i]) > 0:
								min_lun = min(lun[i])
							if min_lun < position[i]:
								direction[i] = -1
								car_clock[i] += transport_time
							else:
								direction[i] = 0
							# ### 以下考慮轉成上樓電梯 ###
							# ### 這層樓是我的lun ###
							# elif min_lun == position[i]:
							# 	direction[i] = 1
							# 	# rm_finish_num += 1
							# 	index = lun[i].index(position[i])

							# 	### waiting time ###
							# 	pointer = 0
							# 	length = len(waiting_rm_id)
							# 	while pointer < length:
							# 		if rm[waiting_rm_id[pointer]][0] == position[i] and rm[waiting_rm_id[pointer]][1] == 1:
							# 			rm_catch[waiting_rm_id[pointer]] = i
							# 			del waiting_rm_id[pointer]
							# 		else:
							# 			pointer += 1
							# 		length = len(waiting_rm_id)

							# 	### catch rm ###
							# 	del lun[i][index]

							# 	temp = lunt[i][index]
							# 	del lunt[i][index]
							# 	for j in range(len(temp)):
							# 		stc[i][temp[j]] = 1
									
							# 	car_clock[i] = 9
							# ### 這層樓是他人的lun ###
							# else:
							# 	for j in range(elevator_num):
							# 		if position[i] in lun[j]:
							# 			if i != j:
							# 				grab = 1
							# 			# rm_finish_num += 1
							# 			direction[i] = 1
							# 			index = lun[j].index(position[i])

							# 			### waiting time ###
							# 			pointer = 0
							# 			length = len(waiting_rm_id)
							# 			while pointer < length:
							# 				if rm[waiting_rm_id[pointer]][0] == position[i] and rm[waiting_rm_id[pointer]][1] == 1:
							# 					rm_catch[waiting_rm_id[pointer]] = i
							# 					del waiting_rm_id[pointer]
							# 				else:
							# 					pointer += 1
							# 				length = len(waiting_rm_id)

							# 			### catch rm ###
							# 			del lun[j][index]

							# 			temp = lunt[j][index] # stc can be a set of numbers
							# 			del lunt[j][index]
							# 			for k in range(len(temp)):
							# 				stc[i][temp[k]] = 1
										
							# 			car_clock[i] = 9
								
							# 	if direction[i] == 0:
							# 		### 下面沒有任務了, 但上面有我的ldn或lun ###
							# 		if len(ldn[i]) > 0 or len(lun[i]) > 0:
							# 			direction[i] = 1
							# 			car_clock[i] += 2

			position_his[i].append(position[i])

		### waiting time ###
		for i in range(len(waiting_rm_id)):
			waiting_time[waiting_rm_id[i]] += 1

		### next state ###
		if scheduling_trigger == 1 or redispatching_up == 1:
			### next state ###
			state_next = np.zeros(parameter_num)
				
			### position ###
			for i in range(elevator_num):
				if direction[i] == 1:
					state_next[i] = position[i] + 1
				elif direction[i] == -1:
					state_next[i] = position[i] - 1
				else:
					state_next[i] = position[i]

			### car clock ###
			for i in range(elevator_num):
				state_next[i+elevator_num] = car_clock[i]

			### lun ###:
			for i in range(elevator_num):
				for j in range(len(lun[i])):
					state_next[elevator_num+elevator_num+(floor*i)+lun[i][j]] += 1

			### target floor ###
			for i in range(elevator_num):
				for j in range(floor):
					state_next[elevator_num+elevator_num+(floor*elevator_num)+i*floor+j] = stc[i][j]

			RL.store_transition(state_up, action_up, reward_up, state_next)

		if scheduling_trigger == 1 or redispatching_down == 1:
			### next state ###
			state_next = np.zeros(parameter_num)

			### position ###
			for i in range(elevator_num):
				if direction[i] == 1:
					state_next[i] = position[i] + 1
				elif direction[i] == -1:
					state_next[i] = position[i] - 1
				else:
					state_next[i] = position[i]

			### car clock ###
			for i in range(elevator_num):
				state_next[i+elevator_num] = car_clock[i]

			### lun ###:
			for i in range(elevator_num):
				for j in range(len(lun[i])):
					state_next[elevator_num+elevator_num+(floor*i)+lun[i][j]] += 1

			### ldn ###
			for i in range(elevator_num):
				for j in range(len(ldn[i])):
					state_next[elevator_num+elevator_num+(floor*i)+ldn[i][j]] += 2

			### target floor ###
			for i in range(elevator_num):
				for j in range(floor):
					state_next[elevator_num+elevator_num+(floor*elevator_num)+i*floor+j] = stc[i][j]

			RL.store_transition(state_down, action_down, reward_down, state_next)
			
		if (run > 200 and run % 5 == 0) and (restore == 0):
			RL.learn()

		if record_file == 1 and iteration == max_iteration:
		# if record_file == 1:
			f = open("trake.txt", 'a')
			f.write("run: " + str(run) + '\n')
			f.write("lun: " + str(lun) + '\n')
			f.write("lunt: " + str(lunt) + '\n')
			f.write("ldn: " + str(ldn) + '\n')
			f.write("ldnt: " + str(ldnt) + '\n')
			f.write("position: " + str(position) + '\n')
			f.write("direction: " + str(direction) + '\n')
			f.write("target floor1: " + str(stc[0]) + '\n')
			f.write("target floor2: " + str(stc[1]) + '\n')
			f.write("target floor3: " + str(stc[2]) + '\n')
			f.write("target floor4: " + str(stc[3]) + '\n\n')
			f.close()

		# print("run: " + str(run))
		# print("waiting_rm_id: " + str(waiting_rm_id))
		# print("lun: " + str(lun))
		# print("lunt: " + str(lunt))
		# print("ldn: " + str(ldn))
		# print("ldnt: " + str(ldnt))
		# print("waiting_time:" + str(waiting_time))
		# print("position: " + str(position))
		# print("direction: " + str(direction))
		# print("car_clock: " + str(car_clock))
		# print("target floor1: " + str(stc[0]))
		# print("target floor2: " + str(stc[1]))
		# print("")
		# print("target floor3: " + str(stc[2]))
		# print("target floor4: " + str(stc[3]) + '\n')
		# print("reward: " + str(reward))
		# os.system("pause")
		# print("rm_finish_num: " + str(rm_finish_num))
	# if save == 1:
	# 	RL.save_this_model()

	total_waiting_time = 0
	for i in range(total_request_num):
		total_waiting_time += waiting_time[i]
	avg_waiting_time += total_waiting_time
	
	if record_file == 1 and iteration == max_iteration:
	# if record_file == 1:
		f = open("trake.txt", 'a')
		f.write("waiting_time: " + str(waiting_time) + '\n')
		f.write("total_waiting_time: " + str(total_waiting_time) + '\n')
		f.write("rm_catch: " + str(rm_catch) + '\n')
		f.write("shortest_waiting_time: " + str(shortest_waiting_time) + '\n')
		f.write("shortest_total_waiting_time: " + str(shortest_total_waiting_time) + '\n')
		f.write("shortest_rm_catch: " + str(shortest_rm_catch) + '\n')
		f.write('\n')
		f.close()

	### record the best experience ###
	if total_waiting_time < shortest_total_waiting_time:
		shortest_total_waiting_time = total_waiting_time
		# for i in range(total_request_num):
		# 	shortest_rm_catch[i] = rm_catch[i]
		# 	shortest_waiting_time[i] = waiting_time[i]

	print("position: " + str(position))
	print("step_counter: " + str(step_counter/iteration))
	# print("waiting_time: " + str(waiting_time))
	print("avg_waiting_time: " + str(avg_waiting_time/(total_request_num*iteration)))
	print("total_waiting_time: " + str(total_waiting_time))
	# print("rm_catch: " + str(rm_catch))
	# print("shortest_waiting_time: " + str(shortest_waiting_time))
	print("shortest_total_waiting_time: " + str(shortest_total_waiting_time))
	# print("shortest_rm_catch: " + str(shortest_rm_catch))
	print("")
	total_waiting_time_his.append(total_waiting_time)
		# os.system("pause")

	if save == 1 and iteration > 1 and total_waiting_time == shortest_total_waiting_time:
		RL.save_this_model()
				
	# break
	# os.system("pause")
if save == 1:
	# RL.save_this_model()
	RL.plot_cost()
	RL.plot_q()

plt.plot(np.arange(len(total_waiting_time_his)), total_waiting_time_his)
plt.annotate(str(total_waiting_time_his[len(total_waiting_time_his)-1]), xy=(max_iteration, total_waiting_time_his[len(total_waiting_time_his)-1]+10), xytext=(max_iteration, total_waiting_time_his[len(total_waiting_time_his)-1]+50),arrowprops=dict(facecolor='black', shrink=0.05))
plt.ylabel('total_waiting_time')
plt.xlabel('iterations')
plt.show()

plt.plot(np.arange(len(position_his[0])), position_his[0], 'bo:', label = 'car1')
plt.plot(np.arange(len(position_his[1])), position_his[1], 'cs:', label = 'car2')
# plt.plot(np.arange(len(position_his[2])), position_his[2], 'gp:', label = 'car3')
# plt.plot(np.arange(len(position_his[3])), position_his[3], 'r*:', label = 'car4')
plt.legend(loc = 'best')
plt.ylabel('position')
plt.xlabel('run')
plt.show()