import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

#plots the data structure that comes out of the parameter
#sweep
def plot(points_tuple,
             title,
             xaxis,
             yaxis,
             legend=[],
             loc = 'upper right',
             filename="output.png",
             ymin=0,
             ymax=100,
             xlim=0):

	rcParams.update({'figure.autolayout': True})
	rcParams.update({'font.size': 16})
	#fprop = font_manager.FontProperties(fname='/Library/Fonts/Microsoft/Gill Sans MT.ttf') 

	plt.figure() 
	colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
	shape = ['s-', 'o-', '^-', 'v-', 'x-', 'h-']

	X = points_tuple[0]
	Y = points_tuple[1]
	i = 0

	for ya in Y:
		print(ya)
		plt.plot(X, ya, shape[i], linewidth=2.5,markersize=7,color=colors[i])
		i += 1

	plt.legend(legend,loc=loc, prop={"size": 13})
	plt.title(title)
	plt.xlabel(xaxis)
	plt.ylabel(yaxis)
	plt.ylim(ymin=ymin, ymax=ymax) 
	plt.xlim(xmin=xlim, xmax=X[len(X)-1])
	plt.grid(True)
	plt.savefig(filename)


def get_experiment_values(csvFilePath, propertyName):
	with open(csvFilePath, 'r') as csvFile:
		dataReader = csv.reader(csvFile, delimiter=',', quotechar='|')
		i = 0
		propertyIndex = 0
		x = []
		y = []
		for row in dataReader:
			if (i == 0):
				propertyIndex = row.index(propertyName)
			else:
				x.append(i)
				y.append(float(row[propertyIndex]))
			i += 1

		return x, y

def get_experiment_values_averaged(csv_file_paths, property_name):
	vals = []
	x = []
	file_num = 0
	for file_path in csv_file_paths:
		x, y = get_experiment_values(file_path, property_name)
		if file_num == 0:
			vals = y
		else:
			for i in range(0, len(vals)):
				vals[i] += y[i]
		file_num += 1
	return x, [v / file_num for v in vals]

def get_performance_value(csv_file_path):
	vals = []
	x = []
	print(csv_file_path)
	iterations, rewards = get_experiment_values(csv_file_path, "AverageReturn")
	return sum(rewards) / len(iterations)

def plot_performance_values(csv_file_base, policies, noise, parameters, ymin, ymax, xLabel, yLabel="Performance"):
	filename = "/progress.csv"
	performances = []
	title = "Algorithm Performance as a function of " + noise + " noise"
	for policy in policies:
		performance = []
		for param in parameters:
			progressFile = csv_file_base + noise + "_" + policy + "_" + str(round(param,3)) + filename
			perf = get_performance_value(progressFile)
			performance.append(perf)
		performances.append(performance)

	graphFileName = csv_file_base + "Performance_" + noise + ".png"
	rounded = [round(x, 3) for x in parameters]
	plot((rounded, performances), title, xLabel, yLabel, legend=policies, ymin=ymin, ymax=ymax, filename=graphFileName)

def plot_performance_median_trpo_vpg(csv_trpos, csv_vpgs, parameters, policies, noise, ymin, ymax, xLabel, yLabel="Performance"):
	filename = "/progress.csv"
	trpo_performances = []
	vpg_performances = []
	title = "Algorithm Performance"
	for policy in policies:
		trpo_performance = []
		vpg_performance = []
		for param in parameters:
			trpo_param = []
			for csv_trpo in csv_trpos:
				trpo_progress = "/home/tejas/Documents/rllab/data/local/experiment/lunar_lander_trpo_v" + str(csv_trpo) + "/" + noise + "_GaussianMLP_32_32_" + str(round(param, 3)) + filename
				trpo_perf = get_performance_value(trpo_progress)
				trpo_param.append(trpo_perf)
			trpo_val = np.median(np.array(trpo_param))
			trpo_performance.append(trpo_val)

			vpg_param = []
			for csv_vpg in csv_vpgs:
				vpg_progress = "/home/tejas/Documents/rllab/data/local/experiment/lunar_lander_v" + str(csv_vpg) + "/" + noise + "_GaussianMLP_32_32_" + str(round(param, 3)) + filename
				print(vpg_progress)
				vpg_perf = get_performance_value(vpg_progress)
				vpg_param.append(vpg_perf)
			vpg_val = np.median(np.array(vpg_param))
			vpg_performance.append(vpg_val)

		trpo_performances.append(trpo_performance)
		vpg_performances.append(vpg_performance)
	graphFileName = "/home/tejas/Documents/rllab/data/local/experiment/TRPO_Median_Performance_" + noise + ".png"
	rounded = [round(x, 3) for x in parameters]
	plot((rounded, [trpo_performance, vpg_performance]), title, xLabel, yLabel, legend=["MLP", "GRU"], ymin=ymin, ymax=ymax, filename=graphFileName)
	#graphFileName = "/home/tejas/Documents/rllab/data/local/experiment/VPG_Performance_" + noise + ".png"
	#plot((rounded, vpg_performances), title, xLabel, yLabel, legend=["VPG, Gaussian MLP"], ymin=ymin, ymax=ymax, filename=graphFileName)

def plot_performance_values_multiple(folders, graph_name, parameters, policies, noise, ymin, ymax, series, xLabel, yLabel="Performance Metric Value"):
	filename = "/progress.csv"
	performances = []
	for _ in range(0, len(folders)):
		performances.append([])

	#title = "Algorithm Performance as " + noise + " Noise Increases" 
	title = "Performance with Gaussian Noise"
	for param in parameters:
		for csv_num in range(0, len(folders)):
			progress = "/home/tejas/Documents/rllab/data/local/experiment/" + folders[csv_num] + "/" + noise + "_" + policies[csv_num] + "_" + str(round(param, 3)) + filename
			perf = get_performance_value(progress)
			performances[csv_num].append(perf)

	graphFileName = "/home/tejas/Documents/rllab/data/local/experiment/" + graph_name + ".png"
	rounded = [round(x, 3) for x in parameters]
	plot((rounded, performances), title, xLabel, yLabel, legend=series, ymin=ymin, ymax=ymax, filename=graphFileName)

# Calculate Area using Trapezoids
def area_under_curve(lstX, lstY):
	first = 0
	second = 1
	area = 0

	while second < len(lstX):
		dx = lstX[second] - lstX[first]
		averageOfBases = ((lstY[first] + lstY[second]) / 2)
		area += dx * averageOfBases
		first += 1
		second += 1

	return area
 


def plot_areas(path, noise_names, policy_names, variables, ymin, ymax, xLabel):

	fieldName = "AverageDiscountedReturn"	
	yLabel = "Area Under Curve for " + fieldName
	filename = "/progress.csv"

	for noise in noise_names:
		areas = []
		title = "Area Under Curve For " + noise + " Noise"
		for policy in policy_names:
			a = []
			for var in variables:
				progressFile = path + noise + "_" + policy + "_" + str(round(var,3)) + filename
				iterations, rewards = get_experiment_values(progressFile, fieldName)
				area = area_under_curve(iterations, rewards)
				a.append(area)
			areas.append(a)

		graphFileName = path + "Area_" + noise + ".png"
		rounded = [round(x, 3) for x in variables]
		plot((rounded, areas), title, xLabel, yLabel, legend=policy_names, ymin=ymin, ymax=ymax, filename=graphFileName)

def plot_areas_averaged(paths, noise_names, policy_names, variables, ymin, ymax, xLabel):
	fieldName = "AverageDiscountedReturn"	
	yLabel = "Area Under Curve for " + fieldName
	filename = "/progress.csv"

	for noise in noise_names:
		aggregated = []
		title = "Area Under Curve For " + noise + " Noise"
		for policy in policy_names:
			areas = []
			file_num = 0
			for path in paths:
				index = 0
				for var in variables:
					progressFile = path + noise + "_" + policy + "_" + str(round(var,3)) + filename
					iterations, rewards = get_experiment_values(progressFile, fieldName)
					area = area_under_curve(iterations, rewards)
					if file_num == 0:
						areas.append(area)
					else:
						areas[index] += area
					index += 1
				file_num += 1
			averaged = [a / file_num for a in areas]
			aggregated.append(averaged)

		path = paths[0];
		graphFileName =  path + "Area_" + noise + ".png"
		rounded = [round(x, 3) for x in variables]
		plot((rounded, aggregated), title, xLabel, yLabel, legend=policy_names, ymin=ymin, ymax=ymax, filename=graphFileName)

def plot_learning_curve(path, noise_name, policy_name, parameter, ymin, ymax):
	fieldName = "AverageDiscountedReturn"	
	xLabel = "Iteration"
	yLabel = "Average Discounted Return"
	title = "Learning Rate"
	filename = "/progress.csv"

	progressFile = path + noise_name + "_" + policy_name + "_" + str(parameter) + filename
	iterations, rewards = get_experiment_values(progressFile, fieldName)
	graphFileName = path + "Learning_Rates_" + policy_name + "_" + noise_name + "_" + str(parameter) + ".png"
	plot((iterations, [rewards]), title, xLabel, yLabel, legend=policy_name, ymin=ymin, ymax=ymax, loc='lower right', filename=graphFileName)	


def plot_learning_curves(path, noise, policy, paramStart, paramEnd, paramIncr, ymin, ymax):
	fieldName = "AverageReturn"	
	xLabel = "Iteration"
	yLabel = "Average Return"
	title = "Learning Rate"
	filename = "/progress.csv"

	legend = []
	iterations = []
	rewards = []
	while paramStart <= paramEnd:
		progressFile = path + noise + "_" + policy + "_" + str(paramStart) + filename
		iterations, r = get_experiment_values(progressFile, fieldName)
		rewards.append(r)
		legend.append(policy + "_" + str(paramStart))
		paramStart += paramIncr
		paramStart = round(paramStart, 3)

	graphFileName = path + "Learning_Rates_" + policy + "_" + noise + ".png"
	plot((iterations, rewards), title, xLabel, yLabel, legend=legend, ymin=ymin, ymax=ymax, loc='lower right', filename=graphFileName)	



def plot_learning_curve_averaged(paths, noise_name, policy_name, parameter, ymin, ymax):
	fieldName = "AverageDiscountedReturn"	
	xLabel = "Iteration"
	yLabel = "Average Discounted Return"
	title = "Learning Rate"
	filename = "/progress.csv"

	files = []
	for path in paths:
		progressFile = path + noise_name + "_" + policy_name + "_" + str(round(parameter, 3)) + filename
		files.append(progressFile)
	
	iterations, rewards = get_experiment_values_averaged(files, fieldName)

	graphFileName = path + "Learning_Rates_" + policy_name + "_" + noise_name + "_" + str(round(parameter, 3)) + ".png"
	plot((iterations, [rewards]), title, xLabel, yLabel, legend=policy_name, ymin=ymin, ymax=ymax, loc='lower right', filename=graphFileName)




file_path_base = "/home/tejas/Documents/rllab/data/local/experiment/lunar_lander_vpq_v1/"
variables = [x*0.1 for x in range(0,6)]
#policies = ["GaussianGRU_32_0"]
policies = ["GaussianMLP_32_32", "GaussianGRU_32_0", "GaussianMLP_32_32", "GaussianMLP_32_32", "GaussianMLP_32_32", "GaussianMLP_32_32", "GaussianMLP_32_32"]
#plot_performance_values(file_path_base, policies, "Gaussian", variables, -100, 100, "SNR")
# plot_learning_curves(file_path_base, "Gaussian", "GaussianMLP_32_32", 0.0, 0.3, 0.1, -500, 300)
#plot_performance_values(file_path_base, policies, "Laplace", variables, iterations, -1200, 0, "Scale Factor")
#plot_performance_values(file_path_base, policies, "DroppedObservations", variables, iterations, -1200, 0, "Probability")
#plot_performance_values(file_path_base, policies, "DroppedObservationsReplace", variables, iterations, -1200, 0, "Probability")

#plot_performance_median_trpo_vpg([1,2,3], [5], variables, policies, "Gaussian", -150, 150, "SNR")

#folders = ["pendulum_trpo_mlp_v1", "pendulum_trpo_mlp_v2", "pendulum_trpo_mlp_v3", "pendulum_vpg_mlp_v5", "pendulum_vpg_mlp_v6", "pendulum_vpg_mlp_v7"]
#folders = ["lunar_lander_trpo_gru_v1", "lunar_lander_trpo_gru_v2"]
#series = ["TRPO, GRU, Trial 1", "TRPO, GRU, Trial 2"]
#series = ["TRPO, Trial 1", "TRPO, Trial 2", "TRPO, Trial 3", "VPG, Trial 1", "VPG, Trial 2", "VPG, Trial 3"]

folders = ["lunar_lander_v6", "lunar_lander_v8"]
series = ["TRPO, MLP", "TRPO, GRU"]

plot_performance_values_multiple(folders, "Lander_Dropped_Performances", variables, policies, "DroppedObservations", -205, 150, series, "Probability")


#plot_performance_values_multiple("lunar_lander_trpo_v", 3, "TRPO_All_Performances", variables, "GaussianMLP_32_32", "Gaussian", -175, 175, "SNR")


#plot_performance_values_multiple("lunar_lander_trpo_v", 3, "TRPO_All_Performances", variables, "GaussianMLP_32_32", "Gaussian", -175, 175, "SNR")
#plot_performance_values_multiple("lunar_lander_vpg_v", 3, "VPG_All_Performances", variables, "GaussianMLP_32_32", "Gaussian", -175, 175, "SNR")


# noises = ["DroppedObservations", "DroppedObservationsReplace"]
# variables = [x*0.1 for x in range(0, 11)]
# x_label = "Probability"
# ymin = -250
# ymax = 100
# noises = ["Laplace"]
# variables = [x*0.1 for x in range(0,6)]
# x_label = "Scale Factor" 
# ymin = -800
# ymax = -300
# mlp_single = "GaussianMLP_8_0"
# mlp_double = "GaussianMLP_8_8"
# gru_single = "GaussianGRU_8_0"
# gaussian = "Gaussian"
# laplace = "Laplace"

# path = "/Users/Tejas/Desktop/Research/rllab/data/local/experiment/mountaincar_v4/"
# paths = ["/Users/Tejas/Desktop/Research/rllab/data/local/experiment/pendulum_vpg_v1/"]
# policy_names = [mlp_single, mlp_double, gru_single]
# #policy_names = ["CategoricalMLP_8_0", "CategoricalMLP_8_8"]

# incr = 0.1
# start = 0.0
# end = 0.5
# #plot_areas_averaged(paths, noises, policy_names, variables, ymin, ymax, x_label)
# for noise in noises:
# 	for policy in policy_names:
# 		param = start
# 		while param <= end:
# 			plot_learning_curve_averaged(paths, noise, policy, param, ymin, ymax)
# 			param += incr



# plot_learning_curve(path, "Gaussian", "Gaussian_GRU_8", 0.25, ymin, ymax)
# plot_learning_curve(path, "Gaussian", "Gaussian_MLP_8", 0, ymin, ymax)
# plot_learning_curve(path, "Gaussian", "Gaussian_MLP_8", 0.25, ymin, ymax)

# plot_learning_curve(path, "Laplace", "Gaussian_GRU_8", 0, ymin, ymax)
# plot_learning_curve(path, "Laplace", "Gaussian_GRU_8", 0.25, ymin, ymax)
# plot_learning_curve(path, "Laplace", "Gaussian_MLP_8", 0, ymin, ymax)
# plot_learning_curve(path, "Laplace", "Gaussian_MLP_8", 0.25, ymin, ymax)
#plot_learning_curve(path, "Laplace", "GaussianMLP_8_0", 0.3, -25, 25)

