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
	rcParams.update({'font.size': 12})
	fprop = font_manager.FontProperties(fname='/Library/Fonts/Microsoft/Gill Sans MT.ttf') 

	plt.figure() 
	colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
	shape = ['s-', 'o-', '^-', 'v-', 'x-', 'h-']

	X = points_tuple[0]
	Y = points_tuple[1]
	i = 0

	for ya in Y:
		plt.plot(X, ya, shape[i], linewidth=2.5,markersize=7,color=colors[i])
		i += 1

	plt.legend(legend,loc=loc)
	plt.title(title)
	plt.xlabel(xaxis,fontproperties=fprop)
	plt.ylabel(yaxis,fontproperties=fprop)
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

		graphFileName = "/Users/Tejas/Desktop/Research/Area_" + noise + ".png"
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


# noises = ["DroppedObservations", "DroppedObservationsReplace"]
# variables = [x*0.1 for x in range(0, 11)]
# x_label = "Probability"
# ymin = -250
# ymax = 100
noises = ["Laplace", "Gaussian"]
variables = [x*0.1 for x in range(0,6)]
x_label = "Scale Factor" 
ymin = -35000
ymax = -20000
mlp_single = "GaussianMLP_8_0"
mlp_double = "GaussianMLP_8_8"
gru_single = "GaussianGRU_8_0"
gaussian = "Gaussian"
laplace = "Laplace"

path = "/Users/Tejas/Desktop/Research/rllab/data/local/experiment/mountaincar_v4/"
paths = ["/Users/Tejas/Desktop/Research/rllab/data/local/experiment/mountaincar_v5/", "/Users/Tejas/Desktop/Research/rllab/data/local/experiment/mountaincar_v6/", "/Users/Tejas/Desktop/Research/rllab/data/local/experiment/mountaincar_v7/"]
policy_names = [mlp_single, mlp_double, gru_single]
#policy_names = ["CategoricalMLP_8_0", "CategoricalMLP_8_8"]

incr = 0.1
start = 0.0
end = 0.5
plot_areas_averaged(paths, noises, policy_names, variables, ymin, ymax, x_label)
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

