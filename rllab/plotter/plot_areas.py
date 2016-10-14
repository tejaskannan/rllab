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


def getExperimentValues(csvFilePath, propertyName):
	with open(csvFilePath, 'rb') as csvFile:
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

# Calculate Area using Trapezoids
def areaUnderCurve(lstX, lstY):
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
 


def plotAreas(noises, variables, ymin, ymax, xLabel):

	fieldName = "AverageDiscountedReturn"	
	yLabel = "Area Under Curve for " + fieldName

	base = "/Users/Tejas/Desktop/Research/rllab/data/local/experiment/"
	filename = "/progress.csv"

	mlp_single = "Gaussian_MLP_8"
	mlp_double = "Gaussian_MLP_8_8"
	gru = "Gaussian_GRU_8"

	exps = [mlp_single, mlp_double, gru]

	for noise in noises:
		print noise
		areas = []
		title = "Area Under Curve For " + noise + " Noise"
		for exp in exps:
			a = []
			for var in variables:
				progressFile = base + noise + "_" + exp + "_" + str(var) + filename
				iterations, rewards = getExperimentValues(progressFile, fieldName)
				area = areaUnderCurve(iterations, rewards)
				print area
				a.append(area)
			areas.append(a)

		graphFileName = "areas_" + noise + ".png"
		plot((variables, areas), title, xLabel, yLabel, legend=exps, ymin=ymin, ymax=ymax, filename=graphFileName)


def plotLearningCurves():
	fieldName = "AverageDiscountedReturn"	
	xLabel = "Iteration"
	yLabel = "Average Discounted Return"
	title = "Learning Rate"

	base = "/Users/Tejas/Desktop/Research/rllab/data/local/experiment/"
	filename = "/progress.csv"

	mlp_single = "Gaussian_MLP_8"
	mlp_double = "Gaussian_MLP_8_8"
	gru = "Gaussian_GRU_8"

	laplace = "Laplace"
	gaussian = "Gaussian"

	ymin = -750
	ymax = -100

	variances = [0, 0.1, 0.5, 1.0, 1.5, 2]
	noises = [laplace, gaussian]
	exps = [mlp_single, mlp_double, gru]

	for variance in variances:
		for noise in noises:
			learningRates = []
			for exp in exps:
				progressFile = base + noise + "_" + exp + "_" + str(variance) + filename
				iterations, rewards = getExperimentValues(progressFile, fieldName)
				learningRates.append(rewards)
			graphFileName = "learning_rates_" + noise + "_" + str(variance) + ".png"
			plot((iterations, learningRates), title, xLabel, yLabel, legend=exps, ymin=ymin, ymax=ymax, loc='lower right', filename=graphFileName)	

noises = ["DroppedObservations"]
variables = [0] + [x*0.1 for x in range(1, 11)]
ymin = -30000
ymax = -15000

plotAreas(["DroppedObservations"], variables, ymin, ymax, "Probability")


	# filePath = './data/noisy_tests/'
	# limit = 5 # ran 10 experiments
	# gaussianMlp = "Gaussian MLP"
	# deterministicMlp = "Deterministic MLP"

	# gaussianMlpAreas = {}
	# deterministicMlpAreas = {}
	# fileName = filePath + "info.txt"

	# discountedReturnProperty = "AverageDiscountedReturn"
	# progressFileName = filePath + "progress.csv"

	# for i in range(0, limit):
	# 	iterations, rewards = getExperimentValues(progressFileName, discountedReturnProperty)
	# 	area = areaUnderCurve(iterations, rewards)

	# 	with infoFile = open(fileName, 'r') as f:
	# 		information = f.readlines()

	# 	sigma = float(information[1])

	# 	if (information[0] == gaussianMlp):
	# 		if (area == 0):
	# 			gaussianMlp[sigma] = 0
	# 		else:
	# 			gaussianMlp[sigma] = 1 / area
		

	# plotArea(gaussianMlp)


