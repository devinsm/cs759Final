###############################################################################
# This script uses matplotlib and numpy to animate the change in the heat of the
# "rod" overtime. I wrote this after consulting the tutorials at matplotlib.org
# (specifically the tutorials at the URLs below). Though none of the code in
# those tutorials was doing the same thing I'm doing here, the code in those
# tutorials highly influenced my code here. For a complete list of the
# the Python/matplotlib/numpy resources I consulted, please see my final report.
# Also, as noted in the inline comments, my conversion of the variable time to a fixed point
# string representation came from https://stackoverflow.com/questions/21008858/formatting-floats-in-a-numpy-array.
#
# matplotlib.org tutorials:
#   https://matplotlib.org/users/pyplot_tutorial.html
#   https://matplotlib.org/api/animation_api.html
#   https://matplotlib.org/users/artists.html
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

##############################################################################
# Function which reads data into 2D numpy array of floats. Each row is a moment
# in time, and each column is a specific loaction.
# Parameters:
#   file: string
#       name of input file
# Returns:
#   t: float
#       time between succesive rows
#   points: numpy array of floats
#       the points on the rod which were used in the numerical simulation
#   data: numpy array of floats
#       The 2D array of generated temperature values
##############################################################################
def readInput(file):
    inputFile = open(file)
    inputLines = inputFile.readlines()

    # Get metadata
    t = float(inputLines[0])

    # Create numpy array
    dataList = [line.replace(' ', '') for line in inputLines[1:]]
    dataList = [line.replace('\n', '') for line in dataList]
    dataList = [line.split(',') for line in dataList]
    dataList = [[float(string) for string in line] for line in dataList]

    return t, np.array(dataList[0]), np.array(dataList[1:])

##############################################################################
# The body of the script
##############################################################################
t, points, data = readInput("euler/cs759Final/smithProblem.txt")
time = 0

fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label="Time: 0.0")
ax.legend()
ax.set_title("Heat Diffusion in 1D \"Rod\"")
ax.set_xlabel("Position")
ax.set_ylabel("Temperature")

numPoints, = points.shape
ax.set_xlim(0, points[numPoints - 1])
ax.set_ylim(0,0.1) #after reading the first line of data this will be updated

##############################################################################
# The function which updates the graph at each time step.
# Parameters:
#   heatVals: numpy array of floats
#       The calculated values of the temperature at the discrete points.
# Returns:
#   line: a tuple containing the line to plot.
##############################################################################
def animate(heatVals):
    global time
    # First update scale of y axis
    # This is necessary since we don't know who big/small the temps are before hand
    ymin, ymax = ax.get_ylim()
    maxTemp = np.amax(heatVals)
    minTemp = np.amin(heatVals)

    if minTemp < ymin or maxTemp > ymax:
        newMin = minTemp if minTemp < ymin else ymin
        newMax = maxTemp if maxTemp > ymax else ymax
        ax.set_ylim(newMin, newMax)

    #Update time
    time += t

    # Now plot the points on the line
    line.set_data(points, heatVals)

    # The string conversion of time below is based off of the following stack overflow
    # post: https://stackoverflow.com/questions/21008858/formatting-floats-in-a-numpy-array
    line.set_label("Time: " + "%.4f" % time)
    ax.legend()
    return line,


##############################################################################

ani = animation.FuncAnimation(fig, animate, data, blit=False)
plt.show()
