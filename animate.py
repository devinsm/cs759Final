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
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
##############################################################################
# Prints ussage and exits.
##############################################################################
def printUsage():
    print "Usage: " + sys.argv[0] + " fileName [t]"
    print "t is the real time (in milliseconds) between frames during the animation"
    exit()
##############################################################################
# Converts a line of text (floats seperated by a comma and white space), to
# a numpy array of floats
##############################################################################
def toNumpyArray(line):
    return np.array(re.split(',\s*', line.strip()), dtype=np.single)

##############################################################################
# The body of the script
##############################################################################
if len(sys.argv) != 2 and len(sys.argv) != 3:
    printUsage()
if len(sys.argv) == 3:
    try:
        delay = float(sys.argv[2])
    except ValueError:
        print "ERROR: Couldn't convert " + sys.argv[2] + " to a float"
        printUsage()
else:
    delay = 200

try:
    inputFile = open(sys.argv[1])
except IOError:
    print "ERROR: Couldn't open file " + sys.argv[1]
    printUsage()

# Get metadata
t = float(inputFile.readline().strip())
points = toNumpyArray(inputFile.readline())

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
def animate(lineFromFile):
    global time
    heatVals = toNumpyArray(lineFromFile)
    # First update scale of y axis
    # This is necessary since we don't know who big/small the temps are before hand
    ymin, ymax = ax.get_ylim()
    maxTemp = np.amax(heatVals)
    minTemp = np.amin(heatVals)

    if minTemp < ymin or maxTemp > ymax:
        newMin = minTemp if minTemp < ymin else ymin
        newMax = maxTemp if maxTemp > ymax else ymax
        ax.set_ylim(newMin, newMax)

    # Now plot the points on the line
    line.set_data(points, heatVals)

    # The string conversion of time below is based off of the following stack overflow
    # post: https://stackoverflow.com/questions/21008858/formatting-floats-in-a-numpy-array
    line.set_label("Time: " + "%.4f" % time)
    ax.legend()
    time += t

    return line,


##############################################################################
# Function initializes animation
# Returns:
#   line: a tuple containing the line to plot
##############################################################################
def initFunction():
    global time
    time = 0.0
    return line,

##############################################################################

ani = animation.FuncAnimation(fig, animate, inputFile, init_func=initFunction, blit=False, interval=delay)
plt.show()
