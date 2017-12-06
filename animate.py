###############################################################################
# This script uses matplotlib and numpy to animate the change in the heat of the
# "rod" overtime. I wrote this after consulting the tutorials at matplotlib.org
# (specifically the tutorials at the URLs below). Though none of the code in
# those tutorials was doing the same thing I'm doing here, the code in those
# tutorials highly influenced my code here. For a complete list of the
# the Python/matplotlib/numpy resources I consulted, please see my final report.
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
#   l: float
#       length of the rod
#   t: float
#       time between succesive rows
#   deltaX: float
#       distance between succesive points
#   data: numpy array of floats
#       The 2D array of generated temperature values
##############################################################################
def readInput(file):
    inputFile = open(file)
    inputLines = inputFile.readlines()

    # Get metadata
    l = inputLines[0]
    t = inputLines[1]
    deltaX = inputLines[2]

    # Create numpy array
    dataList = [line.replace(' ', '') for line in inputLines[3:]]
    dataList = [line.replace('\n', '') for line in dataList]
    dataList = [line.split(',') for line in dataList]
    dataList = [[float(string) for string in line] for line in dataList]
    print("l: " + l)
    print("t: " + t)
    print("deltaX: " + deltaX)
    print("data:")
    print(dataList)

readInput("euler/cs759Final/testInput.txt")

# fig, ax = plt.subplots()
#
# ax.set_xlim(0, 20)
# ax.set_ylim(0, 20)
#
# line, = ax.plot([], [], 'bo')
#
#
# def animate(i):
#     line.set_data(i, i)  # update the data
#     return line,
#
#
# # Init only required for blitting to give a clean slate.
# def init():
#     line.set_data(0, 0)
#     return line,
#
# ani = animation.FuncAnimation(fig, animate, np.arange(1, 20), init_func=init, blit=False)
# plt.show()
