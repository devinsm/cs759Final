import re
import sys
import numpy as np
import matplotlib.pyplot as plt
def printUsage():
    print("Usage: " + argv[0] + " fileName")

if len(sys.argv) != 2:
    printUsage()
else:
    try:
        inputFile = open(sys.argv[1])
    except IOError:
        print("ERROR: Couldn't open file " + sys.argv[1])
        printUsage()

xpoints = []
ypoints = []
for line in inputFile:
    point = line.split()
    xpoints.append(point[0])
    ypoints.append(point[1])

plt.xlabel('Number of Time Steps')
plt.ylabel('Inclusive Time (ms)')
plt.title('Timming of Solution on Degree 7 Polynomial')
plt.plot(xpoints, ypoints, 'b-o')
plt.tight_layout()
plt.savefig("timmingPlot.pdf")
plt.clf()
