# Final Project for CS 759
I will be simulating heat diffusion by numerically approximating solutions to the heat equation.

### Todos
+ Make sure that numeric types are large enough to handle all relevant problems and
simulation params

+ Add check to make sure r <= 1/2. If r > 1/2 alert the user that the explicit method can not be used for the parameters given and terminate.

+ Improve efficiency
	+ Move "workingMem" from global to shared memory

+ Make sure interface the between animation program and simulation can handle very large
simulations (if you write the data to a file, how much data can the simulation output)

+ Have file printing function take parameter structs, output, and file name

+ Have file printing function print output as follows:
	+ First line difference in time between rows of output
	+ Second line locations of points for whom temp was estimated (comma separated)
	+ Each subsequent line the comma separated temps at a moment in time


+ Add timing code (need to decide what should be included in run time)

+ Update simulation code to label axis and display time
