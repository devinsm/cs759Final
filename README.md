# Final Project for CS 759
I will be simulating heat diffusion by numerically approximating solutions to the heat equation.

### Todos
+ Make sure that numeric types are large enough to handle all relevant problems and
simulation params

+ Add check to make sure r <= 1/2. If r > 1/2 alert the user that the explicit (aka FTCS) method
can not be used for the parameters given and terminate the program.

+ Improve efficiency
	+ Move "workingMem" from global to shared memory

+ Make sure interface the between animation program and simulation can handle very large
simulations (if you write the data to a file, how much data can the simulation output?)

+ Have print function print to file instead of cout

+ Add timing code (need to decide what should be included in run time)

+ Update simulation code to label axis and display time
