# Final Project for CS 759
I will be simulating heat diffusion by numerically approximating solutions to the heat equation.

### Todos
+ Make sure that numeric types are large enough to handle all relevant problems and
simulation params

+ Improve efficiency
	+ Move "workingMem" from global to shared memory

+ Make sure interface the between animation program and simulation can handle very large
simulations (if you write the data to a file, how much data can the simulation output?)
	+ Answer: looking online it seems files likely can be absolutely huge, and this is not
	a concern.
	

+ Add timing code (need to decide what should be included in run time)

+ Update simulation code to label axis and display time
