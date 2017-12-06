# Final Project for CS 759
I will be simulating heat diffusion by numerically approximating solutions to the heat equation.

### Todos
+ Make sure that numeric types are large enough to handle all relevant problems and
simulation params
+ Add check to make sure r <= 1/2. If r > 1/2 alert the user that the explicit method
can not be used for the parameters given and terminate.
+ Improve efficiency
+ Set up GNUPG and share repo with course staff
+ Make reusable/flexible animation program
+ Make sure interface the between animation program and simulation can handle very large
simulations (if you write the data to a file, how much data can the simulation output)
+ Update main to print comma separated values to file with first line t, second line points,
subsequent lines the temps



## Plan
I will be using a finite difference approximation of the heat equation. Specifically
I will start by implementing a simulation of the diffusion of heat in a 1d "rod"
using the "explicit" difference equation.

### Plan For Project Structure
I know have options as to how I go about this.

Here's what I'm thinking:
+ I will make a struct type to hold all the relevant information about a 1D diffusion
problem:
	+ Length of the "rod"
	+ alpha
	+ k
	+ h
	+ temperature at ends of rod
	+ function to calculate starting values


+ I will put this struct type in the header (of course). In main I will make an
object of this type for each problem I want to solve. The functions I write in
the header, can except one of these structs as a parameter, making the code in main
very clean and simple.


+ This leaves how I will retrieve the generated data. This will really depend on
how I go about making the animations/graphs that will allow the data to be easily
displayed. For now I think it would be best if the function I call from main simply
takes the name of a file, and writes the state of the rod (i.e. temperatures at discrete
points) periodically to the file.

##### Potential Problems With The Above:
+ Can you have a struct store a function object which is supplied at runtime?
	- Answer: Yes (just use a function pointer).
+ Printing out the data (to a file or to the console) from the GPU would absolutely
destroy the performance of the program. So in order to avoid that I should have
the device code store the data in global memory and then the host code can copy it back
over to the CPU. This also will make the code more flexible when I write the code to
do the graphics.

### Miscellaneous Notes
+ I added a struct to hold parameters of the simulation. I did this because the problem
(specified by length, starting temp, alpha, etc.) is independent of how I will solve it.
Thus I removed k and h from original struct and made a new struct with all the simulation params.

### Thoughts on Animating Results
+ I will use pyplot/matplotlib to make the animation
+ The results of the simulation will be stored in a file (I will have to check the
max possible size of a file)
+ The user will have to copy the file to their local machine and run the animation
script
+ The x axis should range from 0 to L (L should be first line of file)
+ In order to display time, include time step as second line in file
+ Since one can not know the min/max y value before stepping through all the data,
the min/max value should be updated as the animation runs. I don't think the magnitude
of the range for the y-axis should change as the simulation progresses (since that could
appear misleading and make it seem like the values are not getting smaller). So at least
at first only resize to increase the max or decrease the min. In each iteration I
will have to look at the max/min value for that time in order to do that.
