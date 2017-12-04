# Final Project for CS 759
I will be simulating heat diffusion by numerically approximating solutions to the heat equation.

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
