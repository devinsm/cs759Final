# Final Project for CS 759
This code numerically approximates solutions to the 1D heat equation. It does so
using the Forwards-Time Centered-Space (FTCS) method. The FTCS method gives
an explicit finite difference equation, which lends itself very easily to a parallel
implementation.

Specifically, if U(x, t) is the numerically approximated solution to the 1D heat equation, then:

U(j, n + 1) = (1 - 2r) U(j, n) + r U(j - 1, n) + r U(j + 1, n)

where r = alpha * (deltaT / deltaX<sup>2</sup>), deltaT is the time between
calculations, and deltaX is the distance between discrete points used for the
calculations.

This difference equation clearly is easily parallelizable, since at each time step
a single thread can calculate the value of U for each location.

In order to be numerically stable (i.e. floating point errors don't grow out of
control as time progresses), it must be that r <= 1/2.

I found the derivation of the above difference equation at the [Finite Difference Method
Wikipedia page](https://en.wikipedia.org/wiki/Finite_difference_method). Following
one of its sources, I found [an old book](http://www-eng.lbl.gov/~shuman/NEXT/MATERIALS%26COMPONENTS/Xe_damage/Crank-The-Mathematics-of-Diffusion.pdf) which had a table of outputs generated using the FTCS method
with a particular set of initial and boundary conditions (see figure 8.1
in the linked book). This made rudimentary testing of my implementation very easy.

In the references section below, I list other online material I used to understand
how to numerically approximate the heat equation. None of the material I referenced
included code.

## Current Architecture
All the functions which actually carry out the numerical approximation are in
heatEquation.hpp. Currently the main entry point is solveProblemInstance. The user
of my functions populates two structs and passes them to this method.

The first struct, of type HeatProblem1d, includes the length of the rod, alpha,
the boundary conditions, and a functor to calculate the initial temperature at any
point on the rod (excluding the temperature at the ends, which is given in the
boundary conditions). The functor was inspired by thrust, and the struct type is
templated to allow for any functor (though it's operator() must take a float and
return a float).

The second struct is of type SimulationParams1D, and tells the simulation functions
how the simulation is to be carried out.

I chose to pass this information in as structs, because I felt with so many parameters
(9 to be exact) setting struct members was much more readable then long parameter lists.
It also made it easy for me to pass the information on to helper functions.

I chose to have two structs since I thought one might want to solve the same problem
with multiple different simulation parameters (e.g. time step sizes), and decoupling
the problem form the simulation parameters would better facilitate this.

The temperature at all points being used in the simulation is periodically recorded,
and stored in the GPUs global memory. After the simulation is complete, this data is
then written to the file specified when solveProblemInstance was called. The data is
written a way that it is both easy for a human to read, and a program to parse.

After this file is generated (i.e. main has completed), the user then can run
animate.py on their local machine to see an animation of the change in temperature
in the "rod" over time. Currently the file name used in both main.cu and
animate.py is hard coded. This means that when running animate.py on your local
machine, you will have to change animate.py to have the correct path to the
generated file.

## Heat Equation and Numerical Methods References
* https://en.wikipedia.org/wiki/Numerical_stability
* https://en.wikipedia.org/wiki/Heat_equation
* https://en.wikipedia.org/wiki/FTCS_scheme
* http://mathworld.wolfram.com/TaylorSeries.html
* http://www-eng.lbl.gov/~shuman/NEXT/MATERIALS%26COMPONENTS/Xe_damage/Crank-The-Mathematics-of-Diffusion.pdf
* https://www.rsmas.miami.edu/users/miskandarani/Courses/MSC321/lectfiniteDifference.pdf
* https://en.wikipedia.org/wiki/Finite_difference_method
* https://en.wikipedia.org/wiki/Finite_difference


## Todos
* Make sure that numeric types are large enough to handle all relevant problems and
simulation params
* Add timing code (need to decide what should be included in run time)
* Make main ask the user for the initial conditions, boundary conditions, and
other necessary parameters
* Have the simulation script accept the name of the file to read from as a command
line argument (currently the file name is hard coded)
* Improve efficiency
	* Move "workingMem" from global to shared memory
