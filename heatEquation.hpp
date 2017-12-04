/*
 *This file holds the (usually CUDA) functions which I wrote to numerically simulate
 *the diffusion of heat using the heat equation. Please see my final project report for
 *a list of resources I consulted while writing the following code.
 *
 *@author Devin McAllester
*/

#ifndef HEATEQUATION_HPP
#define HEATEQUATION_HPP
struct HeatProblem1d {
	int l; //Length of the rod
	float alpha; //Thermal diffusivity
	float deltaT; //Distance between time points
	float deltaX; //distance between discrete points on rod
	float leftTemp; //Temperature at left end of rod (held constant)
	float rightTemp; //Temperature at right end of rod (held constant)
	float (*initFunction)(float position); //Pointer to function which provides starting temps
};
#endif
