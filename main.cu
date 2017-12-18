/*
 *This file contains the main program which calls the functions in heatEquation.hpp.
 *
 *@author Devin McAllester
*/

//C++ include statements
#include <iostream>
#include <exception>
#include <cmath>

//Project specific include statements
#include "heatEquation.hpp"

//Using statements for convience
using namespace std;

struct TestFunctor {
	int i = 1;
	__host__ __device__
	float operator() (float position) {
		return position * i++;
	};
};

struct SmithFunctor {
	__host__ __device__
	float operator() (float position) {
		if (position <= .5) {
			return 2 * position;
		} else {
			return 2 * (1 - position);
		}
	};
};

struct SineFunctor {
	const float pi = acos(-1.0);
	__host__ __device__
	float operator() (float position) {
		return 1 + sin(position - pi / 2);
	};
};

int main(int argc, char const *argv[]) {
	HeatProblem1d<SmithFunctor> smithsProblem;
	smithsProblem.l = 1;
	smithsProblem.alpha = 1;
	smithsProblem.leftTemp = 0;
	smithsProblem.rightTemp = 0;

	HeatProblem1d<SineFunctor> sineProblem;
	sineProblem.l = acos(-1.0);
	sineProblem.alpha = 3.352;
	sineProblem.leftTemp = 0;
	sineProblem.rightTemp = 2;

	SimulationParams1D superSmallSim;
	superSmallSim.deltaX = .1;
	superSmallSim.deltaT = .001;
	superSmallSim.numIterations = 20;
	superSmallSim.periodOfRecordings = 1;

	SimulationParams1D mediumSizedSim;
	mediumSizedSim.deltaX = .01;
	mediumSizedSim.deltaT = .00001;
	mediumSizedSim.numIterations = 4000000;
	mediumSizedSim.periodOfRecordings = 5;


	try {
		sloveProblemInstance(smithsProblem, superSmallSim, "smithProblem.txt");
		sloveProblemInstance(sineProblem, mediumSizedSim, "sineProblem.txt");
	} catch (exception& e) {
		cout << "Simulation on line " << __LINE__ - 2 << "threw an exception" << endl;
		cout << e.what() << endl;
	}
	return 0;
}
