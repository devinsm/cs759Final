/*
 *This file contains the main program which calls the functions in heatEquation.hpp.
 *
 *@author Devin McAllester
*/

//C++ include statements
#include <iostream>
#include <exception>
#include <cmath>
#include <string>

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

struct Deg7Poly {
	__host__ __device__
	float operator() (float position) {
		float x = (position / 2) - 3;
		return (x - 3) * (x - 2) * (x - 1) * x * (x + 3) * (x + 2) * (x + 1);
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

	//A complex polynomial to use for timming
	HeatProblem1d<Deg7Poly> strangeProblem;
	strangeProblem.l = 10.23;
	strangeProblem.alpha = 4;
	strangeProblem.leftTemp = 0;
	strangeProblem.rightTemp = -15.7362;
	//should use this timming sim with alpha <= 5
	SimulationParams1D timmingSim;
	timmingSim.deltaX = .01;
	timmingSim.deltaT = .00001;
	timmingSim.numIterations = 2;
	timmingSim.periodOfRecordings = 1;

	try {
		sloveProblemInstance(smithsProblem, superSmallSim, "simData/smithProblem.txt");
		//sloveProblemInstance(sineProblem, mediumSizedSim, "sineProblem.txt");
		for (size_t i = 2; i <= 1 << 24; i <<= 1) {
			timmingSim.numIterations = i;
			timmingSim.periodOfRecordings = i / 256 > 1 ? i / 256 : 1; //we will have 257 moments in output for i >= 256
			float inclusiveTime = sloveProblemInstance(sineProblem, timmingSim, "simData/sineProblem_" + to_string(i) + ".txt");
			cout << i << " " << inclusiveTime << endl;
		}
	} catch (exception& e) {
		cout << "Simulation on line " << __LINE__ - 2 << " threw an exception" << endl;
		cout << e.what() << endl;
	}
	return 0;
}
