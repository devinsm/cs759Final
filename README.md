# Final Project for CS 759
This code numerically approximates solutions to the 1D heat equation. It does so
using the Forwards-Time Centered-Space (FTCS) method.

The simulation really consists of two parts. The first part is the CUDA program
which generates the data. The second part is a python script which animates the
results. The CUDA program should be run on Euler. The simulation should be run
on your local machine, and you'll need python 2.7 as well as matplotlib
(don't worry, I'll hold your hand :) )

# Building and Running

## Generating the Data on Euler
1. Clone this repository on Euler.
1. Run `make` in the top level directory of this project (that is to say the
the directory that this README is in).
1. Run `mkdir simData`
1. The generated executable is named `generate_data`. This is the CUDA program
which will be run on Euler to generate the data.
1. Run `./generate_data` either via a slurm script or in interactive mode
1. The generated data is now in a folder called `simData`
1. In order to copy this data to your local machine you will have to compress it.
1. Run `zip -r simData simData`
1. After the compression completes, there will be a file called `simData.zip`
1. Copy `simData.zip` and `animate.py` to your local machine. I
trust you already have some way you usually copy files back and forth.

IMPORTANT: Even if you use sshfs or sftpnetdrive, you will still have to manually
move the compressed file over to your local machine. You can drag and drop it if
you'd like, you can use `scp` if you'd like, but `simData.zip` needs to be on your
local machine's hard drive.

## Running The Animation
Unfortunately the animation requires matplotlib, which can't be used on Euler
at the moment. You will need to install matplotlib if you don't already have it.

#### Installing Matplotlib
The best way to install matplotlib is to install the Anaconda package manager.
1. Go to https://www.anaconda.com/download/
2. Download the Python 2.7 version of Anaconda. This comes with everything you need.

#### Running the Animation
1. `cd` into the directory with `simData.zip` and `animate.py`
1. Unzip `simData.zip`
1. `animate.py` takes a file name as its first argument. For its second argument
you can optionally give it the time in milliseconds that the animation should pause
between frames (the default value is 200). The minimum value for the second argument is 1.
1. For each file `aFile` in the unziped `simData` run `python animate.py aFile`.
If you want to speed things up or slow them down you can run `python animate.py aFile timeStep`.
