/**
 *This code is based off of the header file posted by William Jen in @267.
 *I seperated out the implementation of the class from its decleration. I also
 *modified the code to use exceptions instead of boolean return types.
*/


#ifndef _CUDA_STOPWATCH_HPP
#define _CUDA_STOPWATCH_HPP

#include <stdexcept>
#include <cuda.h>

class CudaException: public std::runtime_error {
public:
  CudaException(cudaError_t error): std::runtime_error(cudaGetErrorString(error)), error(error) {}
  cudaError_t error;
};

class CudaStopwatch {
public:
    CudaStopwatch();
    ~CudaStopwatch();
    void start();
    void stop();
    bool isRunning() const;
    float elapsedTime();
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

private:
    bool running;
    bool initialized;
    void initializeEvent(cudaEvent_t& event);
    void destroyEvent(cudaEvent_t& event);
    void recordEvent(cudaEvent_t& event);
};

/**
 *Helper for constructor. Initalizes given event.
 *
 *@param event The cuda event to be initialized.
 *@throws CudaException
*/
void CudaStopwatch::initializeEvent(cudaEvent_t& event) {
  cudaError_t returnError = cudaEventCreate(&event);
  if (returnError != cudaSuccess) {
    throw CudaException(returnError);
  }
}

/**
 *Destroys the given event throwing an error if it fails
 *
 *@param event The cuda event to be destroyed
 *@throws CudaException
*/
void CudaStopwatch::destroyEvent(cudaEvent_t& event) {
  cudaError_t returnError = cudaEventDestroy(event);
  if (returnError != cudaSuccess) {
    throw CudaException(returnError);
  }
}

/**
 *Constructor for a stopwatch. Initalizes the events.
 *@throws CudaException
*/
CudaStopwatch::CudaStopwatch() {
  initializeEvent(startEvent);
  try {
    initializeEvent(stopEvent);
  } catch(CudaException& e) {
    destroyEvent(startEvent);
    throw e; //if the above throws an error, I don't care to throw this one too
  }

  initialized = true;
  running = false;
}

/**
 *Destructor for CudaStopwatch.
*/
CudaStopwatch::~CudaStopwatch() {
  if (initialized) {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
  }
}

/**
 *Starts/records the given event.
 *
 *@param event A cuda event to start/record.
 *@throws CudaException
 *@throws std::runtime_error
*/
void CudaStopwatch::recordEvent(cudaEvent_t& event) {
  if (!initialized) {
    throw std::runtime_error("Stopwatch not initialized!");
  }

  cudaError_t returnError = cudaEventRecord(event);
  if (returnError != cudaSuccess) {
    throw CudaException(returnError);
  }
}

/**
 *Starts the stopwatch
 *
 *@throws CudaException
 *@throws std::runtime_error
*/
void CudaStopwatch::start() {
  if (running) {
    throw std::runtime_error("Stopwatch already running!");
  }

  recordEvent(startEvent);
  running = true;
}

/**
 *Is the stopwatch running?
*/
bool CudaStopwatch::isRunning() const {
  return running;
}

/**
 *Stops the stopwatch/
 *
 *@throws CudaException
 *@throws std::runtime_error
*/
void CudaStopwatch::stop() {
  recordEvent(stopEvent);

  running = false;
}


/**
 *Returns the elapsed time.
 *
 *@throws CudaException
 *@throws std::runtime_error
*/
float CudaStopwatch::elapsedTime() {
  cudaError_t returnError = cudaEventSynchronize(stopEvent);
  if (returnError != cudaSuccess) {
    throw CudaException(returnError);
  }

  if (running) {
    throw std::runtime_error("Stopwatch running!");
  }

  if (!initialized) {
    throw std::runtime_error("Stopwatch not initialized!");
  }

  float elapsed = 0;

  returnError = cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
  if (returnError != cudaSuccess) {
    throw CudaException(returnError);
  }

  return elapsed;
}


#endif
