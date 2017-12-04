/*
 *This code is NOT mine. It was written by Ashwin Nanjappa and posted publically
 *at https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/. I found
 *the link to that page on Piaza post @403.
 *
 *@author Ashwin Nanjappa.
*/
#ifndef ERRORMACROS_HPP
#define ERRORMACROS_HPP
#include <cuda.h>

// Define CUDA_ERROR_CHECK to turn on error checking

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line ) {
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

#endif
