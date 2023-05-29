#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include <thrust/async/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>
#include <chrono>

#define TOL 0.000001

int main(int argc, char **argv)
{
    std::cout << argv[0] << std::endl;
    
    int N=atoi(argv[1]);
    float A{2.0};
 
    thrust::device_vector<float> X(N);
    thrust::device_vector<float> Y(N);
    thrust::device_vector<float> Z(N);

    auto t1 = std::chrono::steady_clock::now();   // Start timing     
         
    auto e1=thrust::async::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0),X.begin(),Y.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(N),X.end(),Y.end())),
        [=]__device__ (auto tup)
        {
            int i=thrust::get<0>(tup);
            thrust::get<1>(tup)=1.0/float(i+1);
            thrust::get<2>(tup)=-2.0/float(i+1);            
        }
    );    
      
    auto e2=thrust::async::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(X.begin(),Y.begin(),Z.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(X.end(),Y.end(),Z.end())),
        [=]__device__ (auto tup)
        {
            thrust::get<2>(tup)=A*thrust::get<0>(tup)+thrust::get<1>(tup);    
        }
    );
   
   auto t2 = std::chrono::steady_clock::now();   // Start timing
   cudaDeviceSynchronize();   
   auto t3 = std::chrono::steady_clock::now();   // Start timing

   thrust::host_vector<float> Xh(X);
   thrust::host_vector<float> Yh(Y);
   thrust::host_vector<float> Zh(Z);

   auto t4 = std::chrono::steady_clock::now();   // Start timing

    // check
    for (int i = 0; i < N; i++) {
    assert(fabs(A * Xh[i] + Yh[i] - Zh[i]) < TOL);
    }

    std::cout
    << "before Q.wait "
    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
    << std::endl;
    
    std::cout
    << "after Q.wait "
    << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t1).count()
    << std::endl;

    std::cout
      << "after device to host copy "
      << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count()
      << std::endl;
}