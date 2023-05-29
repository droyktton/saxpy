#include <CL/sycl.hpp>
#define TOL 0.000001


int main(int argc, char **argv)
{
  std::cout << argv[0] << std::endl;

  size_t N = atoi(argv[1]);
  float A{2.0};

  sycl::queue Q;
  std::cout << "\n\nRunning on "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  sycl::buffer<float> X{N};
  sycl::buffer<float> Y{N};
  sycl::buffer<float> Z{N};
     
  auto t1 = std::chrono::steady_clock::now(); // Start timing

  auto e_fill = Q.submit([&](sycl::handler &h) {
    sycl::accessor aX{X,h,sycl::write_only};
    sycl::accessor aY(Y,h,sycl::write_only);
    h.parallel_for(sycl::range<1>{N},
                   [=](sycl::id<1> i) { aX[i] = aY[i] = 1.0 / float(i + 1); });
  });

  auto e_saxpy = Q.submit([&](sycl::handler &h) {
    sycl::accessor aX(X,h,sycl::read_only);
    sycl::accessor aY(Y,h,sycl::read_only);
    sycl::accessor aZ(Z,h,sycl::write_only);
    h.parallel_for(sycl::range<1>{N},
                   [=](sycl::id<1> i) { aZ[i] = A * aX[i] + aY[i]; });
  });


  auto t2 = std::chrono::steady_clock::now(); 
  Q.wait();
  auto t3 = std::chrono::steady_clock::now(); 

  sycl::host_accessor Xh(X);
  sycl::host_accessor Yh(Y);
  sycl::host_accessor Zh(Z);

  auto t4 = std::chrono::steady_clock::now(); 

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