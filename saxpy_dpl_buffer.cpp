#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/async>
//#include <sycl/sycl.hpp>
#include <CL/sycl.hpp>

#define TOL 0.000001

using namespace oneapi;

int main(int argc, char **argv)
{
  std::cout << argv[0] << std::endl;

  size_t N = 1024;
  if(argc > 1) N = atoi(argv[1]);

  float A{2.0};

  sycl::queue Q;
  std::cout << "\n\nRunning on "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  sycl::buffer<float> X{N};
  sycl::buffer<float> Y{N};
  sycl::buffer<float> Z{N};

  auto t1 = std::chrono::steady_clock::now(); // Start timing

  auto zipped_begin = dpl::make_zip_iterator(dpl::begin(X), dpl::begin(Y));

  auto polfill = dpl::execution::make_device_policy<class Fill>(Q);
  auto e_fill = dpl::experimental::transform_async(
    polfill,
    dpl::counting_iterator<int>(0),dpl::counting_iterator<int>(N), 
    zipped_begin,
    [=](auto i) 
    { 
        return std::make_tuple(1.0 / float(i + 1),-2.0 / float(i + 1)); 
    }
  );

  auto polsaxpy = dpl::execution::make_device_policy<class Saxpy>(Q);
  
  auto e_saxpy = dpl::experimental::transform_async(
    polsaxpy,dpl::begin(X), dpl::begin(X)+N, dpl::begin(Y), dpl::begin(Z), 
    [=](auto x, auto y) { return A * x + y; },e_fill
  );

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
    if(i<5) std::cout << Xh[i] << "," << Yh[i] << ",    saxpy: " << A * Xh[i] + Yh[i] << "=?" << Zh[i] << std::endl;
  }
  std::cout << "\n\n";

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
