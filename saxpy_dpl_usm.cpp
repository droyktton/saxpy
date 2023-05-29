#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/async>
#include <sycl/sycl.hpp>

#define TOL 0.000001

using namespace oneapi;

int main(int argc, char **argv)
{
  std::cout << argv[0] << std::endl;

  size_t N = atoi(argv[1]);
  float A{2.0};

  sycl::queue Q;
  std::cout << "\n\nRunning on "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  
  auto X = sycl::malloc_device<float>(N, Q);
  auto Y = sycl::malloc_device<float>(N, Q);
  auto Z = sycl::malloc_device<float>(N, Q);

  auto t1 = std::chrono::steady_clock::now(); // Start timing

  dpl::counting_iterator<int> count_a(0);
  dpl::counting_iterator<int> count_b = count_a + 10;
  auto zipped_begin = dpl::make_zip_iterator(X, Y);

  auto polfill = dpl::execution::make_device_policy<class Fill>(Q);
  auto e_fill = dpl::experimental::transform_async(
    polfill,count_a,count_b, zipped_begin,
    [=](auto i) 
    { 
        return std::make_tuple(1.0 / float(i + 1),42.0); 
    }
  );

  auto polsaxpy = dpl::execution::make_device_policy<class Saxpy>(Q);
  auto e_saxpy = dpl::experimental::transform_async(
    polsaxpy, X, X+N, Y, Z, 
    [=](auto x, auto y) { return A * x + y; },e_fill
  );

  auto t2 = std::chrono::steady_clock::now(); 
  Q.wait();
  auto t3 = std::chrono::steady_clock::now(); 

  auto Xh = (float *)malloc(sizeof(float)*N);
  auto Yh = (float *)malloc(sizeof(float)*N);
  auto Zh = (float *)malloc(sizeof(float)*N);

  auto e_copy1 = Q.memcpy(Xh, X, sizeof(float)*N, e_saxpy);
  auto e_copy2 = Q.memcpy(Yh, Y, sizeof(float)*N, e_saxpy);
  auto e_copy3 = Q.memcpy(Zh, Z, sizeof(float)*N, e_saxpy);

  Q.wait();
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
      
  sycl::free(X, Q);
  sycl::free(Y, Q);
  sycl::free(Z, Q);
}
