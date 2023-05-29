#include <CL/sycl.hpp>
#define TOL 0.000001

int main(int argc, char **argv)
{
  std::cout << argv[0] << std::endl;

  size_t N = 1024;
  if(argc > 1) N = atoi(argv[1]);

  float A{2.0};

  sycl::queue Q;
  std::cout << "\n\nRunning on "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  
  auto X = sycl::malloc_device<float>(N, Q);
  auto Y = sycl::malloc_device<float>(N, Q);
  auto Z = sycl::malloc_device<float>(N, Q);

  auto t1 = std::chrono::steady_clock::now(); // Start timing

  auto e_fill = Q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{N},
                   [=](sycl::id<1> i) { X[i] = Y[i] = 1.0 / float(i + 1); });
  });

  auto e_saxpy = Q.submit([&](sycl::handler &h) {
    h.depends_on(e_fill);
    h.parallel_for(sycl::range<1>{N},
                   [=](sycl::id<1> i) { Z[i] = A * X[i] + Y[i]; });
  });

  auto t2 = std::chrono::steady_clock::now(); 
  Q.wait();
  auto t3 = std::chrono::steady_clock::now(); // Start timing

  auto Xh = (float *)malloc(sizeof(float)*N);
  auto Yh = (float *)malloc(sizeof(float)*N);
  auto Zh = (float *)malloc(sizeof(float)*N);

  auto e_copy1 = Q.memcpy(Xh, X, sizeof(float)*N, e_saxpy);
  auto e_copy2 = Q.memcpy(Yh, Y, sizeof(float)*N, e_saxpy);
  auto e_copy3 = Q.memcpy(Zh, Z, sizeof(float)*N, e_saxpy);

  Q.wait();

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

  sycl::free(X, Q);
  sycl::free(Y, Q);
  sycl::free(Z, Q);
}
