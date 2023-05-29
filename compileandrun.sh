clang++ -v -fsycl -fsycl-targets=nvptx64-nvidia-cuda $1 \
-I/usr/local/cuda/targets/x86_64-linux/include/ \
-L/usr/local/cuda/targets/x86_64-linux/lib \
-I/opt/intel/oneapi/dpl/2022.0.0/linux/include/ \
-I/opt/intel/oneapi/mkl/2023.0.0/include/ \
-lnvToolsExt -o $1_gpu.out
#clang++ -fsycl $1 -I/usr/local/cuda/targets/x86_64-linux/include/ -L/usr/local/cuda/targets/x86_64-linux/lib -I/opt/intel/oneapi/dpl/2022.0.0/linux/include/ -lnvToolsExt  -o $1_cpu.out

#clang++ -fsycl fill.cpp -D$1 
#echo "EN GPU"
#./a.out
#echo 
#echo "EN CPU"
#./b.out
