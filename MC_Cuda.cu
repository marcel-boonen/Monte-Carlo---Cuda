#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <curand_kernel.h>




void print(float* a, int a_size){
    std::cout<<"[ ";

    for(int i = 0; i!= a_size; i++){
        std::cout<<a[i]<<" ";
    }
    std::cout<<"]"<<std::endl;
}

float* arr_gen(const float lower_lim, const float upper_lim, const int N){
    float* arr =new float[N];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lower_lim, upper_lim);
    for (int i = 0; i < N; ++i){
        arr[i] = dis(gen);
    }
    return arr;
}

float func(float x, float y){
    return std::pow(x,2)*y;
}

__global__ void summed(const float* input, float* output,const int N){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    output[j]=0;
    __syncthreads();
    for (int i = j; i < N; i += blockDim.x * gridDim.x){
            output[j]+= input[i];
    }
}
__device__ void wrap_reduction(volatile float* sh_data, int thread_idx){
    sh_data[thread_idx]+= sh_data[thread_idx+32];
    sh_data[thread_idx]+= sh_data[thread_idx+16];
    sh_data[thread_idx]+= sh_data[thread_idx+8];
    sh_data[thread_idx]+= sh_data[thread_idx+4];
    sh_data[thread_idx]+= sh_data[thread_idx+2];
    sh_data[thread_idx]+= sh_data[thread_idx+1];
    
}
__global__ void lastwrap_MC (const float* a,float* sum_data, const int N){
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int j = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    if(j<N){
        sh[tid] = a[j] + a[j+blockDim.x];
    }
    else{
        sh[tid]=0;
    }   
    __syncthreads();
    
    for (unsigned int s=blockDim.x/2; s>32; s>>=1){
        if (tid < s) {
        sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }
    if(tid <32){
        wrap_reduction(sh,tid);
    }

    if (tid == 0) {
        sum_data[blockIdx.x] = sh[0];
    }
    
}

__global__ void addload_MC(const float* a, float* sum_data, const int N){
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int j = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    if(j<N){
        sh[tid] = a[j] + a[j+blockDim.x];
    }
    else{
        sh[tid]=0;
    }   
    __syncthreads();
    
    for (unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s) {
        sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sum_data[blockIdx.x] = sh[0];
    }

}
__global__ void seq_MC(const float* a, float* sum_data, const int N){
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){
        sh[tid]=a[j];
    }
    else{
        sh[tid]=0;
    }
    __syncthreads();
    
    for (unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s) {
        sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sum_data[blockIdx.x] = sh[0];
    }

}

__global__ void interleav_MC (const float* a,float* sum_data, const int N){
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<N){
        sh[tid]=a[j];
    }
    else{
        sh[tid]=0;
    }
    __syncthreads();
    
    for(int s=1; s < blockDim.x; s *= 2 ) {
            if (tid % (2*s) == 0) {
                sh[tid] += sh[tid + s];
            }
            __syncthreads();
    }
    if (tid == 0) {
        sum_data[blockIdx.x] = sh[0];
    }

}


__global__ void naive_MC (const float* a,float* sum_data, const int N){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    sum_data[j]=0;
    __syncthreads();
    if(j<N){
            sum_data[j] = a[j];
    }
    __syncthreads();
    if(j <blockDim.x){
        for (int i = threadIdx.x+blockDim.x; i < blockDim.x*gridDim.x; i+=blockDim.x){
            sum_data[j]+=sum_data[i];
        }
    }
    __syncthreads();
}





__global__ void generate_random(float* out, const int N, const float lower_lim, const float upper_lim, const unsigned long seed) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed, j, 0, &state); 

    if(blockDim.x * gridDim.x < N){
        for (int i = j; i < N; i += blockDim.x * gridDim.x){
            out[i]= lower_lim + (upper_lim - lower_lim) * curand_uniform(&state);
        }
    }

    else{
        if(j<N){
            out[j] = lower_lim + (upper_lim - lower_lim) * curand_uniform(&state);  // Generates random float in (0, 1]
        }
    }
}

__global__ void func_GPU(float* x, float* y, float* output, const int N){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = j; i < N; i += blockDim.x * gridDim.x){
            output[i]=pow(x[i],2)*y[i];
    }
}



int main(){
    std::chrono::steady_clock::time_point total_begin = std::chrono::steady_clock::now();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 3000);
    int the_seed_x= dis(gen);
    int the_seed_y= dis(gen);
   
    int numSMs;
    int devId = 0; 
    cudaError_t err = cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    int t=256*32*numSMs;
    
    int N=100000000;   
    //int N=10000000;
   
    float a_x=2;
    float b_x=10;
    float a_y=3;
    float b_y=5;



    float* d_x;
    float* d_y;
    cudaMalloc((void**)&d_x, N*sizeof(float));
    cudaMalloc((void**)&d_y, N*sizeof(float));
    generate_random<<<32*numSMs, 256>>>(d_x, N, a_x, b_x, the_seed_x);
    generate_random<<<32*numSMs, 256>>>(d_y, N, a_y, b_y, the_seed_y);
    cudaDeviceSynchronize();

    float* h_sum_data=new float[10];
    

    float* d_f;
    float* d_intgr;
    float* d_sum_data;
    float* d_first_sum;
    float* d_second_sum;
    
    cudaMalloc((void**)&d_f, N*sizeof(float));
    cudaMalloc((void**)&d_intgr, sizeof(float));
    cudaMalloc((void**)&d_first_sum, t*sizeof(float));
    cudaMalloc((void**)&d_second_sum, numSMs*32*sizeof(float));
    cudaMalloc((void**)&d_sum_data, 10*sizeof(float));

    func_GPU<<<32*numSMs, 256>>>(d_x,d_y,d_f,N);
    summed<<<32*numSMs,256>>>(d_f, d_first_sum, N);
    float integral=0;
    std::string method="seq_MC";

    

    if(method=="naive_MC"){
        float* d_naive_sum;
        cudaMalloc((void**)&d_naive_sum, 256*sizeof(float));
        naive_MC<<<32*numSMs, 256>>>(d_first_sum,d_naive_sum,32*numSMs*256);
        cudaDeviceSynchronize();
        cudaMemcpy(h_sum_data,d_naive_sum, 256*sizeof(float), cudaMemcpyDeviceToHost);
        for(int i=0; i<256; i++){integral+=h_sum_data[i];}
        std::cout<<integral*(b_x-a_x)*(b_y-a_y)/N<<std::endl;
        cudaFree(d_naive_sum);

    }

    if(method=="interleav_MC"){
        interleav_MC<<<32*numSMs,256,256*sizeof(float)>>>(d_first_sum,d_second_sum, 32*numSMs*256);
        interleav_MC<<<10,256,256*sizeof(float)>>>(d_second_sum,d_sum_data,32*numSMs);
        cudaDeviceSynchronize();
        cudaMemcpy(h_sum_data,d_sum_data, 10*sizeof(float), cudaMemcpyDeviceToHost);
        for(int i=0; i<10; i++){integral+=h_sum_data[i];}
        std::cout<<integral*(b_x-a_x)*(b_y-a_y)/N<<std::endl;
    }

    if(method=="seq_MC"){
        seq_MC<<<32*numSMs,256,256*sizeof(float)>>>(d_first_sum,d_second_sum, 32*numSMs*256);
        seq_MC<<<10,256,256*sizeof(float)>>>(d_second_sum,d_sum_data,32*numSMs);
        cudaDeviceSynchronize();
        cudaMemcpy(h_sum_data,d_sum_data, 10*sizeof(float), cudaMemcpyDeviceToHost);
        for(int i=0; i<10; i++){integral+=h_sum_data[i];}
        std::cout<<integral*(b_x-a_x)*(b_y-a_y)/N<<std::endl;
    
    }

    if(method=="addload_MC"){
        addload_MC<<<32*numSMs,256,256*sizeof(float)>>>(d_first_sum,d_second_sum, 32*numSMs*256);
        addload_MC<<<10,256,256*sizeof(float)>>>(d_second_sum,d_sum_data,32*numSMs);
        cudaDeviceSynchronize();
        cudaMemcpy(h_sum_data,d_sum_data, 10*sizeof(float), cudaMemcpyDeviceToHost);
        for(int i=0; i<10; i++){integral+=h_sum_data[i];}
        std::cout<<integral*(b_x-a_x)*(b_y-a_y)/N<<std::endl;    
    }

    if(method=="lastwrap_MC"){
        lastwrap_MC<<<32*numSMs,256,256*sizeof(float)>>>(d_first_sum,d_second_sum, 32*numSMs*256);
        lastwrap_MC<<<10,256,256*sizeof(float)>>>(d_second_sum,d_sum_data,32*numSMs);
        cudaDeviceSynchronize();
        cudaMemcpy(h_sum_data,d_sum_data, 10*sizeof(float), cudaMemcpyDeviceToHost);
        for(int i=0; i<10; i++){integral+=h_sum_data[i];}
        std::cout<<integral*(b_x-a_x)*(b_y-a_y)/N<<std::endl;
    
    }
    

    cudaFree(d_f);
    cudaFree(d_intgr);
    cudaFree(d_first_sum);
    cudaFree(d_second_sum);
    cudaFree(d_sum_data);
    cudaFree(d_x);
    cudaFree(d_y);
    delete [] h_sum_data;

    
    std::chrono::steady_clock::time_point total_end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_begin).count() << "[µs]" << std::endl;

    return 0;
}
