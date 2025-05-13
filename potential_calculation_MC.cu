#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <curand_kernel.h>
#include <math.h>
#include <iomanip>



void print(float* a, int a_size){
    std::cout<<"[";

    for(int i = 0; i< a_size-1; i++){
        std::cout<<a[i]<<",";
    }
    std::cout<<a[a_size-1];
    std::cout<<"]"<<std::endl;
}


__global__ void summed(const float* input, float* output,const int N){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    output[j]=0;
    __syncthreads();
    for (int i = j; i < N; i += blockDim.x * gridDim.x){
            output[j]+= input[i];
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

__device__ float rho(float x, float y, float z){
    float mu_x=0;float sigma_x=0.1;
    float mu_y=0;float sigma_y=0.1;
    float mu_z=0;float sigma_z=0.1;

    /*
    if(pow(x,2)+pow(y,2)+pow(z,2)<26 && pow(x,2)+pow(y,2)+pow(z,2)>24){
        return 10;
    }
    else{
        return 0;
    */
    
    return expf(-pow((x-mu_x)/sigma_x,2))*cos(0.5*y)*cos(z)*(1/(z+1)+(z+z+z*y)+y*y*z);
    //return  expf(-pow((x-mu_x)/sigma_x,2))*expf(-pow((y-mu_y)/sigma_y,2))*expf(-pow((z-mu_z)/sigma_z,2))*10*z*pow((1-pow(z,2)),4);
    //return  expf(-pow((x-mu_x)/sigma_x,2))*(pow(y+pow(z,2),4));
    float f=expf(-pow((y-3)/sigma_y,2))*expf(-pow((z-3)/sigma_z,2));
    float s=-expf(-pow((y-3)/sigma_y,2))*expf(-pow((z+3)/sigma_z,2));
    float t=-expf(-pow((y+3)/sigma_y,2))*expf(-pow((z-3)/sigma_z,2));
    float q=expf(-pow((y+3)/sigma_y,2))*expf(-pow((z+3)/sigma_z,2));
   
    //float t=-expf(-pow((y+3)/sigma_y,2))*expf(-pow((z-0)/sigma_z,2));
    //float q=expf(-pow((y-3)/sigma_y,2))*expf(-pow((z+0)/sigma_z,2));
    //return f+t+s+q;
}   

__global__ void func(float* R, float* x, float* y, float* z, float* output, const int N){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = j; i < N; i += blockDim.x * gridDim.x){
        float norm=sqrtf(pow(x[i]-R[0],2)+pow(y[i]-R[1],2)+pow(z[i]-R[2],2));
        if(norm==0){
            output[i]= rho(x[i], y[i], z[i])*(1e20);
            }
        else{
            output[i]= rho(x[i], y[i], z[i])/norm;
        }
    }
}



int main(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_q(0, 3000);
    std::uniform_int_distribution<> dis_p(0, 3000);

    int the_seed[3]; 
    for(int i=0; i<3;i++){
        the_seed[i]= dis_p(gen);
    }
  
    int numSMs;
    int devId = 0; 
    cudaError_t err = cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    int t=256*32*numSMs;

    int N=10000000;   
    float a=-20;
    float b=20;

    
    float* d_R;
    float* d_rx; float* d_ry; float* d_rz;

    cudaMalloc((void**)&d_R, 3*sizeof(float));

    cudaMalloc((void**)&d_rx, N*sizeof(float)); 
    cudaMalloc((void**)&d_ry, N*sizeof(float)); 
    cudaMalloc((void**)&d_rz, N*sizeof(float));

    float* d_output; 
    float* d_sum_data;
    float* d_first_sum;
    float* d_second_sum;
   

    cudaMalloc((void**)&d_output, N*sizeof(float)); 
    cudaMalloc((void**)&d_first_sum, t*sizeof(float));
    cudaMalloc((void**)&d_second_sum, numSMs*32*sizeof(float));
    cudaMalloc((void**)&d_sum_data, (32*numSMs/256)*sizeof(float));
   

    generate_random<<<32*numSMs, 256>>>(d_rx, N, a, b, the_seed[0]);
    generate_random<<<32*numSMs, 256>>>(d_ry, N, a, b, the_seed[1]);
    generate_random<<<32*numSMs, 256>>>(d_rz, N, a, b, the_seed[2]);
    int index=0;
    int index_x=0;
    int index_y=0;
    int index_z=0;
    float* phi= new float[1764];
    //float* R_x= new float[1];
    float* R_y= new float[1764];
    float* R_z= new float[1764];

    float* h_sum_data=new float[(32*numSMs/256)];
    for(float i=1; i<2;i++){
        for(float j=-10.25; j<10.75;j=j+0.5){
            for(float k=-10.25; k<10.75;k=k+0.5){  
                float R[3] = {0,j,k};
                cudaMemcpy(d_R,R, 3*sizeof(float), cudaMemcpyHostToDevice);
            
                func<<<32*numSMs, 256>>>(d_R,d_rx, d_ry, d_rz,d_output, N);
                summed<<<32*numSMs,256>>>(d_output, d_first_sum, N);
                seq_MC<<<32*numSMs,256,256*sizeof(float)>>>(d_first_sum,d_second_sum, 32*numSMs*256);
                seq_MC<<<32*numSMs/256,256,256*sizeof(float)>>>(d_second_sum,d_sum_data, 32*numSMs);
                cudaDeviceSynchronize();
                
            
                float integral=0.0;
                cudaMemcpy(h_sum_data,d_sum_data, (32*numSMs/256)*sizeof(float), cudaMemcpyDeviceToHost);
                for(int i=0; i<(32*numSMs/256); i++){integral+=h_sum_data[i];}
               
                
                phi[index]=integral*pow(b-a,3)/(N);
                //R_x[index_x]=i;
                R_y[index_y]=j;
                R_z[index_z]=k; 

                index++;
                index_x++;
                index_y++;
                index_z++;
            }
            
        }   
        
    }
    

        

    print(phi,1764);
    //print(R_x,1);
    std::cout<<std::endl<<std::endl;

    print(R_y,1764);
    std::cout<<std::endl<<std::endl;
    print(R_z,1764);


    cudaFree(d_rx);
    cudaFree(d_ry);
    cudaFree(d_rz);
    cudaFree(d_output);
    cudaFree(d_sum_data);
    cudaFree(d_first_sum);
    cudaFree(d_second_sum);
    cudaFree(d_R);
    delete [] phi;
    //delete [] R_x;
    delete [] R_y;
    delete [] R_z;
    delete [] h_sum_data;
    

    

    
    


    //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
    std::cout << numSMs <<std::endl;

    
    return 0;
}
