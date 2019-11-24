#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
           // labwork.labwork5_CPU();
           // labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
#pragma omp parallel for
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }                  
    }
 

}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
		printf("Fermi");
		printf("%d", cores);
            break;
        case 3: // Kepler
            cores = mp * 192;
		
		printf("%d", cores);		
            break;
        case 5: // Maxwell
            cores = mp * 128;
		printf("Maxwell");
		printf("%d", cores);
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
		printf("Pascal");
		printf("%d", cores);
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
	
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
printf("Number of core");
	getSPcores(prop);
        // something more here
        printf("\n Core clock rate %d\n", prop.clockRate);
	printf("Multiprocessor Core count %d\n", prop.multiProcessorCount);
	printf("WarpSize %d\n",prop.warpSize);
    }

}
__global__ void grayscale(uchar3 *input, uchar3 *output) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
output[tid].x = (input[tid].x + input[tid].y +
input[tid].z) / 3;
output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU() {
	 int pixelCount = inputImage->width * inputImage->height;
   	 outputImage = static_cast<char *>(malloc(pixelCount * 3));
	
	
    // Calculate number of pixels

    // Allocate CUDA memory
	uchar3 *devGray;    
	uchar3 *devInput;
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devGray, pixelCount * sizeof(uchar3));


    // Copy CUDA Memory from CPU to GPU
//cudaMemcpy()
//cudaMemcpy(devGray, outputImage, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
    // Processing
//rgb2grayCUDA<<<dimGrid, dimBlock>>>(devInput,devGray,regionSize);
	int blockSize =128;
	int numBlock = pixelCount / blockSize;
	grayscale<<<numBlock, blockSize>>>(devInput, devGray);
    // Copy CUDA Memory from GPU to CPU
//cudaMencpy()
	cudaMemcpy(outputImage, devGray , pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
	//cudaMemcpy(inputImage->buffer, devInput, pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
    // Cleaning
	cudaFree(devInput);
	cudaFree(devGray);
}

//void Labwork::labwork4_GPU() {}

//void Labwork::labwork5_CPU() {}

//void Labwork::labwork5_GPU() {}

//void Labwork::labwork6_GPU() {}

//void Labwork::labwork7_GPU() {}

//void Labwork::labwork8_GPU() {}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}

__global__ void grayscale2(uchar3 *input, uchar3 *output) {
int tx = threadIdx.x + blockIdx.x * blockDim.x;
int ty = threadIdx.y + blockIdx.y * blockDim.y;
int w = blockDim.x * gridDim.x;
int tid = ty*w + tx;
output[tid].x = (input[tid].x + input[tid].y +
input[tid].z) / 3;
output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU() {
         int pixelCount = inputImage->width * inputImage->height;
         outputImage = static_cast<char *>(malloc(pixelCount * 3));
                                                                   
       // Calculate number of pixels
    // Allocate CUDA memory
	 uchar3 *devGray;
	 uchar3 *devInput;
	 cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
 // Copy CUDA Memory from CPU to GPU 
//cudaMemcpy(devGray, outputImage, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
 cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
 // Processing
 //rgb2grayCUDA<<<dimGrid, dimBlock>>>(devInput,devGray,regionSize);
	dim3 blockSize = dim3(16,16);
	dim3 gridSize = dim3(inputImage->width/blockSize.x,inputImage->height/blockSize.y);

	grayscale2<<<gridSize, blockSize>>>(devInput, devGray);
        //int blockSize =128;
       // int numBlock = pixelCount / blockSize;
       // grayscale<<<numBlock, blockSize>>>(devInput, devGray);
    // Copy CUDA Memory from GPU to CPU //cudaMencpy()
        cudaMemcpy(outputImage, devGray , pixelCount*sizeof(uchar3),cudaMemcpyDeviceToHost);
        //cudaMemcpy(inputImage->buffer, devInput, pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
    // Cleaning
        cudaFree(devInput);
        cudaFree(devGray);
}                   

__global__ void Blurredfunction(uchar3 *input, uchar3 *output, int width, int height){
	__shared__ int red[16][16] ;
	__shared__ int green[16][16] ;
	__shared__ int blue[16][16] ;
	const int kernel[7][7] = {
						{0,0,1,2,1,0,0},
                        {0,3,13,22,13,3,0},
                        {1,13,59,97,59,13,1},
                        {2,22,97,159,97,22,2},
                        {1,13,59,97,59,13,1},
                        {0,3,13,22,13,3,0},
                        {0,0,1,2,1,0,0} 
	} ;  
	

        int tx = (threadIdx.x +blockIdx.x*16) -3;
        int ty = (threadIdx.y +blockIdx.y*16)-3; // + blockIdx.y * blockDim.y)-3;
        int index = (ty+3) *width ;
        int tid = tx +3 + index ;
        int tempx = 0 ;
        int tempy = 0;
        int tempz = 0; 
        int i ;
        int j ;
        int somme = 0 ;
        red[threadIdx.x][threadIdx.y] = input[tid].x ;   
        green[threadIdx.x][threadIdx.y] = input[tid].y ;   
        blue[threadIdx.x][threadIdx.y] = input[tid].z ;   
	__syncthreads() ;
        if ((tx < blockDim.x -4) and (tx > 2) and (ty< blockDim.y -4) and (ty>2)){
	        for  (i=0;i<7;i++){
 	               for (j=0;j<7;j++){
	                      tempx += (kernel[i][j]) * (red[threadIdx.x+i][threadIdx.y+j]) ;
                              tempy += (kernel[i][j]) * (green[threadIdx.x+i][threadIdx.y +j ]) ;
                              tempz += (kernel[i][j]) * (blue[threadIdx.x+i][threadIdx.y + j]) ;
                              somme += kernel[i][j] ;
                       }
                 }
        	tempx = tempx / somme ;
	        tempy = tempy / somme ;
        	tempz = tempz / somme ;
	        output[tid].x = tempx ;
        	output[tid].y = tempy ;
	        output[tid].z = tempz ;
        }
	__syncthreads() ;

}

void Labwork::labwork5_GPU(){
        int pixelCount =  inputImage->width * inputImage->height ;
       //allocate memory for the output on the host
       outputImage = static_cast<char *>(malloc(pixelCount * 3));
       // Allocate CUDA memory
       uchar3 *devInput ;
       uchar3 *devGray ;
       cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
       cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
       // Copy CUDA Memory from CPU to GPU
       cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
       // Processing
       dim3 blockSize = dim3(16,16);
       int numBlockx = inputImage-> width / (blockSize.x) ;
       int numBlocky = inputImage-> height / (blockSize.y) ;
       if ((inputImage-> width % (blockSize.x)) > 0) {
	       numBlockx++ ;
       }
       if ((inputImage-> height % (blockSize.y)) > 0){
	       numBlocky++ ;
	}
       dim3 gridSize = dim3(numBlockx,numBlocky) ;
       Blurredfunction<<<gridSize,blockSize>>>(devInput,devGray, inputImage->width, inputImage->height) ;
       // Copy CUDA Memory from GPU to CPU
       cudaMemcpy(outputImage, devGray,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
       // Cleaning
       cudaFree(devInput) ;
       cudaFree(devGray) ;
}



__global__ void binarization(uchar3 *input,uchar3 *output,int width, int threshold){
	int tx = (threadIdx.x + blockIdx.x * blockDim.x);
        int ty = (threadIdx.y + blockIdx.y * blockDim.y);
        int index = ty *width ;
		int tid = tx + index ;
		if (input[tid].x < threshold){
			output[tid].x = 0 ;
		} else {
			output[tid].x = 255 ;
		}
		output[tid].z = output[tid].y = output[tid].x ;
}

__global__ void brightnessControl(uchar3 *input,uchar3 *output, int width,  int pourcentage){
        int tx = (threadIdx.x + blockIdx.x * blockDim.x);
        int ty = (threadIdx.y + blockIdx.y * blockDim.y);
        int index = ty *width ;
        int tid = tx + index ;
		int a ;
		int incValue = pourcentage*255/100 ;
		a = input[tid].x + incValue ;
		if (a > 255){ output[tid].x = 255 ;}
			else{ output[tid].x = a ;}
		output[tid].z = output[tid].y = output[tid].x ;

}


void Labwork::labwork6_GPU() {
         int pixelCount = inputImage->width * inputImage->height;
         outputImage = static_cast<char *>(malloc(pixelCount * 3));
                                                                   
       // Calculate number of pixels
    // Allocate CUDA memory
	 uchar3 *devGray;
	 uchar3 *devInput;
	 cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
 // Copy CUDA Memory from CPU to GPU 
//cudaMemcpy(devGray, outputImage, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
 cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
 // Processing
 //rgb2grayCUDA<<<dimGrid, dimBlock>>>(devInput,devGray,regionSize);
	dim3 blockSize = dim3(16,16);
        int numBlockx = inputImage-> width / (blockSize.x) ;
        int numBlocky = inputImage-> height / (blockSize.y) ;
        if ((inputImage-> width % (blockSize.x)) > 0) {
	        numBlockx++ ;
	} 
        if ((inputImage-> height % (blockSize.y)) > 0){
		numBlocky++ ;
	}
       dim3 gridSize = dim3(numBlockx,numBlocky) ;
       binarization<<<gridSize,blockSize>>>(devInput,devGray, inputImage->width,128) ;
//       brightnessControl<<<gridSize,blockSize>>>(devInput,devGray, inputImage->width,50) ;

        //int blockSize =128;
       // int numBlock = pixelCount / blockSize;
       // grayscale<<<numBlock, blockSize>>>(devInput, devGray);
    // Copy CUDA Memory from GPU to CPU //cudaMencpy()
        cudaMemcpy(outputImage, devGray,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
        //cudaMemcpy(inputImage->buffer, devInput, pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
    // Cleaning
        cudaFree(devInput);
        cudaFree(devGray);
}  


__global__ void  grayscale3(uchar3 *input,uchar3*output){
	extern  __shared__ int cache[];
	int max= 0  ;
	int min=255 ;
	
	unsigned int tid2 = threadIdx.x;
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int w = blockDim.x * gridDim.x;
	int tid = ty*w + tx;
	
	cache[tid2] = input[tid].x ;
	__syncthreads() ;
	for (int i=1; i< blockDim.x; i*=2){
		if (tid2 % (i*2) == 0) {
			if (cache[tid2] < cache[tid2+i]){
				cache[tid2] = cache[tid2 +i] ;
				
			}
		}
	__syncthreads() ;
	}
	if (tid2 == 0) {max = cache[0] ;}

	cache[tid2] = input[tid].x;
	__syncthreads() ;
	for (int i=1; i< blockDim.x; i*=2){
		if (tid2 % (i*2) == 0) {
			if (cache[tid2] > cache[tid2+i]){
				cache[tid2] = cache[tid2 +i] ;
				
			}
		}
	__syncthreads() ;
	}
	if (tid2 == 0) {min = cache[0] ;}
	__syncthreads() ;
	output[tid].x = (255*(input[tid].x - min))/(max-min) ;
	output[tid].z = output[tid].y = output[tid].x ;
	
}


void Labwork::labwork7_GPU() {
       int pixelCount = inputImage->width * inputImage->height ;
        //allocate memory for the output on the host
        outputImage = static_cast<char *>(malloc(pixelCount * 3));
        // Allocate CUDA memory
        uchar3 *devInput ;
		uchar3 * devInputBis ;
		uchar3 *devGray ;
		
		cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
        cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
        cudaMalloc(&devInputBis, pixelCount * sizeof(uchar3));
        // Copy CUDA Memory from CPU to GPU
        cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
        // Processing
        dim3 blockSize = dim3(16,16);
	int sharedMemSize = 256 ;
        int numBlockx = inputImage-> width / (blockSize.x) ;
        int numBlocky = inputImage-> height / (blockSize.y) ;
        if ((inputImage-> width % (blockSize.x)) > 0) {
	        numBlockx++ ;
	} 
        if ((inputImage-> height % (blockSize.y)) > 0){
		numBlocky++ ;
	}
       dim3 gridSize = dim3(numBlockx,numBlocky) ;
       grayscale2<<<gridSize,blockSize>>>(devInput,devInputBis) ;
       grayscale3<<<gridSize,blockSize,sharedMemSize>>>(devInputBis,devGray) ;
       
       // Copy CUDA Memory from GPU to CPU
       cudaMemcpy(outputImage, devGray,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
       // Cleaning
       cudaFree(devInput) ;
       cudaFree(devGray) ;

}


struct hsv {
    float *h, *s, *v;
};

__global__ void RGB2HSV(uchar3* input, hsv output, int imageWidth, int imageHeight){
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    if(tx >= imageWidth || ty >= imageHeight) return;
    int tid = tx + ty * imageWidth;

    float r = (float)input[tid].x/255.0;
    float g = (float)input[tid].y/255.0;
    float b = (float)input[tid].z/255.0;

    float Max = max(r, max(g,b));
    float Min = min(r, min(g,b));
    float delta = Max - Min;

    float h = 0;
    float s = 0;
    float v = 0;

    if (Max != 0){
        s = delta/Max;
        if (Max == r) h = 60 * fmodf(((g-b)/delta),6.0);
        if (Max == g) h = 60 * ((b-r)/delta+2);
        if (Max == b) h = 60 * ((r-g)/delta+4);
    }

    if (Max == 0) s = 0;
    if (delta == 0) h = 0;
    v = Max;

    output.h[tid] = h;
    output.s[tid] = s;
    output.v[tid] = v;
}

__global__ void HSV2RGB(hsv input, uchar3* output, int imageWidth, int imageHeight){
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    if(tx >= imageWidth || ty >= imageHeight) return;
    int tid = tx + ty * imageWidth;
    
    float h = input.h[tid];
    float s = input.s[tid];
    float v = input.v[tid];

    float d = h/60;
    float hi = (int)d % 6;
    float f = d - hi;
    float l = v * (1-s);
    float m = v * (1-f*s);
    float n = v * (1-(1-f)*s);

    float r,g,b;
    if (h >= 0 && h < 60){
        r = v;
        g = n;
        b = l;
    }

    if (h >= 60 && h < 120){
        r = m;
        g = v;
        b = l;
    }

    if (h >= 120 && h < 180){
        r = l;
        g = v;
        b = n;
    }

    if (h >= 180 && h < 240){
        r = l;
        g = m;
        b = v;
    }

    if (h >= 240 && h < 300){
        r = n;
        g = l;
        b = v;
    }

    if (h >= 300 && h < 360){
        r = v;
        g = l;
        b = m;
    }

    output[tid].x = r*255;
    output[tid].y = g*255;
    output[tid].z = b*255;
}


void Labwork::labwork8_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devGray;
    hsv devHSV;
    cudaMalloc((void**)&devHSV.h, pixelCount *sizeof(float));
    cudaMalloc((void**)&devHSV.s, pixelCount *sizeof(float));
    cudaMalloc((void**)&devHSV.v, pixelCount *sizeof(float));
    cudaMalloc(&devInput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount *sizeof(uchar3));

    // Copy InputImage from CPU (host) to GPU (device)
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);

    // Processing 
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((inputImage->width + blockSize.x -1) / blockSize.x, (inputImage->height + blockSize.y -1) / blockSize.y);

    RGB2HSV<<<gridSize, blockSize>>>(devInput, devHSV, inputImage->width, inputImage->height);
    HSV2RGB<<<gridSize, blockSize>>>(devHSV, devGray, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU

    outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));  
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);   

    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
    cudaFree(devHSV.h);
    cudaFree(devHSV.s);
    cudaFree(devHSV.v);    
}
