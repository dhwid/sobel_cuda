#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <ctime>;
#include <iostream>;
#include "opencv2/highgui.hpp"

using namespace std;

#define BLOCK_SIZE 32

void generate_sobel_mask(int *mask) {
	int i = 0;

	mask[i++] = -1; mask[i++] = 0; mask[i++] = 1;
	mask[i++] = -2; mask[i++] = 0; mask[i++] = 2;
	mask[i++] = -1; mask[i++] = 0; mask[i++] = 1;

	mask[i++] = 0; mask[i++] = 1; mask[i++] = 2;
	mask[i++] = -1; mask[i++] = 0; mask[i++] = 1;
	mask[i++] = -2; mask[i++] = -1; mask[i++] = 0;

	mask[i++] = 1; mask[i++] = 2; mask[i++] = 1;
	mask[i++] = 0; mask[i++] = 0; mask[i++] = 0;
	mask[i++] = -1; mask[i++] = -2; mask[i++] = -1;

	mask[i++] = 2; mask[i++] = 1; mask[i++] = 0;
	mask[i++] = 1; mask[i++] = 0; mask[i++] = -1;
	mask[i++] = 0; mask[i++] = -1; mask[i++] = -2;


	for (int i = 0; i < 36; i++) {
		mask[i + 36] = -mask[i];
	}

}

__global__ void convolve(int *Sobel, int *R, int *G, int *B,
	uchar *data, int height, int width, int step, int nchannels) {

	int j= blockDim.x * blockIdx.x + threadIdx.x + 1;


	if (j < (width - 1)) {
		for (int i = 1; i < height - 1; i++) {

			int temp_pixel[8];
			int temp_pixels_sum = 0;

			for (int l = 0; l < 8; l++)
				temp_pixel[l] = 0;


			for (int l = 0; l < 72; l += 9) {
				for (int m = 0; m < 3; m++) {
					for (int n = 0; n < 3; n++) {
						temp_pixel[l / 9] += Sobel[l + (m * 3 + n)] * data[(i - 1 + m)*(step)+(j - 1 + n)*nchannels + 0];
					}
				}
			}

			for (int l = 0; l < 8; l++)
				temp_pixels_sum += abs(temp_pixel[l]);

			R[i*width + j] = temp_pixels_sum / 8;

			///////////////////////

			//clear variables
			temp_pixels_sum = 0;
			for (int l = 0; l < 8; l++)
				temp_pixel[l] = 0;

			for (int l = 0; l < 72; l += 9) {
				for (int m = 0; m < 3; m++) {
					for (int n = 0; n < 3; n++) {
						temp_pixel[l / 9] += Sobel[l + (m * 3 + n)] * data[(i - 1 + m)*(step)+(j - 1 + n)*nchannels + 1];
					}
				}
			}

			for (int l = 0; l < 8; l++)
				temp_pixels_sum += abs(temp_pixel[l]);

			G[i*width + j] = temp_pixels_sum / 8;
			/////////////////////////////
			
			//clear variables
			temp_pixels_sum = 0;
			for (int l = 0; l < 8; l++)
				temp_pixel[l] = 0;
			
			for (int l = 0; l < 72; l += 9) {
				for (int m = 0; m < 3; m++) {
					for (int n = 0; n < 3; n++) {
						temp_pixel[l / 9] += Sobel[l + (m * 3 + n)] * data[(i - 1 + m)*(step)+(j - 1 + n)*nchannels + 2];
					}
				}
			}

			for (int l = 0; l < 8; l++)
				temp_pixels_sum += abs(temp_pixel[l]);

			B[i*width + j] = temp_pixels_sum / 8;
		}
	}

	if (j < (width - 1)) {
		for (int i = 1; i < height - 1; i++) {
			data[i*step + j * nchannels + 0] = R[i* width + j];
			data[i*step + j * nchannels + 1] = G[i* width + j];
			data[i*step + j * nchannels + 2] = B[i *width + j];
		}
	}
}


void detect_edges(IplImage* ramka, int *mask) {
	int width = ramka->width;
	int height = ramka->height;
	int nchannels = ramka->nChannels;
	int step = ramka->widthStep;
	int GRID_SIZE = width * nchannels / BLOCK_SIZE;
	cudaError_t cudaStatus;

	int*d_Sobel, *d_R, *d_G, *d_B;
	uchar *d_data;
	int SobelSize = 8 * 9 * sizeof(int);
	int RGBsize = width * step * sizeof(int);
	int dataSize = (step*height + (nchannels*width + 2)) * sizeof(uchar);


	//alloc space for device
	cudaMalloc((void**)&d_Sobel, SobelSize);
	cudaMalloc((void**)&d_R, RGBsize);
	cudaMalloc((void**)&d_G, RGBsize);
	cudaMalloc((void**)&d_B, RGBsize);
	cudaMalloc((void**)&d_data, dataSize);

	//setup initial values
	uchar *data = (uchar*)ramka->imageData;

	//copy inputs to device
	cudaMemcpy(d_data, data, dataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Sobel, mask, SobelSize, cudaMemcpyHostToDevice);

	convolve << <GRID_SIZE, BLOCK_SIZE >> >(d_Sobel, d_R, d_G, d_B, d_data, height, width, step, nchannels);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(data, d_data, dataSize, cudaMemcpyDeviceToHost);

	//free(data);
	cudaFree(d_R); cudaFree(d_G); cudaFree(d_B); cudaFree(d_data); cudaFree(d_Sobel);
}

int main() {

	int *Sobel;
	int SobelSize = 8 * 9 * sizeof(int);
	Sobel = (int*)malloc(SobelSize);
	generate_sobel_mask(Sobel);

	// odczytanie pliku avi
	CvCapture* vid = cvCreateFileCapture("small.mp4");

	// odczytanie pierwszej klatki 
	cvQueryFrame(vid);
	IplImage* frame = cvQueryFrame(vid);

	double fps = cvGetCaptureProperty(vid, CV_CAP_PROP_FPS);
	int delay = 1000 / fps;

	CvVideoWriter* vr = cvCreateVideoWriter("out.avi", 
		CV_FOURCC_DEFAULT, fps, cvSize(frame->width, frame->height));
	clock_t begin = clock();

	for (;;)
	{
		// pobranie kolejnej ramki
		IplImage* frame = cvQueryFrame(vid);

		if (frame != 0) {
			cvShowImage("input", frame);
			detect_edges(frame, Sobel);
			cvShowImage("output", frame);
			cvWriteFrame(vr, frame);
		}
		else
			break;

		cvWaitKey(delay);

	}
	clock_t end = clock();

	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Obliczenia trwaly:  ";
	cout << elapsed_secs;
	cout << " sekund \n\n";

	system("Pause");

	// zwolnienie zasobw
	cvDestroyAllWindows();
	cvReleaseCapture(&vid);
	cvReleaseVideoWriter(&vr);
}