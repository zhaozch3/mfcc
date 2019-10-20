/*
zhaozch3
2019.10.20

计算语音帧的mfcc特征
不包含差分计算
*/

#ifndef MFCC_H
#define MFCC_H

#include <fftw3.h>
#include <stdint.h>

//44100hz 10ms
#define FRAME_LEN 441
//滤波器数目
#define FILTER_NUM 40

class mfcc {
public:
	float* _window;

	float* _energyspectrum;
	float* _mel;
	//结果
	float _ans[13];

	fftw_complex *in, *out;
	fftw_plan plan;

	mfcc();
	~mfcc();

	bool process(int16_t* frame, int32_t samplerate);
};

#endif
