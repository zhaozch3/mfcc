/*
zhaozch3
2019.10.20

��������֡��mfcc����
��������ּ���
*/

#ifndef MFCC_H
#define MFCC_H

#include <fftw3.h>
#include <stdint.h>

//44100hz 10ms
#define FRAME_LEN 441
//�˲�����Ŀ
#define FILTER_NUM 40

class mfcc {
public:
	float* _window;

	float* _energyspectrum;
	float* _mel;
	//���
	float _ans[13];

	fftw_complex *in, *out;
	fftw_plan plan;

	mfcc();
	~mfcc();

	bool process(int16_t* frame, int32_t samplerate);
};

#endif
