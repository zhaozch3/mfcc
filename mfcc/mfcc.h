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
//�������ά��
#define NDCTS 13

class mfcc {
public:
	float* _window;
	float* _lift_window;

	float* _energyspectrum;
	float* _mel;
	//���
	float _ans[NDCTS];

	fftw_complex *in, *out;
	fftw_plan plan;

	mfcc();
	~mfcc();

	bool process(float* frame, int32_t samplerate);
};

#endif
