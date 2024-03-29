#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include "mfcc.h"

float pi = 3.14159265358979323846;

void lift_window(float* window, int len) {
	float max = 0.0f;
	for (int i = 0; i < len; i++) {
		window[i] = 1 + 0.5 * len * sin(pi * (i + 1) / len);
		if (window[i] > max) {
			max = window[i];
		}
	}
	for (int i = 0; i < len; i++) {
		window[i] /= max;
	}
}

mfcc::mfcc() {
	_window = (float*)malloc(sizeof(float) * FRAME_LEN);
	for (int i = 0; i < FRAME_LEN; i++) {
		_window[i] = 0.54 - 0.46 * cos(2 * pi * i / (FRAME_LEN - 1));
	}
	
	_lift_window = (float*)malloc(sizeof(float) * FRAME_LEN);
	lift_window(_lift_window, NDCTS);

	_energyspectrum = (float*)malloc(sizeof(float) * FRAME_LEN);
	_mel = (float*)malloc(sizeof(float) * FILTER_NUM);

	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * FRAME_LEN);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * FRAME_LEN);
	plan = fftw_plan_dft_1d(FRAME_LEN, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
}

mfcc::~mfcc() {
	if (_window) {
		free(_window);
	}
	if (_lift_window) {
		free(_lift_window);
	}
	if (_energyspectrum) {
		free(_energyspectrum);
	}
	if (_mel) {
		free(_mel);
	}

	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
}

void computeMel(float *mel, const int sampleRate, const float *energySpectrum) {
	int fmax = sampleRate / 2;
	float maxMelFreq = 1127 * log(1 + fmax / 700);
	float melFilters[FILTER_NUM][3];
	float delta = maxMelFreq / (FILTER_NUM + 1.0);
	float *m = new float[FILTER_NUM + 2];
	float *h = new float[FILTER_NUM + 2];
	float *f = new float[FILTER_NUM + 2];
	for (int i = 0; i < FILTER_NUM + 2; i++)
	{
		m[i] = i * delta;
		h[i] = 700 * (exp(m[i] / 1127) - 1);
		f[i] = floor((FRAME_LEN + 1) * h[i] / sampleRate);  //*********//
	}
	//get start, peak, end point of every trigle filter
	for (int i = 0; i < FILTER_NUM; i++)
	{
		melFilters[i][0] = f[i];
		melFilters[i][1] = f[i + 1];
		melFilters[i][2] = f[i + 2];
	}
	delete[] m;
	delete[] h;
	delete[] f;
	//calculate the output of every trigle filter
	for (int i = 0; i < FILTER_NUM; i++)
	{
		for (int j = 0; j < FRAME_LEN; j++)
		{
			if (j > melFilters[i][0] && j <= melFilters[i][1]) {
				mel[i] += ((j - melFilters[i][0]) / (melFilters[i][1] - melFilters[i][0])) * energySpectrum[j];
			}
			if (j > melFilters[i][1] && j <= melFilters[i][2]) {
				mel[i] += ((j - melFilters[i][2]) / (melFilters[i][1] - melFilters[i][2])) * energySpectrum[j];
			}
		}
	}
}

void DCT(const float *mel, float *c) {
	for (int i = 1; i < NDCTS + 1; i++) {
		for (int j = 0; j < FILTER_NUM; j++) {
			if (mel[j] <= -0.00000000001 || mel[j] >= 0.00000000001) {
				c[i - 1] += log(mel[j]) * cos(pi * i / (2 * FILTER_NUM) * (2 * j + 1));
			}
		}
	}
}

bool mfcc::process(float* frame, int32_t samplerate) {
	if (frame == NULL) {
		return false;
	}
	for (int i = 0; i < FRAME_LEN; i++) {
		in[i][0] = frame[i];
		in[i][1] = 0;
	}
	for (int i = 1; i < FRAME_LEN; i++) {
		in[i][0] = in[i][0] - 0.98 * in[i - 1][0];
	}
	for (int i = 0; i < FRAME_LEN; i++) {
		in[i][0] = in[i][0] * _window[i];
	}
	fftw_execute_dft(plan, in, out);
	//������
	for (int i = 0; i < FRAME_LEN; i++) {
		_energyspectrum[i] = pow(out[i][0], 2) + pow(out[i][1], 2);
	}
	memset(_mel, 0, sizeof(float) * FILTER_NUM);
	computeMel(_mel, samplerate, _energyspectrum);
	memset(_ans, 0, sizeof(float) * 13);
	DCT(_mel, _ans);
	for (int i = 0; i < NDCTS; i++) {
		_ans[i] *= _lift_window[i];
	}
	return true;
}
