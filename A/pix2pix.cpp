#include "pix2pix.h"

#include "util.h"

#include <cstring>
#include <string>
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>
#include <algorithm>

#include <xmmintrin.h>
#include <immintrin.h>

using namespace std;

typedef unsigned int uint;

#define PAD 1
#define STRIDE 2
#define NUMWEIGHTS 54431363

class Tensor 
{
public:
  Tensor();
  Tensor(float *buf_, vector<uint> shape_);
  void alloc_once(vector<uint> shape_);
  void set_sz();

  // For real world application, one should use smart pointer to prevent possible memory leak.
  // However, we just use raw pointer for simplicity. Memory will be freed anyway when process exits.
  float* buf;

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., [[1, 2, 3], [4, 5, 6]] => shape = [2, 3]
  vector<uint> shape;

  // Size of tensor; product of all dimensions
  uint sz;
};

// Helpers
static void register_weight(map<string, Tensor>& weights, float* (*buf), string name, vector<uint> shape);
static map<string, Tensor> register_weights(float* weight_buf);
static Tensor preprocess(uint8_t *in, uint num_image);
static void postprocess(const Tensor& input, uint8_t *out);

// Operators
static void conv2d(const Tensor& input, const Tensor& filter, const Tensor& bias, Tensor& output);
static void conv2d_v2(const Tensor& input, const Tensor& filter, const Tensor& bias, Tensor& output);
static void conv2d_transposed(const Tensor& input, const Tensor& filter, const Tensor& bias, Tensor& output);
static void conv2d_transposed_v2(const Tensor& input, const Tensor& filter, const Tensor& bias, Tensor& output);
static void leaky_relu(const Tensor& input, Tensor& output, float alpha);
static void relu(const Tensor& input, Tensor& output);
static void batchnorm(Tensor& inout, const Tensor& scale, const Tensor& offset);
static void concat(const Tensor& input0, const Tensor& input1, Tensor& output);

void pix2pix_init() 
{
  /*
   * You can do input-independent and input-size-independent jobs here.
   * e.g., Getting OpenCL platform, Compiling OpenCL kernel, ...
   * Execution time of this function is not measured, so do as much as possible!
   */
}

uint num_images;

//#define TIME_CHECK

#ifdef TIME_CHECK

const int num_operations = 6;
const string name_operations[] = {
		"Convolution",
		"Leaky ReLU",
		"Batch Normalization",
		"Concatenation",
		"ReLU",
		"Convolution Transposed"
	};
double timers[num_operations] = { 0.0, };

enum Operations
{
	CONV2D,
	LEAKYRELU,
	BATCHNORM,
	CONCAT,
	RELU,
	CONV2DT
};

double st = 0.0;

void StartTimer()
{
	st = get_time();
}

void StopTimer(enum Operations op)
{
	timers[op] += get_time() - st;
}

void PrintTimer()
{
	cout.setf(ios::fixed); cout << setprecision(4) << endl;

	for (int i = 0; i < num_operations; ++i) {
		cout.unsetf(ios::right); cout.setf(ios::left);
		cout << setw(25) << name_operations[i];
		
		cout.unsetf(ios::left); cout.setf(ios::right);
		cout << setw(10) << timers[i] << " sec" << endl;
	}
}

#endif

void pix2pix(uint8_t *input_buf, float *weight_buf, uint8_t *output_buf, size_t num_image) 
{
  /*
   * !!!!!!!! Caution !!!!!!!!
   * In MPI program, all buffers and num_image are only given to rank 0 process.
   * You should manually:
   *   1. allocate buffers on others
   *   2. send inputs from rank 0 to others
   *   3. gather outputs from others to rank 0
   */

	num_images = (uint) num_image;

  map<string, Tensor> weights = register_weights(weight_buf);
  const Tensor input = preprocess(input_buf, num_image);

  Tensor encoder_layer_input[9];
  Tensor encoder_layer_rectified[9];
  Tensor encoder_layer[9];
  Tensor decoder_layer_input[9];
  Tensor decoder_layer_rectified[9];
  Tensor decoder_layer[9];

	// Dimension Prediction & Allocation
	
	for (uint i = 1; i <= 8; ++i) {
		const Tensor& filter = weights["generator/encoder_" + to_string(i) + "/conv2d/kernel"];
		if (i == 1) {
			uint H = input.shape[1], W = input.shape[2], K = filter.shape[3];
			encoder_layer[i].alloc_once({ num_images, H / STRIDE, W / STRIDE, K });
		}
		else {
			encoder_layer_input[i] = encoder_layer[i - 1];
			encoder_layer_rectified[i].alloc_once(encoder_layer_input[i].shape);
			uint H = encoder_layer_rectified[i].shape[1], W = encoder_layer_rectified[i].shape[2];
			uint K = filter.shape[3];
			encoder_layer[i].alloc_once({ num_images, H / STRIDE, W / STRIDE, K });
		}
	}

	for (uint i = 8; i >= 1; --i) {
		const Tensor& filter = weights["generator/decoder_" + to_string(i) + "/conv2d_transpose/kernel"];
		if (i == 8) {
			decoder_layer_input[i] = encoder_layer[8];
		}
		else {
			uint H = decoder_layer[i + 1].shape[1], W = decoder_layer[i + 1].shape[2];
			uint C0 = decoder_layer[i + 1].shape[3], C1 = encoder_layer[i].shape[3];

			decoder_layer_input[i].alloc_once({ num_images, H, W, C0 + C1 });
		}

		decoder_layer_rectified[i].alloc_once(decoder_layer_input[i].shape);
		
		uint H = decoder_layer_rectified[i].shape[1], W = decoder_layer_rectified[i].shape[2];
		uint K = filter.shape[2];
	
		decoder_layer[i].alloc_once({ num_images, H * STRIDE, W * STRIDE, K });
	}
	

	/* Encoder Phase */

  const Tensor& filter = weights["generator/encoder_1/conv2d/kernel"];
  const Tensor& bias = weights["generator/encoder_1/conv2d/bias"];

#ifdef TIME_CHECK
	StartTimer();
#endif
  conv2d(input, filter, bias, encoder_layer[1]);
#ifdef TIME_CHECK
	StopTimer(CONV2D);
#endif

  for (int i = 2; i <= 8; ++i) {
    const string scope = "generator/encoder_" + to_string(i);
    const Tensor& filter = weights[scope + "/conv2d/kernel"];
    const Tensor& bias = weights[scope + "/conv2d/bias"];
    const Tensor& scale = weights[scope + "/batch_normalization/gamma"];
    const Tensor& offset = weights[scope + "/batch_normalization/beta"];
    encoder_layer_input[i] = encoder_layer[i - 1];

#ifdef TIME_CHECK
		StartTimer();
#endif
    leaky_relu(encoder_layer_input[i], encoder_layer_rectified[i], 0.2);
#ifdef TIME_CHECK
		StopTimer(LEAKYRELU);
#endif

#ifdef TIME_CHECK
		StartTimer();
#endif
    conv2d_v2(encoder_layer_rectified[i], filter, bias, encoder_layer[i]);
#ifdef TIME_CHECK
		StopTimer(CONV2D);
#endif

#ifdef TIME_CHECK
		StartTimer();
#endif
    batchnorm(encoder_layer[i], scale, offset);
#ifdef TIME_CHECK
		StopTimer(BATCHNORM);
#endif
  }

	/* Decoder Phase */

  for (int i = 8; i >= 1; --i) {
    const string scope = "generator/decoder_" + to_string(i);
    const Tensor& filter = weights[scope + "/conv2d_transpose/kernel"];
    const Tensor& bias = weights[scope + "/conv2d_transpose/bias"];
    const Tensor& scale = weights[scope + "/batch_normalization/gamma"];
    const Tensor& offset = weights[scope + "/batch_normalization/beta"];
    if (i == 8) {
      decoder_layer_input[i] = encoder_layer[8];
    } 
		else {
#ifdef TIME_CHECK
			StartTimer();
#endif
      concat(decoder_layer[i + 1], encoder_layer[i], decoder_layer_input[i]);
#ifdef TIME_CHECK
			StopTimer(CONCAT);
#endif
    }

#ifdef TIME_CHECK
		StartTimer();
#endif
    relu(decoder_layer_input[i], decoder_layer_rectified[i]);
#ifdef TIME_CHECK
		StopTimer(RELU);
#endif


#ifdef TIME_CHECK
		StartTimer();
#endif
		if (i == 1) {
	    conv2d_transposed(decoder_layer_rectified[i], filter, bias, decoder_layer[i]);
		}
		else {
			conv2d_transposed_v2(decoder_layer_rectified[i], filter, bias, decoder_layer[i]);
		}
#ifdef TIME_CHECK
		StopTimer(CONV2DT);
#endif

    if (i == 1) break;
#ifdef TIME_CHECK
		StartTimer();
#endif
    batchnorm(decoder_layer[i], scale, offset);
#ifdef TIME_CHECK
		StopTimer(BATCHNORM);
#endif
  }

  postprocess(decoder_layer[1], output_buf);

#ifdef TIME_CHECK
	PrintTimer();
#endif
}

Tensor::Tensor() : buf(NULL) {}

// If buf is given, use it. If not, allocate new one.
Tensor::Tensor(float *buf_, vector<uint> shape_) : buf(buf_), shape(shape_) 
{
  set_sz();
  if (buf == NULL) {
    buf = (float*)malloc(sz * sizeof(float));
  }
}

// If buf is not allocated, allocate new one.
void Tensor::alloc_once(vector<uint> shape_) 
{
  if (buf == NULL) {
    shape = shape_;
    set_sz();
    buf = (float*)malloc(sz * sizeof(float));
  }
}

void Tensor::set_sz() 
{
  sz = 1;
  for (auto x : shape) {
    sz *= x;
  }
}

// Make a new tensor from buffer and put the tensor into map. Advance buffer pointer by size.
void register_weight(map<string, Tensor>& weights, float* (*buf), string name, vector<uint> shape) 
{
  Tensor tensor(*buf, shape);
  weights[name] = tensor;
  *buf += tensor.sz;
}

// Put all predefined weights into map. Order should not be changed.
map<string, Tensor> register_weights(float* weight_buf) 
{
  map<string, Tensor> weights;
  // auto generated
  register_weight(weights, &weight_buf, "generator/decoder_1/conv2d_transpose/bias", {3});
  register_weight(weights, &weight_buf, "generator/decoder_1/conv2d_transpose/kernel", {4, 4, 3, 128});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/beta", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/gamma", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/moving_mean", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/moving_variance", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/conv2d_transpose/bias", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/conv2d_transpose/kernel", {4, 4, 64, 256});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/conv2d_transpose/bias", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/conv2d_transpose/kernel", {4, 4, 128, 512});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/conv2d_transpose/bias", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/conv2d_transpose/kernel", {4, 4, 256, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/conv2d_transpose/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_1/conv2d/bias", {64});
  register_weight(weights, &weight_buf, "generator/encoder_1/conv2d/kernel", {4, 4, 3, 64});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/bias", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/kernel", {4, 4, 64, 128});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/bias", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/kernel", {4, 4, 128, 256});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/kernel", {4, 4, 256, 512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/kernel", {4, 4, 512, 512});
  return weights;
}

// Convert 8-bit depth images (value range [0, 255]) into floating-point ones (value range [-1, 1])
Tensor preprocess(uint8_t *in, uint num_image) 
{
  Tensor out(NULL, {num_image, 256, 256, 3});

	#pragma omp parallel for
  for (uint i = 0; i < out.sz; ++i) {
    out.buf[i] = in[i] / 255.0f * 2 - 1;
  }
  return out;
}

// Inverse of preprocess
void postprocess(const Tensor& input, uint8_t *out) 
{
	#pragma omp parallel for
	for (uint i = 0; i < input.sz; ++i) {
		float x = (tanhf(input.buf[i]) + 1) / 2 * 255;
		out[i] = x < 0 ? 0 : (x > 255 ? 225 : x);
	}

}

// Convolution (2-dimension, stride = 2, pad = 1)
void conv2d(const Tensor& input, const Tensor& filter, const Tensor& bias, Tensor& output) 
{
  uint H = input.shape[1], W = input.shape[2], C = input.shape[3];
  uint R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
  uint OH = H / STRIDE, OW = W / STRIDE;

	const uint KSIZE = 32, CSIZE = 3;

	#pragma omp parallel for schedule(guided) collapse(3)
	for (uint img = 0; img < num_images; ++img) {
		for (uint oh = 0; oh < OH; ++oh) {
  	  for (uint ow = 0; ow < OW; ++ow) {
				for (uint k = 0; k < K; k += KSIZE) {
					uint inp_offset = img * H * W * C;
					uint out_offset = img * OH * OW * K;

      	  float x[KSIZE]; memcpy(x, &bias.buf[k], KSIZE * sizeof(float));
				
					uint rbeg = max((int) 0, PAD - (int) oh * STRIDE);
					uint rend = min((int) R, PAD + (int) H - (int) oh * STRIDE);
					uint sbeg = max((int) 0, PAD - (int) ow * STRIDE);
					uint send = min((int) S, PAD + (int) W - (int) ow * STRIDE);
        	
					for (uint r = rbeg; r < rend; ++r) {
						uint ih = oh * STRIDE - PAD + r;
          	for (uint s = sbeg; s < send; ++s) {
							uint iw = ow * STRIDE - PAD + s;
							for (uint c = 0; c < C; c += CSIZE) {
								float ii[CSIZE], ff[CSIZE][KSIZE];

								memcpy(ii, &input.buf[inp_offset + ih * W * C + iw * C + c], CSIZE * sizeof(float));

								for (uint cc = 0; cc < CSIZE; ++cc) {
									memcpy(ff[cc], &filter.buf[r * S * C * K + s * C * K + (c + cc) * K + k],
										KSIZE * sizeof(float));
								}
								for (uint kk = 0; kk < KSIZE; ++kk) {
									for (uint cc = 0; cc < CSIZE; ++cc) {
										x[kk] += ii[cc] * ff[cc][kk];
									}
								}
							}
          	}
        	}
        	memcpy(&output.buf[out_offset + oh * OW * K + ow * K + k], x, KSIZE * sizeof(float));
      	}
    	}
  	}
	}
}

// Convolution (2-dimension, stride = 2, pad = 1)
void conv2d_v2(const Tensor& input, const Tensor& filter, const Tensor& bias, Tensor& output) 
{
  uint H = input.shape[1], W = input.shape[2], C = input.shape[3];
  uint R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
  uint OH = H / STRIDE, OW = W / STRIDE;

	const uint KSIZE = 16, CSIZE = 4;

	#pragma omp parallel for schedule(guided) collapse(3)
	for (uint img = 0; img < num_images; ++img) {
		for (uint oh = 0; oh < OH; ++oh) {
  	  for (uint ow = 0; ow < OW; ++ow) {
				for (uint k = 0; k < K; k += KSIZE) {
					uint inp_offset = img * H * W * C;
					uint out_offset = img * OH * OW * K;
					
					float x[KSIZE]; memcpy(x, &bias.buf[k], KSIZE * sizeof(float));

					uint rbeg = max((int) 0, PAD - (int) oh * STRIDE);
					uint rend = min((int) R, PAD + (int) H - (int) oh * STRIDE);
					uint sbeg = max((int) 0, PAD - (int) ow * STRIDE);
					uint send = min((int) S, PAD + (int) W - (int) ow * STRIDE);
        	
					for (uint r = rbeg; r < rend; ++r) {
						uint ih = oh * STRIDE - PAD + r;
          	for (uint s = sbeg; s < send; ++s) {
							uint iw = ow * STRIDE - PAD + s;
							for (uint c = 0; c < C; c += CSIZE) {
								
								float ii[CSIZE], ff[CSIZE][KSIZE];

								memcpy(ii, &input.buf[inp_offset + ih * W * C + iw * C + c], CSIZE * sizeof(float));
								
								for (uint cc = 0; cc < CSIZE; ++cc) {
									memcpy(ff[cc], &filter.buf[r * S * C * K + s * C * K + (c + cc) * K + k],
										KSIZE * sizeof(float));
								}
								for (uint kk = 0; kk < KSIZE; ++kk) {
									for (uint cc = 0; cc < CSIZE; ++cc) {
										x[kk] += ii[cc] * ff[cc][kk];
									}
								}
							}
          	}
        	}
        	memcpy(&output.buf[out_offset + oh * OW * K + ow * K + k], x, KSIZE * sizeof(float));
      	}
    	}
  	}
	}
}

// Transposed convolution (2-dimension, stride = 2, pad = 1)
void conv2d_transposed(const Tensor& input, const Tensor& filter, const Tensor& bias, Tensor& output) 
{
  uint H = input.shape[1], W = input.shape[2], C = input.shape[3];
  uint R = filter.shape[0], S = filter.shape[1], K = filter.shape[2];
  uint OH = H * STRIDE, OW = W * STRIDE;

	const uint KSIZE = 3, CSIZE = 8;

	#pragma omp parallel for schedule(guided) collapse(4)
	for (uint img = 0; img < num_images; ++img) {
  	for (uint k = 0; k < K; k += KSIZE) {
    	for (uint oh = 0; oh < OH; ++oh) {
      	for (uint ow = 0; ow < OW; ++ow) {
					uint inp_pos = img * H * W * C;
					uint out_pos = img * OH * OW * K;

        	float x[KSIZE]; memcpy(x, &bias.buf[k], KSIZE * sizeof(float));

					uint rbeg = max((int) 0, (int) oh + PAD - STRIDE * (int) H + 1);
					uint rend = min(R, oh + PAD + 1);
					uint sbeg = max((int) 0, (int) ow + PAD - STRIDE * (int) W + 1);
					uint send = min(S, ow + PAD + 1);

					if ((oh - rbeg + PAD) % STRIDE != 0) ++rbeg;
					if ((ow - sbeg + PAD) % STRIDE != 0) ++sbeg;

        	for (uint r = rbeg; r < rend; r += STRIDE) {
        		uint ih = (oh - r + PAD) / STRIDE;
          	for (uint s = sbeg; s < send; s += STRIDE) {
          		uint iw = (ow - s + PAD) / STRIDE;
							for (uint c = 0; c < C; c += CSIZE) {			
								float ii[CSIZE], ff[KSIZE][CSIZE];

								memcpy(ii, &input.buf[inp_pos + ih * W * C + iw * C + c], CSIZE * sizeof(float));

								for (uint kk = 0; kk < KSIZE; ++kk) {
									memcpy(ff[kk], &filter.buf[r * S * K * C + s * K * C + (k + kk) * C + c],
										CSIZE * sizeof(float));
								}

								for (uint kk = 0; kk < KSIZE; ++kk) {
									for (uint cc = 0; cc < CSIZE; ++cc) {
										x[kk] += ii[cc] * ff[kk][cc];
									}
								}
            	}
          	}
        	}
        	memcpy(&output.buf[out_pos + oh * OW * K + ow * K + k], x, KSIZE * sizeof(float));
      	}
    	}
  	}
	}
}

void conv2d_transposed_v2(const Tensor& input, const Tensor& filter, const Tensor& bias, Tensor& output) 
{
  uint H = input.shape[1], W = input.shape[2], C = input.shape[3];
  uint R = filter.shape[0], S = filter.shape[1], K = filter.shape[2];
  uint OH = H * STRIDE, OW = W * STRIDE;

	const uint KSIZE = 4, CSIZE = 16;

	#pragma omp parallel for schedule(guided) collapse(4)
	for (uint img = 0; img < num_images; ++img) {
  	for (uint k = 0; k < K; k += KSIZE) {
    	for (uint oh = 0; oh < OH; ++oh) {
      	for (uint ow = 0; ow < OW; ++ow) {
					uint inp_pos = img * H * W * C;
					uint out_pos = img * OH * OW * K;

					float x[KSIZE]; memcpy(x, &bias.buf[k], KSIZE * sizeof(float));

					uint rbeg = max((int) 0, (int) oh + PAD - STRIDE * (int) H + 1);
					uint rend = min(R, oh + PAD + 1);
					uint sbeg = max((int) 0, (int) ow + PAD - STRIDE * (int) W + 1);
					uint send = min(S, ow + PAD + 1);

					if ((oh - rbeg + PAD) % STRIDE != 0) ++rbeg;
					if ((ow - sbeg + PAD) % STRIDE != 0) ++sbeg;

        	for (uint r = rbeg; r < rend; r += STRIDE) {
        		uint ih = (oh - r + PAD) / STRIDE;
          	for (uint s = sbeg; s < send; s += STRIDE) {
          		uint iw = (ow - s + PAD) / STRIDE;
							for (uint c = 0; c < C; c += CSIZE) {			
								float ii[CSIZE], ff[KSIZE][CSIZE];
								memcpy(ii, &input.buf[inp_pos + ih * W * C + iw * C + c], CSIZE * sizeof(float));

								for (uint kk = 0; kk < KSIZE; ++kk) {
									memcpy(ff[kk], &filter.buf[r * S * K * C + s * K * C + (k + kk) * C + c], 
										CSIZE * sizeof(float));
								}
								for (uint kk = 0; kk < KSIZE; ++kk) {
									for (uint cc = 0; cc < CSIZE; ++cc) {
										x[kk] += ii[cc] * ff[kk][cc];
									}
								}
            	}
          	}
        	}
					memcpy(&output.buf[out_pos + oh * OW * K + ow * K + k], x, KSIZE * sizeof(float));
      	}
    	}
  	}
	}
}

// Leaky ReLU
void leaky_relu(const Tensor& input, Tensor &output, float alpha) 
{
  uint H = input.shape[1], W = input.shape[2], C = input.shape[3];
  
	uint I = H * W * C;

	#pragma omp parallel for collapse(2)
	for (uint img = 0; img < num_images; ++img) {
		for (uint i = 0; i < I; ++i) {
    	output.buf[img * I + i] = input.buf[img * I + i] >= 0 ? 
				input.buf[img * I + i] : alpha * input.buf[img * I + i];
  	}
	}
}

// ReLU
void relu(const Tensor& input, Tensor &output) 
{
  uint H = input.shape[1], W = input.shape[2], C = input.shape[3];
	
	uint I = H * W * C;
  
	#pragma omp parallel for collapse(2)
	for (uint img = 0; img < num_images; ++img) {
		for (uint i = 0; i < H * W * C; ++i) {
   		output.buf[img * I + i] = input.buf[img * I + i] >= 0 ? input.buf[img * I + i] : 0;
  	}
	}
}

// Batch normalization (channel-wise)
void batchnorm(Tensor& inout, const Tensor& scale, const Tensor& offset) 
{
  uint H = inout.shape[1], W = inout.shape[2], C = inout.shape[3];
  
	#pragma omp parallel for schedule(guided) collapse(2)
	for (uint img = 0; img < num_images; ++img) {
		for (uint c = 0; c < C; ++c) {
			uint pos = img * H * W * C;
    	float sum = 0;
    	for (uint h = 0; h < H; ++h) {
      	for (uint w = 0; w < W; ++w) {
        	float ii = inout.buf[pos + h * W * C + w * C + c];
        	sum += ii;
      	}
    	}
    	float mean = sum / (H * W);

    	float sqsum = 0;
    	for (uint h = 0; h < H; ++h) {
      	for (uint w = 0; w < W; ++w) {
        	float ii = inout.buf[pos + h * W * C + w * C + c];
        	sqsum += (ii - mean) * (ii - mean);
      	}
    	}
    	float variance = sqsum / (H * W);

    	const float epsilon = 1e-5;
    	for (uint h = 0; h < H; ++h) {
      	for (uint w = 0; w < W; ++w) {
        	uint idx = pos + h * W * C + w * C + c;
        	inout.buf[idx] = offset.buf[c] + (inout.buf[idx] - mean) * scale.buf[c] / sqrtf(variance + epsilon);
      	}
    	}
  	}
	}
}

// Concatenation (along channel dimension)
void concat(const Tensor& input0, const Tensor& input1, Tensor& output)
{
  uint H = input0.shape[1], W = input0.shape[2], C0 = input0.shape[3];
  uint C1 = input1.shape[3];

	#pragma omp parallel for schedule(guided) collapse(3)
	for (uint img = 0; img < num_images; ++img) {
  	for (uint h = 0; h < H; ++h) {
    	for (uint w = 0; w < W; ++w) {
				uint inp0 = img * H * W * C0, inp1 = img * H * W * C1, out = img * H * W * (C0 + C1);
      	for (uint c = 0; c < C0; ++c) {
        	output.buf[out + h * W * (C0 + C1) + w * (C0 + C1) + c] = 
						input0.buf[inp0 + h * W * C0 + w * C0 + c];
      	}
      	for (uint c = 0; c < C1; ++c) {
        	output.buf[out + h * W * (C0 + C1) + w * (C0 + C1) + (C0 + c)] = 
						input1.buf[inp1 + h * W * C1 + w * C1 + c];
      	}
    	}
  	}
	}
}
