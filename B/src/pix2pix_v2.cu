#include "control.cuh"
#include "operator.cuh"
#include "control.h"

#define STRIDE 2
#define PAD 1

namespace Const
{
__constant__ uint num_images;
__constant__ float alpha;
__constant__ float epsilon;
}

template <typename T>
__global__ void cuFill(uint n, const T v, T* buf)
{
	uint i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n) return;

	buf[i] = v;
}

template <typename T>
void Tensor_t<T>::Fill(const T v)
{
	dim3 threads, blocks;

	threads = dim3(1024, 1, 1);
	blocks = dim3(sz / threads.x + 1, 1, 1);
	
	cuFill <<< blocks, threads >>> (sz, v, Entry);

	cudaCheckError( cudaDeviceSynchronize() );
}

template struct Tensor_t<float4>;

__global__ void cuPreprocWeights_v2(uint n,
	const float *W,
	Tensor_t<float4> *Wv,
	const uint *position)
{

	uint i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n) return;

	float w = W[i];

	i = position[i];

	uint p = i / 4, r = i % 4;

	if (r == 0)	Wv->Entry[p].x = w;
	else if (r == 1) Wv->Entry[p].y = w;
	else if (r == 2) Wv->Entry[p].z = w;
	else Wv->Entry[p].w = w;
}

__global__ void cuPreprocInput_v2(uint n, 
	const Tensor_t<uint8_t> *input_buf, 
	Tensor_t<float4> *input)
{
	uint i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n) return;

	input->Entry[i].x = input_buf->Entry[3 * i + 0] / 255.0f * 2 - 1;
	input->Entry[i].y = input_buf->Entry[3 * i + 1] / 255.0f * 2 - 1;
	input->Entry[i].z = input_buf->Entry[3 * i + 2] / 255.0f * 2 - 1;
	input->Entry[i].w = 0.0f;
}

__global__ void cuConv2D_v2(const Tensor_t<float4> *input, 
	Tensor_t<float4> *output, 
	const Phase_t *Phase)
{
	uint H = input->nz, W = input->ny, Cv = input->nx, C = Cv * 4;
	uint R = Phase->GetKernelShape(0), S = Phase->GetKernelShape(1);
	uint Kv = output->nx, K = Kv * 4;

	uint OH = H / STRIDE, OW = W / STRIDE;

	uint kv = threadIdx.x + blockIdx.x * blockDim.x; uint k = kv * 4;
	uint ow = threadIdx.y + blockIdx.y * blockDim.y;
	uint oh = (threadIdx.z + blockIdx.z * blockDim.z) % OH;
	uint img = (threadIdx.z + blockIdx.z * blockDim.z) / OH;

	if (kv >= Kv || ow >= OW || oh >= OH || img >= Const::num_images) return;
	
	uint lidx = oh * OW * Kv + ow * Kv + kv;
	uint gidx = img * OH * OW * Kv + lidx;
	uint inp_offset = img * (H * W * Cv);

	float4 x = Phase->GetBias_v2(kv);
	
	uint rbeg = max((int) 0, PAD - (int) oh * STRIDE);
	uint rend = min(R, PAD + (int) H - (int) oh * STRIDE);
	uint sbeg = max((int) 0, PAD - (int) ow * STRIDE);
	uint send = min(S, PAD + (int) W - (int) ow * STRIDE);

	for (uint r = rbeg; r < rend; ++r) {
		uint ih = oh * STRIDE - PAD + r;
		for (uint s = sbeg; s < send; ++s) {
			uint iw = ow * STRIDE - PAD + s;
			for (uint cv = 0; cv < Cv; ++cv) {
				uint c = cv * 4;
				float4 ii = input->Entry[inp_offset + ih * W * Cv + iw * Cv + cv];
				float4 ff[4];

				ff[0] = Phase->GetKernel_v2(r * S * C * Kv + s * C * Kv + (c + 0) * Kv + kv);
				ff[1] = Phase->GetKernel_v2(r * S * C * Kv + s * C * Kv + (c + 1) * Kv + kv);
				ff[2] = Phase->GetKernel_v2(r * S * C * Kv + s * C * Kv + (c + 2) * Kv + kv);
				ff[3] = Phase->GetKernel_v2(r * S * C * Kv + s * C * Kv + (c + 3) * Kv + kv);

				x.x += ii.x * ff[0].x + ii.y * ff[1].x + ii.z * ff[2].x + ii.w * ff[3].x;
				x.y += ii.x * ff[0].y + ii.y * ff[1].y + ii.z * ff[2].y + ii.w * ff[3].y;
				x.z += ii.x * ff[0].z + ii.y * ff[1].z + ii.z * ff[2].z + ii.w * ff[3].z;
				x.w += ii.x * ff[0].w + ii.y * ff[1].w + ii.z * ff[2].w + ii.w * ff[3].w;
			}
		}
	}
	output->Entry[gidx] = x;
}

__global__ void cuLeakyRelu_v2(uint n, 
	const Tensor_t<float4> *input,
	Tensor_t<float4> *output)
{

	uint idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= n) return;
	
	float4 v = input->Entry[idx];

	v.x = v.x >= 0.0f ? v.x : Const::alpha * v.x;
	v.y = v.y >= 0.0f ? v.y : Const::alpha * v.y;
	v.z = v.z >= 0.0f ? v.z : Const::alpha * v.z;
	v.w = v.w >= 0.0f ? v.w : Const::alpha * v.w;

	output->Entry[idx] = v;
}

__global__ void cuBatchNorm_v2(uint n,
	Tensor_t<float4> *inout,
	const Phase_t *Phase)
{
	extern __shared__ float4 sum[];
	
	sum[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	__syncthreads();

	uint H = inout->nz, W = inout->ny, Cv = inout->nx, C = Cv * 4;

	uint size = H * W * Cv;
	uint img = blockIdx.x / Cv;
	uint cv = blockIdx.x % Cv;

	uint img_offset = H * W * Cv * img;

	for (uint i = 0; i < H * W; i += blockDim.x) {
		uint idx = (i + threadIdx.x) * Cv + cv;
		if (idx < size) sum[threadIdx.x] += inout->Entry[img_offset + idx];
	}

	__syncthreads();

	for (uint s = blockDim.x / 2; s >= 1; s >>= 1) {
		if (threadIdx.x < s) sum[threadIdx.x] += sum[threadIdx.x + s];
		__syncthreads();
	}

	float4 mean = sum[0] / (float) (H * W);

	__syncthreads();

	sum[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	__syncthreads();

	for (uint i = 0; i < H * W; i += blockDim.x) {
		uint idx = (i + threadIdx.x) * Cv + cv;
		if (idx < size) {
			float4 b = inout->Entry[img_offset + idx];
			sum[threadIdx.x] += (b - mean) * (b - mean);
		}
	}

	__syncthreads();

	for (uint s = blockDim.x / 2; s >= 1; s >>= 1) {
		if (threadIdx.x < s) sum[threadIdx.x] += sum[threadIdx.x + s];
		__syncthreads();
	}

	float4 variance = sum[0] / (float) (H * W);

	float4 offset = Phase->GetOffset_v2(cv);
	float4 scale = Phase->GetScale_v2(cv);

	for (uint i = 0; i < H * W; i += blockDim.x) {
		uint idx = (i + threadIdx.x) * Cv + cv;
		if (idx < size) {
			float4 v = inout->Entry[img_offset + idx];
			inout->Entry[img_offset + idx] = offset + (v - mean) * scale / __fsqrt_rn(variance + Const::epsilon);
		}
	}
}

__global__ void cuConcat_v2(uint n, 
	const Tensor_t<float4> *input0,
	const Tensor_t<float4> *input1,
	Tensor_t<float4> *output)
{
	uint H = input0->nz, W = input0->ny, C0v = input0->nx, C0 = C0v * 4;
	uint C1v = input1->nx, C1 = C1v * 4, Cv = C0v + C1v;

	uint cv = threadIdx.x + blockIdx.x * blockDim.x, c = cv * 4;
	uint w = threadIdx.y + blockIdx.y * blockDim.y;
	uint h = (threadIdx.z + blockIdx.z * blockDim.z) % H;
	uint img = (threadIdx.z + blockIdx.z * blockDim.z) / H;

	if (cv >= Cv || w >= W || h >= H || img >= Const::num_images) return;

	uint gidx = img * H * W * Cv + h * W * Cv + w * Cv + cv;

	if (cv < C0v) 
		output->Entry[gidx] = input0->Entry[img * H * W * C0v + h * W * C0v + w * C0v + cv];
	else
		output->Entry[gidx] = input1->Entry[img * H * W * C1v + h * W * C1v + w * C1v + cv - C0v];
}

__global__ void cuRelu_v2(uint n, 
	const Tensor_t<float4> *inout)
{

	uint idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= n) return;

	float4 v = inout->Entry[idx];

	v.x = v.x >= 0.0f ? v.x : 0.0f;
	v.y = v.y >= 0.0f ? v.y : 0.0f;
	v.z = v.z >= 0.0f ? v.z : 0.0f;
	v.w = v.w >= 0.0f ? v.w : 0.0f;

	inout->Entry[idx] = v;
}

__global__ void cuConv2DT_v2(const Tensor_t<float4> *input,
	Tensor_t<float4> *output,
	const Phase_t *Phase)
{
	
	uint H = input->nz, W = input->ny, Cv = input->nx, C = Cv * 4;
	uint R = Phase->GetKernelShape(0), S = Phase->GetKernelShape(1);
	uint Kv = output->nx, K = Kv * 4;

	uint OH = H * STRIDE, OW = W * STRIDE;
	
	uint ow = threadIdx.x + blockIdx.x * blockDim.x;
	uint oh = threadIdx.y + blockIdx.y * blockDim.y;
	uint kv = (threadIdx.z + blockIdx.z * blockDim.z) % Kv, k = kv * 4;
	uint img = (threadIdx.z + blockIdx.z * blockDim.z) / Kv;


	if (threadIdx.z + blockIdx.z * blockDim.z >= Kv * Const::num_images) return;
	if (ow >= OW || oh >= OH) return;
	
	uint gidx = img * OH * OW * Kv + oh * OW * Kv + ow * Kv + kv;
	uint inp_offset = img * H * W * Cv;

	float4 x = Phase->GetBias_v2(kv);

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
			for (uint cv = 0; cv < Cv; ++cv) {
				float4 ii = input->Entry[inp_offset + ih * W * Cv + iw * Cv + cv];
				float4 ff[4];

				ff[0] = Phase->GetKernel_v2(r * S * K * Cv + s * K * Cv + (k + 0) * Cv + cv);
				ff[1] = Phase->GetKernel_v2(r * S * K * Cv + s * K * Cv + (k + 1) * Cv + cv);
				ff[2] = Phase->GetKernel_v2(r * S * K * Cv + s * K * Cv + (k + 2) * Cv + cv);
				ff[3] = Phase->GetKernel_v2(r * S * K * Cv + s * K * Cv + (k + 3) * Cv + cv);

				x.x += dot(ii, ff[0]);
				x.y += dot(ii, ff[1]);
				x.z += dot(ii, ff[2]);
				x.w += dot(ii, ff[3]);
			}
		}
	}

	output->Entry[gidx] = x;
}

__global__ void cuPostProc_v2(uint n, 
	const Tensor_t<float4> *input, 
	Tensor_t<uint8_t> *output)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= n) return;

	float4 v = (tanhf(input->Entry[idx]) + 1.0f) / 2.0f * 255.0f;
	
	output->Entry[idx * 3 + 0] = v.x < 0 ? 0.0f : (v.x > 255.0f ? 255 : (uint8_t) v.x);
	output->Entry[idx * 3 + 1] = v.y < 0 ? 0.0f : (v.y > 255.0f ? 255 : (uint8_t) v.y);
	output->Entry[idx * 3 + 2] = v.z < 0 ? 0.0f : (v.z > 255.0f ? 255 : (uint8_t) v.z);
}

__host__ __device__ __forceinline__
ostream& operator<<(ostream& os, const float4& a)
{
	os << a.x << " "
		 << a.y << " "
		 << a.z << " " 
		 << a.w;
	return os;
}

template <>
void Control:: PrintPix(const Tensor_t<float4>& input, string filename)
{
	uint H = input.nz, W = input.ny, C = input.nx;

	uint n = num_images * H * W * C;

	float *buf = (float *) malloc(n * sizeof(float4));

	cudaMemcpy(buf, input.Entry, n * sizeof(float4), cudaMemcpyDeviceToHost);

	C *= 4;

	for (uint n = 0; n < num_images; ++n) {
		ofstream fout;
		fout.open("img" + to_string(n) + filename);
		for (uint h = 0; h < H; ++h) {
			for (uint w = 0; w < W; ++w) {
				for (uint c = 0; c < C; ++c) {
					uint idx = n * H * W * C + h * W * C + w * C + c;
					fout << h << " "
							 << w << " "
							 << c << " " 
							 << buf[idx] << endl;
				}
			}
		}
		fout.close();
	}

	cout << "img<n>" + filename << " Printed!" << endl;
	
	delete[] buf;
}

void Control::PreprocInput_v2()
{
	dim3 blocks, threads;
	uint n = 256 * 256 * num_images;

	threads = dim3(1024, 1, 1);
	blocks = dim3(n / threads.x + (n % threads.x == 0 ? 0 : 1), 1, 1);

	cuPreprocInput_v2 <<< blocks, threads, 0, stream >>>
		(n, input_buf.devHandle, input.devHandle);
#ifdef CUDA_DEBUG
	cudaCheckError( cudaDeviceSynchronize() );
	cudaCheckError( cudaPeekAtLastError() );
#endif
}

void Control::PreprocWeights_v2(const float *_weights)
{
	dim3 blocks, threads;
	uint n = NUM_WEIGHTS;

	threads = dim3(1024, 1, 1);
	blocks = dim3(n / threads.x + 1, 1, 1);

	cuPreprocWeights_v2 <<< blocks, threads, 0, stream >>> 
		(n, _weights, weights.devHandle, WeightInfo.position);
#ifdef CUDA_DEBUG
	cudaCheckError( cudaDeviceSynchronize() );
	cudaCheckError( cudaPeekAtLastError() );
#endif
}

void Control::Conv2D_v2(const Tensor_t<float4>& input, Tensor_t<float4>& output, const Phase_t& Phase)
{
	uint H = input.nz, W = input.ny, Kv = output.nx;
	
	uint OH = H / STRIDE, OW = W / STRIDE;
	
	uint n = num_images * OH * OW * Kv;

	dim3 blocks, threads;

	threads = dim3(16, 4, 4);	
	blocks.x = Kv / threads.x + (Kv % threads.x == 0 ? 0 : 1);
	blocks.y = OW / threads.y + (OW % threads.y == 0 ? 0 : 1);
	blocks.z = (num_images * OH) / threads.z + ((OH * num_images) % threads.z == 0 ? 0 : 1);
	
	TimerStart(CONV2D);

	cuConv2D_v2 <<< blocks, threads, 0, stream >>>
		(input.devHandle, output.devHandle, Phase.devHandle);
#ifdef CUDA_DEBUG
	cudaCheckError( cudaDeviceSynchronize() );
	cudaCheckError( cudaPeekAtLastError() );
#endif
	
	TimerStop(CONV2D);
}

void Control::LeakyRelu_v2(const Tensor_t<float4>& input, Tensor_t<float4>& output)
{

	uint H = input.nz, W = input.ny, Cv = input.nx;

	dim3 blocks, threads;
	uint n = num_images * H * W * Cv;

	threads = dim3(1024, 1, 1);
	blocks = dim3(n / threads.x + (n % threads.x == 0 ? 0 : 1), 1, 1);

	TimerStart(LEAKYRELU);
	
	cuLeakyRelu_v2 <<< blocks, threads, 0, stream >>> (n, input.devHandle, output.devHandle);
#ifdef CUDA_DEBUG
	cudaCheckError( cudaDeviceSynchronize() );
	cudaCheckError( cudaPeekAtLastError() );
#endif

	TimerStop(LEAKYRELU);
}

void Control::BatchNorm_v2(Tensor_t<float4>& inout, const Phase_t& Phase)
{
	uint H = inout.nz, W = inout.ny, Cv = inout.nx;

	dim3 blocks, threads;
	const uint TPB = 64;
	uint n = num_images * H * W * Cv;

	threads = dim3(TPB, 1, 1);
	blocks = dim3(Cv * num_images, 1, 1);

	TimerStart(BATCHNORM);

	cuBatchNorm_v2 <<< blocks, threads, TPB * sizeof(float4), stream >>> (n, inout.devHandle, Phase.devHandle);
#ifdef CUDA_DEBUG
	cudaCheckError( cudaDeviceSynchronize() );
	cudaCheckError( cudaPeekAtLastError() );
#endif

	TimerStop(BATCHNORM);
}

void Control::Concat_v2(const Tensor_t<float4>& input0, const Tensor_t<float4>& input1, Tensor_t<float4>& output)
{
	uint H = input0.nz, W = input0.ny, C0v = input0.nx;
	uint C1v = input1.nx;
	uint Cv = output.nx;

	dim3 blocks, threads;
	uint n = num_images * H * W * Cv;

	threads = dim3(16, 8, 8);
	blocks.x = Cv / threads.x + (Cv % threads.x == 0 ? 0 : 1);
	blocks.y = W / threads.y + (W % threads.y == 0 ? 0 : 1);
	blocks.z = (H * num_images) / threads.z + ((H * num_images) % threads.z == 0 ? 0 : 1);

	TimerStart(CONCAT);

	cuConcat_v2 <<< blocks, threads, 0, stream >>> (n, input0.devHandle, input1.devHandle, output.devHandle);
#ifdef CUDA_DEBUG
	cudaCheckError( cudaDeviceSynchronize() );
	cudaCheckError( cudaPeekAtLastError() );
#endif

	TimerStop(CONCAT);
}

void Control::Relu_v2(Tensor_t<float4>& inout)
{

	uint H = inout.nz, W = inout.ny, Cv = inout.nx;

	dim3 blocks, threads;
	uint n = num_images * H * W * Cv;  

	threads = dim3(1024, 1, 1);
	blocks = dim3(n / threads.x + (n % threads.x == 0 ? 0 : 1), 1, 1);

	TimerStart(RELU);

	cuRelu_v2 <<< blocks, threads, 0, stream >>> (n, inout.devHandle);
#ifdef CUDA_DEBUG
	cudaCheckError( cudaDeviceSynchronize() );
	cudaCheckError( cudaPeekAtLastError() );
#endif

	TimerStop(RELU);
}

void Control::Conv2DT_v2(const Tensor_t<float4>& input, Tensor_t<float4>& output, const Phase_t& Phase)
{

	uint H = input.nz, W = input.ny, Cv = input.nx, K = Phase.conv_dim[2];
	uint Kv = output.nx;

	uint OH = H * STRIDE, OW = W * STRIDE;

	dim3 blocks, threads;

	threads = dim3(4, 4, 16);
	blocks.x = OW / threads.x + (OW % threads.x == 0 ? 0 : 1);
	blocks.y = OH / threads.y + (OH % threads.y == 0 ? 0 : 1);
	blocks.z = (Kv * num_images) / threads.z + ((Kv * num_images) % threads.z == 0 ? 0 : 1);

	TimerStart(CONV2DT);

	cuConv2DT_v2 <<< blocks, threads, 0, stream >>> (input.devHandle, output.devHandle, Phase.devHandle);
#ifdef CUDA_DEBUG
	cudaCheckError( cudaDeviceSynchronize() );
	cudaCheckError( cudaPeekAtLastError() );
#endif

	TimerStop(CONV2DT);
}

void Control::PostProc_v2(const Tensor_t<float4>& input, Tensor_t<uint8_t>& output)
{

	dim3 blocks, threads;

	uint n = 256 * 256 * num_images;

	threads = dim3(1024, 1, 1);
	blocks = dim3(n / threads.x + (n % threads.x == 0 ? 0 : 1), 1, 1);

	cuPostProc_v2 <<< blocks, threads, 0, stream >>> (n, input.devHandle, output.devHandle);
#ifdef CUDA_DEBUG
	cudaCheckError( cudaDeviceSynchronize() );
	cudaCheckError( cudaPeekAtLastError() );
#endif

}

void Control::EncodePhase_v2()
{
	Conv2D_v2(input, encoder_layer[1], Encoder[1]);

//	PrintPix(encoder_layer[1], "conv" + to_string(1));

	for (int i = 2; i <= 8; ++i) {
		encoder_layer_input[i] = encoder_layer[i - 1];

		LeakyRelu_v2(encoder_layer_input[i], encoder_layer_rectified[i]);

//		PrintPix(encoder_layer_rectified[i], "leaky_relu" + to_string(i));

		Conv2D_v2(encoder_layer_rectified[i], encoder_layer[i], Encoder[i]);

//		PrintPix(encoder_layer[i], "conv" + to_string(i));
		
		BatchNorm_v2(encoder_layer[i], Encoder[i]);
		
//		PrintPix(encoder_layer[i], "batch" + to_string(i));
	}
}

void Control::DecodePhase_v2()
{
	for (int i = 8; i >= 1; --i) {

		if (i == 8) {
			decoder_layer_input[i] = encoder_layer[8];
		}
		else {
			Concat_v2(decoder_layer[i + 1], encoder_layer[i], decoder_layer_input[i]);

//			PrintPix(decoder_layer_input[i], "concat" + to_string(i));
		}
		
		Relu_v2(decoder_layer_input[i]);

//		PrintPix(decoder_layer_rectified[i], "relu" + to_string(i));

		Conv2DT_v2(decoder_layer_input[i], decoder_layer[i], Decoder[i]);

//		PrintPix(decoder_layer[i], "convt" + to_string(i));

		if (i == 1) break;

		BatchNorm_v2(decoder_layer[i], Decoder[i]);
		
//		PrintPix(decoder_layer[i], "dbatch" + to_string(i));
	}

	PostProc_v2(decoder_layer[1], output_buf);
}
