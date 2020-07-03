#include "control.h"

void LayerInfo_t::Predict(map<string, vector<uint_t>>& shapes)
{
	initial = { 256, 256, 3 };

	for (int i = 1; i <= 8; ++i) {
		string scope = "generator/encoder_" + to_string(i) + "/";

		string filter = scope + encodeNames[KERNEL];
		
		if (i == 1) conv[i] = { initial[0] / STRIDE, initial[1] / STRIDE, shapes[filter][3] };
		else 				conv[i] = { conv[i - 1][0] / STRIDE, conv[i - 1][1] / STRIDE, shapes[filter][3] };

		uint_t sz = 1;
		for (auto x : conv[i]) sz *= x;
		conv_sz[i] = sz;

	}

	for (int i = 8; i >= 1; --i) {
		string scope = "generator/decoder_" + to_string(i) + "/";

		string filter = scope + decodeNames[KERNEL];

		if (i == 8) {
			concat[i] = conv[8];
		}
		else {
			concat[i] = { convt[i + 1][0], convt[i + 1][1], convt[i + 1][2] + conv[i][2] };
		}

		concat_sz[i] = 1;
		for (auto x : concat[i]) concat_sz[i] *= x;

		convt[i] = { concat[i][0] * STRIDE, concat[i][1] * STRIDE, shapes[filter][2] };

		convt_sz[i] = 1;
		for (auto x : convt[i]) convt_sz[i] *= x;
	}

	for (int i = 1; i <= 8; ++i) {
		conv[i][2] = conv[i][2] / 4 + (conv[i][2] % 4 == 0 ? 0 : 1);
	}

	for (int i = 8; i >= 1; --i) {
		concat[i][2] = concat[i][2] / 4 + (concat[i][2] % 4 == 0 ? 0 : 1);
		convt[i][2] = convt[i][2] / 4 + (convt[i][2] % 4 == 0 ? 0 : 1);
	}

	uint_t max_conv_sz = 0, max_convt_sz = 0, max_concat_sz = 0;

	for (int i = 1; i <= 8; ++i) {
		if (max_conv_sz < conv_sz[i]) { conv_max = i; max_conv_sz = conv_sz[i]; }
		if (max_convt_sz < convt_sz[i]) { convt_max = i; max_convt_sz = convt_sz[i]; }
		if (max_concat_sz < concat_sz[i]) { concat_max = i; max_concat_sz = concat_sz[i]; }
	}

	// Reverse
	for (int i = 1; i <= 8; ++i) {
		reverse(conv[i].begin(), conv[i].end());
	}

	for (int i = 8; i >= 1; --i) {
		reverse(concat[i].begin(), concat[i].end());
		reverse(convt[i].begin(), convt[i].end());
	}
}

Phase_t::Phase_t()
{
	num_weights = tot_num_entries = tot_num_shapes = 0;

	num_entries = num_shapes = pos_entries = pos_shapes = NULL;
	shapes = NULL;
	Entry = NULL;
}

void Phase_t::Alloc(string names[], map<string, vector<uint_t>>& shape_map)
{
	num_entries = new uint_t[num_weights];
	num_shapes = new uint_t[num_weights];
	pos_entries = new uint_t[num_weights + 1]; *pos_entries = 0;
	pos_shapes = new uint_t[num_weights + 1]; *pos_shapes = 0;

	for (uint_t i = 0; i < num_weights; ++i) {
		string name = names[i];
		vector<uint_t>& shape = shape_map[name];
		uint_t sz = 1;
		for (auto x : shape) sz *= x;
		
		num_entries[i] = sz;
		num_shapes[i] = shape.size();
		pos_entries[i + 1] = pos_entries[i] + num_entries[i];
		pos_shapes[i + 1] = pos_shapes[i] + num_shapes[i];
	}

	tot_num_entries = pos_entries[num_weights];
	tot_num_shapes = pos_shapes[num_weights];

	shapes = new uint_t[tot_num_shapes];

	for (uint_t i = 0, p = 0; i < num_weights; ++i) {
		string name = names[i];
		vector<uint_t>& shape = shape_map[name];
		
		for (auto x : shape) {
			shapes[p++] = x;
		}	
	}
	
	if (order == 1) {
		bias_index = 0;
		kernel_index = 1;
		memcpy(conv_dim, &shapes[pos_shapes[kernel_index]], 4 * sizeof(uint_t));
	}

	if (order > 1) {
		beta_index = 0;
		gamma_index = 1;
		bias_index = 4;
		kernel_index = 5;
		memcpy(conv_dim, &shapes[pos_shapes[kernel_index]], 4 * sizeof(uint_t));
	}
}

void Phase_t::AliasWeights(const Tensor_t<float>& weights, const uint_t *pos_weights)
{
/*
	uint_t tot_order;
	if (type == ENCODE) tot_order = order + 8;
	else tot_order = order;

	uint_t pos = pos_weights[tot_order - 1];
	
	if (order == 1) {
		bias = &weights.Entry[pos + pos_entries[bias_index]];
		kernel = &weights.Entry[pos + pos_entries[kernel_index]];
	}
	else {
		beta = &weights.Entry[pos + pos_entries[beta_index]];
		gamma = &weights.Entry[pos + pos_entries[gamma_index]];
		bias = &weights.Entry[pos + pos_entries[bias_index]];
		kernel = &weights.Entry[pos + pos_entries[kernel_index]];
	}
*/
}

void Phase_t::AliasWeights(const Tensor_t<float4>& weights, const uint_t *pos_weights)
{
	uint_t tot_order;
	if (type == ENCODE) tot_order = order + 8;
	else tot_order = order;

	uint_t pos = pos_weights[tot_order - 1];

	if (order == 1) {
		bias_v2 = &weights.Entry[pos + pos_entries[bias_index]];
		kernel_v2 = &weights.Entry[pos + pos_entries[kernel_index]];
	}
	else {
		beta_v2 = &weights.Entry[pos + pos_entries[beta_index]];
		gamma_v2 = &weights.Entry[pos + pos_entries[gamma_index]];
		bias_v2 = &weights.Entry[pos + pos_entries[bias_index]];
		kernel_v2 = &weights.Entry[pos + pos_entries[kernel_index]];
	}

}

void Phase_t::AliasShapes(const uint_t *shapes, const uint_t *pos_shapes)
{
	uint_t tot_order;
	if (type == ENCODE) tot_order = order + 8;
	else tot_order = order;

	uint_t pos = pos_shapes[tot_order - 1];

	kernel_shapes = &shapes[pos + this->pos_shapes[kernel_index]];
}

void Phase_t::Relocate()
{
	RelocateBuffer(num_weights, num_entries);
	RelocateBuffer(num_weights, num_shapes);
	RelocateBuffer(num_weights + 1, pos_entries);
	RelocateBuffer(num_weights + 1, pos_shapes);

	RelocateBuffer(tot_num_shapes, shapes);

	cudaCheckError( cudaMalloc(&devHandle, sizeof(Phase_t)) );
	cudaCheckError( cudaMemcpy(devHandle, this, sizeof(Phase_t), cudaMemcpyHostToDevice) );
}

template <typename T>
void Tensor_t<T>::Destroy()
{
	dim = sz = 0;
	nx = ny = nz = nw = 0;

	if (Entry != NULL) {
		cudaPointerAttributes attr;
		cudaCheckError( cudaPointerGetAttributes(&attr, Entry) );
		if (attr.type == cudaMemoryTypeHost) delete[] Entry;
		else if (attr.type == cudaMemoryTypeDevice) cudaCheckError( cudaFree(Entry) );
		Entry = NULL;
	}

	lAlloc = false;
}

template <typename T>
void Tensor_t<T>::SetSize(const vector<uint_t>& shape) 
{

	dim = shape.size();
	this->shape = shape;

	nx = ny = nz = nw = 1;

	int p = 0;

	for (vector<uint_t>::const_iterator i = shape.begin(); i != shape.end(); ++i, ++p) {
		if (p == 0) nx = *i;
		else if (p == 1) ny = *i;
		else if (p == 2) nz = *i;
		else if (p == 3) nw = *i;
		else { cout << "\nDimension overflow\n"; exit(EXIT_FAILURE); }
	}

	sz = nx * ny * nz * nw;
}

template <typename T>
void Tensor_t<T>::SetSize(uint_t x, uint_t y, uint_t z, uint_t w)
{
	nx = x; ny = y; nz = z; nw = w;
	sz = nx * ny * nz * nw;
	dim = 0;
	if (nx > 1) ++dim;
	if (ny > 1) ++dim;
	if (nz > 1) ++dim;
	if (nw > 1) ++dim;
}

template <typename T>
void Tensor_t<T>::Dalloc(const vector<uint_t>& shape, const T *buf)
{
	if (lAlloc) Destroy();

	SetSize(shape);

	cudaCheckError( cudaMalloc(&Entry, sz * sizeof(T)) );

	if (buf != NULL) {
		cudaCheckError( cudaMemcpy(Entry, buf, sz * sizeof(T), cudaMemcpyHostToDevice) );
	}

	cudaCheckError( cudaMalloc(&devHandle, sizeof(Tensor_t<T>)) );
	cudaCheckError( cudaMemcpy(devHandle, this, sizeof(Tensor_t<T>), cudaMemcpyHostToDevice) );
}

void Control::RegisterWeight(const string& name, vector<uint_t> shape)
{ 
	shape_map[name] = shape;

	vector<uint_t> pad = shape;

	for (vector<uint_t>::iterator i = pad.begin(); i != pad.end(); ++i) {
		*i = (*i / 4 + (*i % 4 == 0 ? 0 : 1)) * 4;
	}
	
	uint_t dim = (uint_t) pad.size();

	pad[dim - 1] /= 4;

	uint_t sz = 1;
	for (auto x : pad) sz *= x;
	WeightInfo.nwv += sz;

	pad_map[name] = pad;

	pad[dim - 1] *= 4;

	static uint_t p = 0;
	static uint_t q = 0;
	uint_t S[4] = { 1, 1, 1, 1 };
	uint_t P[4] = { 1, 1, 1, 1 };
	uint_t s = 3;

	for (vector<uint_t>::reverse_iterator i = shape.rbegin(), j = pad.rbegin(); i != shape.rend(); ++i, ++j, --s) {
		S[s] = *i;
		P[s] = *j;
	}

	uint_t A = P[0];
	uint_t B = P[1];
	uint_t C = P[2];
	uint_t D = P[3];

	for (uint_t a = 0; a < S[0]; ++a) {
		for (uint_t b = 0; b < S[1]; ++b) {
			for (uint_t c = 0; c < S[2]; ++c) {
				for (uint_t d = 0; d < S[3]; ++d, ++p) {
					WeightInfo.position[p] = q + a * B * C * D + b * C * D + c * D + d;
				}
			}
		}
	}
	q += A * B * C * D;
}

void Control::RegisterWeightLocations()
{
	RegisterWeight("generator/decoder_1/conv2d_transpose/bias", {3});
  RegisterWeight("generator/decoder_1/conv2d_transpose/kernel", {4, 4, 3, 128});
  RegisterWeight("generator/decoder_2/batch_normalization/beta", {64});
  RegisterWeight("generator/decoder_2/batch_normalization/gamma", {64});
  RegisterWeight("generator/decoder_2/batch_normalization/moving_mean", {64});
  RegisterWeight("generator/decoder_2/batch_normalization/moving_variance", {64});
  RegisterWeight("generator/decoder_2/conv2d_transpose/bias", {64});
  RegisterWeight("generator/decoder_2/conv2d_transpose/kernel", {4, 4, 64, 256});
  RegisterWeight("generator/decoder_3/batch_normalization/beta", {128});
  RegisterWeight("generator/decoder_3/batch_normalization/gamma", {128});
  RegisterWeight("generator/decoder_3/batch_normalization/moving_mean", {128});
  RegisterWeight("generator/decoder_3/batch_normalization/moving_variance", {128});
  RegisterWeight("generator/decoder_3/conv2d_transpose/bias", {128});
  RegisterWeight("generator/decoder_3/conv2d_transpose/kernel", {4, 4, 128, 512});
  RegisterWeight("generator/decoder_4/batch_normalization/beta", {256});
  RegisterWeight("generator/decoder_4/batch_normalization/gamma", {256});
  RegisterWeight("generator/decoder_4/batch_normalization/moving_mean", {256});
  RegisterWeight("generator/decoder_4/batch_normalization/moving_variance", {256});
  RegisterWeight("generator/decoder_4/conv2d_transpose/bias", {256});
  RegisterWeight("generator/decoder_4/conv2d_transpose/kernel", {4, 4, 256, 1024});
  RegisterWeight("generator/decoder_5/batch_normalization/beta", {512});
  RegisterWeight("generator/decoder_5/batch_normalization/gamma", {512});
  RegisterWeight("generator/decoder_5/batch_normalization/moving_mean", {512});
  RegisterWeight("generator/decoder_5/batch_normalization/moving_variance", {512});
  RegisterWeight("generator/decoder_5/conv2d_transpose/bias", {512});
  RegisterWeight("generator/decoder_5/conv2d_transpose/kernel", {4, 4, 512, 1024});
  RegisterWeight("generator/decoder_6/batch_normalization/beta", {512});
  RegisterWeight("generator/decoder_6/batch_normalization/gamma", {512});
  RegisterWeight("generator/decoder_6/batch_normalization/moving_mean", {512});
  RegisterWeight("generator/decoder_6/batch_normalization/moving_variance", {512});
  RegisterWeight("generator/decoder_6/conv2d_transpose/bias", {512});
  RegisterWeight("generator/decoder_6/conv2d_transpose/kernel", {4, 4, 512, 1024});
  RegisterWeight("generator/decoder_7/batch_normalization/beta", {512});
  RegisterWeight("generator/decoder_7/batch_normalization/gamma", {512});
  RegisterWeight("generator/decoder_7/batch_normalization/moving_mean", {512});
  RegisterWeight("generator/decoder_7/batch_normalization/moving_variance", {512});
  RegisterWeight("generator/decoder_7/conv2d_transpose/bias", {512});
  RegisterWeight("generator/decoder_7/conv2d_transpose/kernel", {4, 4, 512, 1024});
  RegisterWeight("generator/decoder_8/batch_normalization/beta", {512});
  RegisterWeight("generator/decoder_8/batch_normalization/gamma", {512});
  RegisterWeight("generator/decoder_8/batch_normalization/moving_mean", {512});
  RegisterWeight("generator/decoder_8/batch_normalization/moving_variance", {512});
  RegisterWeight("generator/decoder_8/conv2d_transpose/bias", {512});
  RegisterWeight("generator/decoder_8/conv2d_transpose/kernel", {4, 4, 512, 512});
  RegisterWeight("generator/encoder_1/conv2d/bias", {64});
  RegisterWeight("generator/encoder_1/conv2d/kernel", {4, 4, 3, 64});
  RegisterWeight("generator/encoder_2/batch_normalization/beta", {128});
  RegisterWeight("generator/encoder_2/batch_normalization/gamma", {128});
  RegisterWeight("generator/encoder_2/batch_normalization/moving_mean", {128});
  RegisterWeight("generator/encoder_2/batch_normalization/moving_variance", {128});
  RegisterWeight("generator/encoder_2/conv2d/bias", {128});
  RegisterWeight("generator/encoder_2/conv2d/kernel", {4, 4, 64, 128});
  RegisterWeight("generator/encoder_3/batch_normalization/beta", {256});
  RegisterWeight("generator/encoder_3/batch_normalization/gamma", {256});
  RegisterWeight("generator/encoder_3/batch_normalization/moving_mean", {256});
  RegisterWeight("generator/encoder_3/batch_normalization/moving_variance", {256});
  RegisterWeight("generator/encoder_3/conv2d/bias", {256});
  RegisterWeight("generator/encoder_3/conv2d/kernel", {4, 4, 128, 256});
  RegisterWeight("generator/encoder_4/batch_normalization/beta", {512});
  RegisterWeight("generator/encoder_4/batch_normalization/gamma", {512});
  RegisterWeight("generator/encoder_4/batch_normalization/moving_mean", {512});
  RegisterWeight("generator/encoder_4/batch_normalization/moving_variance", {512});
  RegisterWeight("generator/encoder_4/conv2d/bias", {512});
  RegisterWeight("generator/encoder_4/conv2d/kernel", {4, 4, 256, 512});
  RegisterWeight("generator/encoder_5/batch_normalization/beta", {512});
  RegisterWeight("generator/encoder_5/batch_normalization/gamma", {512});
  RegisterWeight("generator/encoder_5/batch_normalization/moving_mean", {512});
  RegisterWeight("generator/encoder_5/batch_normalization/moving_variance", {512});
  RegisterWeight("generator/encoder_5/conv2d/bias", {512});
  RegisterWeight("generator/encoder_5/conv2d/kernel", {4, 4, 512, 512});
  RegisterWeight("generator/encoder_6/batch_normalization/beta", {512});
  RegisterWeight("generator/encoder_6/batch_normalization/gamma", {512});
  RegisterWeight("generator/encoder_6/batch_normalization/moving_mean", {512});
  RegisterWeight("generator/encoder_6/batch_normalization/moving_variance", {512});
  RegisterWeight("generator/encoder_6/conv2d/bias", {512});
  RegisterWeight("generator/encoder_6/conv2d/kernel", {4, 4, 512, 512});
  RegisterWeight("generator/encoder_7/batch_normalization/beta", {512});
  RegisterWeight("generator/encoder_7/batch_normalization/gamma", {512});
  RegisterWeight("generator/encoder_7/batch_normalization/moving_mean", {512});
  RegisterWeight("generator/encoder_7/batch_normalization/moving_variance", {512});
  RegisterWeight("generator/encoder_7/conv2d/bias", {512});
  RegisterWeight("generator/encoder_7/conv2d/kernel", {4, 4, 512, 512});
  RegisterWeight("generator/encoder_8/batch_normalization/beta", {512});
  RegisterWeight("generator/encoder_8/batch_normalization/gamma", {512});
  RegisterWeight("generator/encoder_8/batch_normalization/moving_mean", {512});
  RegisterWeight("generator/encoder_8/batch_normalization/moving_variance", {512});
  RegisterWeight("generator/encoder_8/conv2d/bias", {512});
  RegisterWeight("generator/encoder_8/conv2d/kernel", {4, 4, 512, 512});
}

void Control::PadWeights()
{
	uint_t num_funcs = WeightInfo.num_funcs;
	
	WeightInfo.nw = NUM_WEIGHTS;

	WeightInfo.pos = new uint_t[num_funcs + 1]; *WeightInfo.pos = 0;
	WeightInfo.pos_padded = new uint_t[num_funcs + 1]; *WeightInfo.pos_padded = 0;

	uint_t p = 0;

	for (uint_t i = 1; i <= 8; ++i) {
		string scope = "generator/decoder_" + to_string(i) + "/", weights[6];
		uint_t num_weights;

		if (i == 1) {
			num_weights = 2;
			weights[0] = scope + decodeNames[BIAS];
			weights[1] = scope + decodeNames[KERNEL];
		}
		else {
			num_weights = 6;			
			weights[0] = scope + decodeNames[BETA];
			weights[1] = scope + decodeNames[GAMMA];
			weights[2] = scope + decodeNames[MEAN];
			weights[3] = scope + decodeNames[VARIANCE];
			weights[4] = scope + decodeNames[BIAS];
			weights[5] = scope + decodeNames[KERNEL];
		}

		for (uint_t j = 0; j < num_weights; ++j, ++p) {
			vector<uint_t>& pad = pad_map[weights[j]];
			vector<uint_t>& shape = shape_map[weights[j]];
			
			uint_t psz = 1, sz = 1;
			
			for (vector<uint_t>::iterator k = pad.begin(), l = shape.begin(); k != pad.end(); ++k, ++l) {
				psz *= (*k);
				sz *= (*l);
			}
			
			WeightInfo.pos[p + 1] = WeightInfo.pos[p] + sz;
			WeightInfo.pos_padded[p + 1] = WeightInfo.pos_padded[p] + psz;
		}
	}

	for (uint_t i = 1; i <= 8; ++i) {
		string scope = "generator/encoder_" + to_string(i) + "/", weights[6];
		uint_t num_weights;

		if (i == 1) {
			num_weights = 2;
			weights[0] = scope + encodeNames[BIAS];
			weights[1] = scope + encodeNames[KERNEL];
		}
		else {
			num_weights = 6;
			weights[0] = scope + encodeNames[BETA];
			weights[1] = scope + encodeNames[GAMMA];
			weights[2] = scope + encodeNames[MEAN];
			weights[3] = scope + encodeNames[VARIANCE];
			weights[4] = scope + encodeNames[BIAS];
			weights[5] = scope + encodeNames[KERNEL];
		}
		
		for (uint_t j = 0; j < num_weights; ++j, ++p) {
			vector<uint_t>& pad = pad_map[weights[j]];
			vector<uint_t>& shape = shape_map[weights[j]];
			
			uint_t psz = 1, sz = 1;

			for (vector<uint_t>::iterator k = pad.begin(), l = shape.begin(); k != pad.end(); ++k, ++l) {
				psz *= (*k);
				sz *= (*l);
			}

			WeightInfo.pos[p + 1] = WeightInfo.pos[p] + sz;
			WeightInfo.pos_padded[p + 1] = WeightInfo.pos_padded[p] + psz;

		}
	}
}

void Control::SortPhase()
{
	for (uint_t i = 1; i <= 8; ++i) {

		string scope = "generator/decoder_" + to_string(i) + "/", weights[6];
		uint_t num_weights;

		if (i == 1) {
			num_weights = 2;
			weights[0] = scope + decodeNames[BIAS];
			weights[1] = scope + decodeNames[KERNEL];
		}
		else {
			num_weights = 6;
			weights[0] = scope + decodeNames[BETA];
			weights[1] = scope + decodeNames[GAMMA];
			weights[2] = scope + decodeNames[MEAN];
			weights[3] = scope + decodeNames[VARIANCE];
			weights[4] = scope + decodeNames[BIAS];
			weights[5] = scope + decodeNames[KERNEL];
		}
		
		Decoder[i].num_weights = num_weights;
		Decoder[i].order = i;
		Decoder[i].type = DECODE;
		Decoder[i].Alloc(weights, pad_map);

	}

	for (uint_t i = 1; i <= 8; ++i) {

		string scope = "generator/encoder_" + to_string(i) + "/", weights[6];
		uint_t num_weights;

		if (i == 1) {
			num_weights = 2;
			weights[0] = scope + encodeNames[BIAS];
			weights[1] = scope + encodeNames[KERNEL];
		}
		else {
			num_weights = 6;
			weights[0] = scope + encodeNames[BETA];
			weights[1] = scope + encodeNames[GAMMA];
			weights[2] = scope + encodeNames[MEAN];
			weights[3] = scope + encodeNames[VARIANCE];
			weights[4] = scope + encodeNames[BIAS];
			weights[5] = scope + encodeNames[KERNEL];
		}
	
		Encoder[i].num_weights = num_weights;
		Encoder[i].order = i;
		Encoder[i].type = ENCODE;
		Encoder[i].Alloc(weights, pad_map);

	}

	*pos_data = 0;
	
	uint_t p = 0;

	for (uint_t i = 1; i <= 8; ++i, ++p) 
		pos_data[p + 1] = pos_data[p] + Decoder[i].tot_num_entries;
	for (uint_t i = 1; i <= 8; ++i, ++p) 
		pos_data[p + 1] = pos_data[p] + Encoder[i].tot_num_entries;

	*pos_shapes = 0;

	p = 0;
	for (uint_t i = 1; i <= 8; ++i, ++p) 
		pos_shapes[p + 1] = pos_shapes[p] + Decoder[i].tot_num_shapes;
	for (uint_t i = 1; i <= 8; ++i, ++p)
		pos_shapes[p + 1] = pos_shapes[p] + Encoder[i].tot_num_shapes;

	tot_num_data = NUM_WEIGHTS;
	tot_num_shapes = pos_shapes[16];

	shapes = new uint_t[tot_num_shapes];
	p = 0;
	for (uint_t i = 1; i <= 8; ++i, ++p)
		memcpy(&shapes[pos_shapes[p]], Decoder[i].shapes, Decoder[i].tot_num_shapes * sizeof(uint_t));
	for (uint_t i = 1; i <= 8; ++i, ++p)
		memcpy(&shapes[pos_shapes[p]], Encoder[i].shapes, Encoder[i].tot_num_shapes * sizeof(uint_t));
}

void Multimessage(const string mesg)
{
#ifdef USE_MPI
	
	int rank, size;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) cout << mesg;

	if (rank != 0) {
		MPI_Send(mesg.c_str(), mesg.size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	}
	else {
		for (int i = 1; i < size; ++i) {
			char mesg[200] = "";
			MPI_Recv(mesg, 200, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			cout << string(mesg);
		}
	}

#else

	cout << mesg;

#endif

}

void Control::MPI_Init()
{
#ifdef USE_MPI

	MPI_COMM = MPI_COMM_WORLD;
	MPI_Comm_size(MPI_COMM, &MPI_GLOBAL_SIZE);
	MPI_Comm_rank(MPI_COMM, &MPI_GLOBAL_RANK);

	MPI_Comm_split_type(MPI_COMM, MPI_COMM_TYPE_SHARED, MPI_GLOBAL_RANK, MPI_INFO_NULL, &MPI_SHARED_COMM);
	MPI_Comm_size(MPI_SHARED_COMM, &MPI_SHARED_SIZE);
	MPI_Comm_rank(MPI_SHARED_COMM, &MPI_SHARED_RANK);

	MPI_Type_contiguous(256 * 256 * 3, MPI_UINT8_T, &MPI_PIXEL_T);
	MPI_Type_commit(&MPI_PIXEL_T);
	
#else

	MPI_GLOBAL_SIZE = MPI_SHARED_SIZE = 1;
	MPI_GLOBAL_RANK = MPI_SHARED_RANK = 0;

#endif

	if (MPI_GLOBAL_RANK == 0) {
		cout << endl
				 << "# of GPUs per Node"
				 << endl;
	}

	int count = 0;
	cudaCheckError( cudaGetDeviceCount(&count) );
	string mesg = ""; 

	if (MPI_SHARED_RANK == 0) {
		mesg = " Node # " + to_string(MPI_GLOBAL_RANK / MPI_SHARED_SIZE) + 
					 " / # of Required GPUs : " + to_string(MPI_SHARED_SIZE) + 
					 " / # of Detected GPUs : " + to_string(count) + "\n";
	}

	Multimessage(mesg);

	cudaCheckError( cudaSetDevice(MPI_SHARED_RANK) );
	cudaCheckError( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
}

void Control::Initialize()
{
	// Predict weights' locations	
	
	WeightInfo.position = new uint_t[NUM_WEIGHTS];

	RegisterWeightLocations();

	RelocateBuffer(NUM_WEIGHTS, WeightInfo.position);

	num_weights = shape_map.size();

	PadWeights();

	// Alloc Phase Info.
	SortPhase();

	weights.Dalloc({ WeightInfo.nwv });
	weights.Fill(make_float4(0.0f, 0.0f, 0.0f, 0.0f));

	RelocateBuffer(tot_num_shapes, shapes);	

	for (uint_t i = 1; i <= 8; ++i) {
		Decoder[i].AliasWeights(weights, pos_data);
		Decoder[i].AliasShapes(shapes, pos_shapes); 
		Decoder[i].Relocate();
	}

	for (uint_t i = 1; i <= 8; ++i) {
		Encoder[i].AliasWeights(weights, pos_data);
		Encoder[i].AliasShapes(shapes, pos_shapes);
		Encoder[i].Relocate();
	}

	Layer.Predict(shape_map);

	// Constant memory
	
	float alpha = 0.2, epsilon = 1.e-5;

	cudaCheckError( cudaMemcpyToSymbol(&Const::alpha, &alpha, sizeof(float)) );
	cudaCheckError( cudaMemcpyToSymbol(&Const::epsilon, &epsilon, sizeof(float)) );
}

void Control::SetLoadBalance(size_t _num_image)
{
	uint_t num_images = (uint_t) _num_image;

	sendcounts = new uint_t[MPI_GLOBAL_SIZE];
	displs = new uint_t[MPI_GLOBAL_SIZE + 1]; *displs = 0;

	sendimages = new uint_t[MPI_GLOBAL_SIZE];
	displimages = new uint_t[MPI_GLOBAL_SIZE + 1]; *displimages = 0;

#ifdef USE_MPI
	MPI_Bcast(&num_images, 1, MPI_INT, 0, MPI_COMM);
#endif

	for (uint_t i = 0; i < (uint_t) MPI_GLOBAL_SIZE; ++i) {
		sendimages[i] = (uint_t) num_images / (uint_t) MPI_GLOBAL_SIZE;
		if (i < num_images % (uint_t) MPI_GLOBAL_SIZE) ++sendimages[i];
		sendcounts[i] = (uint_t) num_pixels * sendimages[i];
	}
	
	num_images = sendimages[MPI_GLOBAL_RANK];
	
	for (uint_t i = 0; i < (uint_t) MPI_GLOBAL_SIZE; ++i) {
		displs[i + 1] = displs[i] + sendcounts[i];
		displimages[i + 1] = displimages[i] + sendimages[i];
	}
	
	this->num_images = (uint_t) num_images;
}

void Control::ScatterImages(uint8_t *&_input_buf, float *&_weights)
{

#ifdef USE_MPI

	if (MPI_GLOBAL_RANK != 0) {
		_input_buf = new uint8_t[ (uint_t) num_images * 256 * 256 * 3];
		_weights = new float[NUM_WEIGHTS];
	}

	num_requests = 0;

	MPI_Ibcast(_weights, NUM_WEIGHTS, MPI_FLOAT, 0, MPI_COMM, &requests[num_requests++]);

	if (MPI_GLOBAL_RANK != 0) {
		MPI_Irecv(_input_buf, num_images, MPI_PIXEL_T, 0, MPI_GLOBAL_RANK, MPI_COMM, &requests[num_requests++]);
	}
	else {
		uint8_t *send_input = _input_buf;
		for (int i = 1; i < MPI_GLOBAL_SIZE; ++i) {
			send_input += sendcounts[i - 1];
			MPI_Isend(send_input, sendimages[i], MPI_PIXEL_T, i, i, MPI_COMM, &requests[num_requests++]);
		}
	}

	WaitAll();

#endif

	cudaCheckError(
		cudaMemcpy(input_buf.Entry, _input_buf, 
			num_images * 256 * 256 * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice)
	);

	RelocateBuffer(NUM_WEIGHTS, _weights);

#ifdef USE_MPI
	if (MPI_GLOBAL_RANK != 0) {
		delete[] _input_buf; 
	}
#endif

}

void Control::Alloc(uint8_t *_input_buf, float *_weights, size_t _num_image)
{
	SetLoadBalance(_num_image);

	num_data = num_images * num_pixels;

	stream = cudaDefaultStream;

	cudaCheckError( cudaMemcpyToSymbol(&Const::num_images, &num_images, sizeof(uint_t)) );

	input_buf.Dalloc({ 3, 256, 256, num_images });
	
	output_buf.Dalloc({ 3, 256, 256, num_images });

	input.Dalloc({ 1, 256, 256, num_images });

	ScatterImages(_input_buf, _weights);
	
	// uint8_t -> float
	PreprocInput_v2();

	// Weight Padding
	PreprocWeights_v2(_weights);

	// Intermediate output

	for (int i = 1; i <= 8; ++i) {
		Layer.conv[i].push_back(num_images);
		Layer.convt[i].push_back(num_images);
		Layer.concat[i].push_back(num_images);
	}

	for (int i = 1; i <= 8; ++i) {
 		vector<uint_t>& conv = Layer.conv[i];
		if (i != 1) {
			encoder_layer_input[i] = encoder_layer[i - 1];
			if (i == 2) encoder_layer_rectified[i].Dalloc(encoder_layer_input[i].shape);
			else {
				encoder_layer_rectified[i].Entry = encoder_layer_rectified[2].Entry;
				encoder_layer_rectified[i].SetSize(encoder_layer_input[i].shape);
				cudaMalloc(&encoder_layer_rectified[i].devHandle, sizeof(Tensor_t<float4>));
				cudaMemcpy(encoder_layer_rectified[i].devHandle, &encoder_layer_rectified[i],
					sizeof(Tensor_t<float4>), cudaMemcpyHostToDevice);
			}
		}
		encoder_layer[i].Dalloc(conv);
	}

	int convt_max = (int) Layer.convt_max;
	
	decoder_layer[convt_max].Dalloc(Layer.convt[convt_max]);

	for (int i = 1; i <= 8; ++i) {
		vector<uint_t>& convt = Layer.convt[i];
		if (i != convt_max) {
			decoder_layer[i].Entry = decoder_layer[convt_max].Entry;
			decoder_layer[i].SetSize(convt);
			cudaMalloc(&decoder_layer[i].devHandle, sizeof(Tensor_t<float4>));
			cudaMemcpy(decoder_layer[i].devHandle, &decoder_layer[i],
				sizeof(Tensor_t<float4>), cudaMemcpyHostToDevice);
		}
	}

	for (int i = 1; i <= 8; ++i) {
		vector<uint_t>& concat = Layer.concat[i];
		if (i == 1) {
			decoder_layer_input[i].Dalloc(concat);
		}
		else if (i == 8) {
			decoder_layer_input[i] = encoder_layer[8];
		}
		else {
			decoder_layer_input[i].Entry = decoder_layer_input[1].Entry;
			decoder_layer_input[i].SetSize(concat);
			cudaMalloc(&decoder_layer_input[i].devHandle, sizeof(Tensor_t<float4>));
			cudaMemcpy(decoder_layer_input[i].devHandle, &decoder_layer_input[i],
				sizeof(Tensor_t<float4>), cudaMemcpyHostToDevice);
		}
	}

	cudaCheckError( cudaDeviceSynchronize() );

	if (MPI_GLOBAL_RANK != 0) cudaCheckError( cudaFree(_weights) );
}


void Control::PushOutput(uint8_t *_output_buf)
{
#ifdef USE_MPI
	if (MPI_GLOBAL_RANK != 0) _output_buf = new uint8_t[(uint_t) num_images * 256 * 256 * 3];
#endif

	cudaCheckError( 
		cudaMemcpy(_output_buf, output_buf.Entry, 
			(uint_t) num_images * 256 * 256 * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost) 
	);

#ifdef USE_MPI
	if (MPI_GLOBAL_RANK != 0) {
		MPI_Isend(_output_buf, num_images, MPI_PIXEL_T, 0, MPI_GLOBAL_RANK, MPI_COMM, &requests[num_requests++]);
	}
	else {
		uint8_t *recv_out = _output_buf;
		for (int i = 1; i < MPI_GLOBAL_SIZE; ++i) {
			recv_out += sendcounts[i - 1];
			MPI_Irecv(recv_out, sendimages[i], MPI_PIXEL_T, i, i, MPI_COMM, &requests[num_requests++]);
		}
	}

	WaitAll();

	if (MPI_GLOBAL_RANK != 0) delete[] _output_buf;
#endif
}

void Control::WaitAll()
{

#ifdef USE_MPI

	MPI_Waitall(num_requests, requests, MPI_STATUS_IGNORE);

	num_requests = 0;

#endif

}

void Control::TimerStart(enum Operations op)
{
#ifdef CUDA_DEBUG
	start_times[op] = get_time();
#endif
}

void Control::TimerStop(enum Operations op)
{
#ifdef CUDA_DEBUG
	timers[op] += get_time() - start_times[op];
#endif
}

void Control::PrintTimerInfo()
{
#ifdef CUDA_DEBUG
	static const string names[] = {
		"convolution 2d",
		"leaky relu",
		"batch normalization",
		"relu",
		"convolution 2d transposed",
		"concatenation"
	};

#ifdef USE_MPI

	double buf[NUM_OPERATIONS];

	MPI_Reduce(timers, buf, NUM_OPERATIONS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM);

	if (MPI_GLOBAL_RANK == 0) memcpy(buf, timers, NUM_OPERATIONS * sizeof(double));

#endif

	if (MPI_GLOBAL_RANK == 0) {
		const uint_t num_timers = NUM_OPERATIONS;
		cout << "\n";	
		cout.setf(ios::fixed); cout << setprecision(4);
		for (uint_t i = 0; i < num_timers; ++i) {
			cout.unsetf(ios::right); cout.setf(ios::left);
			cout << setw(32) << names[i];
			cout.unsetf(ios::left); cout.setf(ios::right);
			cout << " " << timers[i] << " sec." << endl;
		}
	}

#endif
}

void Control::PrintDeviceMemoryUsage()
{
	size_t free_memory, total_memory;

	cudaMemGetInfo(&free_memory, &total_memory);

	cout << " Device " + to_string(MPI_GLOBAL_RANK) + " : " 
			 << to_string((total_memory - free_memory) / 1048576) + "MB Used / "
			 << to_string(total_memory / 1048576) + "MB Total" << endl;
}
