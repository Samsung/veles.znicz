extern "C"
__global__ void fill_minibatch_data_labels(
    const original_data_dtype    /* IN */    *original_data,
    minibatch_data_dtype        /* OUT */    *minibatch_data,
    const int                    /* IN */    start_offset,
    const int                    /* IN */    count,
#if LABELS > 0
    const int                    /* IN */    *original_labels,
    int                         /* OUT */    *minibatch_labels,
#endif
    const int                    /* IN */    *shuffled_indices,
    int                         /* OUT */    *minibatch_indices) {

  int sample_number = blockDim.x * blockIdx.x + threadIdx.x;
  int real_sample_number = sample_number < count ? shuffled_indices[start_offset + sample_number] : -1;

  int offs_in_sample = blockDim.y * blockIdx.y + threadIdx.y;
  int offs_in_data = real_sample_number * SAMPLE_SIZE + offs_in_sample;
  int offs_in_minibatch = sample_number * SAMPLE_SIZE + offs_in_sample;

  minibatch_data[offs_in_minibatch] = sample_number < count ? (minibatch_data_dtype)original_data[offs_in_data] : 0;
#if LABELS > 0
  minibatch_labels[sample_number] = sample_number < count ? original_labels[real_sample_number] : -1;
#endif
  minibatch_indices[sample_number] = real_sample_number;
}


#if TARGET > 0
extern "C"
__global__ void fill_minibatch_target(
    const original_target_dtype    /* IN */    *original_target,
    minibatch_target_dtype        /* OUT */    *minibatch_target,
    const int                      /* IN */    start_offset,
    const int                      /* IN */    count,
    int                            /* IN */    *shuffled_indices) {

  int sample_number = blockDim.x * blockIdx.x + threadIdx.x;
  int real_sample_number = sample_number < count ? shuffled_indices[start_offset + sample_number] : -1;

  int offs_in_sample = blockDim.y * blockIdx.y + threadIdx.y;
  int offs_in_target = real_sample_number * TARGET_SIZE + offs_in_sample;
  int offs_in_minibatch = sample_number * TARGET_SIZE + offs_in_sample;

  minibatch_target[offs_in_minibatch] = sample_number < count ? (minibatch_target_dtype)original_target[offs_in_target] : 0;
}
#endif
