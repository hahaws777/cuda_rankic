#pragma once

// Include <cub/warp/warp_merge_sort.cuh> outside the enclosing namespace.
// This header follows Moments, midrank(), and correlation() in rankic_cuda.cu.
struct WarpRankLess {
  __device__ __forceinline__ bool operator()(float a, float b) const {
    return a < b;
  }
};

template<int I>
union WarpRankWorkspace {
  typename cub::WarpMergeSort<float,I,32,float>::TempStorage sort;
  float sorted[32*I];
};

// Launch with 32*WARPS threads and ceil(t/WARPS) blocks; n must be <=32*I.
// Each warp owns all storage and synchronization for one row, so unused warps
// in the last block can return without participating in any block collective.
template<int I, int WARPS=4>
__global__ void warp_rankic(const float* x, const float* y, float* out,
                          int t, int n) {
  static_assert(I>0, "items per thread must be positive");
  static_assert(WARPS>0 && WARPS<=32, "invalid warps per block");
  constexpr unsigned mask=0xffffffffu;
  using Sort=cub::WarpMergeSort<float,I,32,float>;
  __shared__ WarpRankWorkspace<I> workspace[WARPS];
  const int warp=threadIdx.x/32, lane=threadIdx.x%32;
  const int row=blockIdx.x*WARPS+warp;
  if(row>=t) return;

  auto& tmp=workspace[warp];
  float* sorted=tmp.sorted;
  const int64_t base=int64_t(row)*n;
  float keys[I], payload[I];
  int count=0;
  #pragma unroll
  for(int j=0;j<I;++j) {
    // Coalesced loads; the full-tile sort accepts this permutation of pairs.
    const int i=lane+j*32;
    const float a=i<n?x[base+i]:CUDART_INF_F;
    const float b=i<n?y[base+i]:CUDART_INF_F;
    const bool valid=isfinite(a)&&isfinite(b);
    keys[j]=valid?(a==0?0.f:a):CUDART_INF_F;
    payload[j]=valid?(b==0?0.f:b):CUDART_INF_F;
    count+=valid;
  }
  #pragma unroll
  for(int offset=16;offset>0;offset/=2)
    count+=__shfl_down_sync(mask,count,offset);
  count=__shfl_sync(mask,count,0);
  if(count<2) {
    if(lane==0) out[row]=nanf("");
    return;
  }
  const float mean=0.5f*float(count-1);

  Sort(tmp.sort).Sort(keys,payload,WarpRankLess{});
  __syncwarp(mask);
  // WarpMergeSort returns a blocked arrangement, unlike the block radix
  // path's striped output: item j in lane l has sorted position l*I+j.
  #pragma unroll
  for(int j=0;j<I;++j) sorted[lane*I+j]=keys[j];
  __syncwarp(mask);
  #pragma unroll
  for(int j=0;j<I;++j) {
    const int p=lane*I+j;
    const float rank=p<count?midrank(sorted,p,count)-mean:0.f;
    keys[j]=payload[j];
    payload[j]=rank;
  }
  // All factor-key readers must finish before sort reuses their storage.
  __syncwarp(mask);
  Sort(tmp.sort).Sort(keys,payload,WarpRankLess{});
  __syncwarp(mask);
  #pragma unroll
  for(int j=0;j<I;++j) sorted[lane*I+j]=keys[j];
  __syncwarp(mask);

  Moments m={0,0,0};
  #pragma unroll
  for(int j=0;j<I;++j) {
    const int p=lane*I+j;
    if(p<count) {
      const float a=payload[j];
      const float b=midrank(sorted,p,count)-mean;
      m.xx+=a*a; m.yy+=b*b; m.xy+=a*b;
    }
  }
  #pragma unroll
  for(int offset=16;offset>0;offset/=2) {
    m.xx+=__shfl_down_sync(mask,m.xx,offset);
    m.yy+=__shfl_down_sync(mask,m.yy,offset);
    m.xy+=__shfl_down_sync(mask,m.xy,offset);
  }
  if(lane==0) out[row]=correlation(m,count);
}
