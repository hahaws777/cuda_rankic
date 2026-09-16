#pragma once

// Include <cub/block/block_scan.cuh> outside the enclosing namespace, together
// with the existing block radix-sort/reduce headers. This header follows the
// Moments, Add, and correlation definitions in rankic_cuda.cu.
struct ScanRankMax {
  __device__ __forceinline__ int operator()(int a, int b) const {
    return a>b?a:b;
  }
};

template<int I>
union ScanRankWorkspace {
  typename cub::BlockRadixSort<float,256,I,float>::TempStorage sort;
  typename cub::BlockScan<int,256>::TempStorage scan;
  typename cub::BlockReduce<int,256>::TempStorage count;
  typename cub::BlockReduce<Moments,256>::TempStorage reduce;
  float sorted[256*I];
  int endpoints[256*I];
};

template<int I>
constexpr int scan_rankic_shared_bytes() {
  return int(sizeof(ScanRankWorkspace<I>));
}

// Sort output and the scan both use blocked positions threadIdx.x*I+j.
// On return, heads contains each valid item's group start; when ties exist,
// tmp.endpoints[heads[j]] contains that group's inclusive end. The caller must
// finish reading endpoints and synchronize before reusing the shared union.
template<int I>
__device__ __forceinline__ bool scan_rank_heads(
    const float (&keys)[I], int (&heads)[I], int count,
    ScanRankWorkspace<I>& tmp) {
  const int first=threadIdx.x*I;
  // CUB sort may still be reading its temporary storage in another warp.
  __syncthreads();
  #pragma unroll
  for(int j=0;j<I;++j) tmp.sorted[first+j]=keys[j];
  __syncthreads();

  bool tails[I];
  bool thread_has_ties=false;
  #pragma unroll
  for(int j=0;j<I;++j) {
    const int p=first+j;
    const bool valid=p<count;
    const bool head=valid && (p==0 || keys[j]!=tmp.sorted[p-1]);
    tails[j]=valid && (p+1==count || keys[j]!=tmp.sorted[p+1]);
    heads[j]=head?p:0;
    thread_has_ties|=valid && !head;
  }
  // This vote also completes every sorted-key read before scan overwrites it.
  const bool has_ties=__syncthreads_or(thread_has_ties)!=0;
  if(!has_ties) {
    #pragma unroll
    for(int j=0;j<I;++j) heads[j]=first+j;
    return false;
  }

  cub::BlockScan<int,256>(tmp.scan).InclusiveScan(heads,heads,ScanRankMax{});
  __syncthreads();
  // Exactly one tail writes each group endpoint. Other entries need no init.
  #pragma unroll
  for(int j=0;j<I;++j)
    if(tails[j]) tmp.endpoints[heads[j]]=first+j;
  __syncthreads();
  return true;
}

// Launch one 256-thread block per row with scan_rankic_shared_bytes<I>() bytes
// of dynamic shared memory. The dispatcher must guarantee n<=256*I.
template<int I>
__global__ void scan_rankic(const float* x, const float* y, float* out, int n) {
  static_assert(I>0, "items per thread must be positive");
  using Sort=cub::BlockRadixSort<float,256,I,float>;
  using Count=cub::BlockReduce<int,256>;
  using Reduce=cub::BlockReduce<Moments,256>;
  extern __shared__ __align__(16) unsigned char scan_workspace[];
  auto& tmp=*reinterpret_cast<ScanRankWorkspace<I>*>(scan_workspace);
  __shared__ int count;
  const int lane=threadIdx.x;
  const int64_t base=int64_t(blockIdx.x)*n;
  float keys[I], payload[I];
  int valid_count=0;
  #pragma unroll
  for(int j=0;j<I;++j) {
    // Input permutation is irrelevant; keep original pairings and coalescing.
    const int i=lane+j*256;
    const float a=i<n?x[base+i]:CUDART_INF_F;
    const float b=i<n?y[base+i]:CUDART_INF_F;
    const bool valid=isfinite(a)&&isfinite(b);
    keys[j]=valid?(a==0?0.f:a):CUDART_INF_F;
    payload[j]=valid?(b==0?0.f:b):CUDART_INF_F;
    valid_count+=valid;
  }
  const int total=Count(tmp.count).Sum(valid_count);
  if(lane==0) count=total;
  __syncthreads();
  if(count<2) {
    if(lane==0) out[blockIdx.x]=nanf("");
    return;
  }
  const float mean=0.5f*float(count-1);
  int heads[I];

  Sort(tmp.sort).Sort(keys,payload);
  const bool factor_ties=scan_rank_heads(keys,heads,count,tmp);
  #pragma unroll
  for(int j=0;j<I;++j) {
    const int p=lane*I+j;
    float rank=0;
    if(p<count) {
      const int end=factor_ties?tmp.endpoints[heads[j]]:p;
      rank=0.5f*float(heads[j]+end)-mean;
    }
    keys[j]=payload[j];
    payload[j]=rank;
  }
  __syncthreads();
  Sort(tmp.sort).Sort(keys,payload);
  const bool return_ties=scan_rank_heads(keys,heads,count,tmp);
  Moments m={0,0,0};
  #pragma unroll
  for(int j=0;j<I;++j) {
    const int p=lane*I+j;
    if(p<count) {
      const int end=return_ties?tmp.endpoints[heads[j]]:p;
      const float a=payload[j];
      const float b=0.5f*float(heads[j]+end)-mean;
      m.xx+=a*a; m.yy+=b*b; m.xy+=a*b;
    }
  }
  __syncthreads();
  const Moments sum=Reduce(tmp.reduce).Reduce(m,Add{});
  if(lane==0) out[blockIdx.x]=correlation(sum,count);
}
