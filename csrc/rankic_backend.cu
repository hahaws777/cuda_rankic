#include "rankic.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <climits>
#include <cmath>
#include <math_constants.h>

namespace {
constexpr int B = 256;
struct Moments { float xx, yy, xy; };
struct Add {
  __device__ Moments operator()(Moments a, Moments b) const {
    return {a.xx+b.xx, a.yy+b.yy, a.xy+b.xy};
  }
};

// Binary search is only needed on tied sides. This also handles ties spanning
// tiles/warps and arbitrarily long runs without serial walks through the run.
__device__ float midrank(const float* keys, int p, int n) {
  float v = keys[p];
  int lo=p, hi=p+1;
  if (p && keys[p-1] == v) {
    int l=0, r=p;
    while(l<r) { int m=l+(r-l)/2; if(keys[m]<v) l=m+1; else r=m; }
    lo=l;
  }
  if (p+1<n && keys[p+1] == v) {
    int l=p+1, r=n;
    while(l<r) { int m=l+(r-l)/2; if(keys[m]<=v) l=m+1; else r=m; }
    hi=l;
  }
  return 0.5f*(float(lo)+float(hi-1));
}
__device__ float correlation(Moments m, int count) {
  if(count<2 || m.xx<=0 || m.yy<=0) return nanf("");
  return fminf(1.f, fmaxf(-1.f, (m.xy / sqrtf(m.xx)) / sqrtf(m.yy)));
}

#include "warp_rankic.cuh"
#include "scan_rankic.cuh"

template<int I>
union BlockWorkspace {
  typename cub::BlockRadixSort<float,B,I,float>::TempStorage sort;
  typename cub::BlockReduce<int,B>::TempStorage count;
  typename cub::BlockReduce<Moments,B>::TempStorage reduce;
  float sorted[B*I];
};

template<int I>
__global__ void block_rankic(const float* x, const float* y, float* out, int n) {
  using Sort=cub::BlockRadixSort<float,B,I,float>;
  using Count=cub::BlockReduce<int,B>;
  using Reduce=cub::BlockReduce<Moments,B>;
  extern __shared__ __align__(16) unsigned char workspace[];
  auto& tmp=*reinterpret_cast<BlockWorkspace<I>*>(workspace);
  float* sorted=tmp.sorted;
  __shared__ int count;
  const int row=blockIdx.x, lane=threadIdx.x;
  const int64_t base=int64_t(row)*n;
  float keys[I], payload[I]; int valid_count=0;
  #pragma unroll
  for(int j=0;j<I;++j) {
    int i=lane+j*B; float a=i<n?x[base+i]:CUDART_INF_F, b=i<n?y[base+i]:CUDART_INF_F;
    bool valid=isfinite(a)&&isfinite(b);
    keys[j]=valid?(a==0?0.f:a):CUDART_INF_F;
    payload[j]=valid?(b==0?0.f:b):CUDART_INF_F;
    valid_count+=valid;
  }
  int total=Count(tmp.count).Sum(valid_count);
  if(lane==0) count=total;
  __syncthreads();
  if(count<2) { if(lane==0) out[row]=nanf(""); return; }
  // Input items may be striped: sorting accepts any permutation of the tile.
  Sort(tmp.sort).SortBlockedToStriped(keys,payload);
  __syncthreads();
  #pragma unroll
  for(int j=0;j<I;++j) sorted[lane+j*B]=keys[j];
  __syncthreads();
  if(sorted[0]==sorted[count-1]) { if(lane==0) out[row]=nanf(""); return; }
  float mean=0.5f*float(count-1);
  #pragma unroll
  for(int j=0;j<I;++j) {
    int p=lane+j*B;
    float r=p<count?midrank(sorted,p,count)-mean:0.f;
    keys[j]=payload[j];
    payload[j]=r;
  }
  __syncthreads();
  Sort(tmp.sort).SortBlockedToStriped(keys,payload);
  __syncthreads();
  #pragma unroll
  for(int j=0;j<I;++j) sorted[lane+j*B]=keys[j];
  __syncthreads();
  Moments m={0,0,0};
  #pragma unroll
  for(int j=0;j<I;++j) {
    int p=lane+j*B;
    float b=p<count?midrank(sorted,p,count)-mean:0.f;
    if(p<count) {
      float a=payload[j];
      m.xx+=a*a; m.yy+=b*b; m.xy+=a*b;
    }
  }
  // Every reader must finish before the union becomes reduction scratch.
  __syncthreads();
  Moments sum=Reduce(tmp.reduce).Reduce(m,Add{});
  if(lane==0) out[row]=correlation(sum,count);
}

__global__ void prepare(const float* x,const float* y,float* keys,int* ids,int64_t size,int n) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size;i+=int64_t(blockDim.x)*gridDim.x) {
    float a=x[i],b=y[i]; bool valid=isfinite(a)&&isfinite(b);
    keys[i]=valid?(a==0?0.f:a):CUDART_INF_F;
    keys[size+i]=valid?(b==0?0.f:b):CUDART_INF_F;
    ids[i]=ids[size+i]=int(i%n);
  }
}
__global__ void offsets_kernel(int* offsets,int segments,int n) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<=segments;i+=blockDim.x*gridDim.x) offsets[i]=i*n;
}
__device__ int finite_count(const float* keys,int n) {
  int l=0,r=n;
  while(l<r) { int m=l+(r-l)/2; if(isfinite(keys[m])) l=m+1; else r=m; }
  return l;
}
__global__ void factor_ranks(const float* keys,const int* ids,float* rx,int n) {
  int64_t base=int64_t(blockIdx.x)*n;
  __shared__ int count;
  if(threadIdx.x==0) count=finite_count(keys+base,n);
  __syncthreads();
  float mean=0.5f*float(count-1);
  for(int p=threadIdx.x;p<count;p+=B) rx[base+ids[base+p]]=midrank(keys+base,p,count)-mean;
}
__global__ void return_reduce(const float* keys,const int* ids,const float* rx,float* out,int n) {
  int64_t base=int64_t(blockIdx.x)*n;
  __shared__ int count;
  __shared__ cub::BlockReduce<Moments,B>::TempStorage tmp;
  if(threadIdx.x==0) count=finite_count(keys+base,n);
  __syncthreads();
  float mean=0.5f*float(count-1);
  Moments m={0,0,0};
  for(int p=threadIdx.x;p<count;p+=B) {
    float a=rx[base+ids[base+p]],b=midrank(keys+base,p,count)-mean;
    m.xx+=a*a; m.yy+=b*b; m.xy+=a*b;
  }
  Moments sum=cub::BlockReduce<Moments,B>(tmp).Reduce(m,Add{});
  if(threadIdx.x==0) out[blockIdx.x]=correlation(sum,count);
}
__global__ void fill_nan(float* out,int rows) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows;i+=int64_t(blockDim.x)*gridDim.x) out[i]=nanf("");
}
int fail(int code,const char* message,char* error,size_t capacity) {
  if(error && capacity) std::snprintf(error,capacity,"%s",message);
  return code;
}
int cuda_status(cudaError_t status,char* error,size_t capacity) {
  return status==cudaSuccess ? RANKIC_SUCCESS : fail(RANKIC_CUDA_ERROR,cudaGetErrorString(status),error,capacity);
}
int validate(int64_t rows,int64_t cols,int strategy,char* error,size_t capacity) {
  if(rows<0 || cols<0 || rows>INT_MAX/2 || cols>INT_MAX/2 ||
     (cols && rows>(INT_MAX/2)/cols))
    return fail(RANKIC_INVALID_ARGUMENT,"dimensions exceed per-launch indexing limit; chunk rows",error,capacity);
  if(strategy<0 || strategy>2 || (strategy==1 && cols>6144))
    return fail(RANKIC_INVALID_ARGUMENT,"invalid strategy or block width exceeds 6144",error,capacity);
  return RANKIC_SUCCESS;
}
bool segmented(int rows,int cols,int strategy) {
  return rows>0 && cols>=2 && (strategy==2 || cols>6144);
}
struct Layout {
  size_t k1,k2,i1,i2,rx,offsets,scratch,sort_bytes,total;
};
size_t aligned(size_t n) { return (n+255)&~size_t(255); }
cudaError_t layout_for(int rows,int cols,Layout& layout,cudaStream_t stream) {
  const size_t size=size_t(rows)*cols;
  size_t cursor=0;
  auto reserve=[&](size_t bytes) { size_t pos=cursor; cursor+=aligned(bytes); return pos; };
  layout.k1=reserve(2*size*sizeof(float));
  layout.k2=reserve(2*size*sizeof(float));
  layout.i1=reserve(2*size*sizeof(int));
  layout.i2=reserve(2*size*sizeof(int));
  layout.rx=reserve(size*sizeof(float));
  layout.offsets=reserve(size_t(2*rows+1)*sizeof(int));
  layout.sort_bytes=0;
  cudaError_t status=cub::DeviceSegmentedRadixSort::SortPairs(
    nullptr,layout.sort_bytes,static_cast<const float*>(nullptr),static_cast<float*>(nullptr),
    static_cast<const int*>(nullptr),static_cast<int*>(nullptr),int(2*size),2*rows,
    static_cast<const int*>(nullptr),static_cast<const int*>(nullptr),0,32,stream);
  layout.scratch=reserve(layout.sort_bytes);
  layout.total=cursor;
  return status;
}
template<int I>
cudaError_t launch_block(const float* x,const float* y,float* out,int rows,int cols,cudaStream_t stream) {
  constexpr int bytes=sizeof(BlockWorkspace<I>);
  if(bytes>48*1024-16) {
    auto status=cudaFuncSetAttribute(block_rankic<I>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(status!=cudaSuccess) return status;
  }
  block_rankic<I><<<rows,B,bytes,stream>>>(x,y,out,cols);
  return cudaGetLastError();
}
} // namespace

extern "C" int rankic_workspace_size(int64_t rows,int64_t cols,int strategy,
    size_t* bytes,char* error,size_t error_size) {
  if(error && error_size) error[0]=0;
  if(!bytes) return fail(RANKIC_INVALID_ARGUMENT,"bytes pointer is required",error,error_size);
  *bytes=0;
  int status=validate(rows,cols,strategy,error,error_size);
  if(status) return status;
  if(!segmented(int(rows),int(cols),strategy)) return RANKIC_SUCCESS;
  Layout layout{};
  status=cuda_status(layout_for(int(rows),int(cols),layout,nullptr),error,error_size);
  if(!status) *bytes=layout.total;
  return status;
}

extern "C" int rankic_cuda_f32(const float* x,const float* y,float* out,
    int64_t rows,int64_t cols,void* workspace,size_t workspace_bytes,
    int strategy,void* stream_pointer,char* error,size_t error_size) {
  if(error && error_size) error[0]=0;
  int status=validate(rows,cols,strategy,error,error_size);
  if(status || rows==0) return status;
  if(!out || (cols>=2 && (!x || !y)))
    return fail(RANKIC_INVALID_ARGUMENT,"required device pointer is null",error,error_size);
  int t=int(rows),n=int(cols);
  cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_pointer);
  if(n<2) {
    fill_nan<<<std::min((t+255)/256,65535),256,0,stream>>>(out,t);
    return cuda_status(cudaGetLastError(),error,error_size);
  }
  if(strategy==0 && n<=256 && t>=1024) {
    #define WARP_TILE(I) warp_rankic<I,4><<<(t+3)/4,128,0,stream>>>(x,y,out,t,n)
    if(n<=32) { WARP_TILE(1); }
    else if(n<=64) { WARP_TILE(2); }
    else if(n<=128) { WARP_TILE(4); }
    else { WARP_TILE(8); }
    #undef WARP_TILE
    return cuda_status(cudaGetLastError(),error,error_size);
  }
  if(!segmented(t,n,strategy)) {
    cudaError_t result=cudaSuccess;
    if(n<=256) result=launch_block<1>(x,y,out,t,n,stream);
    else if(n<=512) result=launch_block<2>(x,y,out,t,n,stream);
    else if(n<=1024) result=launch_block<4>(x,y,out,t,n,stream);
    else if(n<=2048) result=launch_block<8>(x,y,out,t,n,stream);
    else if(n<=3072) scan_rankic<12><<<t,B,scan_rankic_shared_bytes<12>(),stream>>>(x,y,out,n);
    else if(n<=4096) result=launch_block<16>(x,y,out,t,n,stream);
    else if(n<=5120) scan_rankic<20><<<t,B,scan_rankic_shared_bytes<20>(),stream>>>(x,y,out,n);
    else result=launch_block<24>(x,y,out,t,n,stream);
    if(result!=cudaSuccess) return cuda_status(result,error,error_size);
    return cuda_status(cudaGetLastError(),error,error_size);
  }
  Layout layout{};
  status=cuda_status(layout_for(t,n,layout,stream),error,error_size);
  if(status) return status;
  if(!workspace || workspace_bytes<layout.total)
    return fail(RANKIC_WORKSPACE_TOO_SMALL,"workspace is smaller than rankic_workspace_size",error,error_size);
  if(reinterpret_cast<uintptr_t>(workspace)%256)
    return fail(RANKIC_INVALID_ARGUMENT,"workspace must be 256-byte aligned",error,error_size);
  auto* base=static_cast<unsigned char*>(workspace);
  float* k1=reinterpret_cast<float*>(base+layout.k1);
  float* k2=reinterpret_cast<float*>(base+layout.k2);
  int* i1=reinterpret_cast<int*>(base+layout.i1);
  int* i2=reinterpret_cast<int*>(base+layout.i2);
  float* rx=reinterpret_cast<float*>(base+layout.rx);
  int* offsets=reinterpret_cast<int*>(base+layout.offsets);
  int64_t size=rows*cols;
  prepare<<<int(std::min<int64_t>((size+255)/256,65535)),256,0,stream>>>(x,y,k1,i1,size,n);
  offsets_kernel<<<std::min((2*t+256)/256,65535),256,0,stream>>>(offsets,2*t,n);
  status=cuda_status(cub::DeviceSegmentedRadixSort::SortPairs(base+layout.scratch,
    layout.sort_bytes,k1,k2,i1,i2,int(2*size),2*t,offsets,offsets+1,0,32,stream),error,error_size);
  if(status) return status;
  factor_ranks<<<t,B,0,stream>>>(k2,i2,rx,n);
  return_reduce<<<t,B,0,stream>>>(k2+size,i2+size,rx,out,n);
  return cuda_status(cudaGetLastError(),error,error_size);
}
