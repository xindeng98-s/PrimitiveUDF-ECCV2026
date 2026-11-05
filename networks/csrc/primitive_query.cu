#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include <vector>

using torch_tensor = at::Tensor;

static void check(const torch_tensor& t, at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dtype,
              "primitive kernel: wrong device, contiguity or dtype");
}

__device__ float norm3(float x, float y, float z) {
  return sqrtf((x*x + y*y) + z*z);
}

// A block owns one K=16 neighborhood. Only final primitives leave the block.
// 16 * C(6,2) = 240 raw candidates; C(16,3) = 560 identity slots.
__global__ void select_kernel(
    const float* xyz, const float* query, const float* uv,
    const int64_t* graph, const bool* graph_mask, const bool* point_mask,
    const float* area_eps, int keep, bool build_graph,
    int64_t* output, bool* output_mask, bool* uncertain, int64_t* counts) {
  int row = blockIdx.x, lane = threadIdx.x;
  __shared__ int first[560];
  __shared__ float scores[256];
  __shared__ int candidate_j[240], candidate_k[240];
  __shared__ float warp_score[8];
  __shared__ int warp_index[8];
  __shared__ int count, winner, ambiguous;
  __shared__ int64_t uv_graph[96];
  __shared__ bool uv_mask[96];
  for (int s=lane; s<560; s+=256) first[s]=240;
  if (lane==0) { count=0; ambiguous=0; }
  xyz += row*48; query += row*3; uv += row*32;
  if (!build_graph) { graph += row*96; graph_mask += row*96; }
  point_mask += row*16;
  __syncthreads();
  if (build_graph && lane<16) {
    float distances[16];
    for (int target=0;target<16;++target) {
      float du=uv[lane*2]-uv[target*2], dv=uv[lane*2+1]-uv[target*2+1];
      distances[target]=(lane!=target && point_mask[lane] && point_mask[target]) ? du*du+dv*dv : CUDART_INF_F;
    }
    float previous=-CUDART_INF_F;
    for (int edge=0;edge<7;++edge) {
      int best=0;
      for (int target=1;target<16;++target) if (distances[target]<distances[best]) best=target;
      float d=distances[best];
      if (edge>0 && isfinite(d) && d-previous<=8*FLT_EPSILON*fmaxf(d,1e-12f)) atomicOr(&ambiguous,1);
      if (edge<6) { uv_graph[lane*6+edge]=best; uv_mask[lane*6+edge]=isfinite(d); }
      previous=d; distances[best]=CUDART_INF_F;
    }
  }
  __syncthreads();
  if (build_graph) { graph=uv_graph; graph_mask=uv_mask; }
  int i=0, j=0, k=0, code=0;
  bool distinct=false, valid=false;
  float area2=0, score=-1e9f;
  if (lane < 240) {
    i=lane/15;
    int pair=lane%15, a=0;
    while (pair >= 5-a) { pair-=5-a; ++a; }
    int b=a+1+pair;
    j=(int)graph[i*6+a]; k=(int)graph[i*6+b];
    distinct=i!=j && i!=k && j!=k;
    valid=distinct && graph_mask[i*6+a] && graph_mask[i*6+b]
          && point_mask[i] && point_mask[j] && point_mask[k];
    float x1=uv[j*2]-uv[i*2], y1=uv[j*2+1]-uv[i*2+1];
    float x2=uv[k*2]-uv[i*2], y2=uv[k*2+1]-uv[i*2+1];
    area2=x1*y2-y1*x2;
    valid=valid && fabsf(area2)>area_eps[row];
    if (area2<0) { int t=j; j=k; k=t; }
    candidate_j[lane]=j; candidate_k[lane]=k;
    int lo=min(i,min(j,k)), hi=max(i,max(j,k)), mid=i+j+k-lo-hi;
    code=hi*(hi-1)*(hi-2)/6 + mid*(mid-1)/2 + lo;
  }
  __syncthreads();
  // Match the reference's FIRST occurrence rule even if that occurrence has
  // an invalid edge/mask/area. Invalid distinct triples must still claim IDs.
  if (lane<240 && distinct) atomicMin(first+code, lane);
  __syncthreads();
  if (lane<240 && distinct && first[code]==lane && valid) {
    float ax=xyz[j*3]-xyz[i*3], ay=xyz[j*3+1]-xyz[i*3+1], az=xyz[j*3+2]-xyz[i*3+2];
    float bx=xyz[k*3]-xyz[i*3], by=xyz[k*3+1]-xyz[i*3+1], bz=xyz[k*3+2]-xyz[i*3+2];
    float cx=xyz[k*3]-xyz[j*3], cy=xyz[k*3+1]-xyz[j*3+1], cz=xyz[k*3+2]-xyz[j*3+2];
    float cross_norm=norm3(ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx);
    float edge=fmaxf(1e-8f, fmaxf(norm3(ax,ay,az),fmaxf(norm3(bx,by,bz),norm3(cx,cy,cz))));
    float quality=(0.5f*cross_norm)/(edge*edge+1e-8f);
    float dx=(xyz[i*3]+xyz[j*3]+xyz[k*3])/3.0f-query[0];
    float dy=(xyz[i*3+1]+xyz[j*3+1]+xyz[k*3+1])/3.0f-query[1];
    float dz=(xyz[i*3+2]+xyz[j*3+2]+xyz[k*3+2])/3.0f-query[2];
    score=-norm3(dx,dy,dz)+0.05f*quality;
    atomicAdd(&count,1);
  }
  scores[lane]=score;
  __syncthreads();
  float previous=CUDART_INF_F;
  // One extra winner checks the top-k boundary. Close scores are delegated to
  // torch.topk, preserving its tie behavior instead of inventing a new order.
  for (int slot=0; slot<=keep; ++slot) {
    float best=scores[lane]; int pos=lane;
    for (int step=16; step>0; step>>=1) {
      float other=__shfl_down_sync(0xffffffff,best,step);
      int oi=__shfl_down_sync(0xffffffff,pos,step);
      if (other>best || (other==best && oi<pos)) { best=other; pos=oi; }
    }
    if ((lane&31)==0) { warp_score[lane>>5]=best; warp_index[lane>>5]=pos; }
    __syncthreads();
    if (lane==0) {
      int p=warp_index[0]; float s=warp_score[0];
      for (int w=1; w<8; ++w)
        if (warp_score[w]>s || (warp_score[w]==s && warp_index[w]<p)) { s=warp_score[w]; p=warp_index[w]; }
      winner=p;
      if (slot>0 && s>-1e8f && previous-s<=1e-7f*(1.0f+fabsf(s))) ambiguous=1;
      previous=s;
      if (slot<keep) {
        int out=(row*keep+slot)*3;
        output[out]=p<240 ? p/15 : 0;
        output[out+1]=p<240 ? candidate_j[p] : 0;
        output[out+2]=p<240 ? candidate_k[p] : 0;
        output_mask[row*keep+slot]=s>-1e8f;
      }
    }
    __syncthreads();
    if (lane==0) scores[winner]=-1e9f;
    __syncthreads();
  }
  if (lane==0) { uncertain[row]=ambiguous || count<keep; counts[row]=count; }
}

// One thread owns a small candidate point cache. No [Nq,M,3] tensor is needed.
__global__ void certified_kernel(
    const float* support, const float* query, const int64_t* rows,
    const int64_t* indices, const float* centers, int n, int m, float tolerance,
    int64_t* output, bool* certified) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if (row>=n) return;
  int64_t cache=rows[row];
  for (int s=0;s<16;++s) output[row*16+s]=-1;
  certified[row]=false;
  if (cache<0) return;
  float best[17]; int64_t ids[17];
  for (int s=0;s<17;++s) { best[s]=CUDART_INF_F; ids[s]=-1; }
  float qx=query[row*3], qy=query[row*3+1], qz=query[row*3+2];
  for (int candidate=0;candidate<m;++candidate) {
    int64_t id=indices[cache*(m+1)+candidate];
    if (id<0) continue;
    float dx=support[id*3]-qx, dy=support[id*3+1]-qy, dz=support[id*3+2]-qz;
    float d=(dx*dx+dy*dy)+dz*dz;
    int slot=16;
    if (d>=best[slot]) continue;
    while (slot>0 && d<best[slot-1]) { best[slot]=best[slot-1]; ids[slot]=ids[slot-1]; --slot; }
    best[slot]=d; ids[slot]=id;
  }
  if (ids[15]<0) return;
  bool ties=false;
  for (int s=0;s<16;++s) {
    output[row*16+s]=ids[s];
    if (best[s+1]-best[s] <= 8*FLT_EPSILON*fmaxf(best[s+1],1e-8f)) ties=true;
  }
  float cx=centers[cache*3], cy=centers[cache*3+1], cz=centers[cache*3+2];
  int64_t omitted=indices[cache*(m+1)+m];
  float d=omitted<0 ? CUDART_INF_F : norm3(support[omitted*3]-cx,support[omitted*3+1]-cy,support[omitted*3+2]-cz);
  float r=norm3(qx-cx,qy-cy,qz-cz), t=sqrtf(best[15]);
  float scale=fmaxf(fmaxf(fabsf(cx),fabsf(cy)),fabsf(cz))+r+t;
  float guard=tolerance+32*FLT_EPSILON*scale;
  certified[row]=!ties && t+r+guard<d;
}

std::vector<torch_tensor> primitive_select_cuda(
    torch_tensor xyz, torch_tensor query, torch_tensor uv,
    torch_tensor graph, torch_tensor graph_mask, torch_tensor point_mask,
    torch_tensor area_eps, int64_t keep, bool build_graph) {
  check(xyz,at::kFloat); check(query,at::kFloat); check(uv,at::kFloat);
  check(graph,at::kLong); check(graph_mask,at::kBool); check(point_mask,at::kBool); check(area_eps,at::kFloat);
  TORCH_CHECK(xyz.dim()==3 && xyz.size(1)==16 && xyz.size(2)==3 && graph.size(2)==6 && keep>0 && keep<=32,
              "primitive selection requires K=16, degree=6 and 1<=T<=32");
  c10::cuda::CUDAGuard guard(xyz.device());
  int n=xyz.size(0);
  auto output=at::empty({n,keep,3}, graph.options());
  auto mask=at::empty({n,keep},graph_mask.options());
  auto uncertain=at::empty({n},graph_mask.options());
  auto counts=at::empty({n},graph.options());
  if (n) select_kernel<<<n,256,0,at::cuda::getCurrentCUDAStream()>>>(
      xyz.data_ptr<float>(),query.data_ptr<float>(),uv.data_ptr<float>(),graph.data_ptr<int64_t>(),
      graph_mask.data_ptr<bool>(),point_mask.data_ptr<bool>(),area_eps.data_ptr<float>(),keep,build_graph,
      output.data_ptr<int64_t>(),mask.data_ptr<bool>(),uncertain.data_ptr<bool>(),counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output,mask,uncertain,counts};
}

std::vector<torch_tensor> certified_knn_cuda(
    torch_tensor support, torch_tensor query, torch_tensor rows,
    torch_tensor indices, torch_tensor centers, double tolerance) {
  check(support,at::kFloat); check(query,at::kFloat); check(rows,at::kLong);
  check(indices,at::kLong); check(centers,at::kFloat);
  TORCH_CHECK(indices.dim()==2 && indices.size(1)>=17, "point cache must contain M>=16 plus one bound point");
  c10::cuda::CUDAGuard guard(support.device());
  int n=query.size(0), m=indices.size(1)-1;
  auto output=at::empty({n,16},indices.options());
  auto accepted=at::empty({n},indices.options().dtype(at::kBool));
  if (n) certified_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
      support.data_ptr<float>(),query.data_ptr<float>(),rows.data_ptr<int64_t>(),indices.data_ptr<int64_t>(),
      centers.data_ptr<float>(),n,m,tolerance,output.data_ptr<int64_t>(),accepted.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output,accepted};
}

__device__ void insert_support(const float* support, int64_t id, const float* center,
                               int count, float* best, int64_t* ids) {
  float dx=support[id*3]-center[0], dy=support[id*3+1]-center[1], dz=support[id*3+2]-center[2];
  float d=(dx*dx+dy*dy)+dz*dz;
  int slot=count-1;
  if (d>=best[slot]) return;
  while (slot>0 && d<best[slot-1]) { best[slot]=best[slot-1]; ids[slot]=ids[slot-1]; --slot; }
  best[slot]=d; ids[slot]=id;
}

// Exact nearest supports at a voxel center. Expand disjoint voxel shells until
// the current farthest selected support is closer than every unvisited voxel.
// Sparse/out-of-range cases are explicitly returned for the original KNN.
__global__ void center_kernel(
    const float* support, const float* centers, const int64_t* codes, const int64_t* batches,
    const int64_t* sizes, const int64_t* bases, const int64_t* ranges, const int64_t* support_ids,
    const int64_t* offsets, int n, int count, float voxel_size, int64_t* output, bool* complete) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if (row>=n) return;
  int batch=batches[row];
  int64_t begin=batch==0 ? 0 : offsets[batch-1], end=offsets[batch];
  float best[97]; int64_t ids[97];
  for (int j=0;j<count;++j) { best[j]=CUDART_INF_F; ids[j]=-1; }
  const float* center=centers+row*3;
  bool done=false;
  if (end-begin<=count) {
    for (int64_t id=begin;id<end;++id) insert_support(support,id,center,count,best,ids);
    done=true;
  } else {
    int sx=sizes[batch*3], sy=sizes[batch*3+1], sz=sizes[batch*3+2];
    int64_t base=bases[batch], local=codes[row]-base;
    int cx=local/(sy*sz), cy=(local/sz)%sy, cz=local%sz;
    int64_t visited=0;
    float guard=1e-6f+32*FLT_EPSILON*fmaxf(fabsf(center[0]),fmaxf(fabsf(center[1]),fabsf(center[2])));
    for (int radius=0;radius<=12 && !done;++radius) {
      for (int dx=-radius;dx<=radius;++dx) {
        int x=cx+dx; if (x<0 || x>=sx) continue;
        for (int dy=-radius;dy<=radius;++dy) {
          int y=cy+dy; if (y<0 || y>=sy) continue;
          int step=(abs(dx)==radius || abs(dy)==radius) ? 1 : max(1,2*radius);
          for (int dz=-radius;dz<=radius;dz+=step) {
            int z=cz+dz; if (z<0 || z>=sz) continue;
            int64_t code=base+((int64_t)x*sy+y)*sz+z;
            int64_t first=ranges[code*2], last=ranges[code*2+1];
            visited+=last-first;
            for (int64_t p=first;p<last;++p)
              insert_support(support,support_ids[p],center,count,best,ids);
          }
        }
      }
      float bound=(radius+0.5f)*voxel_size-guard;
      done=visited==end-begin || (ids[count-1]>=0 && bound>0 && best[count-1]<bound*bound);
    }
  }
  for (int j=0;j<count;++j) output[row*count+j]=ids[j];
  complete[row]=done;
}

std::vector<torch_tensor> cache_centers_cuda(
    torch_tensor support, torch_tensor centers, torch_tensor codes, torch_tensor batches,
    torch_tensor sizes, torch_tensor bases, torch_tensor ranges, torch_tensor support_ids,
    torch_tensor offsets, int64_t count, double voxel_size) {
  check(support,at::kFloat); check(centers,at::kFloat); check(codes,at::kLong); check(batches,at::kLong);
  check(sizes,at::kLong); check(bases,at::kLong); check(ranges,at::kLong); check(support_ids,at::kLong); check(offsets,at::kLong);
  TORCH_CHECK(count>0 && count<=97 && voxel_size>0, "invalid center-cache dimensions");
  c10::cuda::CUDAGuard guard(support.device());
  int n=centers.size(0);
  auto output=at::empty({n,count},codes.options());
  auto complete=at::empty({n},codes.options().dtype(at::kBool));
  if (n) center_kernel<<<(n+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(
      support.data_ptr<float>(),centers.data_ptr<float>(),codes.data_ptr<int64_t>(),batches.data_ptr<int64_t>(),
      sizes.data_ptr<int64_t>(),bases.data_ptr<int64_t>(),ranges.data_ptr<int64_t>(),support_ids.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),n,count,voxel_size,output.data_ptr<int64_t>(),complete.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output,complete};
}
