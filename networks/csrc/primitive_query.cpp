#include <torch/extension.h>

std::vector<torch::Tensor> primitive_select_cuda(
    torch::Tensor xyz, torch::Tensor query, torch::Tensor uv,
    torch::Tensor graph, torch::Tensor graph_mask, torch::Tensor point_mask,
    torch::Tensor area_eps, int64_t keep, bool build_graph);
std::vector<torch::Tensor> certified_knn_cuda(
    torch::Tensor support, torch::Tensor query, torch::Tensor rows,
    torch::Tensor indices, torch::Tensor centers, double tolerance);
std::vector<torch::Tensor> cache_centers_cuda(
    torch::Tensor support, torch::Tensor centers, torch::Tensor codes, torch::Tensor batches,
    torch::Tensor sizes, torch::Tensor bases, torch::Tensor ranges, torch::Tensor support_ids,
    torch::Tensor offsets, int64_t count, double voxel_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select", &primitive_select_cuda);
  m.def("certified_knn", &certified_knn_cuda);
  m.def("cache_centers", &cache_centers_cuda);
}
