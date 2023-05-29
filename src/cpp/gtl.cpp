#include <torch/extension.h>
using namespace torch::indexing;

// torch::Tensor get_triangles(torch::Tensor us,torch::Tensor vs, torch::Tensor ego,torch::Tensor neighbors,int fanout,bool cuda){
//   auto neighbor_count = neighbors.sizes()[0];
//   auto edge_count = us.sizes()[0];
//   auto triangles_found = 0;
//   torch::TensorOptions options;
// 
//   if (cuda){
//     options = torch::TensorOptions()
//       .dtype(torch::kInt64)
//       .device(torch::kCUDA, 1 )
//       .requires_grad(false);
//   }
//   else {
//     options = torch::TensorOptions()
//       .dtype(torch::kInt64)
//       .device(torch::kCPU)
//       .requires_grad(false);
//   }
// 
//   auto triangle_nodes = torch::empty({0},options);
//     
//   // go over neighbors
//   for (int i = 0; i<neighbor_count && triangles_found < fanout; i++){
//     for (int j = i; j< neighbor_count && triangles_found < fanout ;j++) {
//       auto neighbor1 = neighbors.index({i}).item();
//       auto neighbor2 = neighbors.index({j}).item();
//       
//       // find edges between neighbor1 and neighbor2
//       for (int k=0;k<edge_count; k++){
//         int u = us.index({k}).item();
//         int v = vs.index({k}).item();
//         if(((u == neighbor1) && (v == neighbor2)) || ((v == neighbor1) && (u == neighbor2))) {
//             torch::Tensor new_nodes = torch::tensor({u,v},options);
//             triangle_nodes = torch::cat({triangle_nodes,new_nodes});
//             break;
//             }
//       }
//     }
//   }
// 
//   return triangle_nodes;
// }
// 
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("get_triangles", &get_triangles, "get_triangles");
// }
// 
