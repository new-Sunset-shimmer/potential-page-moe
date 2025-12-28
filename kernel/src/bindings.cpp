#include <pybind11/pybind11.h>
#include "moe_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libmoe, m) {
    py::class_<MoEManager>(m, "MoEManager")
        // 생성자 인자 추가 (num_layers 기본값 32)
        .def(py::init<int, size_t, int>(), py::arg("max_gpu_slots"), py::arg("slot_size"), py::arg("num_layers")=32)
        
        .def("register_expert", &MoEManager::register_expert, 
             py::arg("layer_id"), py::arg("expert_id"), py::arg("size_bytes"), py::arg("cpu_ptr"), py::arg("version") = 0)
        .def("request_expert", &MoEManager::request_expert,
             py::arg("layer_id"), py::arg("expert_id"), py::arg("version") = 0)
        
        .def("lock_expert", &MoEManager::lock_expert,
             py::arg("layer_id"), py::arg("expert_id"), py::arg("version") = 0)
        
        .def("unlock_expert", &MoEManager::unlock_expert,
             py::arg("layer_id"), py::arg("expert_id"), py::arg("version") = 0)
             
        .def("wait_for_transfer", &MoEManager::wait_for_transfer)
        .def("transfer_waits_for_compute", &MoEManager::transfer_waits_for_compute)
        .def("evict_one", &MoEManager::evict_one)
        .def("print_stats", &MoEManager::print_stats);
}