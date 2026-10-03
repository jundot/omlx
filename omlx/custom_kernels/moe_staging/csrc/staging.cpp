// SPDX-License-Identifier: Apache-2.0
// Bounded expert read buffers; only owner threads allocate or hand off to MLX.
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <mlx/array.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace nb = nanobind;
namespace mx = mlx::core;

struct Slot {
  mx::allocator::Buffer buffer;
  void* address;
  size_t size;
  bool busy = false;
};
struct State {
  std::mutex mutex;
  std::vector<Slot> slots;
  size_t reads = 0, fallbacks = 0, high_water = 0, busy = 0;
  std::thread::id owner = std::this_thread::get_id();
  ~State() {
    for (auto& slot : slots) mx::allocator::free(slot.buffer);
  }
  void release(size_t index) {
    std::lock_guard<std::mutex> lock(mutex);
    slots[index].busy = false;
    --busy;
  }
  void check_owner() const {
    if (std::this_thread::get_id() != owner)
      throw std::runtime_error("MLX handoff must run on the pool owner");
  }
};
struct Lease : std::enable_shared_from_this<Lease> {
  std::shared_ptr<State> state;
  size_t index;
  Lease(std::shared_ptr<State> state, size_t index)
      : state(std::move(state)), index(index) {}
  ~Lease() { state->release(index); }
};
struct Pool {
  std::shared_ptr<State> state = std::make_shared<State>();
  explicit Pool(const std::vector<size_t>& sizes) {
    state->slots.reserve(sizes.size());
    for (auto size : sizes) {
      if (!size) throw std::invalid_argument("Zero-size staging buffer");
      auto buffer = mx::allocator::malloc(size);
      // Resolve Metal contents only on owner; workers use this raw address.
      state->slots.push_back({buffer, buffer.raw_ptr(), size});
    }
  }
  std::shared_ptr<Lease> read(int fd, int64_t offset, size_t size) {
    std::shared_ptr<Lease> lease;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      for (size_t i = 0; i < state->slots.size(); ++i) {
        auto& slot = state->slots[i];
        if (slot.size != size || slot.busy) continue;
        // Construct before marking busy so allocation failure changes no state.
        lease = std::make_shared<Lease>(state, i);
        slot.busy = true;
        state->high_water = std::max(state->high_water, ++state->busy);
        ++state->reads;
        break;
      }
      if (!lease) { ++state->fallbacks; return nullptr; }
    }
    // Match CPython os.pread: retry positive partial reads and EINTR unless
    // the Python signal handler raises; fail at EOF and propagate other errno.
    size_t got = 0;
    while (got < size) {
      auto address = static_cast<char*>(state->slots[lease->index].address);
      auto n = ::pread(fd, address + got, size - got, offset + got);
      if (n < 0) {
        auto error = errno;
        nb::gil_scoped_acquire gil;
        if (error == EINTR) {
          if (PyErr_CheckSignals() < 0) throw nb::python_error();
          continue;
        }
        errno = error;
        PyErr_SetFromErrno(PyExc_OSError);
        throw nb::python_error();
      }
      if (!n) {
        std::ostringstream message;
        message << "short read of " << size << " bytes at " << offset;
        nb::gil_scoped_acquire gil;
        PyErr_SetString(PyExc_OSError, message.str().c_str());
        throw nb::python_error();
      }
      got += n;
    }
    return lease;
  }
};

NB_MODULE(_ext, m) {
  nb::class_<Lease>(m, "Lease")
      .def("__len__", [](const Lease& l) { return l.state->slots[l.index].size; })
      .def_prop_ro("address", [](const Lease& l) {
        return reinterpret_cast<uintptr_t>(l.state->slots[l.index].address);
      });
  nb::class_<Pool>(m, "Pool")
      .def(nb::init<const std::vector<size_t>&>())
      .def("read", &Pool::read, nb::call_guard<nb::gil_scoped_release>())
      .def("stats", [](const Pool& p) {
        nb::dict out;
        std::lock_guard<std::mutex> lock(p.state->mutex);
        out["reads"] = p.state->reads; out["fallbacks"] = p.state->fallbacks;
        out["busy"] = p.state->busy; out["high_water"] = p.state->high_water;
        size_t logical = 0, allocated = 0;
        for (auto& slot : p.state->slots) {
          logical += slot.size;
          allocated += mx::allocator::allocator().size(slot.buffer);
        }
        out["logical_bytes"] = logical; out["allocated_bytes"] = allocated;
        return out;
      });
  m.def("to_array", [](Lease& borrowed,
                       const std::vector<int>& shape, mx::Dtype dtype) {
    // A Python->shared_ptr caster can install a GIL-taking py_deleter.
    // Recover the ORIGINAL C++ control block; completion must never need GIL.
    auto lease = borrowed.shared_from_this();
    lease->state->check_owner();
    size_t count = 1;
    for (auto dim : shape) {
      if (dim <= 0 || count > SIZE_MAX / static_cast<size_t>(dim))
        throw std::invalid_argument("Invalid staging shape");
      count *= dim;
    }
    if (count > SIZE_MAX / mx::size_of(dtype) ||
        count * mx::size_of(dtype) != lease->state->slots[lease->index].size)
      throw std::invalid_argument("Staging shape/dtype byte length mismatch");
    auto buffer = lease->state->slots[lease->index].buffer;
    // MLX's lazy graph and Metal completion handlers retain shared Data.
    // No reuse on Python-wrapper death, install return, or async_eval return.
    return mx::array(buffer, mx::Shape(shape.begin(), shape.end()), dtype,
        [lease = std::move(lease)](mx::allocator::Buffer) {});
  });
  m.def("address", [](mx::array& a) {
    if (!a.is_available())
      throw std::invalid_argument("Evaluate array before inspecting its address");
    return reinterpret_cast<uintptr_t>(a.data<void>());
  });
}
