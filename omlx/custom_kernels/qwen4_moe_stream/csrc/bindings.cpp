// qwen4_moe_stream: stream qwen4_exp MoE routed-expert (switch_mlp) weights off
// the wired/phys budget by wrapping a page-aligned mmap'd artifact in MLX arrays.
//
// No custom Metal kernel: the wrapped arrays feed the STOCK mlx::core::gather_qmm
// path unchanged (validated bit-exact, 2-bit + 4-bit, decode + sorted-prefill).
// This module only (1) mmaps the streaming artifact, (2) hands out mx.array views
// over individual page-aligned tensor regions via allocator::make_buffer, and
// (3) tears the mapping down on unload. make_buffer wrappers MUST be released with
// allocator::release (mlx/allocator.h) -- a no-op deleter leaks one wrapper handle
// per wrapped tensor per load/unload cycle. release() frees only the lightweight
// MTL::Buffer wrapper object, never the mmap'd bytes (the mapping owns those).

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/allocator.h"
#include "mlx/array.h"

namespace omlx::qwen4_moe_stream {
namespace mx = mlx::core;

constexpr size_t kPage = 16384;  // artifact page size (must match repack.py PAGE)

static size_t align_up(size_t n, size_t a = kPage) { return (n + a - 1) / a * a; }

struct Mapping {
  void* base = nullptr;
  size_t size = 0;
  int fd = -1;
  long refcount = 0;    // outstanding wrapped arrays pointing into this mapping
  bool closed = false;  // close_artifact called; unmap deferred until refcount 0
};

static std::mutex g_mu;
static std::unordered_map<int, Mapping> g_maps;
static int g_next_id = 1;

// Tear down a mapping (must hold g_mu). Safe no-op if already gone.
static void unmap_locked(std::unordered_map<int, Mapping>::iterator it) {
  ::munmap(it->second.base, it->second.size);
  ::close(it->second.fd);
  g_maps.erase(it);
}

// Called by each wrapped array's deleter when it is destroyed: drop one ref and,
// if close_artifact already ran and this was the last live array, unmap now.
static void release_ref(int id) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_maps.find(id);
  if (it == g_maps.end()) return;
  if (it->second.refcount > 0) it->second.refcount--;
  if (it->second.closed && it->second.refcount == 0) unmap_locked(it);
}

// mmap the artifact read-only/shared; returns an opaque id used by wrap_tensor.
// No madvise: the default policy is left in place. Experts are gathered sparsely
// at serving time, but the initial warm-up/canary sweep faults in sequentially,
// and MADV_RANDOM would penalize that first pass -- default advice measured well
// (3.1-4.3 GB/s fault-in). Revisit under Phase-A measurement if needed.
int mmap_artifact(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("mmap_artifact: open failed: " + path);
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("mmap_artifact: fstat failed: " + path);
  }
  void* base = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                      MAP_SHARED, fd, 0);
  if (base == MAP_FAILED) {
    ::close(fd);
    throw std::runtime_error("mmap_artifact: mmap failed: " + path);
  }
  std::lock_guard<std::mutex> lk(g_mu);
  int id = g_next_id++;
  g_maps[id] = Mapping{base, static_cast<size_t>(st.st_size), fd, 0, false};
  return id;
}

// Mark the mapping closed. Unmap immediately only if no wrapped arrays still
// point into it; otherwise defer the munmap to the last array's deleter. This
// removes the close-while-alive hazard: a GPU touch of a wrapped array after
// close() can no longer fault on unmapped memory (Fable review 2, issue 4).
void close_artifact(int id) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_maps.find(id);
  if (it == g_maps.end()) return;
  it->second.closed = true;
  if (it->second.refcount == 0) unmap_locked(it);
}

// Total bytes currently mmap'd across all live artifacts. The Python side feeds
// this to the enforcer's register_external_wired_provider() as the worst-case
// externally-wired figure (all expert pages resident under sustained serving).
size_t mapped_bytes() {
  std::lock_guard<std::mutex> lk(g_mu);
  size_t total = 0;
  for (auto& kv : g_maps) total += kv.second.size;
  return total;
}

static mx::Dtype dtype_from_str(const std::string& s) {
  if (s == "uint32" || s == "U32") return mx::uint32;
  if (s == "bfloat16" || s == "BF16") return mx::bfloat16;
  if (s == "float16" || s == "F16") return mx::float16;
  if (s == "uint16" || s == "U16") return mx::uint16;
  if (s == "uint8" || s == "U8") return mx::uint8;
  throw std::runtime_error("wrap_tensor: unsupported dtype: " + s);
}

// Return an mx.array viewing [offset, offset+len) of artifact `id` as `shape`/
// `dtype`, backed by external mmap memory (no copy). offset MUST be page-aligned
// (guaranteed by repack.py) so the underlying pointer is page-aligned for
// newBufferWithBytesNoCopy. The MTLBuffer is created over a page-rounded length
// (each tensor's region is page-padded in the artifact, so the rounded span
// stays inside the file and never overlaps the next tensor); the array itself
// only ever addresses the logical `length` bytes described by shape*itemsize.
mx::array wrap_tensor(int id, size_t offset, size_t length,
                      std::vector<int> shape, const std::string& dtype) {
  void* base = nullptr;
  size_t buf_len = align_up(length);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_maps.find(id);
    if (it == g_maps.end())
      throw std::runtime_error("wrap_tensor: unknown artifact id");
    if (it->second.closed)
      throw std::runtime_error("wrap_tensor: artifact already closed");
    if (offset % kPage != 0)
      throw std::runtime_error("wrap_tensor: offset not page-aligned");
    if (offset + buf_len > it->second.size)
      throw std::runtime_error("wrap_tensor: region out of bounds");
    base = it->second.base;
    it->second.refcount++;  // one ref per wrapped array; deleter drops it
  }
  mx::allocator::Buffer buf =
      mx::allocator::make_buffer(static_cast<char*>(base) + offset, buf_len);
  mx::Shape sh(shape.begin(), shape.end());
  // Sanctioned single-step constructor: array(Buffer, Shape, Dtype, Deleter).
  // Marks the array available; no set_status needed. The deleter releases the
  // make_buffer wrapper AND drops this mapping's refcount (deferred-unmap safe).
  return mx::array(buf, std::move(sh), dtype_from_str(dtype),
                   [id](mx::allocator::Buffer b) {
                     mx::allocator::release(b);
                     release_ref(id);
                   });
}

}  // namespace omlx::qwen4_moe_stream

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "qwen4_exp MoE expert weight streaming (mmap'd page-aligned artifact)";

  // ABI canary (matches glm_moe_dsa/qwen35_prefill): if the nanobind ABI does
  // not match the mlx wheel, passing an mx.array here fails and fast.py disables
  // the native path instead of crashing.
  m.def(
      "abi_probe",
      [](const mlx::core::array& a) { return static_cast<int64_t>(a.size()); },
      "a"_a);

  m.def("mmap_artifact", &omlx::qwen4_moe_stream::mmap_artifact, "path"_a);
  m.def("close_artifact", &omlx::qwen4_moe_stream::close_artifact, "id"_a);
  m.def("mapped_bytes", &omlx::qwen4_moe_stream::mapped_bytes);
  m.def("wrap_tensor", &omlx::qwen4_moe_stream::wrap_tensor, "id"_a, "offset"_a,
        "length"_a, "shape"_a, "dtype"_a);
}
