#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mlx/array.h>
#include <mlx/backend/metal/device.h>
#include <mlx/device.h>
#include <mlx/mlx.h>
#include <mlx/primitives.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

namespace mx = mlx::core;

struct AsyncRouteState {
  explicit AsyncRouteState(nb::handle callback_handle)
      : callback(callback_handle.ptr()) {
    Py_INCREF(callback);
  }

  ~AsyncRouteState() {
    if (callback != nullptr && Py_IsInitialized()) {
      nb::gil_scoped_acquire acquire;
      Py_DECREF(callback);
    }
  }

  PyObject* callback = nullptr;
  __strong id<MTLSharedEvent> event = nil;
  uint64_t value = 0;
};

static std::mutex g_async_route_event_mutex;
static __strong id<MTLSharedEvent> g_async_route_event = nil;
static __weak id<MTLDevice> g_async_route_event_device = nil;
static std::atomic<uint64_t> g_async_route_event_value{1};
static std::mutex g_async_route_error_mutex;
static std::string g_async_route_error;
static dispatch_queue_t g_async_route_accounting_queue = dispatch_queue_create(
    "com.omlx.expert-streaming.route-accounting", DISPATCH_QUEUE_SERIAL);

static std::string async_route_metal_source() {
  return R"METAL(
    #include <metal_stdlib>
    using namespace metal;
    kernel void async_route_remap(
        device const uint* source [[buffer(0)]],
        device const int* slot_map [[buffer(1)]],
        device const uchar* resident [[buffer(2)]],
        device int* destination [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
      const uint expert = source[index];
      destination[index] = resident[expert] ? slot_map[expert] : -1;
    }
    kernel void async_route_copy(
        device const int* source [[buffer(0)]],
        device int* destination [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      destination[index] = source[index];
    }
  )METAL";
}

static NSString* resource_conversion_metal_source() {
  return [NSString stringWithUTF8String:R"METAL(
    #include <metal_stdlib>
    using namespace metal;
    kernel void frl_bf16_to_fp16(
        device const ushort* source [[buffer(0)]],
        device half* destination [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      const uint fp32_bits = uint(source[index]) << 16;
      destination[index] = half(as_type<float>(fp32_bits));
    }
  )METAL"];
}

static void record_async_route_error(const std::string& message) {
  std::lock_guard<std::mutex> lock(g_async_route_error_mutex);
  if (g_async_route_error.empty()) g_async_route_error = message;
}

class AsyncRoutePrimitive : public mx::Primitive {
 public:
  AsyncRoutePrimitive(mx::Stream stream, std::shared_ptr<AsyncRouteState> state)
      : Primitive(stream), state_(std::move(state)) {}

  const char* name() const override { return "AsyncRoutePrimitive"; }

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("Async route resolution requires Metal");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 3 || outputs.size() != 2) {
      throw std::invalid_argument("Async route input/output mismatch");
    }
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    outputs[1].set_data(mx::allocator::malloc(outputs[1].nbytes()));
    const mx::array indices = inputs[0];
    const mx::array slot_map = inputs[1];
    const mx::array resident = inputs[2];
    mx::array entry = outputs[0];
    mx::array final = outputs[1];
    int32_t* entry_ptr = entry.data<int32_t>();
    int32_t* final_ptr = final.data<int32_t>();
    const uint32_t* index_ptr = indices.data<uint32_t>();
    const size_t count = indices.size();

    auto& device = mx::metal::device(stream().device);
    auto& encoder = mx::metal::get_command_encoder(stream());
    id<MTLDevice> metal_device = (__bridge id<MTLDevice>)(
        static_cast<void*>(device.mtl_device()));
    {
      std::lock_guard<std::mutex> lock(g_async_route_event_mutex);
      if (g_async_route_event == nil ||
          g_async_route_event_device != metal_device) {
        g_async_route_event = [metal_device newSharedEvent];
        g_async_route_event_device = metal_device;
        g_async_route_event_value.store(1, std::memory_order_relaxed);
      }
      state_->event = g_async_route_event;
      state_->value =
          g_async_route_event_value.fetch_add(1, std::memory_order_relaxed);
    }

    auto library = device.get_library(
        "omlx_async_route", [] { return async_route_metal_source(); });
    auto remap = device.get_kernel("async_route_remap", library);
    encoder.set_compute_pipeline_state(remap);
    encoder.set_input_array(indices, 0);
    encoder.set_input_array(slot_map, 1);
    encoder.set_input_array(resident, 2);
    encoder.set_output_array(outputs[0], 3);

    // Attach before dispatch: MLX may automatically commit a full command
    // buffer from dispatch_threads(). The callback must observe the buffer
    // which contains the upstream router and this route copy.
    id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)(
        static_cast<void*>(encoder.get_command_buffer()));
    auto state = state_;
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
      @autoreleasepool {
        // The remap kernel above read the pool's live slot map and resident
        // mask (updated in place by publish_route_maps, never re-allocated),
        // so its verdict reflects residency at execution time.  Ordering:
        // this layer's previous callback wrote the maps before signalling
        // the event its MoE waits on, and this command buffer was encoded
        // after that MoE, so it observes those writes.  A resident verdict
        // therefore needs no Python and no I/O: release the MoE at once and
        // account the hits asynchronously.  Anything else resolves through
        // the callback under the pool lock.
        bool all_resident = true;
        for (size_t index = 0; index < count; ++index) {
          all_resident = all_resident && entry_ptr[index] >= 0;
        }
        if (all_resident) {
          std::memcpy(final_ptr, entry_ptr, count * sizeof(int32_t));
          [state->event setSignaledValue:state->value];
          auto route_values = std::make_shared<std::vector<uint32_t>>(
              index_ptr, index_ptr + count);
          dispatch_async(g_async_route_accounting_queue, ^{
            @autoreleasepool {
              try {
                nb::gil_scoped_acquire acquire;
                nb::object callback = nb::borrow<nb::object>(state->callback);
                nb::list values;
                for (uint32_t value : *route_values) {
                  values.append(nb::int_(value));
                }
                callback(values);
              } catch (const std::exception& error) {
                record_async_route_error(error.what());
              }
            }
          });
          return;
        }
        try {
          nb::gil_scoped_acquire acquire;
          nb::object callback = nb::borrow<nb::object>(state->callback);
          nb::list values;
          for (size_t index = 0; index < count; ++index) {
            values.append(nb::int_(index_ptr[index]));
          }
          nb::object result = callback(values);
          nb::sequence slots = nb::cast<nb::sequence>(result);
          if (nb::len(slots) != count) {
            throw std::runtime_error(
                "Async route callback returned the wrong slot count");
          }
          for (size_t index = 0; index < count; ++index) {
            final_ptr[index] = nb::cast<int32_t>(slots[index]);
          }
        } catch (const std::exception& error) {
          std::fill(final_ptr, final_ptr + count, -1);
          record_async_route_error(error.what());
        }
        // CPU cache updates and writes to final_ptr happen-before the MoE
        // consumer encoded after the matching SharedEvent wait.
        [state->event setSignaledValue:state->value];
      }
    }];
    encoder.dispatch_threads(
        MTL::Size(count, 1, 1),
        MTL::Size(std::min<size_t>(count, 256), 1, 1));
  }

 private:
  std::shared_ptr<AsyncRouteState> state_;
};

class AsyncRouteWaitPrimitive : public mx::Primitive {
 public:
  AsyncRouteWaitPrimitive(
      mx::Stream stream, std::shared_ptr<AsyncRouteState> state)
      : Primitive(stream), state_(std::move(state)) {}

  const char* name() const override { return "AsyncRouteWaitPrimitive"; }

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("Async route wait requires Metal");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.end_encoding();
    id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)(
        static_cast<void*>(encoder.get_command_buffer()));
    [command_buffer encodeWaitForEvent:state_->event value:state_->value];
    auto library = device.get_library(
        "omlx_async_route", [] { return async_route_metal_source(); });
    auto copy = device.get_kernel("async_route_copy", library);
    encoder.set_compute_pipeline_state(copy);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_output_array(outputs[0], 1);
    encoder.dispatch_threads(
        MTL::Size(outputs[0].size(), 1, 1),
        MTL::Size(std::min<size_t>(outputs[0].size(), 256), 1, 1));
  }

 private:
  std::shared_ptr<AsyncRouteState> state_;
};

static std::pair<mx::array, mx::array> resolve_route_async(
    const mx::array& indices,
    const mx::array& slot_map,
    const mx::array& resident,
    nb::handle callback) {
  if (indices.dtype() != mx::uint32 && indices.dtype() != mx::int32) {
    throw std::invalid_argument("Async route indices must be uint32 or int32");
  }
  mx::Stream stream = mx::to_stream(mx::Device::gpu);
  mx::array contiguous = mx::contiguous(indices, false, stream);
  // The slot map and resident mask are the pool's live route maps: the
  // pool overwrites them in place (publish_route_maps) so the remap kernel
  // reads the current residency when it executes, not a snapshot taken
  // when the decode graph was built. Bind them directly when they are
  // already row-contiguous; a copy would freeze them again.
  mx::array contiguous_slots = slot_map.flags().row_contiguous
      ? slot_map
      : mx::contiguous(slot_map, false, stream);
  mx::array contiguous_resident = resident.flags().row_contiguous
      ? resident
      : mx::contiguous(resident, false, stream);
  if (contiguous_slots.dtype() != mx::int32 ||
      contiguous_resident.dtype() != mx::bool_) {
    throw std::invalid_argument(
        "Async route map must be int32 with a boolean resident mask");
  }
  auto state = std::make_shared<AsyncRouteState>(callback);
  auto raw = mx::array::make_arrays(
      {contiguous.shape(), contiguous.shape()},
      {mx::int32, mx::int32},
      std::make_shared<AsyncRoutePrimitive>(stream, state),
      {contiguous, contiguous_slots, contiguous_resident});
  mx::array gated(
      raw[1].shape(), mx::int32,
      std::make_shared<AsyncRouteWaitPrimitive>(stream, state),
      {raw[1]});
  return {raw[0], gated};
}

static void publish_route_maps(
    const mx::array& slot_map,
    const mx::array& resident,
    nb::bytes slots,
    nb::bytes mask) {
  // Overwrite the pool's route maps in place. Both arrays live in shared
  // Metal storage, so the CPU write is visible to the next remap kernel
  // that binds them; see AsyncRoutePrimitive for the ordering argument.
  if (slot_map.dtype() != mx::int32 || resident.dtype() != mx::bool_) {
    throw std::invalid_argument(
        "Route maps must be an int32 slot map and a boolean resident mask");
  }
  if (!slot_map.flags().row_contiguous || !resident.flags().row_contiguous) {
    throw std::invalid_argument("Route maps must be row-contiguous");
  }
  if (slot_map.status() == mx::array::unscheduled ||
      resident.status() == mx::array::unscheduled) {
    throw std::invalid_argument("Route maps must be evaluated before publishing");
  }
  if (slots.size() != slot_map.nbytes() || mask.size() != resident.nbytes()) {
    throw std::invalid_argument("Route map payload size mismatch");
  }
  std::memcpy(
      const_cast<int32_t*>(slot_map.data<int32_t>()), slots.c_str(), slots.size());
  std::memcpy(
      const_cast<bool*>(resident.data<bool>()), mask.c_str(), mask.size());
}

static void check_async_route_error() {
  std::lock_guard<std::mutex> lock(g_async_route_error_mutex);
  if (!g_async_route_error.empty()) {
    std::string error = std::move(g_async_route_error);
    g_async_route_error.clear();
    throw std::runtime_error(error);
  }
}

static void eval_with_gil_released(const nb::args& values) {
  std::vector<mx::array> arrays;
  arrays.reserve(values.size());
  for (nb::handle value : values) {
    arrays.push_back(nb::cast<const mx::array&>(value));
  }
  // A native-demand miss is resolved by a Metal completion handler which
  // briefly enters Python.  The ordinary Python mx.eval()/tolist() barriers
  // can retain the GIL while waiting for that handler, creating a circular
  // wait.  Convert the Python arguments first, then materialize without the
  // GIL so the exact-demand callback can publish the missing rows.
  nb::gil_scoped_release release;
  mx::eval(std::move(arrays));
}

static void synchronize_with_gil_released(nb::object stream) {
  // mx.synchronize() holds the GIL while it blocks in waitUntilCompleted.
  // If the pending command buffer carries an exact-demand miss, its Metal
  // completion handler needs the GIL to publish the missing rows, and the
  // process deadlocks.  Resolve the stream while the GIL is held, then wait
  // without it.  See eval_with_gil_released for the eval() counterpart.
  // Accept everything mx.synchronize() accepts: a Stream, a
  // ThreadLocalStream (mlx-lm/mlx-vlm generation streams), a Device, or None.
  std::optional<mx::Stream> target;
  std::optional<mx::ThreadLocalStream> thread_local_target;
  if (nb::isinstance<mx::ThreadLocalStream>(stream)) {
    thread_local_target = nb::cast<mx::ThreadLocalStream>(stream);
  } else if (nb::isinstance<mx::Stream>(stream)) {
    target = nb::cast<mx::Stream>(stream);
  } else if (nb::isinstance<mx::Device>(stream)) {
    target = mx::default_stream(nb::cast<mx::Device>(stream));
  } else if (nb::isinstance<mx::Device::DeviceType>(stream)) {
    target = mx::default_stream(
        mx::Device(nb::cast<mx::Device::DeviceType>(stream)));
  } else if (!stream.is_none()) {
    throw nb::type_error(
        "synchronize_with_gil_released expects a Stream, ThreadLocalStream, "
        "Device, or None");
  }
  nb::gil_scoped_release release;
  if (thread_local_target) {
    mx::synchronize(*thread_local_target);
  } else if (target) {
    mx::synchronize(*target);
  } else {
    mx::synchronize();
  }
}

// Wait for a Metal IO command buffer without holding the GIL.  Loads can
// take seconds for multi-GiB tickets, and holding the GIL for that long
// freezes every other Python thread (the asyncio server included).  A
// bounded wait also turns a stalled IO queue into a load error with the
// ticket's accounting attached instead of a process that never returns.
static double io_wait_clock_now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct LoadTicket;
static void wait_io_ticket(
    const std::shared_ptr<LoadTicket>& ticket,
    double timeout_seconds,
    const char* what);

static void wait_io_command_buffer(
    id<MTLIOCommandBuffer> command_buffer,
    double timeout_seconds,
    const char* what) {
  nb::gil_scoped_release release;
  const double started = io_wait_clock_now();
  while (true) {
    const MTLIOStatus status = command_buffer.status;
    if (status == MTLIOStatusComplete ||
        status == MTLIOStatusError ||
        status == MTLIOStatusCancelled) {
      return;
    }
    if (io_wait_clock_now() - started > timeout_seconds) {
      throw std::runtime_error(
          std::string(what) + " did not complete within " +
          std::to_string(static_cast<long>(timeout_seconds)) + " s");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
}

struct LoadTicket {
  id<MTLBuffer> staging = nil;
  id<MTLBuffer> status = nil;
  id<MTLIOCommandBuffer> command_buffer = nil;
  uint64_t bytes = 0;
  uint64_t commands = 0;
  double queue_acquire_seconds = 0.0;
  double submitted_at = 0.0;
  std::atomic<double> completed_at{0.0};
  uint64_t inflight_at_submit = 0;
  std::atomic<bool> completion_recorded{false};
  bool direct = false;
  bool finished = false;
  dispatch_semaphore_t completed = nil;
};

static void wait_io_ticket(
    const std::shared_ptr<LoadTicket>& ticket,
    double timeout_seconds,
    const char* what) {
  // The completion handler signals the ticket's semaphore, so the wait is
  // event-driven and bounded.  A 200 us polling sleep here cost about a
  // millisecond per wait under macOS timer coalescing, which the inference
  // thread paid on every scratch group and every decode miss.
  static const bool force_polling = [] {
    const char* value = std::getenv("OMLX_FRL_POLL_WAIT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  if (ticket->completed == nil || force_polling) {
    wait_io_command_buffer(ticket->command_buffer, timeout_seconds, what);
    return;
  }
  long timed_out = 0;
  {
    nb::gil_scoped_release release;
    timed_out = dispatch_semaphore_wait(
        ticket->completed,
        dispatch_time(DISPATCH_TIME_NOW,
                      static_cast<int64_t>(timeout_seconds * NSEC_PER_SEC)));
  }
  if (timed_out != 0) {
    throw std::runtime_error(
        std::string(what) + " did not complete within " +
        std::to_string(static_cast<long>(timeout_seconds)) + " s");
  }
}

struct IOQueueStats {
  std::atomic<uint64_t> inflight{0};
  std::atomic<uint64_t> max_inflight{0};
};

class FastResourceLoader {
 public:
  explicit FastResourceLoader(const std::string& priority = "high") {
    @autoreleasepool {
      auto& mlx_device = mlx::core::metal::device(mlx::core::Device::gpu);
      device_ = (__bridge id<MTLDevice>)(static_cast<void*>(mlx_device.mtl_device()));
      if (device_ == nil) {
        throw std::runtime_error("MLX did not provide a Metal device");
      }
      MTLIOCommandQueueDescriptor* descriptor =
          [[MTLIOCommandQueueDescriptor alloc] init];
      descriptor.type = MTLIOCommandQueueTypeConcurrent;
      if (priority == "high") {
        descriptor.priority = MTLIOPriorityHigh;
      } else if (priority == "normal") {
        descriptor.priority = MTLIOPriorityNormal;
      } else if (priority == "low") {
        descriptor.priority = MTLIOPriorityLow;
      } else {
        throw std::invalid_argument(
            "Metal I/O priority must be high, normal, or low");
      }
      descriptor.maxCommandBufferCount = 2;
      descriptor.maxCommandsInFlight = 0;
      NSError* error = nil;
      io_queue_ = [device_ newIOCommandQueueWithDescriptor:descriptor error:&error];
      if (io_queue_ == nil) {
        throw std::runtime_error(
            "Could not create Metal IO queue: " + error_string(error));
      }
      blit_queue_ = [device_ newCommandQueue];
      if (blit_queue_ == nil) {
        throw std::runtime_error("Could not create Metal blit queue");
      }
      NSError* conversion_error = nil;
      id<MTLLibrary> conversion_library =
          [device_ newLibraryWithSource:resource_conversion_metal_source()
                                options:nil
                                  error:&conversion_error];
      if (conversion_library == nil) {
        throw std::runtime_error(
            "Could not compile FRL conversion library: " +
            error_string(conversion_error));
      }
      id<MTLFunction> conversion_function =
          [conversion_library newFunctionWithName:@"frl_bf16_to_fp16"];
      bf16_to_fp16_pipeline_ =
          [device_ newComputePipelineStateWithFunction:conversion_function
                                                 error:&conversion_error];
      if (bf16_to_fp16_pipeline_ == nil) {
        throw std::runtime_error(
            "Could not build FRL BF16-to-FP16 pipeline: " +
            error_string(conversion_error));
      }
    }
  }

  std::shared_ptr<LoadTicket> begin(const nb::list& requests) {
    @autoreleasepool {
      if (requests.size() == 0) {
        throw std::invalid_argument("Fast resource load requires requests");
      }
      uint64_t total_bytes = 0;
      for (nb::handle item : requests) {
        nb::tuple request = nb::cast<nb::tuple>(item);
        if (request.size() != 4) {
          throw std::invalid_argument(
              "Load requests must be (path, source_offset, size, destination_offset)");
        }
        const uint64_t size = nb::cast<uint64_t>(request[2]);
        const uint64_t destination = nb::cast<uint64_t>(request[3]);
        total_bytes = std::max(total_bytes, destination + size);
      }

      auto ticket = std::make_shared<LoadTicket>();
      ticket->staging = acquire_staging(total_bytes);
      ticket->status = [device_ newBufferWithLength:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
      if (ticket->staging == nil || ticket->status == nil) {
        throw std::runtime_error("Metal could not allocate FRL staging buffers");
      }
      *static_cast<uint32_t*>(ticket->status.contents) = 0;
      const auto queue_acquire_started = clock_now();
      ticket->command_buffer = [io_queue_ commandBuffer];
      ticket->queue_acquire_seconds = clock_now() - queue_acquire_started;
      if (ticket->command_buffer == nil) {
        throw std::runtime_error("Metal IO queue did not create a command buffer");
      }

      for (nb::handle item : requests) {
        nb::tuple request = nb::cast<nb::tuple>(item);
        const std::string path = nb::cast<std::string>(request[0]);
        const uint64_t source = nb::cast<uint64_t>(request[1]);
        const uint64_t size = nb::cast<uint64_t>(request[2]);
        const uint64_t destination = nb::cast<uint64_t>(request[3]);
        id<MTLIOFileHandle> handle = file_handle(path);
        [ticket->command_buffer loadBuffer:ticket->staging
                                    offset:destination
                                      size:size
                              sourceHandle:handle
                        sourceHandleOffset:source];
        ticket->bytes += size;
        ticket->commands += 1;
      }
      [ticket->command_buffer copyStatusToBuffer:ticket->status offset:0];
      commit_ticket(ticket);
      return ticket;
    }
  }

  std::shared_ptr<LoadTicket> begin_direct(const nb::list& requests) {
    @autoreleasepool {
      if (requests.size() == 0) {
        throw std::invalid_argument("Direct resource load requires requests");
      }
      auto ticket = std::make_shared<LoadTicket>();
      ticket->status = [device_ newBufferWithLength:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
      if (ticket->status == nil) {
        throw std::runtime_error("Metal could not allocate FRL status buffer");
      }
      *static_cast<uint32_t*>(ticket->status.contents) = 0;
      const auto queue_acquire_started = clock_now();
      ticket->command_buffer = [io_queue_ commandBuffer];
      ticket->queue_acquire_seconds = clock_now() - queue_acquire_started;
      if (ticket->command_buffer == nil) {
        throw std::runtime_error("Metal IO queue did not create a command buffer");
      }
      ticket->direct = true;
      for (nb::handle item : requests) {
        nb::tuple request = nb::cast<nb::tuple>(item);
        if (request.size() != 5) {
          throw std::invalid_argument(
              "Direct requests must be (path, source_offset, size, array, destination_offset)");
        }
        const std::string path = nb::cast<std::string>(request[0]);
        const uint64_t source = nb::cast<uint64_t>(request[1]);
        const uint64_t size = nb::cast<uint64_t>(request[2]);
        const mlx::core::array& array =
            nb::cast<const mlx::core::array&>(request[3]);
        const uint64_t destination = nb::cast<uint64_t>(request[4]);
        if (array.status() == mlx::core::array::unscheduled) {
          throw std::invalid_argument(
              "Direct FRL destination MLX array is not evaluated");
        }
        if (array.offset() < 0 ||
            static_cast<uint64_t>(array.offset()) + destination + size >
                array.buffer_size()) {
          throw std::out_of_range("Direct FRL destination exceeds MLX buffer");
        }
        id<MTLBuffer> target =
            (__bridge id<MTLBuffer>)(array.buffer().ptr());
        if (target == nil) {
          throw std::runtime_error("Direct FRL destination has no Metal buffer");
        }
        id<MTLIOFileHandle> handle = file_handle(path);
        [ticket->command_buffer loadBuffer:target
                                    offset:destination + array.offset()
                                      size:size
                              sourceHandle:handle
                        sourceHandleOffset:source];
        ticket->bytes += size;
        ticket->commands += 1;
      }
      [ticket->command_buffer copyStatusToBuffer:ticket->status offset:0];
      commit_ticket(ticket);
      return ticket;
    }
  }

  nb::dict finish(
      const std::shared_ptr<LoadTicket>& ticket,
      const nb::list& copies) {
    @autoreleasepool {
      if (!ticket || ticket->command_buffer == nil) {
        throw std::invalid_argument("Invalid fast resource load ticket");
      }
      if (ticket->finished) {
        throw std::invalid_argument("Fast resource load ticket is already finished");
      }
      const auto io_started = clock_now();
      wait_io_ticket(ticket, 120.0, "Metal IO command buffer");
      const double io_seconds = clock_now() - io_started;
      record_completion(ticket, queue_stats_);
      const uint32_t status = *static_cast<uint32_t*>(ticket->status.contents);
      if (status != static_cast<uint32_t>(MTLIOStatusComplete)) {
        throw std::runtime_error(
            "Metal IO command buffer failed with status " + std::to_string(status));
      }

      if (ticket->direct) {
        if (copies.size() != 0) {
          throw std::invalid_argument(
              "Direct FRL ticket does not accept staging copies");
        }
        ticket->finished = true;
        nb::dict result;
        result["io_wait_seconds"] = io_seconds;
        result["copy_seconds"] = 0.0;
        result["bytes"] = ticket->bytes;
        result["commands"] = ticket->commands;
        result["copied_bytes"] = ticket->bytes;
        result["direct"] = true;
        add_queue_stats(result, ticket);
        return result;
      }

      const auto copy_started = clock_now();
      id<MTLCommandBuffer> command_buffer = [blit_queue_ commandBuffer];
      struct CopyPlan {
        __strong id<MTLBuffer> target;
        uint64_t destination;
        uint64_t source;
        uint64_t size;
        std::string conversion;
      };
      std::vector<CopyPlan> plans;
      plans.reserve(copies.size());
      uint64_t copied_bytes = 0;
      for (nb::handle item : copies) {
        nb::tuple copy = nb::cast<nb::tuple>(item);
        if (copy.size() != 4 && copy.size() != 5) {
          throw std::invalid_argument(
              "Copies must be (array, destination_offset, source_offset, size[, conversion])");
        }
        const mlx::core::array& array = nb::cast<const mlx::core::array&>(copy[0]);
        if (array.status() == mlx::core::array::unscheduled) {
          throw std::invalid_argument("FRL destination MLX array is not evaluated");
        }
        const uint64_t destination = nb::cast<uint64_t>(copy[1]);
        const uint64_t source = nb::cast<uint64_t>(copy[2]);
        const uint64_t size = nb::cast<uint64_t>(copy[3]);
        if (source + size > ticket->staging.length) {
          throw std::out_of_range("FRL source copy exceeds staging buffer");
        }
        if (array.offset() < 0 ||
            static_cast<uint64_t>(array.offset()) + destination + size >
                array.buffer_size()) {
          throw std::out_of_range("FRL destination copy exceeds MLX buffer");
        }
        id<MTLBuffer> target = (__bridge id<MTLBuffer>)(array.buffer().ptr());
        if (target == nil) {
          throw std::runtime_error("MLX destination has no Metal buffer");
        }
        const std::string conversion =
            copy.size() == 5 ? nb::cast<std::string>(copy[4]) : "none";
        if (conversion != "none" && conversion != "bf16_to_fp16") {
          throw std::invalid_argument(
              "Unsupported FRL conversion: " + conversion);
        }
        if (conversion == "bf16_to_fp16" && size % sizeof(uint16_t) != 0) {
          throw std::invalid_argument(
              "BF16-to-FP16 FRL conversion requires an even byte count");
        }
        plans.push_back(
            {target,
             destination + static_cast<uint64_t>(array.offset()),
             source,
             size,
             conversion});
        copied_bytes += size;
      }

      const bool has_raw_copies = std::any_of(
          plans.begin(), plans.end(), [](const CopyPlan& plan) {
            return plan.conversion == "none";
          });
      if (has_raw_copies) {
        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        for (const auto& plan : plans) {
          if (plan.conversion != "none") continue;
          [encoder copyFromBuffer:ticket->staging
                     sourceOffset:plan.source
                         toBuffer:plan.target
                destinationOffset:plan.destination
                             size:plan.size];
        }
        [encoder endEncoding];
      }

      const bool has_conversions = std::any_of(
          plans.begin(), plans.end(), [](const CopyPlan& plan) {
            return plan.conversion == "bf16_to_fp16";
          });
      if (has_conversions) {
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:bf16_to_fp16_pipeline_];
        const NSUInteger max_threads =
            bf16_to_fp16_pipeline_.maxTotalThreadsPerThreadgroup;
        for (const auto& plan : plans) {
          if (plan.conversion != "bf16_to_fp16") continue;
          const NSUInteger elements = plan.size / sizeof(uint16_t);
          [encoder setBuffer:ticket->staging offset:plan.source atIndex:0];
          [encoder setBuffer:plan.target offset:plan.destination atIndex:1];
          [encoder dispatchThreads:MTLSizeMake(elements, 1, 1)
               threadsPerThreadgroup:
                   MTLSizeMake(std::min<NSUInteger>(elements, max_threads), 1, 1)];
        }
        [encoder endEncoding];
      }
      [command_buffer commit];
      {
        nb::gil_scoped_release release;
        [command_buffer waitUntilCompleted];
      }
      const double copy_seconds = clock_now() - copy_started;
      if (command_buffer.status == MTLCommandBufferStatusError) {
        throw std::runtime_error(
            "Metal FRL blit failed: " + error_string(command_buffer.error));
      }
      release_staging(ticket->staging);
      ticket->staging = nil;
      ticket->finished = true;
      nb::dict result;
      result["io_wait_seconds"] = io_seconds;
      result["copy_seconds"] = copy_seconds;
      result["bytes"] = ticket->bytes;
      result["commands"] = ticket->commands;
      result["copied_bytes"] = copied_bytes;
      result["direct"] = false;
      add_queue_stats(result, ticket);
      return result;
    }
  }

 private:
  // Staged loads reuse a small pool of shared staging buffers.  Allocating a
  // fresh 64 MB buffer for every scratch group cost the inference thread
  // milliseconds per group during streamed prefill.
  static constexpr size_t kStagingPoolLimit = 8;
  static constexpr uint64_t kStagingGranularity = 4ull << 20;
  std::mutex staging_mutex_;
  std::vector<id<MTLBuffer>> staging_pool_;

  id<MTLBuffer> acquire_staging(uint64_t total_bytes) {
    const uint64_t rounded =
        (total_bytes + kStagingGranularity - 1) / kStagingGranularity *
        kStagingGranularity;
    {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      size_t best = staging_pool_.size();
      for (size_t index = 0; index < staging_pool_.size(); ++index) {
        const uint64_t length = [staging_pool_[index] length];
        if (length >= total_bytes &&
            (best == staging_pool_.size() ||
             length < [staging_pool_[best] length])) {
          best = index;
        }
      }
      if (best != staging_pool_.size()) {
        id<MTLBuffer> buffer = staging_pool_[best];
        staging_pool_.erase(staging_pool_.begin() + best);
        return buffer;
      }
    }
    return [device_ newBufferWithLength:rounded
                                options:MTLResourceStorageModeShared];
  }

  void release_staging(id<MTLBuffer> buffer) {
    if (buffer == nil) return;
    std::lock_guard<std::mutex> lock(staging_mutex_);
    if (staging_pool_.size() >= kStagingPoolLimit) {
      // Drop the smallest pooled buffer so the pool keeps the most useful sizes.
      size_t smallest = 0;
      for (size_t index = 1; index < staging_pool_.size(); ++index) {
        if ([staging_pool_[index] length] < [staging_pool_[smallest] length]) {
          smallest = index;
        }
      }
      if ([staging_pool_[smallest] length] >= [buffer length]) return;
      staging_pool_.erase(staging_pool_.begin() + smallest);
    }
    staging_pool_.push_back(buffer);
  }

  static void record_completion(
      const std::shared_ptr<LoadTicket>& ticket,
      const std::shared_ptr<IOQueueStats>& queue_stats) {
    bool expected = false;
    if (!ticket->completion_recorded.compare_exchange_strong(expected, true)) {
      return;
    }
    ticket->completed_at.store(clock_now(), std::memory_order_release);
    queue_stats->inflight.fetch_sub(1, std::memory_order_relaxed);
  }

  void commit_ticket(const std::shared_ptr<LoadTicket>& ticket) {
    ticket->submitted_at = clock_now();
    ticket->inflight_at_submit =
        queue_stats_->inflight.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t observed = queue_stats_->max_inflight.load(std::memory_order_relaxed);
    while (observed < ticket->inflight_at_submit &&
           !queue_stats_->max_inflight.compare_exchange_weak(
               observed,
               ticket->inflight_at_submit,
               std::memory_order_relaxed)) {
    }
    std::weak_ptr<LoadTicket> weak_ticket(ticket);
    auto queue_stats = queue_stats_;
    ticket->completed = dispatch_semaphore_create(0);
    dispatch_semaphore_t completed_semaphore = ticket->completed;
    [ticket->command_buffer addCompletedHandler:^(id<MTLIOCommandBuffer>) {
      if (auto completed = weak_ticket.lock()) {
        record_completion(completed, queue_stats);
      }
      dispatch_semaphore_signal(completed_semaphore);
    }];
    [ticket->command_buffer commit];
  }

  void add_queue_stats(
      nb::dict& result, const std::shared_ptr<LoadTicket>& ticket) const {
    const double completed_at =
        ticket->completed_at.load(std::memory_order_acquire);
    result["queue_acquire_seconds"] = ticket->queue_acquire_seconds;
    result["submission_to_completion_seconds"] =
        std::max(0.0, completed_at - ticket->submitted_at);
    result["inflight_at_submit"] = ticket->inflight_at_submit;
    result["overlapped_submission"] = ticket->inflight_at_submit > 1;
    result["queue_max_inflight"] =
        queue_stats_->max_inflight.load(std::memory_order_relaxed);
  }

  static double clock_now() {
    return static_cast<double>(mach_absolute_time()) * timebase_seconds();
  }

  static double timebase_seconds() {
    static double value = [] {
      mach_timebase_info_data_t info{};
      mach_timebase_info(&info);
      return static_cast<double>(info.numer) /
          static_cast<double>(info.denom) / 1e9;
    }();
    return value;
  }

  static std::string error_string(NSError* error) {
    return error == nil ? "unknown error" : std::string(error.localizedDescription.UTF8String);
  }

  id<MTLIOFileHandle> file_handle(const std::string& path) {
    std::lock_guard<std::mutex> lock(handles_mutex_);
    auto found = handles_.find(path);
    if (found != handles_.end()) {
      return found->second;
    }
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    NSError* error = nil;
    id<MTLIOFileHandle> handle = nil;
    if ([device_ respondsToSelector:@selector(newIOFileHandleWithURL:error:)]) {
      handle = [device_ newIOFileHandleWithURL:url error:&error];
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      handle = [device_ newIOHandleWithURL:url error:&error];
#pragma clang diagnostic pop
    }
    if (handle == nil) {
      throw std::runtime_error(
          "Could not open Metal IO file handle for " + path + ": " +
          error_string(error));
    }
    handles_.emplace(path, handle);
    return handle;
  }

  id<MTLDevice> device_ = nil;
  id<MTLIOCommandQueue> io_queue_ = nil;
  id<MTLCommandQueue> blit_queue_ = nil;
  id<MTLComputePipelineState> bf16_to_fp16_pipeline_ = nil;
  std::shared_ptr<IOQueueStats> queue_stats_ = std::make_shared<IOQueueStats>();
  std::mutex handles_mutex_;
  std::unordered_map<std::string, id<MTLIOFileHandle>> handles_;
};

}  // namespace



NB_MODULE(_ext, m) {
  m.doc() = "Metal Fast Resource Loading for oMLX expert banks";
  m.def(
      "abi_probe",
      [](const mlx::core::array& array) {
        return static_cast<int64_t>(array.size());
      },
      "array"_a);
  m.def(
      "resolve_route_async",
      &resolve_route_async,
      "indices"_a,
      "slot_map"_a,
      "resident"_a,
      "callback"_a);
  m.def("check_async_route_error", &check_async_route_error);
  m.def("eval_with_gil_released", &eval_with_gil_released);
  m.def(
      "publish_route_maps",
      &publish_route_maps,
      nb::arg("slot_map"),
      nb::arg("resident"),
      nb::arg("slots"),
      nb::arg("mask"));
  m.def(
      "synchronize_with_gil_released",
      &synchronize_with_gil_released,
      nb::arg("stream").none() = nb::none());
  nb::class_<LoadTicket>(m, "LoadTicket")
      .def_prop_ro("bytes", [](const LoadTicket& ticket) { return ticket.bytes; })
      .def_prop_ro("commands", [](const LoadTicket& ticket) { return ticket.commands; });
  nb::class_<FastResourceLoader>(m, "FastResourceLoader")
      .def(nb::init<const std::string&>(), "priority"_a = "high")
      .def("begin", &FastResourceLoader::begin, "requests"_a)
      .def("begin_direct", &FastResourceLoader::begin_direct, "requests"_a)
      .def("finish", &FastResourceLoader::finish, "ticket"_a, "copies"_a);
}
