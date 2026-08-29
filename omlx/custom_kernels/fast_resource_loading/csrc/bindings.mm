#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mlx/array.h>
#include <mlx/backend/metal/device.h>
#include <mlx/device.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

struct LoadTicket {
  id<MTLBuffer> staging = nil;
  id<MTLBuffer> status = nil;
  id<MTLIOCommandBuffer> command_buffer = nil;
  uint64_t bytes = 0;
  uint64_t commands = 0;
  bool finished = false;
};

class FastResourceLoader {
 public:
  FastResourceLoader() {
    @autoreleasepool {
      auto& mlx_device = mlx::core::metal::device(mlx::core::Device::gpu);
      device_ = (__bridge id<MTLDevice>)(static_cast<void*>(mlx_device.mtl_device()));
      if (device_ == nil) {
        throw std::runtime_error("MLX did not provide a Metal device");
      }
      MTLIOCommandQueueDescriptor* descriptor =
          [[MTLIOCommandQueueDescriptor alloc] init];
      descriptor.type = MTLIOCommandQueueTypeConcurrent;
      descriptor.priority = MTLIOPriorityHigh;
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
      ticket->staging = [device_ newBufferWithLength:total_bytes
                                             options:MTLResourceStorageModeShared];
      ticket->status = [device_ newBufferWithLength:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
      if (ticket->staging == nil || ticket->status == nil) {
        throw std::runtime_error("Metal could not allocate FRL staging buffers");
      }
      *static_cast<uint32_t*>(ticket->status.contents) = 0;
      ticket->command_buffer = [io_queue_ commandBuffer];
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
      [ticket->command_buffer commit];
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
      [ticket->command_buffer waitUntilCompleted];
      const double io_seconds = clock_now() - io_started;
      const uint32_t status = *static_cast<uint32_t*>(ticket->status.contents);
      if (status != static_cast<uint32_t>(MTLIOStatusComplete)) {
        throw std::runtime_error(
            "Metal IO command buffer failed with status " + std::to_string(status));
      }

      const auto copy_started = clock_now();
      id<MTLCommandBuffer> command_buffer = [blit_queue_ commandBuffer];
      id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
      uint64_t copied_bytes = 0;
      for (nb::handle item : copies) {
        nb::tuple copy = nb::cast<nb::tuple>(item);
        if (copy.size() != 4) {
          throw std::invalid_argument(
              "Copies must be (array, destination_offset, source_offset, size)");
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
        [encoder copyFromBuffer:ticket->staging
                   sourceOffset:source
                       toBuffer:target
              destinationOffset:destination + array.offset()
                           size:size];
        copied_bytes += size;
      }
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      const double copy_seconds = clock_now() - copy_started;
      if (command_buffer.status == MTLCommandBufferStatusError) {
        throw std::runtime_error(
            "Metal FRL blit failed: " + error_string(command_buffer.error));
      }
      ticket->finished = true;
      nb::dict result;
      result["io_wait_seconds"] = io_seconds;
      result["copy_seconds"] = copy_seconds;
      result["bytes"] = ticket->bytes;
      result["commands"] = ticket->commands;
      result["copied_bytes"] = copied_bytes;
      return result;
    }
  }

 private:
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
  nb::class_<LoadTicket>(m, "LoadTicket")
      .def_prop_ro("bytes", [](const LoadTicket& ticket) { return ticket.bytes; })
      .def_prop_ro("commands", [](const LoadTicket& ticket) { return ticket.commands; });
  nb::class_<FastResourceLoader>(m, "FastResourceLoader")
      .def(nb::init<>())
      .def("begin", &FastResourceLoader::begin, "requests"_a)
      .def("finish", &FastResourceLoader::finish, "ticket"_a, "copies"_a);
}
