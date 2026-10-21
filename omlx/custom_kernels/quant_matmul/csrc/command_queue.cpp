// SPDX-License-Identifier: Apache-2.0
#include "command_queue.h"

#include <mutex>
#include <stdexcept>

#include "mlx/backend/metal/device.h"

namespace omlx::quant_matmul {

namespace mx = mlx::core;
namespace metal = mlx::core::metal;

namespace {

// metal::CommandEncoder keeps its queue and pending command buffer private.
// An explicit instantiation may name a private member, so these tags hand out
// pointers to the fields; a renamed field fails to compile instead of
// misbehaving.
template <typename Tag, typename Tag::type Member>
struct EncoderField {
  friend typename Tag::type field(Tag) {
    return Member;
  }
};

struct Queue {
  using type = NS::SharedPtr<MTL::CommandQueue> metal::CommandEncoder::*;
  friend type field(Queue);
};

struct Buffer {
  using type = NS::SharedPtr<MTL::CommandBuffer> metal::CommandEncoder::*;
  friend type field(Buffer);
};

struct SetsAttached {
  using type = uint64_t metal::CommandEncoder::*;
  friend type field(SetsAttached);
};

template struct EncoderField<Queue, &metal::CommandEncoder::queue_>;
template struct EncoderField<Buffer, &metal::CommandEncoder::buffer_>;
template struct EncoderField<
    SetsAttached,
    &metal::CommandEncoder::sets_attached_>;

// A queue holds 64 uncompleted command buffers by default; every engine's
// in-flight buffers go to this one.
constexpr NS::UInteger kMaxCommandBuffers = 1024;

std::mutex queue_mutex;
NS::SharedPtr<MTL::CommandQueue> shared_queue;

} // namespace

uintptr_t share_command_queue(mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  if (stream.device != mx::Device::gpu) {
    return 0;
  }
  auto& encoder = metal::get_command_encoder(stream);
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::lock_guard<std::mutex> lock(queue_mutex);
  if (!shared_queue) {
    auto* device = metal::device(stream.device).mtl_device();
    shared_queue = NS::TransferPtr(device->newCommandQueue(kMaxCommandBuffers));
    if (!shared_queue) {
      throw std::runtime_error(
          "[share_command_queue] Failed to make a command queue.");
    }
  }
  auto& queue = encoder.*field(Queue{});
  if (queue.get() != shared_queue.get()) {
    // Retire the buffer the encoder holds on its own queue first.
    encoder.synchronize();
    queue = shared_queue;
    // The next commit attaches every residency set to the shared queue.
    encoder.*field(SetsAttached{}) = 0;
    encoder.*field(Buffer{}) =
        NS::RetainPtr(shared_queue->commandBufferWithUnretainedReferences());
  }
  return reinterpret_cast<uintptr_t>(queue.get());
}

} // namespace omlx::quant_matmul
