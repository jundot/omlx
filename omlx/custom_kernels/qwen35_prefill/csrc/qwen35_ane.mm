#include "qwen35_ane.h"
#include "qwen35_prefill.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/message.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::qwen35_prefill_kernels {
namespace {

using namespace mlx::core;

enum ProfileMetric : size_t {
  kOperations,
  kPackNs,
  kAneRegionNs,
  kAne0EvalNs,
  kAne1EvalNs,
  kAne0LaunchNs,
  kAne1LaunchNs,
  kGpuQmmNs,
  kAneLast,
  kGpuLast,
  kGapBeforeNs,
  kProfileMetricCount,
};

using ProfileCategory =
    std::array<std::atomic<uint64_t>, kProfileMetricCount>;
ProfileCategory g_ane_profile[2]{};
std::atomic<uint64_t> g_previous_ane_done_ns{0};
std::atomic<int> g_ane_profile_override{-1};

bool ane_profile_enabled() {
  const int override =
      g_ane_profile_override.load(std::memory_order_relaxed);
  if (override >= 0) {
    return override != 0;
  }
  static const bool enabled = [] {
    const char *value = std::getenv("OMLX_ANE_PROFILE");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

uint64_t profile_now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void profile_add(int category, ProfileMetric metric, uint64_t value) {
  g_ane_profile[category][metric].fetch_add(value,
                                             std::memory_order_relaxed);
}

std::string binary_dir() {
  static const std::string path = [] {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(&binary_dir), &info)) {
      throw std::runtime_error("Unable to locate the qwen35 native extension.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return path;
}

void load_ane_framework() {
  static const bool loaded = [] {
    return dlopen(
               "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
               "AppleNeuralEngine",
               RTLD_NOW | RTLD_LOCAL) != nullptr;
  }();
  if (!loaded) {
    throw std::runtime_error(
        "AppleNeuralEngine.framework could not be loaded.");
  }
}

std::string error_text(NSString *prefix, NSError *error) {
  NSString *value =
      error ? [NSString stringWithFormat:@"%@: %@", prefix, error] : prefix;
  return std::string([value UTF8String]);
}

NSData *make_blob(const void *bytes, size_t byte_count) {
  const size_t total = 128 + byte_count;
  auto *blob = static_cast<uint8_t *>(calloc(total, 1));
  if (!blob) {
    throw std::bad_alloc();
  }
  blob[0] = 0x01;
  blob[4] = 0x02;
  auto *chunk = blob + 64;
  chunk[0] = 0xef;
  chunk[1] = 0xbe;
  chunk[2] = 0xad;
  chunk[3] = 0xde;
  chunk[4] = 0x01;
  *reinterpret_cast<uint32_t *>(chunk + 8) = static_cast<uint32_t>(byte_count);
  *reinterpret_cast<uint32_t *>(chunk + 16) = 128;
  std::memcpy(blob + 128, bytes, byte_count);
  return [NSData dataWithBytesNoCopy:blob length:total freeWhenDone:YES];
}

uint64_t append_blob_chunk(NSMutableData *blob, const void *bytes,
                           size_t byte_count, uint32_t type) {
  const NSUInteger aligned = (blob.length + 63) & ~static_cast<NSUInteger>(63);
  if (aligned > blob.length) {
    [blob increaseLengthBy:aligned - blob.length];
  }
  const uint64_t chunk_offset = blob.length;
  uint8_t header[64] = {};
  header[0] = 0xef;
  header[1] = 0xbe;
  header[2] = 0xad;
  header[3] = 0xde;
  *reinterpret_cast<uint32_t *>(header + 4) = type;
  *reinterpret_cast<uint64_t *>(header + 8) = byte_count;
  *reinterpret_cast<uint64_t *>(header + 16) = chunk_offset + sizeof(header);
  [blob appendBytes:header length:sizeof(header)];
  [blob appendBytes:bytes length:byte_count];
  return chunk_offset;
}

NSString *int8_linear_mil(int input_dim, int output_dim, int sequence_length) {
  return [NSString
      stringWithFormat:
          @"program(1.3)\n"
           "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", "
           "\"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, "
           "{\"coremltools-component-milinternal\", \"\"}, "
           "{\"coremltools-version\", \"9.0\"}})]\n{\n"
           "  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
           "    tensor<int8, [%d, %d, 1, 1]> wd = const()[name=string(\"wd\"), "
           "val=tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path=string("
           "\"@model_path/weights/weight_data.bin\"), offset=uint64(64)))];\n"
           "    tensor<fp16, [%d, 1, 1, 1]> ws = const()[name=string(\"ws\"), "
           "val=tensor<fp16, [%d, 1, 1, 1]>(BLOBFILE(path=string("
           "\"@model_path/weights/weight_scale.bin\"), offset=uint64(64)))];\n"
           "    tensor<fp16, [%d, %d, 1, 1]> w = "
           "constexpr_blockwise_shift_scale(data=wd, scale=ws)"
           "[name=string(\"dequant\")];\n"
           "    string pt = const()[name=string(\"pt\"), "
           "val=string(\"valid\")];\n"
           "    tensor<int32, [2]> st = const()[name=string(\"st\"), "
           "val=tensor<int32, [2]>([1,1])];\n"
           "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), "
           "val=tensor<int32, [4]>([0,0,0,0])];\n"
           "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), "
           "val=tensor<int32, [2]>([1,1])];\n"
           "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
           "    tensor<fp16, [1, %d, 1, %d]> y = conv(dilations=dl, groups=gr, "
           "pad=pd, pad_type=pt, strides=st, weight=w, x=x)"
           "[name=string(\"conv\")];\n"
           "  } -> (y);\n}\n",
          input_dim, sequence_length, output_dim, input_dim, output_dim,
          input_dim, output_dim, output_dim, output_dim, input_dim, output_dim,
          sequence_length];
}

NSString *int8_linear_bank_mil(
    const std::vector<std::pair<int, int>> &shapes,
    const std::vector<std::pair<uint64_t, uint64_t>> &weight_offsets,
    int sequence_length) {
  NSMutableString *mil = [NSMutableString
      stringWithString:
          @"program(1.3)\n"
           "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", "
           "\"3520.4.1\"}, {\"coremlc-version\", \"3520.5.1\"}})]\n{\n"];
  for (size_t index = 0; index < shapes.size(); ++index) {
    const int input_dim = shapes[index].first;
    const int output_dim = shapes[index].second;
    [mil appendFormat:
             @"  func procedure%03zu<ios18>(tensor<fp16, [1, %d, 1, %d]> "
              "x) {\n"
              "    tensor<fp16, [%d, %d, 1, 1]> w = "
              "constexpr_blockwise_shift_scale(data=tensor<int8, [%d, %d, 1, "
              "1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), "
              "offset=uint64(%llu))), "
              "scale=tensor<fp16, [%d, 1, 1, 1]>(BLOBFILE(path=string("
              "\"@model_path/weights/weight.bin\"), "
              "offset=uint64(%llu))))"
              "[name=string(\"dequant\")];\n"
              "    string pt = const()[name=string(\"pt\"), "
              "val=string(\"valid\")];\n"
              "    tensor<int32, [2]> st = const()[name=string(\"st\"), "
              "val=tensor<int32, [2]>([1,1])];\n"
              "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), "
              "val=tensor<int32, [4]>([0,0,0,0])];\n"
              "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), "
              "val=tensor<int32, [2]>([1,1])];\n"
              "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
              "    tensor<fp16, [1, %d, 1, %d]> y = "
              "conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, "
              "strides=st, weight=w, x=x)[name=string(\"conv\")];\n"
              "  } -> (y);\n",
         index, input_dim, sequence_length, output_dim, input_dim, output_dim,
         input_dim,
         static_cast<unsigned long long>(weight_offsets[index].first),
         output_dim,
         static_cast<unsigned long long>(weight_offsets[index].second),
         output_dim, sequence_length];
  }
  [mil appendString:@"}\n"];
  return mil;
}

NSString *fp16_linear_mil(int input_dim, int output_dim, int sequence_length) {
  return [NSString
      stringWithFormat:
          @"program(1.3)\n"
           "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", "
           "\"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, "
           "{\"coremltools-component-milinternal\", \"\"}, "
           "{\"coremltools-version\", \"9.0\"}})]\n{\n"
           "  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
           "    tensor<fp16, [%d, %d, 1, 1]> w = const()[name=string(\"w\"), "
           "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string("
           "\"@model_path/weights/weight_data.bin\"), offset=uint64(64)))];\n"
           "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
           "    tensor<int32, [2]> st = const()[name=string(\"st\"), "
           "val=tensor<int32, [2]>([1,1])];\n"
           "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), "
           "val=tensor<int32, [4]>([0,0,0,0])];\n"
           "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), "
           "val=tensor<int32, [2]>([1,1])];\n"
           "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
           "    tensor<fp16, [1, %d, 1, %d]> y = conv(dilations=dl, groups=gr, "
           "pad=pd, pad_type=pt, strides=st, weight=w, x=x)"
           "[name=string(\"conv\")];\n"
           "  } -> (y);\n}\n",
          input_dim, sequence_length, output_dim, input_dim, output_dim,
          input_dim, output_dim, sequence_length];
}

NSString *fp16_swiglu_down_mil(int input_dim, int hidden_dim, int output_dim,
                               int sequence_length) {
  return [NSString
      stringWithFormat:
          @"program(1.3)\n"
           "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", "
           "\"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, "
           "{\"coremltools-component-milinternal\", \"\"}, "
           "{\"coremltools-version\", \"9.0\"}})]\n{\n"
           "  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
           "    tensor<fp16, [%d, %d, 1, 1]> gw = const()[name=string(\"gw\"), "
           "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string("
           "\"@model_path/weights/gate.bin\"), offset=uint64(64)))];\n"
           "    tensor<fp16, [%d, %d, 1, 1]> uw = const()[name=string(\"uw\"), "
           "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string("
           "\"@model_path/weights/up.bin\"), offset=uint64(64)))];\n"
           "    tensor<fp16, [%d, %d, 1, 1]> dw = const()[name=string(\"dw\"), "
           "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string("
           "\"@model_path/weights/down.bin\"), offset=uint64(64)))];\n"
           "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
           "    tensor<int32, [2]> st = const()[name=string(\"st\"), "
           "val=tensor<int32, [2]>([1,1])];\n"
           "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), "
           "val=tensor<int32, [4]>([0,0,0,0])];\n"
           "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), "
           "val=tensor<int32, [2]>([1,1])];\n"
           "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
           "    tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations=dl, "
           "groups=gr, pad=pd, pad_type=pt, strides=st, weight=gw, x=x)"
           "[name=string(\"gate\")];\n"
           "    tensor<fp16, [1, %d, 1, %d]> up = conv(dilations=dl, "
           "groups=gr, pad=pd, pad_type=pt, strides=st, weight=uw, x=x)"
           "[name=string(\"up\")];\n"
           "    tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x=gate)"
           "[name=string(\"sigmoid\")];\n"
           "    tensor<fp16, [1, %d, 1, %d]> silu = mul(x=gate, y=sig)"
           "[name=string(\"silu\")];\n"
           "    tensor<fp16, [1, %d, 1, %d]> act = mul(x=silu, y=up)"
           "[name=string(\"swiglu\")];\n"
           "    tensor<fp16, [1, %d, 1, %d]> y = conv(dilations=dl, "
           "groups=gr, pad=pd, pad_type=pt, strides=st, weight=dw, x=act)"
           "[name=string(\"down\")];\n"
           "  } -> (y);\n}\n",
          input_dim, sequence_length,
          hidden_dim, input_dim, hidden_dim, input_dim,
          hidden_dim, input_dim, hidden_dim, input_dim,
          output_dim, hidden_dim, output_dim, hidden_dim,
          hidden_dim, sequence_length, hidden_dim, sequence_length,
          hidden_dim, sequence_length, hidden_dim, sequence_length,
          hidden_dim, sequence_length, output_dim, sequence_length];
}

std::vector<_Float16> fp16_rows(const float *weight, int output_dim,
                                int input_dim) {
  std::vector<_Float16> result(static_cast<size_t>(output_dim) * input_dim);
  std::transform(weight, weight + result.size(), result.begin(),
                 [](float value) { return static_cast<_Float16>(value); });
  return result;
}

struct QuantizedMatrix {
  std::vector<int8_t> data;
  std::vector<_Float16> scales;
};

QuantizedMatrix quantize_rows(const float *weight, int output_dim,
                              int input_dim) {
  QuantizedMatrix result{
      std::vector<int8_t>(static_cast<size_t>(output_dim) * input_dim),
      std::vector<_Float16>(output_dim)};
  for (int row = 0; row < output_dim; ++row) {
    const float *source = weight + static_cast<size_t>(row) * input_dim;
    float maximum = 0.0f;
    for (int col = 0; col < input_dim; ++col) {
      maximum = std::max(maximum, std::abs(source[col]));
    }
    const float scale = std::max(maximum / 127.0f, 1.0e-8f);
    result.scales[row] = static_cast<_Float16>(scale);
    int8_t *destination =
        result.data.data() + static_cast<size_t>(row) * input_dim;
    for (int col = 0; col < input_dim; ++col) {
      destination[col] = static_cast<int8_t>(
          std::clamp(static_cast<int>(std::nearbyint(source[col] / scale)),
                     -127, 127));
    }
  }
  return result;
}

IOSurfaceRef make_surface(size_t byte_count) {
  const size_t allocation = std::max<size_t>(
      65536, (byte_count + 65535) & ~static_cast<size_t>(65535));
  return IOSurfaceCreate((__bridge CFDictionaryRef) @{
    (id)kIOSurfaceWidth : @(allocation),
    (id)kIOSurfaceHeight : @1,
    (id)kIOSurfaceBytesPerElement : @1,
    (id)kIOSurfaceBytesPerRow : @(allocation),
    (id)kIOSurfaceAllocSize : @(allocation),
    (id)kIOSurfacePixelFormat : @0,
  });
}

std::string metal_type_name(Dtype dtype) {
  if (dtype == float16) {
    return "float16_t";
  }
  if (dtype == bfloat16) {
    return "bfloat16_t";
  }
  throw std::invalid_argument("ANE hybrid input must be fp16 or bf16.");
}

bool row_contiguous(const array &value) {
  return value.flags().row_contiguous && value.strides(-1) == 1;
}

NSDictionary *ane_execution_options(int ane_instance) {
  if (ane_instance == 0) {
    return @{};
  }
  if (ane_instance < 1 || ane_instance > 4) {
    throw std::invalid_argument("ANE instance hint must be between 1 and 4.");
  }
  // The private scheduler only accepts an instance hint for its single-ANE
  // procedure variant. M3 Ultra exposes the two physical dies as 1 and 2.
  return @{
    @"kANEFProcedureVariantHint" : @1,
    @"kANEFAneInstanceHint" : @(ane_instance),
  };
}

} // namespace

class SharedAneProgram {
public:
  SharedAneProgram(id model, NSString *directory, NSDictionary *options)
      : model_([model retain]), directory_([directory copy]),
        execution_options_([options copy]) {}

  ~SharedAneProgram() {
    @autoreleasepool {
      if (model_) {
        NSError *error = nil;
        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model_, @selector(unloadWithQoS:error:), 21, &error);
      }
      if (directory_) {
        [[NSFileManager defaultManager] removeItemAtPath:directory_ error:nil];
      }
      [model_ release];
      [directory_ release];
      [execution_options_ release];
    }
  }

  id model_;
  NSString *directory_;
  NSDictionary *execution_options_;
  std::mutex evaluation_mutex_;
};

class AneLinearModel::Impl {
public:
  Impl(const float *weight, int output_dim, int input_dim, int sequence_length,
       bool use_fp16 = false, int ane_instance = 0)
      : input_dim_(input_dim), output_dim_(output_dim),
        sequence_length_(sequence_length) {
    @autoreleasepool {
      load_ane_framework();
      execution_options_ = [ane_execution_options(ane_instance) copy];

      auto quantized = quantize_rows(weight, output_dim_, input_dim_);
      auto dense = use_fp16 ? fp16_rows(weight, output_dim_, input_dim_)
                            : std::vector<_Float16>{};

      NSData *mil = [(use_fp16 ? fp16_linear_mil(
                                      input_dim_, output_dim_, sequence_length_)
                                : int8_linear_mil(
                                      input_dim_, output_dim_, sequence_length_))
          dataUsingEncoding:NSUTF8StringEncoding];
      NSData *data_blob = use_fp16
                              ? make_blob(dense.data(),
                                          dense.size() * sizeof(dense.front()))
                              : make_blob(
                                    quantized.data.data(),
                                    quantized.data.size() *
                                        sizeof(quantized.data.front()));
      NSData *scale_blob = make_blob(
          quantized.scales.data(),
          quantized.scales.size() * sizeof(quantized.scales.front()));
      NSDictionary *weights = @{
        @"@model_path/weights/weight_data.bin" :
            @{@"offset" : @0, @"data" : data_blob},
        @"@model_path/weights/weight_scale.bin" :
            @{@"offset" : @0, @"data" : scale_blob},
      };

      Class descriptor_class =
          NSClassFromString(@"_ANEInMemoryModelDescriptor");
      Class model_class = NSClassFromString(@"_ANEInMemoryModel");
      Class request_class = NSClassFromString(@"_ANERequest");
      Class surface_class = NSClassFromString(@"_ANEIOSurfaceObject");
      if (!descriptor_class || !model_class || !request_class ||
          !surface_class) {
        throw std::runtime_error(
            "Required private ANE classes are unavailable.");
      }

      id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
          descriptor_class, @selector(modelWithMILText:weights:optionsPlist:),
          mil, weights, nil);
      id model = ((id (*)(Class, SEL, id))objc_msgSend)(
          model_class, @selector(inMemoryModelWithDescriptor:), descriptor);
      if (!model) {
        throw std::runtime_error(
            "Private ANE in-memory model creation failed.");
      }

      id identifier = ((id (*)(id, SEL))objc_msgSend)(
          model, @selector(hexStringIdentifier));
      NSString *directory =
          [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
      directory_ = [directory copy];
      [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
      NSString *weight_directory =
          [directory stringByAppendingPathComponent:@"weights"];
      [[NSFileManager defaultManager] createDirectoryAtPath:weight_directory
                                withIntermediateDirectories:YES
                                                 attributes:nil
                                                      error:nil];
      [mil writeToFile:[directory stringByAppendingPathComponent:@"model.mil"]
            atomically:YES];
      [data_blob
          writeToFile:[weight_directory
                          stringByAppendingPathComponent:@"weight_data.bin"]
           atomically:YES];
      [scale_blob
          writeToFile:[weight_directory
                          stringByAppendingPathComponent:@"weight_scale.bin"]
           atomically:YES];

      NSError *error = nil;
      BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
          model, @selector(compileWithQoS:options:error:), 21,
          execution_options_, &error);
      if (!ok) {
        throw std::runtime_error(error_text(@"ANE compilation failed", error));
      }
      ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
          model, @selector(loadWithQoS:options:error:), 21,
          execution_options_, &error);
      if (!ok) {
        throw std::runtime_error(error_text(@"ANE model load failed", error));
      }
      model_ = [model retain];

      input_surface_ = make_surface(static_cast<size_t>(input_dim_) *
                                    sequence_length_ * sizeof(_Float16));
      output_surface_ = make_surface(static_cast<size_t>(output_dim_) *
                                     sequence_length_ * sizeof(_Float16));
      if (!input_surface_ || !output_surface_) {
        throw std::runtime_error("ANE IOSurface allocation failed.");
      }
      input_object_ = [((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
          surface_class, @selector(objectWithIOSurface:),
          input_surface_) retain];
      output_object_ = [((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
          surface_class, @selector(objectWithIOSurface:),
          output_surface_) retain];

      auto &mlx_device = metal::device(Device::gpu);
      id<MTLDevice> device = (__bridge id<MTLDevice>)(static_cast<void *>(
          mlx_device.mtl_device()));
      event_ = [device newSharedEvent];
      input_buffer_ = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
          device, @selector(newBufferWithIOSurface:), input_surface_);
      output_buffer_ = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
          device, @selector(newBufferWithIOSurface:), output_surface_);
      if (!event_ || !input_buffer_ || !output_buffer_) {
        throw std::runtime_error("Metal could not wrap the ANE IOSurfaces.");
      }

      request_ =
          [((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
              request_class,
              @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                        weightsBuffer:perfStats:procedureIndex:),
              @[ input_object_ ], @[ @0 ], @[ output_object_ ], @[ @0 ], nil,
              nil, @0) retain];
      if (!request_) {
        throw std::runtime_error("ANE request creation failed.");
      }
    }
  }

  Impl(const float *gate_weight, const float *up_weight,
       const float *down_weight, int hidden_dim, int output_dim, int input_dim,
       int sequence_length)
      : input_dim_(input_dim), output_dim_(output_dim),
        sequence_length_(sequence_length) {
    @autoreleasepool {
      load_ane_framework();
      execution_options_ = [@{} retain];
      auto gate = fp16_rows(gate_weight, hidden_dim, input_dim);
      auto up = fp16_rows(up_weight, hidden_dim, input_dim);
      auto down = fp16_rows(down_weight, output_dim, hidden_dim);
      NSData *mil = [fp16_swiglu_down_mil(input_dim, hidden_dim, output_dim,
                                          sequence_length)
          dataUsingEncoding:NSUTF8StringEncoding];
      NSData *gate_data =
          make_blob(gate.data(), gate.size() * sizeof(gate.front()));
      NSData *up_data = make_blob(up.data(), up.size() * sizeof(up.front()));
      NSData *down_data =
          make_blob(down.data(), down.size() * sizeof(down.front()));
      NSDictionary *weights = @{
        @"@model_path/weights/gate.bin" :
            @{ @"offset" : @0, @"data" : gate_data },
        @"@model_path/weights/up.bin" :
            @{ @"offset" : @0, @"data" : up_data },
        @"@model_path/weights/down.bin" :
            @{ @"offset" : @0, @"data" : down_data },
      };

      Class descriptor_class =
          NSClassFromString(@"_ANEInMemoryModelDescriptor");
      Class model_class = NSClassFromString(@"_ANEInMemoryModel");
      Class request_class = NSClassFromString(@"_ANERequest");
      Class surface_class = NSClassFromString(@"_ANEIOSurfaceObject");
      if (!descriptor_class || !model_class || !request_class ||
          !surface_class) {
        throw std::runtime_error(
            "Required private ANE classes are unavailable.");
      }

      id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
          descriptor_class, @selector(modelWithMILText:weights:optionsPlist:),
          mil, weights, nil);
      id model = ((id (*)(Class, SEL, id))objc_msgSend)(
          model_class, @selector(inMemoryModelWithDescriptor:), descriptor);
      if (!model) {
        throw std::runtime_error(
            "Private ANE in-memory model creation failed.");
      }

      id identifier = ((id (*)(id, SEL))objc_msgSend)(
          model, @selector(hexStringIdentifier));
      NSString *directory =
          [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
      directory_ = [directory copy];
      [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
      NSString *weight_directory =
          [directory stringByAppendingPathComponent:@"weights"];
      [[NSFileManager defaultManager] createDirectoryAtPath:weight_directory
                                withIntermediateDirectories:YES
                                                 attributes:nil
                                                      error:nil];
      [mil writeToFile:[directory stringByAppendingPathComponent:@"model.mil"]
            atomically:YES];
      NSDictionary *files = @{
        @"gate.bin" : gate_data,
        @"up.bin" : up_data,
        @"down.bin" : down_data,
      };
      for (NSString *name in files) {
        [files[name]
            writeToFile:[weight_directory stringByAppendingPathComponent:name]
             atomically:YES];
      }

      NSError *error = nil;
      BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
          model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
      if (!ok) {
        throw std::runtime_error(error_text(@"ANE compilation failed", error));
      }
      ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
          model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
      if (!ok) {
        throw std::runtime_error(error_text(@"ANE model load failed", error));
      }
      model_ = [model retain];

      input_surface_ = make_surface(static_cast<size_t>(input_dim_) *
                                    sequence_length_ * sizeof(_Float16));
      output_surface_ = make_surface(static_cast<size_t>(output_dim_) *
                                     sequence_length_ * sizeof(_Float16));
      if (!input_surface_ || !output_surface_) {
        throw std::runtime_error("ANE IOSurface allocation failed.");
      }
      input_object_ = [((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
          surface_class, @selector(objectWithIOSurface:),
          input_surface_) retain];
      output_object_ = [((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
          surface_class, @selector(objectWithIOSurface:),
          output_surface_) retain];

      auto &mlx_device = metal::device(Device::gpu);
      id<MTLDevice> device = (__bridge id<MTLDevice>)(static_cast<void *>(
          mlx_device.mtl_device()));
      event_ = [device newSharedEvent];
      input_buffer_ = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
          device, @selector(newBufferWithIOSurface:), input_surface_);
      output_buffer_ = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
          device, @selector(newBufferWithIOSurface:), output_surface_);
      if (!event_ || !input_buffer_ || !output_buffer_) {
        throw std::runtime_error("Metal could not wrap the ANE IOSurfaces.");
      }

      request_ =
          [((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
              request_class,
              @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                        weightsBuffer:perfStats:procedureIndex:),
              @[ input_object_ ], @[ @0 ], @[ output_object_ ], @[ @0 ], nil,
              nil, @0) retain];
      if (!request_) {
        throw std::runtime_error("ANE request creation failed.");
      }
    }
  }

  Impl(std::shared_ptr<SharedAneProgram> program, int input_dim, int output_dim,
       int sequence_length, int procedure_index)
      : input_dim_(input_dim), output_dim_(output_dim),
        sequence_length_(sequence_length), shared_program_(std::move(program)) {
    @autoreleasepool {
      model_ = [shared_program_->model_ retain];
      execution_options_ = [shared_program_->execution_options_ copy];
      Class request_class = NSClassFromString(@"_ANERequest");
      Class surface_class = NSClassFromString(@"_ANEIOSurfaceObject");

      input_surface_ = make_surface(static_cast<size_t>(input_dim_) *
                                    sequence_length_ * sizeof(_Float16));
      output_surface_ = make_surface(static_cast<size_t>(output_dim_) *
                                     sequence_length_ * sizeof(_Float16));
      if (!input_surface_ || !output_surface_) {
        throw std::runtime_error("ANE IOSurface allocation failed.");
      }
      input_object_ = [((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
          surface_class, @selector(objectWithIOSurface:), input_surface_) retain];
      output_object_ = [((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
          surface_class, @selector(objectWithIOSurface:), output_surface_) retain];

      auto &mlx_device = metal::device(Device::gpu);
      id<MTLDevice> device = (__bridge id<MTLDevice>)(static_cast<void *>(
          mlx_device.mtl_device()));
      event_ = [device newSharedEvent];
      input_buffer_ = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
          device, @selector(newBufferWithIOSurface:), input_surface_);
      output_buffer_ = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
          device, @selector(newBufferWithIOSurface:), output_surface_);
      if (!event_ || !input_buffer_ || !output_buffer_) {
        throw std::runtime_error("Metal could not wrap the ANE IOSurfaces.");
      }

      id ane_model =
          ((id (*)(id, SEL))objc_msgSend)(model_, @selector(model));
      NSIndexSet *input_index_set =
          ((id (*)(id, SEL, unsigned int))objc_msgSend)(
              ane_model, @selector(inputSymbolIndicesForProcedureIndex:),
              procedure_index);
      NSIndexSet *output_index_set =
          ((id (*)(id, SEL, unsigned int))objc_msgSend)(
              ane_model, @selector(outputSymbolIndicesForProcedureIndex:),
              procedure_index);
      if (input_index_set.count != 1 || output_index_set.count != 1) {
        throw std::runtime_error("ANE procedure has unexpected I/O symbols.");
      }
      NSArray *input_indices = @[ @(input_index_set.firstIndex) ];
      NSArray *output_indices = @[ @(output_index_set.firstIndex) ];
      request_ =
          [((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
              request_class,
              @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                        weightsBuffer:perfStats:procedureIndex:),
              @[ input_object_ ], input_indices, @[ output_object_ ],
              output_indices, nil, nil, @(procedure_index)) retain];
      if (!request_) {
        throw std::runtime_error("ANE procedure request creation failed.");
      }
    }
  }

  ~Impl() {
    @autoreleasepool {
      {
        std::unique_lock<std::mutex> lock(state_mutex_);
        completion_cv_.wait_for(lock, std::chrono::seconds(10),
                                [this] { return completed_ >= submitted_; });
      }
      if (model_ && !shared_program_) {
        NSError *error = nil;
        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model_, @selector(unloadWithQoS:error:), 21, &error);
      }
      if (directory_) {
        [[NSFileManager defaultManager] removeItemAtPath:directory_ error:nil];
      }
      [request_ release];
      [execution_options_ release];
      [event_ release];
      [input_buffer_ release];
      [output_buffer_ release];
      [input_object_ release];
      [output_object_ release];
      if (input_surface_)
        CFRelease(input_surface_);
      if (output_surface_)
        CFRelease(output_surface_);
      [model_ release];
      [directory_ release];
    }
  }

  AneLinearModel::Ticket begin(MTL::CommandBuffer *command_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!completion_error_.empty()) {
      throw std::runtime_error("Prior ANE evaluation failed: " +
                               completion_error_);
    }
    if (submitted_ != completed_) {
      throw std::runtime_error("The same fixed-shape ANE model cannot have "
                               "overlapping evaluations.");
    }
    const uint64_t ready = next_event_value_++;
    const uint64_t done = next_event_value_++;
    ++submitted_;
    id<MTLCommandBuffer> buffer =
        (__bridge id<MTLCommandBuffer>)(static_cast<void *>(command_buffer));
    [buffer encodeSignalEvent:event_ value:ready];
    return {ready, done};
  }

  void evaluate_and_signal(AneLinearModel::Ticket ticket) {
    while ([event_ signaledValue] < ticket.ready) {
      std::this_thread::yield();
    }
    NSError *error = nil;
    BOOL ok = false;
    if (shared_program_) {
      std::lock_guard<std::mutex> program_lock(
          shared_program_->evaluation_mutex_);
      ok = ((BOOL (*)(id, SEL, unsigned int, id, id,
                      NSError **))objc_msgSend)(
          model_, @selector(evaluateWithQoS:options:request:error:), 21,
          execution_options_, request_, &error);
    } else {
      ok = ((BOOL (*)(id, SEL, unsigned int, id, id,
                      NSError **))objc_msgSend)(
          model_, @selector(evaluateWithQoS:options:request:error:), 21,
          execution_options_, request_, &error);
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!ok) {
        completion_error_ = error_text(@"ANE evaluation failed", error);
      }
      ++completed_;
    }
    // A CPU-side MTLSharedEvent signal wakes the independent merge command
    // buffer without imposing ANE's private request-wide shared-event fence,
    // which serializes concurrent GPU work on current macOS drivers.
    [event_ setSignaledValue:ticket.done];
    completion_cv_.notify_all();
  }

  void end(MTL::CommandBuffer *command_buffer, AneLinearModel::Ticket ticket) {
    id<MTLCommandBuffer> buffer =
        (__bridge id<MTLCommandBuffer>)(static_cast<void *>(command_buffer));
    [buffer encodeWaitForEvent:event_ value:ticket.done];
  }

  void wait(AneLinearModel::Ticket) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    completion_cv_.wait(lock, [this] { return completed_ >= submitted_; });
    if (!completion_error_.empty()) {
      throw std::runtime_error(completion_error_);
    }
  }

  void warmup() {
    AneLinearModel::Ticket ticket{};
    {
      std::unique_lock<std::mutex> lock(state_mutex_);
      completion_cv_.wait(lock, [this] { return submitted_ == completed_; });
      if (!completion_error_.empty()) {
        throw std::runtime_error("Prior ANE evaluation failed: " +
                                 completion_error_);
      }
      ticket.ready = next_event_value_++;
      ticket.done = next_event_value_++;
      ++submitted_;
    }
    // Host-side ready signal: there is no producer command buffer, the
    // input surface holds whatever it holds. evaluate_and_signal's readiness
    // spin-wait passes immediately and the evaluation result is discarded.
    [event_ setSignaledValue:ticket.ready];
    evaluate_and_signal(ticket);
    wait(ticket);
  }

  int input_dim_;
  int output_dim_;
  int sequence_length_;
  id model_ = nil;
  NSString *directory_ = nil;
  IOSurfaceRef input_surface_ = nullptr;
  IOSurfaceRef output_surface_ = nullptr;
  id input_object_ = nil;
  id output_object_ = nil;
  id<MTLBuffer> input_buffer_ = nil;
  id<MTLBuffer> output_buffer_ = nil;
  id<MTLSharedEvent> event_ = nil;
  id request_ = nil;
  NSDictionary *execution_options_ = nil;
  uint64_t next_event_value_ = 1;
  uint64_t submitted_ = 0;
  uint64_t completed_ = 0;
  std::string completion_error_;
  std::mutex state_mutex_;
  std::condition_variable completion_cv_;
  std::shared_ptr<SharedAneProgram> shared_program_;
};

AneLinearModel::AneLinearModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AneLinearModel::~AneLinearModel() = default;
int AneLinearModel::input_dim() const { return impl_->input_dim_; }
int AneLinearModel::output_dim() const { return impl_->output_dim_; }
int AneLinearModel::sequence_length() const { return impl_->sequence_length_; }
MTL::Buffer *AneLinearModel::input_buffer() const {
  return reinterpret_cast<MTL::Buffer *>(impl_->input_buffer_);
}
MTL::Buffer *AneLinearModel::output_buffer() const {
  return reinterpret_cast<MTL::Buffer *>(impl_->output_buffer_);
}
AneLinearModel::Ticket
AneLinearModel::begin(MTL::CommandBuffer *command_buffer) {
  return impl_->begin(command_buffer);
}
void AneLinearModel::execute(AneLinearModel::Ticket ticket) {
  @autoreleasepool {
    impl_->evaluate_and_signal(ticket);
  }
}
void AneLinearModel::wait(AneLinearModel::Ticket ticket) {
  impl_->wait(ticket);
}
void AneLinearModel::warmup() {
  @autoreleasepool {
    impl_->warmup();
  }
}
void AneLinearModel::end(MTL::CommandBuffer *command_buffer,
                         AneLinearModel::Ticket ticket) {
  impl_->end(command_buffer, ticket);
}

bool qwen35_ane_available() {
  @autoreleasepool {
    try {
      load_ane_framework();
      Class descriptor = NSClassFromString(@"_ANEInMemoryModelDescriptor");
      Class model = NSClassFromString(@"_ANEInMemoryModel");
      Class request = NSClassFromString(@"_ANERequest");
      auto &d = metal::device(Device::gpu);
      id<MTLDevice> device =
          (__bridge id<MTLDevice>)(static_cast<void *>(d.mtl_device()));
      return descriptor && model && request &&
             [device respondsToSelector:@selector(newBufferWithIOSurface:)];
    } catch (...) {
      return false;
    }
  }
}

void qwen35_ane_profile_set_enabled(bool enabled) {
  g_ane_profile_override.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

void qwen35_ane_profile_reset() {
  for (auto &category : g_ane_profile) {
    for (auto &metric : category) {
      metric.store(0, std::memory_order_relaxed);
    }
  }
  g_previous_ane_done_ns.store(0, std::memory_order_relaxed);
}

std::vector<double> qwen35_ane_profile_snapshot() {
  std::vector<double> result;
  result.reserve(2 * kProfileMetricCount);
  for (const auto &category : g_ane_profile) {
    for (const auto &metric : category) {
      result.push_back(static_cast<double>(
          metric.load(std::memory_order_relaxed)));
    }
  }
  return result;
}

std::shared_ptr<AneLinearModel> qwen35_ane_compile_linear(const array &weight,
                                                          int sequence_length) {
  return qwen35_ane_compile_linear(weight, sequence_length, 0);
}

std::shared_ptr<AneLinearModel> qwen35_ane_compile_linear(
    const array &weight, int sequence_length, int ane_instance) {
  if (!qwen35_ane_available()) {
    throw std::runtime_error("Private ANE runtime is unavailable.");
  }
  if (weight.dtype() != float32 || weight.ndim() != 2 ||
      !row_contiguous(weight)) {
    throw std::invalid_argument(
        "ANE compile weight must be a contiguous rank-2 float32 MLX array.");
  }
  if (sequence_length <= 1 || weight.shape(0) <= 0 || weight.shape(1) <= 0) {
    throw std::invalid_argument("Invalid fixed shape for ANE linear model.");
  }
  auto impl = std::make_unique<AneLinearModel::Impl>(
      weight.data<float>(), static_cast<int>(weight.shape(0)),
      static_cast<int>(weight.shape(1)), sequence_length, false, ane_instance);
  return std::shared_ptr<AneLinearModel>(new AneLinearModel(std::move(impl)));
}

std::vector<std::shared_ptr<AneLinearModel>> qwen35_ane_compile_linear_bank(
    const std::vector<array> &weights, int sequence_length, int ane_instance) {
  if (!qwen35_ane_available()) {
    throw std::runtime_error("Private ANE runtime is unavailable.");
  }
  if (weights.empty() || weights.size() > 256 || sequence_length <= 1) {
    throw std::invalid_argument("Invalid ANE linear procedure bank shape.");
  }
  std::vector<std::pair<int, int>> shapes;
  shapes.reserve(weights.size());
  for (const auto &weight : weights) {
    if (weight.dtype() != float32 || weight.ndim() != 2 ||
        !row_contiguous(weight) || weight.shape(0) <= 0 ||
        weight.shape(1) <= 0) {
      throw std::invalid_argument(
          "ANE bank weights must be contiguous rank-2 float32 MLX arrays.");
    }
    shapes.emplace_back(static_cast<int>(weight.shape(1)),
                        static_cast<int>(weight.shape(0)));
  }

  @autoreleasepool {
    load_ane_framework();
    NSDictionary *execution_options = ane_execution_options(ane_instance);
    NSMutableData *weight_blob = [NSMutableData dataWithLength:64];
    std::vector<std::pair<uint64_t, uint64_t>> weight_offsets;
    weight_offsets.reserve(weights.size());
    for (size_t index = 0; index < weights.size(); ++index) {
      const auto &weight = weights[index];
      auto quantized = quantize_rows(
          weight.data<float>(), static_cast<int>(weight.shape(0)),
          static_cast<int>(weight.shape(1)));
      uint64_t data_offset = append_blob_chunk(
          weight_blob,
          quantized.data.data(),
          quantized.data.size() * sizeof(quantized.data.front()), 4);
      uint64_t scale_offset = append_blob_chunk(
          weight_blob,
          quantized.scales.data(),
          quantized.scales.size() * sizeof(quantized.scales.front()), 1);
      weight_offsets.emplace_back(data_offset, scale_offset);
    }
    auto *weight_header = static_cast<uint32_t *>(weight_blob.mutableBytes);
    weight_header[0] = static_cast<uint32_t>(weights.size() * 2);
    weight_header[1] = 2;
    NSData *mil =
        [int8_linear_bank_mil(shapes, weight_offsets, sequence_length)
            dataUsingEncoding:NSUTF8StringEncoding];
    NSDictionary *weight_map = @{
      @"@model_path/weights/weight.bin" :
          @{ @"offset" : @0, @"data" : weight_blob },
    };

    Class descriptor_class =
        NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class model_class = NSClassFromString(@"_ANEInMemoryModel");
    id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
        descriptor_class, @selector(modelWithMILText:weights:optionsPlist:),
        mil, weight_map, nil);
    id model = ((id (*)(Class, SEL, id))objc_msgSend)(
        model_class, @selector(inMemoryModelWithDescriptor:), descriptor);
    if (!model) {
      throw std::runtime_error(
          "Private ANE procedure bank model creation failed.");
    }
    id identifier = ((id (*)(id, SEL))objc_msgSend)(
        model, @selector(hexStringIdentifier));
    NSString *directory =
        [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
    [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
    NSString *weight_directory =
        [directory stringByAppendingPathComponent:@"weights"];
    [[NSFileManager defaultManager]
        createDirectoryAtPath:weight_directory
        withIntermediateDirectories:YES
        attributes:nil
        error:nil];
    [mil writeToFile:[directory stringByAppendingPathComponent:@"model.mil"]
          atomically:YES];
    [weight_blob
        writeToFile:[weight_directory stringByAppendingPathComponent:
                                          @"weight.bin"]
         atomically:YES];

    NSError *error = nil;
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21,
        execution_options, &error);
    if (!ok) {
      [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
      throw std::runtime_error(
          error_text(@"ANE procedure bank compilation failed", error));
    }
    ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, execution_options,
        &error);
    if (!ok) {
      [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
      throw std::runtime_error(
          error_text(@"ANE procedure bank load failed", error));
    }

    auto program =
        std::make_shared<SharedAneProgram>(model, directory, execution_options);
    std::vector<std::shared_ptr<AneLinearModel>> result;
    result.reserve(weights.size());
    for (size_t index = 0; index < weights.size(); ++index) {
      auto impl = std::make_unique<AneLinearModel::Impl>(
          program, shapes[index].first, shapes[index].second, sequence_length,
          static_cast<int>(index));
      result.emplace_back(new AneLinearModel(std::move(impl)));
    }
    return result;
  }
}

std::shared_ptr<AneLinearModel>
qwen35_ane_compile_fp16_linear(const array &weight, int sequence_length) {
  if (!qwen35_ane_available()) {
    throw std::runtime_error("Private ANE runtime is unavailable.");
  }
  if (weight.dtype() != float32 || weight.ndim() != 2 ||
      !row_contiguous(weight) || sequence_length <= 1) {
    throw std::invalid_argument(
        "ANE fp16 weight must be a contiguous rank-2 float32 MLX array.");
  }
  auto impl = std::make_unique<AneLinearModel::Impl>(
      weight.data<float>(), static_cast<int>(weight.shape(0)),
      static_cast<int>(weight.shape(1)), sequence_length, true);
  return std::shared_ptr<AneLinearModel>(new AneLinearModel(std::move(impl)));
}

std::shared_ptr<AneLinearModel> qwen35_ane_compile_swiglu_down(
    const array &gate_weight, const array &up_weight, const array &down_weight,
    int sequence_length) {
  if (!qwen35_ane_available()) {
    throw std::runtime_error("Private ANE runtime is unavailable.");
  }
  for (const auto *weight : {&gate_weight, &up_weight, &down_weight}) {
    if (weight->dtype() != float32 || weight->ndim() != 2 ||
        !row_contiguous(*weight)) {
      throw std::invalid_argument(
          "ANE SwiGLU weights must be contiguous rank-2 float32 arrays.");
    }
  }
  const int hidden_dim = static_cast<int>(gate_weight.shape(0));
  const int input_dim = static_cast<int>(gate_weight.shape(1));
  const int output_dim = static_cast<int>(down_weight.shape(0));
  if (sequence_length <= 1 || hidden_dim <= 0 || input_dim <= 0 ||
      output_dim <= 0 || up_weight.shape() != gate_weight.shape() ||
      down_weight.shape(1) != hidden_dim) {
    throw std::invalid_argument("Invalid fixed shape for ANE SwiGLU model.");
  }
  auto impl = std::make_unique<AneLinearModel::Impl>(
      gate_weight.data<float>(), up_weight.data<float>(),
      down_weight.data<float>(), hidden_dim, output_dim, input_dim,
      sequence_length);
  return std::shared_ptr<AneLinearModel>(new AneLinearModel(std::move(impl)));
}

namespace {

std::atomic<bool> g_hybrid_nax_runtime_ok{true};

bool hybrid_nax_enabled() {
  const char *override = std::getenv("OMLX_QWEN35_QMM_NAX");
  if (override && (std::strcmp(override, "0") == 0 ||
                   std::strcmp(override, "false") == 0 ||
                   std::strcmp(override, "off") == 0)) {
    return false;
  }
  return g_hybrid_nax_runtime_ok.load(std::memory_order_relaxed) &&
         is_nax_available() && nax_qmm_kernels_built() &&
         nax_qmm_runtime_active();
}

std::string hybrid_nax_qmm_name(int bits, int group_size,
                                const std::string &type) {
  return "qwen35_q" + std::to_string(bits) +
         (group_size == 128 ? "_affine_qmm128_t_nax_"
                            : "_affine_qmm_t_nax_") +
         type + "_bm_64_bk_64_bn_64_wm_2_wn_2";
}

std::string hybrid_classic_qmm_name(int bits, int group_size,
                                    const std::string &type) {
  return "qwen35_q" + std::to_string(bits) +
         (group_size == 128 ? "_affine_qmm128_t_" : "_affine_qmm_t_") +
         type + "_bm_64_bk_32_bn_64";
}

class AneHybridQ4Primitive : public Primitive {
public:
  AneHybridQ4Primitive(Stream stream, std::shared_ptr<AneLinearModel> model,
                       int bits, int variant, int group_size,
                       bool fuse_swiglu = false, int profile_category = -1)
      : Primitive(stream), model_(std::move(model)), variant_(variant),
        bits_(bits), group_size_(group_size), fuse_swiglu_(fuse_swiglu),
        profile_category_(profile_category >= 0 ? profile_category
                                                : (fuse_swiglu ? 0 : 1)) {}

  void eval_cpu(const std::vector<array> &, std::vector<array> &) override {
    throw std::runtime_error("ANE hybrid qmm has no CPU implementation.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    auto &encoder = metal::get_command_encoder(stream);
    const auto &x = inputs[0];
    const auto &weight = inputs[1];
    const auto &scales = inputs[2];
    const auto &biases = inputs[3];
    auto &output = outputs[0];

    const int K = static_cast<int>(x.shape(-1));
    const int M = static_cast<int>(x.size() / K);
    const int gpu_n = static_cast<int>(weight.shape(0));
    const int ane_n = model_->output_dim();
    const bool profiling = ane_profile_enabled();
    const int profile_category = profile_category_;
    const uint64_t operation_start = profiling ? profile_now_ns() : 0;
    if (profiling) {
      profile_add(profile_category, kOperations, 1);
      const uint64_t previous =
          g_previous_ane_done_ns.load(std::memory_order_relaxed);
      if (previous && operation_start > previous) {
        profile_add(profile_category, kGapBeforeNs,
                    operation_start - previous);
      }
    }
    output.set_data(allocator::malloc(output.nbytes()));
    array gpu_output({M, gpu_n}, x.dtype(), nullptr, {});
    gpu_output.set_data(allocator::malloc(gpu_output.nbytes()));

    auto library =
        device.get_library("omlx_qwen35_prefill_kernels", binary_dir());
    auto pack = device.get_kernel(
        "qwen35_ane_pack_input_" + metal_type_name(x.dtype()), library);
    encoder.set_compute_pipeline_state(pack);
    encoder.set_input_array(x, 0);
    encoder.set_buffer(model_->input_buffer(), 1);
    encoder.set_bytes(M, 2);
    encoder.set_bytes(K, 3);
    encoder.dispatch_threads(MTL::Size(K, M, 1), MTL::Size(16, 16, 1));
    encoder.end_encoding();

    auto *producer_buffer = encoder.get_command_buffer();
    producer_buffer->retain();
    auto ticket = model_->begin(producer_buffer);

    // Submit the pack-and-signal buffer before encoding the GPU remainder.
    // On current Metal drivers, a shared-event signal embedded midway through
    // one command buffer is not made visible to ANE early enough to overlap
    // reliably with later GPU encoders in that same buffer.
    encoder.commit();
    // The public shared-event counter is also delivered late on this driver
    // when later GPU work is queued. Since packing is a short prerequisite for
    // both devices, complete it up front; the expensive ANE and qmm stages are
    // still launched concurrently below.
    producer_buffer->waitUntilCompleted();
    producer_buffer->release();
    if (profiling) {
      profile_add(profile_category, kPackNs,
                  profile_now_ns() - operation_start);
    }

    const std::string qmm_type = metal_type_name(x.dtype());
    auto qmm = [&] {
      if (hybrid_nax_enabled()) {
        try {
          auto nax_library = device.get_library(
              "omlx_qwen35_prefill_kernels_nax", binary_dir());
          return device.get_kernel(
              hybrid_nax_qmm_name(bits_, group_size_, qmm_type), nax_library);
        } catch (const std::exception &) {
          // Older app bundles may carry an extension without the matching
          // group-size-128 metallib. Keep the private ANE path usable by
          // permanently demoting its suffix to the classic kernel.
          g_hybrid_nax_runtime_ok.store(false, std::memory_order_relaxed);
        }
      }
      return device.get_kernel(
          hybrid_classic_qmm_name(bits_, group_size_, qmm_type), library);
    }();
    encoder.set_compute_pipeline_state(qmm);
    encoder.set_input_array(weight, 0);
    encoder.set_input_array(scales, 1);
    encoder.set_input_array(biases, 2);
    encoder.set_input_array(x, 3);
    encoder.set_output_array(gpu_output, 4);
    encoder.set_bytes(K, 5);
    encoder.set_bytes(gpu_n, 6);
    encoder.set_bytes(M, 7);
    encoder.dispatch_threadgroups(
        MTL::Size((gpu_n + 63) / 64, (M + 63) / 64, 1), MTL::Size(32, 2, 2));
    encoder.end_encoding();
    id<MTLCommandBuffer> qmm_buffer =
        (__bridge id<MTLCommandBuffer>)(static_cast<void *>(
            encoder.get_command_buffer()));
    if (profiling) {
      [qmm_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        const double duration = completed.GPUEndTime - completed.GPUStartTime;
        if (duration > 0) {
          profile_add(profile_category, kGpuQmmNs,
                      static_cast<uint64_t>(duration * 1e9));
        }
      }];
    }

    // Keep the wait in a third command buffer. Metal may hold an entire
    // command buffer at an encoded wait, even when compute encoders precede
    // it, which serializes the otherwise independent GPU remainder behind
    // ANE. Submitting the qmm first makes the intended overlap explicit.
    encoder.commit();
    const uint64_t launch = profiling ? profile_now_ns() : 0;
    auto model = model_;
    std::thread([model = std::move(model), ticket, profiling, profile_category,
                 launch] {
      // Launch only after the GPU qmm has been committed. Starting the private
      // ANE evaluation earlier can contend on a driver submission lock and
      // delay the GPU command buffer itself.
      const uint64_t start = profiling ? profile_now_ns() : 0;
      model->execute(ticket);
      if (profiling) {
        const uint64_t end = profile_now_ns();
        profile_add(profile_category, kAne0LaunchNs, start - launch);
        profile_add(profile_category, kAne0EvalNs, end - start);
      }
    }).detach();
    // Do not submit a future Metal wait here: on current M3 Ultra drivers it
    // can stall the already-queued qmm behind ANE. Both devices are running at
    // this point, so a host wait preserves their overlap and immediately
    // exposes asynchronous ANE failures before the merge is encoded.
    model_->wait(ticket);
    if (profiling) {
      const uint64_t done = profile_now_ns();
      profile_add(profile_category, kAneRegionNs, done - launch);
      g_previous_ane_done_ns.store(done, std::memory_order_relaxed);
      if (qmm_buffer.status == MTLCommandBufferStatusCompleted) {
        profile_add(profile_category, kAneLast, 1);
      } else {
        profile_add(profile_category, kGpuLast, 1);
      }
    }

    auto merge = device.get_kernel(
        std::string(fuse_swiglu_ ? "qwen35_ane_merge_swiglu_output_"
                                 : "qwen35_ane_merge_output_") +
            metal_type_name(x.dtype()),
        library);
    encoder.set_compute_pipeline_state(merge);
    encoder.set_buffer(model_->output_buffer(), 0);
    encoder.set_input_array(gpu_output, 1);
    encoder.set_output_array(output, 2);
    encoder.set_bytes(M, 3);
    const int output_n =
        fuse_swiglu_ ? (ane_n + gpu_n) / 2 : ane_n + gpu_n;
    const int merge_ane_n = fuse_swiglu_ ? ane_n / 2 : ane_n;
    const int merge_gpu_n = fuse_swiglu_ ? gpu_n / 2 : gpu_n;
    encoder.set_bytes(merge_ane_n, 4);
    encoder.set_bytes(merge_gpu_n, 5);
    encoder.dispatch_threads(MTL::Size(output_n, M, 1),
                             MTL::Size(16, 16, 1));

    // This primitive deliberately commits intermediate command buffers. MLX's
    // evaluator retains the original buffer and only switches to its safe
    // post-commit path when the active encoder requests a commit. Large target
    // projections cross the byte threshold naturally; smaller supported
    // shapes use one-thread no-ops to cross the operation threshold instead.
    if (!encoder.needs_commit()) {
      auto guard = device.get_kernel("qwen35_ane_commit_guard", library);
      encoder.set_compute_pipeline_state(guard);
      while (!encoder.needs_commit()) {
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
      }
    }
    encoder.add_temporary(std::move(gpu_output));
  }

  DEFINE_NAME(Qwen35AneHybridQ4)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs = static_cast<const AneHybridQ4Primitive &>(other);
    return model_.get() == rhs.model_.get() && bits_ == rhs.bits_ &&
           variant_ == rhs.variant_ &&
           group_size_ == rhs.group_size_ && fuse_swiglu_ == rhs.fuse_swiglu_ &&
           profile_category_ == rhs.profile_category_;
  }
  auto state() const {
    return std::make_tuple(reinterpret_cast<uintptr_t>(model_.get()), bits_,
                           variant_, group_size_, fuse_swiglu_,
                           profile_category_);
  }

private:
  std::shared_ptr<AneLinearModel> model_;
  int variant_;
  int bits_;
  int group_size_;
  bool fuse_swiglu_;
  int profile_category_;
};

class DualAneHybridPrimitive : public Primitive {
public:
  DualAneHybridPrimitive(Stream stream,
                         std::shared_ptr<AneLinearModel> model0,
                         std::shared_ptr<AneLinearModel> model1, int bits,
                         int variant, int group_size, bool fuse_swiglu = false,
                         int profile_category = -1)
      : Primitive(stream), model0_(std::move(model0)),
        model1_(std::move(model1)), variant_(variant), bits_(bits),
        group_size_(group_size), fuse_swiglu_(fuse_swiglu),
        profile_category_(profile_category >= 0 ? profile_category
                                                : (fuse_swiglu ? 0 : 1)) {}

  void eval_cpu(const std::vector<array> &, std::vector<array> &) override {
    throw std::runtime_error("Dual ANE hybrid qmm has no CPU implementation.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    auto &encoder = metal::get_command_encoder(stream);
    const auto &x = inputs[0];
    const auto &weight = inputs[1];
    const auto &scales = inputs[2];
    const auto &biases = inputs[3];
    auto &output = outputs[0];

    const int K = static_cast<int>(x.shape(-1));
    const int M = static_cast<int>(x.size() / K);
    const int gpu_n = static_cast<int>(weight.shape(0));
    const int ane0_n = model0_->output_dim();
    const int ane1_n = model1_->output_dim();
    const bool profiling = ane_profile_enabled();
    const int profile_category = profile_category_;
    const uint64_t operation_start = profiling ? profile_now_ns() : 0;
    if (profiling) {
      profile_add(profile_category, kOperations, 1);
      const uint64_t previous =
          g_previous_ane_done_ns.load(std::memory_order_relaxed);
      if (previous && operation_start > previous) {
        profile_add(profile_category, kGapBeforeNs,
                    operation_start - previous);
      }
    }
    output.set_data(allocator::malloc(output.nbytes()));
    array gpu_output({M, gpu_n}, x.dtype(), nullptr, {});
    gpu_output.set_data(allocator::malloc(gpu_output.nbytes()));

    auto library =
        device.get_library("omlx_qwen35_prefill_kernels", binary_dir());
    auto pack = device.get_kernel(
        "qwen35_ane_pack_input_dual_" + metal_type_name(x.dtype()), library);
    encoder.set_compute_pipeline_state(pack);
    encoder.set_input_array(x, 0);
    encoder.set_buffer(model0_->input_buffer(), 1);
    encoder.set_buffer(model1_->input_buffer(), 2);
    encoder.set_bytes(M, 3);
    encoder.set_bytes(K, 4);
    encoder.dispatch_threads(MTL::Size(K, M, 1), MTL::Size(16, 16, 1));
    encoder.end_encoding();

    auto *producer_buffer = encoder.get_command_buffer();
    producer_buffer->retain();
    auto ticket0 = model0_->begin(producer_buffer);
    auto ticket1 = model1_->begin(producer_buffer);
    encoder.commit();
    producer_buffer->waitUntilCompleted();
    producer_buffer->release();
    if (profiling) {
      profile_add(profile_category, kPackNs,
                  profile_now_ns() - operation_start);
    }

    const std::string qmm_type = metal_type_name(x.dtype());
    auto qmm = [&] {
      if (hybrid_nax_enabled()) {
        try {
          auto nax_library = device.get_library(
              "omlx_qwen35_prefill_kernels_nax", binary_dir());
          return device.get_kernel(
              hybrid_nax_qmm_name(bits_, group_size_, qmm_type), nax_library);
        } catch (const std::exception &) {
          g_hybrid_nax_runtime_ok.store(false, std::memory_order_relaxed);
        }
      }
      return device.get_kernel(
          hybrid_classic_qmm_name(bits_, group_size_, qmm_type), library);
    }();
    encoder.set_compute_pipeline_state(qmm);
    encoder.set_input_array(weight, 0);
    encoder.set_input_array(scales, 1);
    encoder.set_input_array(biases, 2);
    encoder.set_input_array(x, 3);
    encoder.set_output_array(gpu_output, 4);
    encoder.set_bytes(K, 5);
    encoder.set_bytes(gpu_n, 6);
    encoder.set_bytes(M, 7);
    encoder.dispatch_threadgroups(
        MTL::Size((gpu_n + 63) / 64, (M + 63) / 64, 1), MTL::Size(32, 2, 2));
    encoder.end_encoding();
    id<MTLCommandBuffer> qmm_buffer =
        (__bridge id<MTLCommandBuffer>)(static_cast<void *>(
            encoder.get_command_buffer()));
    if (profiling) {
      [qmm_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        const double duration = completed.GPUEndTime - completed.GPUStartTime;
        if (duration > 0) {
          profile_add(profile_category, kGpuQmmNs,
                      static_cast<uint64_t>(duration * 1e9));
        }
      }];
    }
    encoder.commit();
    const uint64_t launch = profiling ? profile_now_ns() : 0;

    auto model0 = model0_;
    auto model1 = model1_;
    std::thread([model0, ticket0, profiling, profile_category, launch] {
      const uint64_t start = profiling ? profile_now_ns() : 0;
      model0->execute(ticket0);
      if (profiling) {
        const uint64_t end = profile_now_ns();
        profile_add(profile_category, kAne0LaunchNs, start - launch);
        profile_add(profile_category, kAne0EvalNs, end - start);
      }
    }).detach();
    std::thread([model1, ticket1, profiling, profile_category, launch] {
      const uint64_t start = profiling ? profile_now_ns() : 0;
      model1->execute(ticket1);
      if (profiling) {
        const uint64_t end = profile_now_ns();
        profile_add(profile_category, kAne1LaunchNs, start - launch);
        profile_add(profile_category, kAne1EvalNs, end - start);
      }
    }).detach();
    model0_->wait(ticket0);
    model1_->wait(ticket1);
    if (profiling) {
      const uint64_t done = profile_now_ns();
      profile_add(profile_category, kAneRegionNs, done - launch);
      g_previous_ane_done_ns.store(done, std::memory_order_relaxed);
      if (qmm_buffer.status == MTLCommandBufferStatusCompleted) {
        profile_add(profile_category, kAneLast, 1);
      } else {
        profile_add(profile_category, kGpuLast, 1);
      }
    }

    auto merge = device.get_kernel(
        std::string(fuse_swiglu_ ? "qwen35_ane_merge_dual_swiglu_output_"
                                 : "qwen35_ane_merge_dual_output_") +
            metal_type_name(x.dtype()),
        library);
    encoder.set_compute_pipeline_state(merge);
    encoder.set_buffer(model0_->output_buffer(), 0);
    encoder.set_buffer(model1_->output_buffer(), 1);
    encoder.set_input_array(gpu_output, 2);
    encoder.set_output_array(output, 3);
    encoder.set_bytes(M, 4);
    const int output_n = fuse_swiglu_ ? (ane0_n + ane1_n + gpu_n) / 2
                                      : ane0_n + ane1_n + gpu_n;
    const int merge_ane0_n = fuse_swiglu_ ? ane0_n / 2 : ane0_n;
    const int merge_ane1_n = fuse_swiglu_ ? ane1_n / 2 : ane1_n;
    const int merge_gpu_n = fuse_swiglu_ ? gpu_n / 2 : gpu_n;
    encoder.set_bytes(merge_ane0_n, 5);
    encoder.set_bytes(merge_ane1_n, 6);
    encoder.set_bytes(merge_gpu_n, 7);
    encoder.dispatch_threads(MTL::Size(output_n, M, 1),
                             MTL::Size(16, 16, 1));

    if (!encoder.needs_commit()) {
      auto guard = device.get_kernel("qwen35_ane_commit_guard", library);
      encoder.set_compute_pipeline_state(guard);
      while (!encoder.needs_commit()) {
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
      }
    }
    encoder.add_temporary(std::move(gpu_output));
  }

  DEFINE_NAME(Qwen35DualAneHybrid)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs = static_cast<const DualAneHybridPrimitive &>(other);
    return model0_.get() == rhs.model0_.get() &&
           model1_.get() == rhs.model1_.get() && bits_ == rhs.bits_ &&
           variant_ == rhs.variant_ && group_size_ == rhs.group_size_ &&
           fuse_swiglu_ == rhs.fuse_swiglu_ &&
           profile_category_ == rhs.profile_category_;
  }
  auto state() const {
    return std::make_tuple(reinterpret_cast<uintptr_t>(model0_.get()),
                           reinterpret_cast<uintptr_t>(model1_.get()), bits_,
                           variant_, group_size_, fuse_swiglu_,
                           profile_category_);
  }

private:
  std::shared_ptr<AneLinearModel> model0_;
  std::shared_ptr<AneLinearModel> model1_;
  int variant_;
  int bits_;
  int group_size_;
  bool fuse_swiglu_;
  int profile_category_;
};

class AneHybridQ4SwiGLUDownPrimitive : public Primitive {
public:
  AneHybridQ4SwiGLUDownPrimitive(Stream stream,
                                 std::shared_ptr<AneLinearModel> model,
                                 int variant, int group_size)
      : Primitive(stream), model_(std::move(model)), variant_(variant),
        group_size_(group_size) {}

  void eval_cpu(const std::vector<array> &, std::vector<array> &) override {
    throw std::runtime_error(
        "ANE hybrid SwiGLU/down has no CPU implementation.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    auto &encoder = metal::get_command_encoder(stream);
    const auto &x = inputs[0];
    const auto &gate_up_weight = inputs[1];
    const auto &gate_up_scales = inputs[2];
    const auto &gate_up_biases = inputs[3];
    const auto &down_weight = inputs[4];
    const auto &down_scales = inputs[5];
    const auto &down_biases = inputs[6];
    auto &output = outputs[0];

    const int input_dim = static_cast<int>(x.shape(-1));
    const int M = static_cast<int>(x.size() / input_dim);
    const int gpu_hidden = static_cast<int>(gate_up_weight.shape(0)) / 2;
    const int output_dim = model_->output_dim();
    output.set_data(allocator::malloc(output.nbytes()));
    array gate_up({M, 2 * gpu_hidden}, x.dtype(), nullptr, {});
    array activation({M, gpu_hidden}, x.dtype(), nullptr, {});
    array gpu_output({M, output_dim}, x.dtype(), nullptr, {});
    gate_up.set_data(allocator::malloc(gate_up.nbytes()));
    activation.set_data(allocator::malloc(activation.nbytes()));
    gpu_output.set_data(allocator::malloc(gpu_output.nbytes()));

    auto library =
        device.get_library("omlx_qwen35_prefill_kernels", binary_dir());
    auto pack = device.get_kernel(
        "qwen35_ane_pack_input_" + metal_type_name(x.dtype()), library);
    encoder.set_compute_pipeline_state(pack);
    encoder.set_input_array(x, 0);
    encoder.set_buffer(model_->input_buffer(), 1);
    encoder.set_bytes(M, 2);
    encoder.set_bytes(input_dim, 3);
    encoder.dispatch_threads(MTL::Size(input_dim, M, 1),
                             MTL::Size(16, 16, 1));
    encoder.end_encoding();

    auto *producer_buffer = encoder.get_command_buffer();
    producer_buffer->retain();
    auto ticket = model_->begin(producer_buffer);
    encoder.commit();
    producer_buffer->waitUntilCompleted();
    producer_buffer->release();

    std::string qmm_name = "qwen35_q4_affine_qmm128_t_" +
                           metal_type_name(x.dtype()) + "_bm_64_bk_32_bn_64";
    auto qmm = device.get_kernel(qmm_name, library);
    encoder.set_compute_pipeline_state(qmm);
    encoder.set_input_array(gate_up_weight, 0);
    encoder.set_input_array(gate_up_scales, 1);
    encoder.set_input_array(gate_up_biases, 2);
    encoder.set_input_array(x, 3);
    encoder.set_output_array(gate_up, 4);
    encoder.set_bytes(input_dim, 5);
    encoder.set_bytes(2 * gpu_hidden, 6);
    encoder.set_bytes(M, 7);
    encoder.dispatch_threadgroups(
        MTL::Size((2 * gpu_hidden + 63) / 64, (M + 63) / 64, 1),
        MTL::Size(32, 2, 2));
    encoder.end_encoding();

    auto swiglu = device.get_kernel(
        "qwen35_ane_swiglu_suffix_" + metal_type_name(x.dtype()), library);
    encoder.set_compute_pipeline_state(swiglu);
    encoder.set_input_array(gate_up, 0);
    encoder.set_output_array(activation, 1);
    encoder.set_bytes(M, 2);
    encoder.set_bytes(gpu_hidden, 3);
    encoder.dispatch_threads(MTL::Size(gpu_hidden, M, 1),
                             MTL::Size(16, 16, 1));
    encoder.end_encoding();

    encoder.set_compute_pipeline_state(qmm);
    encoder.set_input_array(down_weight, 0);
    encoder.set_input_array(down_scales, 1);
    encoder.set_input_array(down_biases, 2);
    encoder.set_input_array(activation, 3);
    encoder.set_output_array(gpu_output, 4);
    encoder.set_bytes(gpu_hidden, 5);
    encoder.set_bytes(output_dim, 6);
    encoder.set_bytes(M, 7);
    encoder.dispatch_threadgroups(
        MTL::Size((output_dim + 63) / 64, (M + 63) / 64, 1),
        MTL::Size(32, 2, 2));
    encoder.end_encoding();

    encoder.commit();
    auto model = model_;
    std::thread([model = std::move(model), ticket] {
      model->execute(ticket);
    }).detach();
    model_->wait(ticket);

    auto sum = device.get_kernel(
        "qwen35_ane_sum_output_" + metal_type_name(x.dtype()), library);
    encoder.set_compute_pipeline_state(sum);
    encoder.set_buffer(model_->output_buffer(), 0);
    encoder.set_input_array(gpu_output, 1);
    encoder.set_output_array(output, 2);
    encoder.set_bytes(M, 3);
    encoder.set_bytes(output_dim, 4);
    encoder.dispatch_threads(MTL::Size(output_dim, M, 1),
                             MTL::Size(16, 16, 1));

    if (!encoder.needs_commit()) {
      auto guard = device.get_kernel("qwen35_ane_commit_guard", library);
      encoder.set_compute_pipeline_state(guard);
      while (!encoder.needs_commit()) {
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
      }
    }
    encoder.add_temporary(std::move(gate_up));
    encoder.add_temporary(std::move(activation));
    encoder.add_temporary(std::move(gpu_output));
  }

  DEFINE_NAME(Qwen35AneHybridQ4SwiGLUDown)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs =
        static_cast<const AneHybridQ4SwiGLUDownPrimitive &>(other);
    return model_.get() == rhs.model_.get() && variant_ == rhs.variant_ &&
           group_size_ == rhs.group_size_;
  }
  auto state() const {
    return std::make_tuple(reinterpret_cast<uintptr_t>(model_.get()), variant_,
                           group_size_);
  }

private:
  std::shared_ptr<AneLinearModel> model_;
  int variant_;
  int group_size_;
};

} // namespace

array qwen35_ane_affine_qmm_t(
    const array &x, const array &gpu_weight, const array &gpu_scales,
    const array &gpu_biases, const std::shared_ptr<AneLinearModel> &ane_model,
    int bits, int variant, int group_size, int profile_category,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!ane_model || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || gpu_weight.dtype() != mlx::core::uint32 ||
      gpu_scales.dtype() != x.dtype() || gpu_biases.dtype() != x.dtype() ||
      gpu_weight.ndim() != 2 || gpu_scales.ndim() != 2 ||
      gpu_biases.shape() != gpu_scales.shape() || !row_contiguous(gpu_weight) ||
      !row_contiguous(gpu_scales) || !row_contiguous(gpu_biases) ||
      (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (group_size != 64 && group_size != 128) || variant != 8) {
    throw std::invalid_argument("Unsupported ANE hybrid qmm configuration.");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / K);
  const int gpu_n = static_cast<int>(gpu_weight.shape(0));
  if (K != ane_model->input_dim() || M != ane_model->sequence_length() ||
      gpu_n <= 0 || gpu_n % 64 != 0 || K % group_size != 0 ||
      gpu_weight.shape(1) * 32 != K * bits ||
      gpu_scales.shape(0) != gpu_n ||
      gpu_scales.shape(1) != K / group_size) {
    throw std::invalid_argument("ANE hybrid qmm shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = ane_model->output_dim() + gpu_n;
  return array(std::move(shape), x.dtype(),
               std::make_shared<AneHybridQ4Primitive>(
                   stream, ane_model, bits, variant, group_size,
                   /* fuse_swiglu */ false, profile_category),
               std::vector<array>{x, gpu_weight, gpu_scales, gpu_biases});
}

array qwen35_ane_q4_affine_qmm_t(
    const array &x, const array &gpu_weight, const array &gpu_scales,
    const array &gpu_biases, const std::shared_ptr<AneLinearModel> &ane_model,
    int variant, int group_size, int profile_category, StreamOrDevice s) {
  return qwen35_ane_affine_qmm_t(x, gpu_weight, gpu_scales, gpu_biases,
                                 ane_model, 4, variant, group_size,
                                 profile_category, s);
}

array qwen35_ane_affine_swiglu_t(
    const array &x, const array &gpu_weight, const array &gpu_scales,
    const array &gpu_biases, const std::shared_ptr<AneLinearModel> &ane_model,
    int bits, int variant, int group_size, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!ane_model || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || gpu_weight.dtype() != mlx::core::uint32 ||
      gpu_scales.dtype() != x.dtype() || gpu_biases.dtype() != x.dtype() ||
      gpu_weight.ndim() != 2 || gpu_scales.ndim() != 2 ||
      gpu_biases.shape() != gpu_scales.shape() || !row_contiguous(gpu_weight) ||
      !row_contiguous(gpu_scales) || !row_contiguous(gpu_biases) ||
      (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (group_size != 64 && group_size != 128) || variant != 8) {
    throw std::invalid_argument(
        "Unsupported ANE hybrid affine SwiGLU configuration.");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / K);
  const int gpu_n = static_cast<int>(gpu_weight.shape(0));
  const int ane_n = ane_model->output_dim();
  if (K != ane_model->input_dim() || M != ane_model->sequence_length() ||
      ane_n <= 0 || ane_n % 2 != 0 || gpu_n <= 0 || gpu_n % 128 != 0 ||
      K % group_size != 0 || gpu_weight.shape(1) * 32 != K * bits ||
      gpu_scales.shape(0) != gpu_n ||
      gpu_scales.shape(1) != K / group_size) {
    throw std::invalid_argument("ANE hybrid affine SwiGLU shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = (ane_n + gpu_n) / 2;
  return array(std::move(shape), x.dtype(),
               std::make_shared<AneHybridQ4Primitive>(
                   stream, ane_model, bits, variant, group_size, true),
               std::vector<array>{x, gpu_weight, gpu_scales, gpu_biases});
}

array qwen35_ane_q4_swiglu_t(
    const array &x, const array &gpu_weight, const array &gpu_scales,
    const array &gpu_biases, const std::shared_ptr<AneLinearModel> &ane_model,
    int variant, int group_size, StreamOrDevice s) {
  return qwen35_ane_affine_swiglu_t(x, gpu_weight, gpu_scales, gpu_biases,
                                    ane_model, 4, variant, group_size, s);
}

array qwen35_ane_dual_affine_qmm_t(
    const array &x, const array &gpu_weight, const array &gpu_scales,
    const array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int bits, int variant,
    int group_size, int profile_category, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!ane_model0 || !ane_model1 || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || gpu_weight.dtype() != mlx::core::uint32 ||
      gpu_scales.dtype() != x.dtype() || gpu_biases.dtype() != x.dtype() ||
      gpu_weight.ndim() != 2 || gpu_scales.ndim() != 2 ||
      gpu_biases.shape() != gpu_scales.shape() || !row_contiguous(gpu_weight) ||
      !row_contiguous(gpu_scales) || !row_contiguous(gpu_biases) ||
      (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (group_size != 64 && group_size != 128) || variant != 8) {
    throw std::invalid_argument(
        "Unsupported dual ANE hybrid qmm configuration.");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / K);
  const int gpu_n = static_cast<int>(gpu_weight.shape(0));
  if (K != ane_model0->input_dim() || K != ane_model1->input_dim() ||
      M != ane_model0->sequence_length() ||
      M != ane_model1->sequence_length() || gpu_n <= 0 || gpu_n % 64 != 0 ||
      K % group_size != 0 || gpu_weight.shape(1) * 32 != K * bits ||
      gpu_scales.shape(0) != gpu_n ||
      gpu_scales.shape(1) != K / group_size) {
    throw std::invalid_argument("Dual ANE hybrid qmm shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() =
      ane_model0->output_dim() + ane_model1->output_dim() + gpu_n;
  return array(
      std::move(shape), x.dtype(),
      std::make_shared<DualAneHybridPrimitive>(
          stream, ane_model0, ane_model1, bits, variant, group_size,
          /* fuse_swiglu */ false, profile_category),
      std::vector<array>{x, gpu_weight, gpu_scales, gpu_biases});
}

array qwen35_ane_dual_affine_swiglu_t(
    const array &x, const array &gpu_weight, const array &gpu_scales,
    const array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int bits, int variant,
    int group_size, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!ane_model0 || !ane_model1 || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || gpu_weight.dtype() != mlx::core::uint32 ||
      gpu_scales.dtype() != x.dtype() || gpu_biases.dtype() != x.dtype() ||
      gpu_weight.ndim() != 2 || gpu_scales.ndim() != 2 ||
      gpu_biases.shape() != gpu_scales.shape() || !row_contiguous(gpu_weight) ||
      !row_contiguous(gpu_scales) || !row_contiguous(gpu_biases) ||
      (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (group_size != 64 && group_size != 128) || variant != 8) {
    throw std::invalid_argument(
        "Unsupported dual ANE affine SwiGLU configuration.");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / K);
  const int gpu_n = static_cast<int>(gpu_weight.shape(0));
  const int ane0_n = ane_model0->output_dim();
  const int ane1_n = ane_model1->output_dim();
  if (K != ane_model0->input_dim() ||
      K != ane_model1->input_dim() || M != ane_model0->sequence_length() ||
      M != ane_model1->sequence_length() || ane0_n <= 0 || ane1_n <= 0 ||
      ane0_n % 2 != 0 || ane1_n % 2 != 0 || gpu_n <= 0 ||
      gpu_n % 128 != 0 || K % group_size != 0 ||
      gpu_weight.shape(1) * 32 != K * bits ||
      gpu_scales.shape(0) != gpu_n ||
      gpu_scales.shape(1) != K / group_size) {
    throw std::invalid_argument(
        "Dual ANE hybrid affine SwiGLU shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = (ane0_n + ane1_n + gpu_n) / 2;
  return array(
      std::move(shape), x.dtype(),
      std::make_shared<DualAneHybridPrimitive>(
          stream, ane_model0, ane_model1, bits, variant, group_size, true),
      std::vector<array>{x, gpu_weight, gpu_scales, gpu_biases});
}

array qwen35_ane_dual_q4_swiglu_t(
    const array &x, const array &gpu_weight, const array &gpu_scales,
    const array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int variant,
    int group_size, StreamOrDevice s) {
  return qwen35_ane_dual_affine_swiglu_t(
      x, gpu_weight, gpu_scales, gpu_biases, ane_model0, ane_model1, 4,
      variant, group_size, s);
}

array qwen35_ane_q4_swiglu_down_t(
    const array &x, const array &gpu_gate_up_weight,
    const array &gpu_gate_up_scales, const array &gpu_gate_up_biases,
    const array &gpu_down_weight, const array &gpu_down_scales,
    const array &gpu_down_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int variant,
    int group_size, StreamOrDevice s) {
  auto stream = to_stream(s);
  const auto q4_valid = [&](const array &weight, const array &scales,
                            const array &biases, int input_dim) {
    return weight.dtype() == mlx::core::uint32 && scales.dtype() == x.dtype() &&
           biases.dtype() == x.dtype() && weight.ndim() == 2 &&
           scales.ndim() == 2 && biases.shape() == scales.shape() &&
           row_contiguous(weight) && row_contiguous(scales) &&
           row_contiguous(biases) && weight.shape(1) * 8 == input_dim &&
           scales.shape(0) == weight.shape(0) &&
           scales.shape(1) == input_dim / 128;
  };
  if (!ane_model || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || group_size != 128 || variant != 8) {
    throw std::invalid_argument(
        "Unsupported ANE hybrid SwiGLU/down configuration.");
  }
  const int input_dim = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / input_dim);
  const int gate_up_n = static_cast<int>(gpu_gate_up_weight.shape(0));
  const int gpu_hidden = gate_up_n / 2;
  const int output_dim = static_cast<int>(gpu_down_weight.shape(0));
  if (input_dim != ane_model->input_dim() ||
      output_dim != ane_model->output_dim() ||
      M != ane_model->sequence_length() || gate_up_n <= 0 ||
      gate_up_n % 128 != 0 || gpu_hidden % 128 != 0 ||
      output_dim % 64 != 0 ||
      !q4_valid(gpu_gate_up_weight, gpu_gate_up_scales, gpu_gate_up_biases,
                input_dim) ||
      !q4_valid(gpu_down_weight, gpu_down_scales, gpu_down_biases,
                gpu_hidden)) {
    throw std::invalid_argument("ANE hybrid SwiGLU/down shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = output_dim;
  return array(
      std::move(shape), x.dtype(),
      std::make_shared<AneHybridQ4SwiGLUDownPrimitive>(
          stream, ane_model, variant, group_size),
      std::vector<array>{x, gpu_gate_up_weight, gpu_gate_up_scales,
                         gpu_gate_up_biases, gpu_down_weight, gpu_down_scales,
                         gpu_down_biases});
}

} // namespace omlx::qwen35_prefill_kernels
