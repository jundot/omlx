#include "qwen35_ane.h"
#include "qwen35_prefill.h"

#import <Accelerate/Accelerate.h>
#import <dispatch/dispatch.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

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
#include <memory>
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
  kGpuCompletionNs,
  kCpuMatmulNs,
  kCpuCompletionNs,
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

// Opt-in persistent compile cache for private-ANE programs. The default
// (unset or any value other than exactly "1") keeps the historical
// temp-directory path that is deleted when the program unloads.
bool ane_compile_cache_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("OMLX_QWEN35_ANE_COMPILE_CACHE");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

NSString *ane_compile_cache_root_directory() {
  NSArray *roots = NSSearchPathForDirectoriesInDomains(
      NSCachesDirectory, NSUserDomainMask, YES);
  NSString *root = roots.firstObject ?: NSTemporaryDirectory();
  return [[[root stringByAppendingPathComponent:@"omlx"]
      stringByAppendingPathComponent:@"ane"]
      stringByAppendingPathComponent:@"v1"];
}

// ~/Library/Caches/omlx/ane/v1/<os-build>/<identifier>. The private ANE
// runtime keys its compiled-program cache by this descriptor content hash.
// oMLX only needs a matching cross-process lock; it must not assign this path
// as the in-memory model URL (#3124).
NSString *ane_compile_cache_lock_entry(NSString *identifier) {
  NSString *os_build =
      [[NSProcessInfo processInfo] operatingSystemVersionString];
  os_build =
      [os_build stringByReplacingOccurrencesOfString:@"/" withString:@"_"];
  os_build =
      [os_build stringByReplacingOccurrencesOfString:@":" withString:@"_"];
  return [[ane_compile_cache_root_directory()
      stringByAppendingPathComponent:os_build]
      stringByAppendingPathComponent:identifier];
}

std::string error_text(NSString *prefix, NSError *error);

// A suspended process holding the entry lock must not hang later loads of the
// same identifier forever. Acquisition is bounded: the lock is taken
// non-blocking and retried until the deadline, then the constructor throws so
// the caller fails open to the historical temp-directory path.
constexpr std::chrono::milliseconds kAneCacheLockTimeout{30000};
constexpr std::chrono::milliseconds kAneCacheLockRetry{50};

// One stable rendezvous file per cache entry, beside the entry directory.
// Never unlink this path while the process is running: a waiter may already
// hold the old inode open, and recreating the path would let a later process
// lock a different inode and enter the same staging directory concurrently.
NSString *ane_compile_cache_lock_path(NSString *entry_directory) {
  return [entry_directory stringByAppendingString:@".lock"];
}

class ScopedAneCacheLock {
public:
  explicit ScopedAneCacheLock(NSString *entry_directory) {
    NSString *parent = [entry_directory stringByDeletingLastPathComponent];
    NSError *error = nil;
    if (![[NSFileManager defaultManager]
            createDirectoryAtPath:parent
      withIntermediateDirectories:YES
                       attributes:nil
                            error:&error]) {
      throw std::runtime_error(
          error_text(@"ANE compile cache parent creation failed", error));
    }
    NSString *lock_path = ane_compile_cache_lock_path(entry_directory);
    fd_ = open(lock_path.fileSystemRepresentation, O_CREAT | O_RDWR, 0600);
    if (fd_ < 0) {
      throw std::runtime_error("ANE compile cache lock open failed.");
    }
    const auto deadline = std::chrono::steady_clock::now() + kAneCacheLockTimeout;
    for (;;) {
      if (flock(fd_, LOCK_EX | LOCK_NB) == 0) {
        return;
      }
      if (errno != EWOULDBLOCK && errno != EAGAIN) {
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        break;
      }
      std::this_thread::sleep_for(kAneCacheLockRetry);
    }
    close(fd_);
    fd_ = -1;
    throw std::runtime_error("ANE compile cache lock acquisition timed out.");
  }


  ~ScopedAneCacheLock() {
    if (fd_ >= 0) {
      flock(fd_, LOCK_UN);
      close(fd_);
    }
  }

  ScopedAneCacheLock(const ScopedAneCacheLock &) = delete;
  ScopedAneCacheLock &operator=(const ScopedAneCacheLock &) = delete;

private:
  int fd_ = -1;
};

std::string error_text(NSString *prefix, NSError *error) {
  NSString *value =
      error ? [NSString stringWithFormat:@"%@: %@", prefix, error] : prefix;
  return std::string([value UTF8String]);
}

void remove_ane_staging_directory(NSString *directory,
                                  NSString *cache_lock_entry) noexcept {
  if (directory == nil) {
    return;
  }
  if (cache_lock_entry == nil) {
    [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
    return;
  }
  try {
    // Compile/load and cleanup mutate the same historical temp path. Keep
    // cleanup in the descriptor-hash lock domain used by cache-enabled loads.
    ScopedAneCacheLock cache_lock(cache_lock_entry);
    [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
  } catch (const std::exception &failure) {
    // Cleanup must not make model teardown fail. Leaving staging behind is
    // safer than deleting another process's in-progress compile inputs.
    NSLog(@"oMLX: ANE compile cache cleanup deferred (%s)", failure.what());
  }
}

struct AneLoadResult {
  NSString *staging_directory;
  NSString *cache_lock_entry;
};

// Cache-first load for one private-ANE program. The in-memory model already
// derives its model URL and compiled-cache key from the descriptor content
// hash. Overriding that URL breaks the per-file bundle hash verification added
// by macOS 27, so both cache modes retain the historical temporary staging
// path. With caching enabled, compiledModelExists probes Apple's persistent
// cache directly; a hit skips materialization and compilation. A restored
// program whose load fails is purged and recompiled exactly once. oMLX owns
// only temporary MIL and weight inputs, deleting them on unload as before.
AneLoadResult
load_or_compile_ane_model(id model, NSString *identifier, int ane_instance,
                          NSDictionary *execution_options, NSData *mil,
                          NSDictionary<NSString *, NSData *> *weight_files,
                          NSString *compile_what, NSString *load_what) {
  NSString *directory =
      [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
  NSString *cache_lock_entry = nil;
  bool restored = false;
  bool staged = false;
  std::unique_ptr<ScopedAneCacheLock> cache_lock;
  if (ane_compile_cache_enabled()) {
    try {
      cache_lock_entry = ane_compile_cache_lock_entry(identifier);
      // Hold the descriptor-hash lock from probe through load so a second
      // process compiling the same program waits instead of racing writes to
      // the framework-selected temporary staging path.
      cache_lock = std::make_unique<ScopedAneCacheLock>(cache_lock_entry);
      SEL exists_selector = @selector(compiledModelExists);
      restored = [model respondsToSelector:exists_selector] &&
                 ((BOOL (*)(id, SEL))objc_msgSend)(model, exists_selector);
      if (restored) {
        NSLog(@"oMLX: ANE compile cache hit identifier=%@ instance=%d",
              identifier, ane_instance);
      } else {
        NSLog(@"oMLX: ANE compile cache miss identifier=%@ instance=%d",
              identifier, ane_instance);
      }
    } catch (const std::exception &failure) {
      // Fail open: the cache only accelerates loads. An unusable root or
      // lock must degrade to the historical temp path, never fail the load.
      NSLog(@"oMLX: ANE compile cache unavailable (%s), using a temporary "
            @"directory",
            failure.what());
      cache_lock.reset();
      cache_lock_entry = nil;
      restored = false;
    }
  }
  auto compile_fresh = [&]() {
    // Clear any stale or half-written entry first: a crash mid-compile can
    // leave incomplete staging inputs, and they must never be reused.
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
    for (NSString *name in weight_files) {
      [weight_files[name]
          writeToFile:[weight_directory stringByAppendingPathComponent:name]
               atomically:YES];
    }
    NSError *error = nil;
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21,
        execution_options, &error);
    if (!ok) {
      [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
      throw std::runtime_error(error_text(
          [compile_what stringByAppendingString:@" compilation failed"],
          error));
    }
    staged = true;
  };

  if (!restored) {
    compile_fresh();
  }
  NSError *error = nil;
  BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
      model, @selector(loadWithQoS:options:error:), 21, execution_options,
      &error);
  if (!ok && restored) {
    // Corrupt or incompatible native entry: purge it before compiling exactly
    // once more, otherwise compiledModelExists would keep returning the same
    // unusable artifact.
    NSLog(@"oMLX: ANE compile cache fallback identifier=%@ instance=%d",
          identifier, ane_instance);
    SEL purge_selector = @selector(purgeCompiledModel);
    if ([model respondsToSelector:purge_selector]) {
      ((void (*)(id, SEL))objc_msgSend)(model, purge_selector);
    }
    compile_fresh();
    ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, execution_options,
        &error);
  }
  if (!ok) {
    [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
    throw std::runtime_error(error_text(
        [load_what stringByAppendingString:@" load failed"], error));
  }
  return {
      staged ? directory : nil,
      staged ? cache_lock_entry : nil,
  };
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
                               int sequence_length, uint64_t gate_offset,
                               uint64_t up_offset, uint64_t down_offset) {
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
           "\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
           "    tensor<fp16, [%d, %d, 1, 1]> uw = const()[name=string(\"uw\"), "
           "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string("
           "\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
           "    tensor<fp16, [%d, %d, 1, 1]> dw = const()[name=string(\"dw\"), "
           "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string("
           "\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
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
           "    tensor<fp16, [1, %d, 1, %d]> silu_out = silu(x=gate)"
           "[name=string(\"silu\")];\n"
           "    tensor<fp16, [1, %d, 1, %d]> act = mul(x=silu_out, y=up)"
           "[name=string(\"swiglu\")];\n"
           "    tensor<fp16, [1, %d, 1, %d]> y = conv(dilations=dl, "
           "groups=gr, pad=pd, pad_type=pt, strides=st, weight=dw, x=act)"
           "[name=string(\"down\")];\n"
           "  } -> (y);\n}\n",
          input_dim, sequence_length,
          hidden_dim, input_dim, hidden_dim, input_dim,
          static_cast<unsigned long long>(gate_offset),
          hidden_dim, input_dim, hidden_dim, input_dim,
          static_cast<unsigned long long>(up_offset),
          output_dim, hidden_dim, output_dim, hidden_dim,
          static_cast<unsigned long long>(down_offset),
          hidden_dim, sequence_length, hidden_dim, sequence_length,
          hidden_dim, sequence_length, hidden_dim, sequence_length,
          output_dim, sequence_length];
}

struct SwiGLUDownShape {
  int input_dim;
  int hidden_dim;
  int output_dim;
};

NSString *fp16_swiglu_down_bank_mil(
    const std::vector<SwiGLUDownShape> &shapes,
    const std::vector<std::array<uint64_t, 3>> &offsets,
    int sequence_length) {
  NSMutableString *mil = [NSMutableString
      stringWithString:
          @"program(1.3)\n"
           "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", "
           "\"3520.4.1\"}, {\"coremlc-version\", \"3520.5.1\"}})]\n{\n"];
  for (size_t index = 0; index < shapes.size(); ++index) {
    const auto &shape = shapes[index];
    const auto &weight_offsets = offsets[index];
    [mil appendFormat:
             @"  func procedure%03zu<ios18>(tensor<fp16, [1, %d, 1, %d]> x) "
              "{\n"
              "    tensor<fp16, [%d, %d, 1, 1]> gw = const()[name=string("
              "\"gw\"), val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string("
              "\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
              "    tensor<fp16, [%d, %d, 1, 1]> uw = const()[name=string("
              "\"uw\"), val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string("
              "\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
              "    tensor<fp16, [%d, %d, 1, 1]> dw = const()[name=string("
              "\"dw\"), val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string("
              "\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
              "    string pt = const()[name=string(\"pt\"), "
              "val=string(\"valid\")];\n"
              "    tensor<int32, [2]> st = const()[name=string(\"st\"), "
              "val=tensor<int32, [2]>([1,1])];\n"
              "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), "
              "val=tensor<int32, [4]>([0,0,0,0])];\n"
              "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), "
              "val=tensor<int32, [2]>([1,1])];\n"
              "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
              "    tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations=dl, "
              "groups=gr, pad=pd, pad_type=pt, strides=st, weight=gw, x=x);\n"
              "    tensor<fp16, [1, %d, 1, %d]> up = conv(dilations=dl, "
              "groups=gr, pad=pd, pad_type=pt, strides=st, weight=uw, x=x);\n"
              "    tensor<fp16, [1, %d, 1, %d]> silu_out = silu(x=gate);\n"
              "    tensor<fp16, [1, %d, 1, %d]> act = mul(x=silu_out, y=up);\n"
              "    tensor<fp16, [1, %d, 1, %d]> y = conv(dilations=dl, "
              "groups=gr, pad=pd, pad_type=pt, strides=st, weight=dw, x=act);\n"
              "  } -> (y);\n",
         index, shape.input_dim, sequence_length,
         shape.hidden_dim, shape.input_dim, shape.hidden_dim, shape.input_dim,
         static_cast<unsigned long long>(weight_offsets[0]),
         shape.hidden_dim, shape.input_dim, shape.hidden_dim, shape.input_dim,
         static_cast<unsigned long long>(weight_offsets[1]),
         shape.output_dim, shape.hidden_dim, shape.output_dim, shape.hidden_dim,
         static_cast<unsigned long long>(weight_offsets[2]),
         shape.hidden_dim, sequence_length, shape.hidden_dim, sequence_length,
         shape.hidden_dim, sequence_length, shape.hidden_dim, sequence_length,
         shape.output_dim, sequence_length];
  }
  [mil appendString:@"}\n"];
  return mil;
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
  SharedAneProgram(id model, const AneLoadResult &load_result,
                   NSDictionary *options)
      : model_([model retain]),
        directory_([load_result.staging_directory copy]),
        cache_lock_entry_([load_result.cache_lock_entry copy]),
        execution_options_([options copy]) {}

  ~SharedAneProgram() {
    @autoreleasepool {
      if (model_) {
        NSError *error = nil;
        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model_, @selector(unloadWithQoS:error:), 21, &error);
      }
      remove_ane_staging_directory(directory_, cache_lock_entry_);
      [model_ release];
      [directory_ release];
      [cache_lock_entry_ release];
      [execution_options_ release];
    }
  }

  id model_;
  NSString *directory_;
  NSString *cache_lock_entry_;
  NSDictionary *execution_options_;
  std::mutex evaluation_mutex_;
};

// Ceiling on every wait for an ANE signal on the live request path. A healthy
// readiness spin resolves in microseconds (the producer buffer is committed
// and waited on before the evaluation thread is spawned) and a healthy
// evaluation in milliseconds-to-seconds (first-eval JIT), but a wedged ANE
// driver never signals at all (#2974) — without a bound, one lost signal
// spins a core forever and hangs the server. OMLX_ANE_WAIT_TIMEOUT_S
// overrides the 30s default; 0 restores the old unbounded behavior.
static std::chrono::seconds ane_wait_timeout() {
  static const std::chrono::seconds timeout = [] {
    if (const char *raw = std::getenv("OMLX_ANE_WAIT_TIMEOUT_S")) {
      char *end = nullptr;
      const long parsed = std::strtol(raw, &end, 10);
      if (end != raw) {
        if (parsed <= 0) {
          return std::chrono::seconds(std::chrono::hours(24 * 365));
        }
        return std::chrono::seconds(parsed);
      }
    }
    return std::chrono::seconds(30);
  }();
  return timeout;
}

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
      AneLoadResult load_result = load_or_compile_ane_model(
          model, identifier, ane_instance, execution_options_, mil, @{
            @"weight_data.bin" : data_blob,
            @"weight_scale.bin" : scale_blob,
          },
          @"ANE", @"ANE model");
      directory_ = [load_result.staging_directory copy];
      cache_lock_entry_ = [load_result.cache_lock_entry copy];
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
       int sequence_length, int ane_instance)
      : input_dim_(input_dim), output_dim_(output_dim),
        sequence_length_(sequence_length) {
    @autoreleasepool {
      load_ane_framework();
      execution_options_ = [ane_execution_options(ane_instance) copy];
      auto gate = fp16_rows(gate_weight, hidden_dim, input_dim);
      auto up = fp16_rows(up_weight, hidden_dim, input_dim);
      auto down = fp16_rows(down_weight, output_dim, hidden_dim);
      NSMutableData *weight_blob = [NSMutableData dataWithLength:64];
      const uint64_t gate_offset = append_blob_chunk(
          weight_blob, gate.data(), gate.size() * sizeof(gate.front()), 1);
      const uint64_t up_offset = append_blob_chunk(
          weight_blob, up.data(), up.size() * sizeof(up.front()), 1);
      const uint64_t down_offset = append_blob_chunk(
          weight_blob, down.data(), down.size() * sizeof(down.front()), 1);
      auto *weight_header = static_cast<uint32_t *>(weight_blob.mutableBytes);
      weight_header[0] = 3;
      weight_header[1] = 2;
      NSData *mil = [fp16_swiglu_down_mil(
                          input_dim, hidden_dim, output_dim, sequence_length,
                          gate_offset, up_offset, down_offset)
          dataUsingEncoding:NSUTF8StringEncoding];
      NSDictionary *weights = @{
        @"@model_path/weights/weight.bin" :
            @{ @"offset" : @0, @"data" : weight_blob },
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
      AneLoadResult load_result = load_or_compile_ane_model(
          model, identifier, ane_instance, @{}, mil, @{
            @"weight.bin" : weight_blob,
          },
          @"ANE", @"ANE model");
      directory_ = [load_result.staging_directory copy];
      cache_lock_entry_ = [load_result.cache_lock_entry copy];

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
      remove_ane_staging_directory(directory_, cache_lock_entry_);
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
      [cache_lock_entry_ release];
    }
  }

  bool has_error() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return !completion_error_.empty();
  }

  AneLinearModel::Ticket begin(MTL::CommandBuffer *command_buffer) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    if (!completion_error_.empty()) {
      throw std::runtime_error("Prior ANE evaluation failed: " +
                               completion_error_);
    }
    if (submitted_ != completed_) {
      // A pending evaluation can be legitimately in flight for a moment
      // (another request aborted between spawn and wait); give it one bounded
      // grace period, then latch — a counter stuck past the timeout is the
      // wedged-driver state and must not stay indistinguishable from a race.
      const bool settled = completion_cv_.wait_for(
          lock, ane_wait_timeout(),
          [this] { return completed_ >= submitted_; });
      if (!settled) {
        if (completion_error_.empty()) {
          completion_error_ = "a prior ANE evaluation never completed";
        }
        throw std::runtime_error("Prior ANE evaluation failed: " +
                                 completion_error_);
      }
      if (!completion_error_.empty()) {
        throw std::runtime_error("Prior ANE evaluation failed: " +
                                 completion_error_);
      }
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
    const auto deadline = std::chrono::steady_clock::now() + ane_wait_timeout();
    while ([event_ signaledValue] < ticket.ready) {
      if (std::chrono::steady_clock::now() >= deadline) {
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          completion_error_ =
              "ANE readiness signal timed out; the producer command buffer "
              "never signaled the shared event";
          ++completed_;
        }
        [event_ setSignaledValue:ticket.done];
        completion_cv_.notify_all();
        return;
      }
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

  void cancel_ticket(AneLinearModel::Ticket ticket) {
    // The ticket's producer command buffer failed, so its encoded ready
    // signal will never fire. Retire the ticket before any evaluation thread
    // is spawned: balance the counters, publish the done value, and wake any
    // waiter. This is a per-submission failure (typically a request abort
    // tearing down Metal state mid-prefill), not a program fault, so
    // completion_error_ is left clear and the next request may use this
    // program again.
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      ++completed_;
    }
    [event_ setSignaledValue:ticket.done];
    completion_cv_.notify_all();
  }

  void wait(AneLinearModel::Ticket) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    const bool finished = completion_cv_.wait_for(
        lock, ane_wait_timeout(), [this] { return completed_ >= submitted_; });
    if (!finished) {
      // The evaluation thread is parked inside the private ANE framework and
      // cannot be unblocked from here; latch the error so begin() refuses
      // further dispatches to this program and the per-module fallback takes
      // over, instead of the caller sleeping forever.
      if (completion_error_.empty()) {
        completion_error_ = "ANE evaluation timed out; the driver never "
                            "signaled completion for this program";
      }
      throw std::runtime_error(completion_error_);
    }
    if (!completion_error_.empty()) {
      throw std::runtime_error(completion_error_);
    }
  }

  void warmup() {
    AneLinearModel::Ticket ticket{};
    {
      std::unique_lock<std::mutex> lock(state_mutex_);
      const bool idle = completion_cv_.wait_for(
          lock, ane_wait_timeout(),
          [this] { return submitted_ == completed_; });
      if (!idle) {
        throw std::runtime_error(
            "Prior ANE evaluation never completed; refusing to warm this "
            "program");
      }
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
  NSString *cache_lock_entry_ = nil;
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
bool AneLinearModel::has_error() { return impl_->has_error(); }
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
void AneLinearModel::cancel_ticket(AneLinearModel::Ticket ticket) {
  impl_->cancel_ticket(ticket);
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

// A pack-and-signal buffer that completed with an error never fired its
// encoded ready signal, so a detached evaluation thread launched for its
// ticket would spin on a value that never arrives (#3025: a request abort
// mid-prefill tears down Metal state and fails the buffer). The dispatch
// sites check this after waitUntilCompleted() and, on failure, retire the
// tickets and fail the dispatch immediately instead of leaving the wait
// timeout as the only backstop.
static bool pack_buffer_failed(MTL::CommandBuffer *producer_buffer) {
  id<MTLCommandBuffer> buffer =
      (__bridge id<MTLCommandBuffer>)(static_cast<void *>(producer_buffer));
  return buffer.status == MTLCommandBufferStatusError;
}

static std::string pack_buffer_error(MTL::CommandBuffer *producer_buffer) {
  id<MTLCommandBuffer> buffer =
      (__bridge id<MTLCommandBuffer>)(static_cast<void *>(producer_buffer));
  return error_text(
      @"ANE input pack command buffer failed; canceling this dispatch",
      buffer.error);
}

struct AneTicketPair {
  AneLinearModel::Ticket first;
  AneLinearModel::Ticket second;
};

static AneTicketPair begin_ane_ticket_pair(
    const std::shared_ptr<AneLinearModel> &first,
    const std::shared_ptr<AneLinearModel> &second,
    MTL::CommandBuffer *producer_buffer) {
  auto first_ticket = first->begin(producer_buffer);
  try {
    return {first_ticket, second->begin(producer_buffer)};
  } catch (...) {
    first->cancel_ticket(first_ticket);
    throw;
  }
}

// Rolls an in-flight ANE dispatch back if the stack unwinds after begin()
// has handed out a ticket but before the detached evaluation thread(s) take
// ownership of it (finding C1). The dangerous window is everything between
// begin() and the spawn — most notably the qmm get_kernel() calls, which can
// throw. Without this guard such a throw would leave submitted_ ahead of
// completed_ forever and leak the retained producer command buffer, so the
// next begin() throws "overlapping evaluations" and the program stays wedged
// until the process restarts. On unwind the guard releases the producer
// buffer (unless producer_released() already retired it at the pack step) and
// cancels the outstanding ticket(s) to rebalance the counters. Call disarm()
// once ownership has transferred to the evaluation thread(s).
class AneDispatchGuard {
public:
  AneDispatchGuard(MTL::CommandBuffer *producer_buffer,
                   std::shared_ptr<AneLinearModel> model0,
                   AneLinearModel::Ticket ticket0,
                   std::shared_ptr<AneLinearModel> model1 = nullptr,
                   AneLinearModel::Ticket ticket1 = {})
      : producer_buffer_(producer_buffer), model0_(std::move(model0)),
        ticket0_(ticket0), model1_(std::move(model1)), ticket1_(ticket1) {}
  AneDispatchGuard(const AneDispatchGuard &) = delete;
  AneDispatchGuard &operator=(const AneDispatchGuard &) = delete;
  // The producer command buffer is released mid-dispatch once packing has
  // completed; tell the guard so it does not double-release on unwind.
  void producer_released() { producer_buffer_ = nullptr; }
  // Transfer one ticket only after its detached evaluation thread exists.
  // Keeping ownership per ticket lets a later thread-constructor failure
  // cancel the tickets that have not acquired a worker yet without racing
  // workers that are already running.
  void transfer_ticket0() noexcept { model0_.reset(); }
  void transfer_ticket1() noexcept { model1_.reset(); }
  void disarm() { armed_ = false; }
  ~AneDispatchGuard() {
    if (!armed_) {
      return;
    }
    if (producer_buffer_) {
      producer_buffer_->release();
    }
    if (model0_) {
      model0_->cancel_ticket(ticket0_);
    }
    if (model1_) {
      model1_->cancel_ticket(ticket1_);
    }
  }

private:
  MTL::CommandBuffer *producer_buffer_;
  std::shared_ptr<AneLinearModel> model0_;
  AneLinearModel::Ticket ticket0_;
  std::shared_ptr<AneLinearModel> model1_;
  AneLinearModel::Ticket ticket1_;
  bool armed_ = true;
};

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

struct AneLinearBankBuilder::Impl {
  struct Chunk {
    QuantizedMatrix quantized;
    int input_dim;
    int output_dim;
  };
  int sequence_length = 0;
  std::vector<Chunk> chunks;
};

AneLinearBankBuilder::AneLinearBankBuilder(int sequence_length)
    : impl_(std::make_unique<Impl>()) {
  if (!qwen35_ane_available()) {
    throw std::runtime_error("Private ANE runtime is unavailable.");
  }
  if (sequence_length <= 1) {
    throw std::invalid_argument("Invalid ANE linear procedure bank shape.");
  }
  impl_->sequence_length = sequence_length;
}

AneLinearBankBuilder::~AneLinearBankBuilder() = default;

void AneLinearBankBuilder::add(const array &weight) {
  if (weight.dtype() != float32 || weight.ndim() != 2 ||
      !row_contiguous(weight) || weight.shape(0) <= 0 ||
      weight.shape(1) <= 0) {
    throw std::invalid_argument(
        "ANE bank weights must be contiguous rank-2 float32 MLX arrays.");
  }
  // Quantize to the INT8 program format now so the caller can drop the fp32
  // staging array immediately; the retained chunk is a quarter of its size.
  Impl::Chunk chunk{
      quantize_rows(weight.data<float>(), static_cast<int>(weight.shape(0)),
                    static_cast<int>(weight.shape(1))),
      static_cast<int>(weight.shape(1)),
      static_cast<int>(weight.shape(0)),
  };
  impl_->chunks.emplace_back(std::move(chunk));
}

int AneLinearBankBuilder::size() const {
  return static_cast<int>(impl_->chunks.size());
}

std::vector<std::shared_ptr<AneLinearModel>>
AneLinearBankBuilder::compile(int ane_instance, int start, int stop) {
  const int count = static_cast<int>(impl_->chunks.size());
  if (start < 0 || stop <= start || stop > count || stop - start > 256) {
    throw std::invalid_argument("Invalid ANE linear procedure bank span.");
  }
  const int sequence_length = impl_->sequence_length;
  const int span = stop - start;
  std::vector<std::pair<int, int>> shapes;
  shapes.reserve(span);
  for (int index = start; index < stop; ++index) {
    const auto &chunk = impl_->chunks[index];
    shapes.emplace_back(chunk.input_dim, chunk.output_dim);
  }

  @autoreleasepool {
    load_ane_framework();
    NSDictionary *execution_options = ane_execution_options(ane_instance);
    NSMutableData *weight_blob = [NSMutableData dataWithLength:64];
    std::vector<std::pair<uint64_t, uint64_t>> weight_offsets;
    weight_offsets.reserve(span);
    for (int index = start; index < stop; ++index) {
      const auto &quantized = impl_->chunks[index].quantized;
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
    weight_header[0] = static_cast<uint32_t>(span * 2);
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
    AneLoadResult load_result = load_or_compile_ane_model(
        model, identifier, ane_instance, execution_options, mil, @{
          @"weight.bin" : weight_blob,
        },
        @"ANE procedure bank", @"ANE procedure bank");
    auto program = std::make_shared<SharedAneProgram>(model, load_result,
                                                      execution_options);
    std::vector<std::shared_ptr<AneLinearModel>> result;
    result.reserve(span);
    for (int index = 0; index < span; ++index) {
      auto impl = std::make_unique<AneLinearModel::Impl>(
          program, shapes[index].first, shapes[index].second, sequence_length,
          index);
      result.emplace_back(new AneLinearModel(std::move(impl)));
    }
    return result;
  }
}

std::vector<std::shared_ptr<AneLinearModel>> qwen35_ane_compile_linear_bank(
    const std::vector<array> &weights, int sequence_length, int ane_instance) {
  AneLinearBankBuilder builder(sequence_length);
  if (weights.empty() || weights.size() > 256) {
    throw std::invalid_argument("Invalid ANE linear procedure bank shape.");
  }
  for (const auto &weight : weights) {
    builder.add(weight);
  }
  return builder.compile(ane_instance, 0, builder.size());
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
    int sequence_length, int ane_instance) {
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
      sequence_length, ane_instance);
  return std::shared_ptr<AneLinearModel>(new AneLinearModel(std::move(impl)));
}

struct AneFusedBankBuilder::Impl {
  struct Chunk {
    std::vector<_Float16> gate;
    std::vector<_Float16> up;
    std::vector<_Float16> down;
    SwiGLUDownShape shape;
  };
  int sequence_length = 0;
  std::vector<Chunk> chunks;
};

AneFusedBankBuilder::AneFusedBankBuilder(int sequence_length)
    : impl_(std::make_unique<Impl>()) {
  if (!qwen35_ane_available()) {
    throw std::runtime_error("Private ANE runtime is unavailable.");
  }
  if (sequence_length <= 1) {
    throw std::invalid_argument("Invalid ANE SwiGLU/down procedure bank.");
  }
  impl_->sequence_length = sequence_length;
}

AneFusedBankBuilder::~AneFusedBankBuilder() = default;

void AneFusedBankBuilder::add(const array &gate_weight, const array &up_weight,
                              const array &down_weight) {
  for (const auto *weight : {&gate_weight, &up_weight, &down_weight}) {
    if (weight->dtype() != float32 || weight->ndim() != 2 ||
        !row_contiguous(*weight) || weight->shape(0) <= 0 ||
        weight->shape(1) <= 0) {
      throw std::invalid_argument(
          "ANE SwiGLU/down bank weights must be contiguous float32 matrices.");
    }
  }
  if (up_weight.shape() != gate_weight.shape() ||
      down_weight.shape(1) != gate_weight.shape(0)) {
    throw std::invalid_argument("ANE SwiGLU/down bank shape mismatch.");
  }
  // Convert to the fp16 program format now so the caller can drop the fp32
  // staging arrays immediately; the retained chunk is half their size.
  const auto to_fp16 = [](const array &weight) {
    return fp16_rows(weight.data<float>(), static_cast<int>(weight.shape(0)),
                     static_cast<int>(weight.shape(1)));
  };
  Impl::Chunk chunk{
      to_fp16(gate_weight),
      to_fp16(up_weight),
      to_fp16(down_weight),
      {static_cast<int>(gate_weight.shape(1)),
       static_cast<int>(gate_weight.shape(0)),
       static_cast<int>(down_weight.shape(0))},
  };
  impl_->chunks.emplace_back(std::move(chunk));
}

int AneFusedBankBuilder::size() const {
  return static_cast<int>(impl_->chunks.size());
}

std::vector<std::shared_ptr<AneLinearModel>>
AneFusedBankBuilder::compile(int ane_instance, int start, int stop) {
  const int count = static_cast<int>(impl_->chunks.size());
  if (start < 0 || stop <= start || stop > count || stop - start > 256) {
    throw std::invalid_argument(
        "Invalid ANE SwiGLU/down procedure bank span.");
  }
  const int sequence_length = impl_->sequence_length;
  std::vector<SwiGLUDownShape> shapes;
  shapes.reserve(stop - start);
  for (int index = start; index < stop; ++index) {
    shapes.push_back(impl_->chunks[index].shape);
  }

  @autoreleasepool {
    load_ane_framework();
    NSDictionary *execution_options = ane_execution_options(ane_instance);
    NSMutableData *weight_blob = [NSMutableData dataWithLength:64];
    std::vector<std::array<uint64_t, 3>> offsets;
    offsets.reserve(shapes.size());
    for (int index = start; index < stop; ++index) {
      const auto append_fp16 = [&](const std::vector<_Float16> &dense) {
        return append_blob_chunk(weight_blob, dense.data(),
                                 dense.size() * sizeof(dense.front()), 1);
      };
      const auto &chunk = impl_->chunks[index];
      offsets.push_back({append_fp16(chunk.gate), append_fp16(chunk.up),
                         append_fp16(chunk.down)});
    }
    auto *weight_header = static_cast<uint32_t *>(weight_blob.mutableBytes);
    weight_header[0] = static_cast<uint32_t>(shapes.size() * 3);
    weight_header[1] = 2;
    NSData *mil =
        [fp16_swiglu_down_bank_mil(shapes, offsets, sequence_length)
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
          "Private ANE SwiGLU/down bank model creation failed.");
    }
    id identifier = ((id (*)(id, SEL))objc_msgSend)(
        model, @selector(hexStringIdentifier));
    AneLoadResult load_result = load_or_compile_ane_model(
        model, identifier, ane_instance, execution_options, mil, @{
          @"weight.bin" : weight_blob,
        },
        @"ANE SwiGLU/down bank", @"ANE SwiGLU/down bank");

    auto program = std::make_shared<SharedAneProgram>(model, load_result,
                                                      execution_options);
    std::vector<std::shared_ptr<AneLinearModel>> result;
    result.reserve(shapes.size());
    for (size_t index = 0; index < shapes.size(); ++index) {
      auto impl = std::make_unique<AneLinearModel::Impl>(
          program, shapes[index].input_dim, shapes[index].output_dim,
          sequence_length, static_cast<int>(index));
      result.emplace_back(new AneLinearModel(std::move(impl)));
    }
    return result;
  }
}

std::vector<std::shared_ptr<AneLinearModel>>
qwen35_ane_compile_swiglu_down_bank(
    const std::vector<array> &gate_weights,
    const std::vector<array> &up_weights,
    const std::vector<array> &down_weights, int sequence_length,
    int ane_instance) {
  if (gate_weights.empty() || gate_weights.size() > 256 ||
      gate_weights.size() != up_weights.size() ||
      gate_weights.size() != down_weights.size()) {
    throw std::invalid_argument("Invalid ANE SwiGLU/down procedure bank.");
  }
  AneFusedBankBuilder builder(sequence_length);
  for (size_t index = 0; index < gate_weights.size(); ++index) {
    builder.add(gate_weights[index], up_weights[index], down_weights[index]);
  }
  return builder.compile(ane_instance, 0, builder.size());
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

template <typename Device, typename Library>
MTL::ComputePipelineState *hybrid_qmm_kernel(
    Device &device, const Library &classic_library, int bits, int group_size,
    const std::string &type, int K, int N) {
  // The bundled hybrid NAX kernel currently uses the same 64x64x64 tile as
  // variant 0 of the standalone qmm op. Select it per projection: the fused
  // gate/up and down suffixes have different K/N shapes, and either one must
  // be able to fall back independently if a future architecture is not tile
  // compatible.
  if (K % 64 == 0 && N % 64 == 0 && hybrid_nax_enabled()) {
    try {
      auto nax_library = device.get_library(
          "omlx_qwen35_prefill_kernels_nax", binary_dir());
      return device.get_kernel(
          hybrid_nax_qmm_name(bits, group_size, type), nax_library);
    } catch (const std::exception &) {
      // Keep the fused ANE path usable with an older/incomplete app bundle.
      // This mirrors the established hybrid suffix behavior and avoids
      // retrying a missing NAX pipeline on every layer.
      g_hybrid_nax_runtime_ok.store(false, std::memory_order_relaxed);
    }
  }
  return device.get_kernel(
      hybrid_classic_qmm_name(bits, group_size, type), classic_library);
}

BNNSNDArrayDescriptor cpu_matrix_descriptor(void *data, int rows, int cols,
                                             BNNSDataType dtype) {
  BNNSNDArrayDescriptor descriptor{};
  descriptor.layout = BNNSDataLayoutRowMajorMatrix;
  descriptor.size[0] = static_cast<size_t>(cols);
  descriptor.size[1] = static_cast<size_t>(rows);
  descriptor.data = data;
  descriptor.data_type = dtype;
  descriptor.data_scale = 1.0f;
  return descriptor;
}

// dispatch_apply_with_attr is present on the macOS versions that expose the
// shared-resource scheduling policy, but older SDKs do not declare the family.
// Resolve the public lifecycle dynamically so one extension binary remains
// loadable on those SDKs without reaching into thread-control syscalls.
using DispatchApplyAttrStorage = uint64_t[8];
using DispatchApplyAttrInitFn = void (*)(DispatchApplyAttrStorage *);
using DispatchApplyAttrDestroyFn = void (*)(DispatchApplyAttrStorage *);
using DispatchApplyAttrEntity = unsigned long;
using DispatchApplyAttrSetParallelismFn = void (*)(DispatchApplyAttrStorage *,
                                                   DispatchApplyAttrEntity,
                                                   size_t);
using DispatchApplyWithAttrFn = void (*)(size_t, DispatchApplyAttrStorage *,
                                         void (^)(size_t));

// Request one worker per CPU cluster. dispatch_apply_attr is libdispatch SPI
// (private/apply_private.h, not shipped in the public SDK), which is why it is
// resolved through dlsym; DISPATCH_APPLY_ATTR_ENTITY_CLUSTER = 2 and the only
// value libdispatch accepts for threads_per_entity is 1, anything else traps.
// The number of requested work shards is still supplied as the iteration
// count to dispatch_apply_with_attr.
constexpr DispatchApplyAttrEntity kDispatchApplyAttrEntityCluster = 2;
constexpr size_t kDispatchApplyThreadsPerCluster = 1;

struct DispatchApplyAttrApi {
  DispatchApplyAttrInitFn init;
  DispatchApplyAttrDestroyFn destroy;
  DispatchApplyAttrSetParallelismFn set_parallelism;
  DispatchApplyWithAttrFn apply;

  explicit operator bool() const {
    return init && destroy && set_parallelism && apply;
  }
};

const DispatchApplyAttrApi &dispatch_apply_attr_api() {
  static const DispatchApplyAttrApi api{
      reinterpret_cast<DispatchApplyAttrInitFn>(
          dlsym(RTLD_DEFAULT, "dispatch_apply_attr_init")),
      reinterpret_cast<DispatchApplyAttrDestroyFn>(
          dlsym(RTLD_DEFAULT, "dispatch_apply_attr_destroy")),
      reinterpret_cast<DispatchApplyAttrSetParallelismFn>(
          dlsym(RTLD_DEFAULT, "dispatch_apply_attr_set_parallelism")),
      reinterpret_cast<DispatchApplyWithAttrFn>(
          dlsym(RTLD_DEFAULT, "dispatch_apply_with_attr")),
  };
  return api;
}

bool cpu_shared_resource_policy_available() {
  return static_cast<bool>(dispatch_apply_attr_api());
}

class ScopedDispatchApplyAttributes {
public:
  ScopedDispatchApplyAttributes() {
    const auto &api = dispatch_apply_attr_api();
    if (!api) {
      return;
    }
    api.init(&attributes_);
    active_ = true;
    api.set_parallelism(&attributes_, kDispatchApplyAttrEntityCluster,
                        kDispatchApplyThreadsPerCluster);
  }

  ~ScopedDispatchApplyAttributes() {
    if (active_) {
      dispatch_apply_attr_api().destroy(&attributes_);
    }
  }

  ScopedDispatchApplyAttributes(const ScopedDispatchApplyAttributes &) = delete;
  ScopedDispatchApplyAttributes &
  operator=(const ScopedDispatchApplyAttributes &) = delete;

  DispatchApplyAttrStorage *get() { return &attributes_; }
  bool active() const { return active_; }

private:
  DispatchApplyAttrStorage attributes_{};
  bool active_{false};
};

void cpu_fp16_matmul(const array &input, const array &weight, array &output,
                     int M, int N, int K, int cpu_threads,
                     bool shared_resource) {
  const char *manual_shards_value =
      std::getenv("OMLX_QWEN35_CPU_MANUAL_SHARDS");
  const int requested_threads =
      shared_resource && cpu_threads == 0 ? 8 : std::max(cpu_threads, 0);
  const int manual_shards =
      shared_resource
          ? requested_threads
          : (manual_shards_value
                 ? (std::strcmp(manual_shards_value, "threads") == 0
                        ? requested_threads
                        : std::max(std::atoi(manual_shards_value), 0))
                 : 0);
  const char *manual_axis_value = std::getenv("OMLX_QWEN35_CPU_MANUAL_AXIS");
  const bool shard_columns =
      manual_axis_value && std::strcmp(manual_axis_value, "columns") == 0;
  const bool use_shared_resource = shared_resource && !shard_columns &&
                                   cpu_shared_resource_policy_available();
  const int shard_extent = shard_columns ? N : M;
  if (manual_shards > 1 && shard_extent > 1) {
    const int shards = std::min(manual_shards, shard_extent);
    auto failure = std::make_shared<std::atomic<int>>(0);
    auto *input_data = const_cast<_Float16 *>(input.data<_Float16>());
    auto *weight_data = const_cast<_Float16 *>(weight.data<_Float16>());
    auto *output_data = output.data<_Float16>();
    void (^work)(size_t) = ^(size_t index) {
      try {
        const int row_blocks = M / 64;
        const bool align_rows =
            use_shared_resource && M % 64 == 0 && row_blocks >= shards;
        const int begin =
            align_rows
                ? static_cast<int>((static_cast<int64_t>(row_blocks) * index) /
                                   shards) *
                      64
                : static_cast<int>(
                      (static_cast<int64_t>(shard_extent) * index) / shards);
        const int end =
            align_rows ? static_cast<int>(
                             (static_cast<int64_t>(row_blocks) * (index + 1)) /
                             shards) *
                             64
                       : static_cast<int>((static_cast<int64_t>(shard_extent) *
                                           (index + 1)) /
                                          shards);
        const int row_begin = shard_columns ? 0 : begin;
        const int row_end = shard_columns ? M : end;
        const int column_begin = shard_columns ? begin : 0;
        const int column_end = shard_columns ? end : N;
        const int rows = row_end - row_begin;
        const int columns = column_end - column_begin;
        auto input_desc = cpu_matrix_descriptor(
            input_data + static_cast<size_t>(row_begin) * K, rows, K,
            BNNSDataTypeFloat16);
        auto weight_desc = cpu_matrix_descriptor(
            weight_data + static_cast<size_t>(column_begin) * K, columns, K,
            BNNSDataTypeFloat16);
        auto output_desc = cpu_matrix_descriptor(
            output_data + static_cast<size_t>(row_begin) * N + column_begin,
            rows, columns, BNNSDataTypeFloat16);
        if (shard_columns) {
          output_desc.stride[0] = 1;
          output_desc.stride[1] = static_cast<size_t>(N);
        }
        BNNSFilterParameters filter_params{};
        filter_params.n_threads = 1;
        const ssize_t workspace_size =
            BNNSMatMulWorkspaceSize(false, true, 1.0f, &input_desc,
                                    &weight_desc, &output_desc, &filter_params);
        if (workspace_size < 0) {
          failure->store(1, std::memory_order_relaxed);
          return;
        }
        std::vector<uint8_t> workspace(static_cast<size_t>(workspace_size));
        if (BNNSMatMul(false, true, 1.0f, &input_desc, &weight_desc,
                       &output_desc,
                       workspace.empty() ? nullptr : workspace.data(),
                       &filter_params) != 0) {
          failure->store(1, std::memory_order_relaxed);
        }
      } catch (const std::bad_alloc &) {
        failure->store(2, std::memory_order_relaxed);
      } catch (...) {
        failure->store(3, std::memory_order_relaxed);
      }
    };
    ScopedDispatchApplyAttributes attributes;
    if (use_shared_resource && attributes.active()) {
      dispatch_apply_attr_api().apply(static_cast<size_t>(shards),
                                      attributes.get(), work);
    } else {
      dispatch_apply(static_cast<size_t>(shards),
                     dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                     work);
    }
    const int failure_code = failure->load(std::memory_order_relaxed);
    if (failure_code == 2) {
      throw std::runtime_error(
          "Unable to allocate an Accelerate CPU fp16 matrix workspace.");
    }
    if (failure_code != 0) {
      throw std::runtime_error(
          "Accelerate rejected a manually sharded CPU fp16 matrix multiply.");
    }
    return;
  }
  auto input_desc =
      cpu_matrix_descriptor(const_cast<_Float16 *>(input.data<_Float16>()), M,
                            K, BNNSDataTypeFloat16);
  auto weight_desc =
      cpu_matrix_descriptor(const_cast<_Float16 *>(weight.data<_Float16>()), N,
                            K, BNNSDataTypeFloat16);
  auto output_desc =
      cpu_matrix_descriptor(output.data<_Float16>(), M, N, BNNSDataTypeFloat16);
  BNNSFilterParameters filter_params{};
  filter_params.n_threads = static_cast<size_t>(std::max(cpu_threads, 0));
  const ssize_t workspace_size = BNNSMatMulWorkspaceSize(
      false, true, 1.0f, &input_desc, &weight_desc, &output_desc,
      &filter_params);
  if (workspace_size < 0) {
    throw std::runtime_error("Accelerate rejected the CPU fp16 matrix shape.");
  }
  std::vector<uint8_t> workspace(static_cast<size_t>(workspace_size));
  if (BNNSMatMul(false, true, 1.0f, &input_desc, &weight_desc, &output_desc,
                 workspace.empty() ? nullptr : workspace.data(),
                 &filter_params) != 0) {
    throw std::runtime_error("Accelerate CPU fp16 matrix multiply failed.");
  }
}

void cpu_fp16_swiglu(const array &gate_up, array &activation, int M, int N,
                     int cpu_threads, bool shared_resource) {
  const int requested_threads =
      shared_resource && cpu_threads == 0 ? 8 : std::max(cpu_threads, 1);
  const int shards = std::min(requested_threads, std::max(M, 1));
  const auto *input = gate_up.data<_Float16>();
  auto *output = activation.data<_Float16>();
  void (^work)(size_t) = ^(size_t index) {
    const int begin = static_cast<int>(
        (static_cast<int64_t>(M) * static_cast<int64_t>(index)) / shards);
    const int end = static_cast<int>(
        (static_cast<int64_t>(M) * static_cast<int64_t>(index + 1)) / shards);
    for (int m = begin; m < end; ++m) {
      const auto *gate = input + static_cast<size_t>(m) * 2 * N;
      const auto *up = gate + N;
      auto *row = output + static_cast<size_t>(m) * N;
      for (int n = 0; n < N; ++n) {
        const float g = static_cast<float>(gate[n]);
        row[n] = static_cast<_Float16>(
            g * static_cast<float>(up[n]) / (1.0f + std::exp(-g)));
      }
    }
  };
  ScopedDispatchApplyAttributes attributes;
  if (shared_resource && attributes.active()) {
    dispatch_apply_attr_api().apply(static_cast<size_t>(shards),
                                    attributes.get(), work);
  } else {
    dispatch_apply(static_cast<size_t>(shards),
                   dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   work);
  }
}

class CpuFp16HybridPrimitive : public Primitive {
public:
  CpuFp16HybridPrimitive(Stream stream, int bits, int variant, int group_size,
                         int cpu_threads, bool cpu_shared_resource)
      : Primitive(stream), bits_(bits), variant_(variant),
        group_size_(group_size), cpu_threads_(cpu_threads),
        cpu_shared_resource_(cpu_shared_resource) {}

  void eval_cpu(const std::vector<array> &, std::vector<array> &) override {
    throw std::runtime_error("CPU/GPU hybrid qmm has no CPU evaluator.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    auto &encoder = metal::get_command_encoder(stream);
    const auto &x = inputs[0];
    const auto &cpu_weight = inputs[1];
    const auto &gpu_weight = inputs[2];
    const auto &gpu_scales = inputs[3];
    const auto &gpu_biases = inputs[4];
    auto &output = outputs[0];
    const int K = static_cast<int>(x.shape(-1));
    const int M = static_cast<int>(x.size() / K);
    const int cpu_n = static_cast<int>(cpu_weight.shape(0));
    const int gpu_n = static_cast<int>(gpu_weight.shape(0));

    output.set_data(allocator::malloc(output.nbytes()));
    std::unique_ptr<array> cpu_input;
    array cpu_output({M, cpu_n}, float16, nullptr, {});
    array gpu_output({M, gpu_n}, x.dtype(), nullptr, {});
    if (x.dtype() != float16) {
      cpu_input = std::make_unique<array>(Shape{M, K}, float16, nullptr,
                                          std::vector<array>{});
      cpu_input->set_data(allocator::malloc(cpu_input->nbytes()));
    }
    cpu_output.set_data(allocator::malloc(cpu_output.nbytes()));
    gpu_output.set_data(allocator::malloc(gpu_output.nbytes()));

    auto library =
        device.get_library("omlx_qwen35_prefill_kernels", binary_dir());
    if (cpu_input) {
      auto pack = device.get_kernel(
          "qwen35_cpu_pack_input_" + metal_type_name(x.dtype()), library);
      encoder.set_compute_pipeline_state(pack);
      encoder.set_input_array(x, 0);
      encoder.set_output_array(*cpu_input, 1);
      encoder.set_bytes(M, 2);
      encoder.set_bytes(K, 3);
      encoder.dispatch_threads(MTL::Size(K, M, 1), MTL::Size(16, 16, 1));
      encoder.end_encoding();
      auto *pack_buffer = encoder.get_command_buffer();
      pack_buffer->retain();
      encoder.commit();
      pack_buffer->waitUntilCompleted();
      pack_buffer->release();
    } else {
      // The caller's fp16 input may still be produced by earlier Metal work.
      encoder.end_encoding();
      auto *ready_buffer = encoder.get_command_buffer();
      ready_buffer->retain();
      encoder.commit();
      ready_buffer->waitUntilCompleted();
      ready_buffer->release();
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
    encoder.set_input_array(gpu_weight, 0);
    encoder.set_input_array(gpu_scales, 1);
    encoder.set_input_array(gpu_biases, 2);
    encoder.set_input_array(x, 3);
    encoder.set_output_array(gpu_output, 4);
    encoder.set_bytes(K, 5);
    encoder.set_bytes(gpu_n, 6);
    encoder.set_bytes(M, 7);
    encoder.dispatch_threadgroups(
        MTL::Size((gpu_n + 63) / 64, (M + 63) / 64, 1),
        MTL::Size(32, 2, 2));
    encoder.end_encoding();
    auto *gpu_buffer = encoder.get_command_buffer();
    gpu_buffer->retain();
    encoder.commit();

    try {
      cpu_fp16_matmul(cpu_input ? *cpu_input : x, cpu_weight, cpu_output,
                      M, cpu_n, K, cpu_threads_, cpu_shared_resource_);
    } catch (...) {
      // The committed qmm still reads this frame's MLX arrays; drain it
      // before unwinding so the allocator cannot recycle them mid-flight.
      gpu_buffer->waitUntilCompleted();
      gpu_buffer->release();
      throw;
    }
    gpu_buffer->waitUntilCompleted();
    gpu_buffer->release();

    auto merge = device.get_kernel(
        "qwen35_cpu_merge_output_" + metal_type_name(x.dtype()), library);
    encoder.set_compute_pipeline_state(merge);
    encoder.set_input_array(cpu_output, 0);
    encoder.set_input_array(gpu_output, 1);
    encoder.set_output_array(output, 2);
    encoder.set_bytes(M, 3);
    encoder.set_bytes(cpu_n, 4);
    encoder.set_bytes(gpu_n, 5);
    encoder.dispatch_threads(MTL::Size(cpu_n + gpu_n, M, 1),
                             MTL::Size(16, 16, 1));
    if (!encoder.needs_commit()) {
      auto guard = device.get_kernel("qwen35_ane_commit_guard", library);
      encoder.set_compute_pipeline_state(guard);
      while (!encoder.needs_commit()) {
        encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
      }
    }
    if (cpu_input) {
      encoder.add_temporary(std::move(*cpu_input));
    }
    encoder.add_temporary(std::move(cpu_output));
    encoder.add_temporary(std::move(gpu_output));
  }

  DEFINE_NAME(Qwen35CpuFp16Hybrid)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs = static_cast<const CpuFp16HybridPrimitive &>(other);
    return bits_ == rhs.bits_ && variant_ == rhs.variant_ &&
           group_size_ == rhs.group_size_ && cpu_threads_ == rhs.cpu_threads_ &&
           cpu_shared_resource_ == rhs.cpu_shared_resource_;
  }
  auto state() const {
    return std::make_tuple(bits_, variant_, group_size_, cpu_threads_,
                           cpu_shared_resource_);
  }

private:
  int bits_;
  int variant_;
  int group_size_;
  int cpu_threads_;
  bool cpu_shared_resource_;
};

class AneHybridQ4Primitive : public Primitive {
public:
  AneHybridQ4Primitive(Stream stream, std::shared_ptr<AneLinearModel> model,
                       int bits, int variant, int group_size,
                       bool fuse_swiglu = false, int profile_category = -1,
                       bool cpu_fp16 = false, int cpu_threads = 0,
                       bool cpu_shared_resource = false)
      : Primitive(stream), model_(std::move(model)), variant_(variant),
        bits_(bits), group_size_(group_size), fuse_swiglu_(fuse_swiglu),
        profile_category_(profile_category >= 0 ? profile_category
                                                : (fuse_swiglu ? 0 : 1)),
        cpu_fp16_(cpu_fp16), cpu_threads_(cpu_threads),
        cpu_shared_resource_(cpu_shared_resource) {}

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
    const array *cpu_weight = cpu_fp16_ ? &inputs[4] : nullptr;
    auto &output = outputs[0];

    const int K = static_cast<int>(x.shape(-1));
    const int M = static_cast<int>(x.size() / K);
    const int gpu_n = static_cast<int>(weight.shape(0));
    const int ane_n = model_->output_dim();
    const int cpu_n = cpu_weight ? static_cast<int>(cpu_weight->shape(0)) : 0;
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
    std::unique_ptr<array> cpu_input;
    std::unique_ptr<array> cpu_output;
    if (cpu_weight) {
      cpu_output = std::make_unique<array>(Shape{M, cpu_n}, float16, nullptr,
                                           std::vector<array>{});
      cpu_output->set_data(allocator::malloc(cpu_output->nbytes()));
      if (x.dtype() != float16) {
        cpu_input = std::make_unique<array>(Shape{M, K}, float16, nullptr,
                                            std::vector<array>{});
        cpu_input->set_data(allocator::malloc(cpu_input->nbytes()));
      }
    }

    auto library =
        device.get_library("omlx_qwen35_prefill_kernels", binary_dir());
    auto pack =
        device.get_kernel(std::string(cpu_input ? "qwen35_ane_pack_input_cpu_"
                                                : "qwen35_ane_pack_input_") +
                              metal_type_name(x.dtype()),
                          library);
    encoder.set_compute_pipeline_state(pack);
    encoder.set_input_array(x, 0);
    encoder.set_buffer(model_->input_buffer(), 1);
    if (cpu_input) {
      encoder.set_output_array(*cpu_input, 2);
      encoder.set_bytes(M, 3);
      encoder.set_bytes(K, 4);
    } else {
      encoder.set_bytes(M, 2);
      encoder.set_bytes(K, 3);
    }
    encoder.dispatch_threads(MTL::Size(K, M, 1), MTL::Size(16, 16, 1));
    encoder.end_encoding();

    auto *producer_buffer = encoder.get_command_buffer();
    producer_buffer->retain();
    AneLinearModel::Ticket ticket{};
    try {
      ticket = model_->begin(producer_buffer);
    } catch (...) {
      // begin() throws before it increments the submission counter, so no
      // ticket exists to cancel here — only the retained buffer must be freed.
      producer_buffer->release();
      throw;
    }
    // Cancel the ticket and free the buffer if anything below throws before the
    // detached evaluation thread takes ownership (finding C1).
    AneDispatchGuard ane_guard(producer_buffer, model_, ticket);

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
    if (pack_buffer_failed(producer_buffer)) {
      // ane_guard releases the buffer and cancels the ticket on unwind.
      throw std::runtime_error(pack_buffer_error(producer_buffer));
    }
    producer_buffer->release();
    ane_guard.producer_released();
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
    const uint64_t launch = profiling ? profile_now_ns() : 0;
    id<MTLCommandBuffer> qmm_buffer =
        (__bridge id<MTLCommandBuffer>)(static_cast<void *>(
            encoder.get_command_buffer()));
    [qmm_buffer retain];
    if (profiling) {
      [qmm_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        const double duration = completed.GPUEndTime - completed.GPUStartTime;
        if (duration > 0) {
          profile_add(profile_category, kGpuQmmNs,
                      static_cast<uint64_t>(duration * 1e9));
        }
        profile_add(profile_category, kGpuCompletionNs,
                    profile_now_ns() - launch);
      }];
    }
    // Keep the wait in a third command buffer. Metal may hold an entire
    // command buffer at an encoded wait, even when compute encoders precede
    // it, which serializes the otherwise independent GPU remainder behind
    // ANE. Submitting the qmm first makes the intended overlap explicit.
    encoder.commit();
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
    // The evaluation thread now owns the ticket and will signal completion.
    ane_guard.disarm();
    try {
      if (cpu_weight) {
        const uint64_t cpu_start = profiling ? profile_now_ns() : 0;
        cpu_fp16_matmul(cpu_input ? *cpu_input : x, *cpu_weight, *cpu_output,
                        M, cpu_n, K, cpu_threads_, cpu_shared_resource_);
        if (profiling) {
          const uint64_t cpu_done = profile_now_ns();
          profile_add(profile_category, kCpuMatmulNs, cpu_done - cpu_start);
          profile_add(profile_category, kCpuCompletionNs, cpu_done - launch);
        }
      }
      // Do not submit a future Metal wait here: on current M3 Ultra drivers
      // it can stall the already-queued qmm behind ANE. Both devices are
      // running at this point, so a host wait preserves their overlap and
      // immediately exposes asynchronous ANE failures before the merge is
      // encoded.
      model_->wait(ticket);
    } catch (...) {
      // The committed qmm buffer references MLX arrays owned by this frame;
      // drain it before unwinding so the allocator cannot recycle them while
      // the GPU is still writing. The detached ANE thread holds its own
      // model reference and signals the ticket on its own.
      [qmm_buffer waitUntilCompleted];
      [qmm_buffer release];
      throw;
    }
    // ANE completion does not order an independently committed GPU suffix
    // command buffer before the merge encoder. This is observable on the
    // first qmm dispatch, when pipeline setup lets ANE finish first: the merge
    // can otherwise read a partially written suffix. Preserve ANE/GPU overlap,
    // then join both branches explicitly before consuming either output.
    const bool gpu_finished_at_ane_done =
        qmm_buffer.status == MTLCommandBufferStatusCompleted;
    [qmm_buffer waitUntilCompleted];
    if (qmm_buffer.status == MTLCommandBufferStatusError) {
      const std::string error = error_text(
          @"ANE hybrid GPU suffix command buffer failed", qmm_buffer.error);
      [qmm_buffer release];
      throw std::runtime_error(error);
    }
    [qmm_buffer release];
    if (profiling) {
      const uint64_t done = profile_now_ns();
      profile_add(profile_category, kAneRegionNs, done - launch);
      g_previous_ane_done_ns.store(done, std::memory_order_relaxed);
      if (gpu_finished_at_ane_done) {
        profile_add(profile_category, kAneLast, 1);
      } else {
        profile_add(profile_category, kGpuLast, 1);
      }
    }

    auto merge = device.get_kernel(
        std::string(cpu_weight
                        ? (fuse_swiglu_ ? "qwen35_ane_merge_cpu_swiglu_output_"
                                        : "qwen35_ane_merge_cpu_output_")
                        : (fuse_swiglu_ ? "qwen35_ane_merge_swiglu_output_"
                                        : "qwen35_ane_merge_output_")) +
            metal_type_name(x.dtype()),
        library);
    encoder.set_compute_pipeline_state(merge);
    encoder.set_buffer(model_->output_buffer(), 0);
    if (cpu_weight) {
      encoder.set_input_array(*cpu_output, 1);
      encoder.set_input_array(gpu_output, 2);
      encoder.set_output_array(output, 3);
      encoder.set_bytes(M, 4);
    } else {
      encoder.set_input_array(gpu_output, 1);
      encoder.set_output_array(output, 2);
      encoder.set_bytes(M, 3);
    }
    const int output_n =
        fuse_swiglu_ ? (ane_n + cpu_n + gpu_n) / 2 : ane_n + cpu_n + gpu_n;
    const int merge_ane_n = fuse_swiglu_ ? ane_n / 2 : ane_n;
    const int merge_cpu_n = fuse_swiglu_ ? cpu_n / 2 : cpu_n;
    const int merge_gpu_n = fuse_swiglu_ ? gpu_n / 2 : gpu_n;
    if (cpu_weight) {
      encoder.set_bytes(merge_ane_n, 5);
      encoder.set_bytes(merge_cpu_n, 6);
      encoder.set_bytes(merge_gpu_n, 7);
    } else {
      encoder.set_bytes(merge_ane_n, 4);
      encoder.set_bytes(merge_gpu_n, 5);
    }
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
    if (cpu_input) {
      encoder.add_temporary(std::move(*cpu_input));
    }
    if (cpu_output) {
      encoder.add_temporary(std::move(*cpu_output));
    }
  }

  DEFINE_NAME(Qwen35AneHybridQ4)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs = static_cast<const AneHybridQ4Primitive &>(other);
    return model_.get() == rhs.model_.get() && bits_ == rhs.bits_ &&
           variant_ == rhs.variant_ && group_size_ == rhs.group_size_ &&
           fuse_swiglu_ == rhs.fuse_swiglu_ &&
           profile_category_ == rhs.profile_category_ &&
           cpu_fp16_ == rhs.cpu_fp16_ && cpu_threads_ == rhs.cpu_threads_ &&
           cpu_shared_resource_ == rhs.cpu_shared_resource_;
  }
  auto state() const {
    return std::make_tuple(reinterpret_cast<uintptr_t>(model_.get()), bits_,
                           variant_, group_size_, fuse_swiglu_,
                           profile_category_, cpu_fp16_, cpu_threads_,
                           cpu_shared_resource_);
  }

private:
  std::shared_ptr<AneLinearModel> model_;
  int variant_;
  int bits_;
  int group_size_;
  bool fuse_swiglu_;
  int profile_category_;
  bool cpu_fp16_;
  int cpu_threads_;
  bool cpu_shared_resource_;
};

class DualAneHybridPrimitive : public Primitive {
public:
  DualAneHybridPrimitive(Stream stream,
                         std::shared_ptr<AneLinearModel> model0,
                         std::shared_ptr<AneLinearModel> model1, int bits,
                         int variant, int group_size, bool fuse_swiglu = false,
                         int profile_category = -1, bool cpu_fp16 = false,
                         int cpu_threads = 0,
                         bool cpu_shared_resource = false)
      : Primitive(stream), model0_(std::move(model0)),
        model1_(std::move(model1)), variant_(variant), bits_(bits),
        group_size_(group_size), fuse_swiglu_(fuse_swiglu),
        profile_category_(profile_category >= 0 ? profile_category
                                                : (fuse_swiglu ? 0 : 1)),
        cpu_fp16_(cpu_fp16), cpu_threads_(cpu_threads),
        cpu_shared_resource_(cpu_shared_resource) {}

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
    const array *cpu_weight = cpu_fp16_ ? &inputs[4] : nullptr;
    auto &output = outputs[0];

    const int K = static_cast<int>(x.shape(-1));
    const int M = static_cast<int>(x.size() / K);
    const int gpu_n = static_cast<int>(weight.shape(0));
    const int ane0_n = model0_->output_dim();
    const int ane1_n = model1_->output_dim();
    const int cpu_n = cpu_weight ? static_cast<int>(cpu_weight->shape(0)) : 0;
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
    std::unique_ptr<array> cpu_input;
    std::unique_ptr<array> cpu_output;
    if (cpu_weight) {
      cpu_output = std::make_unique<array>(Shape{M, cpu_n}, float16, nullptr,
                                           std::vector<array>{});
      cpu_output->set_data(allocator::malloc(cpu_output->nbytes()));
      if (x.dtype() != float16) {
        cpu_input = std::make_unique<array>(Shape{M, K}, float16, nullptr,
                                            std::vector<array>{});
        cpu_input->set_data(allocator::malloc(cpu_input->nbytes()));
      }
    }

    auto library =
        device.get_library("omlx_qwen35_prefill_kernels", binary_dir());
    auto pack = device.get_kernel(
        std::string(cpu_input ? "qwen35_ane_pack_input_dual_cpu_"
                              : "qwen35_ane_pack_input_dual_") +
            metal_type_name(x.dtype()),
        library);
    encoder.set_compute_pipeline_state(pack);
    encoder.set_input_array(x, 0);
    encoder.set_buffer(model0_->input_buffer(), 1);
    encoder.set_buffer(model1_->input_buffer(), 2);
    if (cpu_input) {
      encoder.set_output_array(*cpu_input, 3);
      encoder.set_bytes(M, 4);
      encoder.set_bytes(K, 5);
    } else {
      encoder.set_bytes(M, 3);
      encoder.set_bytes(K, 4);
    }
    encoder.dispatch_threads(MTL::Size(K, M, 1), MTL::Size(16, 16, 1));
    encoder.end_encoding();

    auto *producer_buffer = encoder.get_command_buffer();
    producer_buffer->retain();
    AneTicketPair tickets{};
    try {
      tickets = begin_ane_ticket_pair(model0_, model1_, producer_buffer);
    } catch (...) {
      producer_buffer->release();
      throw;
    }
    const auto ticket0 = tickets.first;
    const auto ticket1 = tickets.second;
    // Cancel both tickets and free the buffer if anything below throws before
    // the detached evaluation threads take ownership (finding C1).
    AneDispatchGuard ane_guard(producer_buffer, model0_, ticket0, model1_,
                               ticket1);
    encoder.commit();
    producer_buffer->waitUntilCompleted();
    if (pack_buffer_failed(producer_buffer)) {
      // ane_guard releases the buffer and cancels both tickets on unwind.
      throw std::runtime_error(pack_buffer_error(producer_buffer));
    }
    producer_buffer->release();
    ane_guard.producer_released();
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
    const uint64_t launch = profiling ? profile_now_ns() : 0;
    id<MTLCommandBuffer> qmm_buffer =
        (__bridge id<MTLCommandBuffer>)(static_cast<void *>(
            encoder.get_command_buffer()));
    [qmm_buffer retain];
    if (profiling) {
      [qmm_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        const double duration = completed.GPUEndTime - completed.GPUStartTime;
        if (duration > 0) {
          profile_add(profile_category, kGpuQmmNs,
                      static_cast<uint64_t>(duration * 1e9));
        }
        profile_add(profile_category, kGpuCompletionNs,
                    profile_now_ns() - launch);
      }];
    }
    encoder.commit();

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
    ane_guard.transfer_ticket0();
    std::thread([model1, ticket1, profiling, profile_category, launch] {
      const uint64_t start = profiling ? profile_now_ns() : 0;
      model1->execute(ticket1);
      if (profiling) {
        const uint64_t end = profile_now_ns();
        profile_add(profile_category, kAne1LaunchNs, start - launch);
        profile_add(profile_category, kAne1EvalNs, end - start);
      }
    }).detach();
    ane_guard.transfer_ticket1();
    ane_guard.disarm();
    try {
      if (cpu_weight) {
        const uint64_t cpu_start = profiling ? profile_now_ns() : 0;
        cpu_fp16_matmul(cpu_input ? *cpu_input : x, *cpu_weight, *cpu_output,
                        M, cpu_n, K, cpu_threads_, cpu_shared_resource_);
        if (profiling) {
          const uint64_t cpu_done = profile_now_ns();
          profile_add(profile_category, kCpuMatmulNs, cpu_done - cpu_start);
          profile_add(profile_category, kCpuCompletionNs, cpu_done - launch);
        }
      }
      model0_->wait(ticket0);
      model1_->wait(ticket1);
    } catch (...) {
      // Same unwind hazard as the single-instance path: drain the committed
      // qmm before this frame's MLX arrays can be recycled. The detached ANE
      // threads hold their own model references.
      [qmm_buffer waitUntilCompleted];
      [qmm_buffer release];
      throw;
    }
    // ANE completion does not order the separately committed GPU suffix
    // before this primitive's merge. Join the already-overlapped branches so
    // the merge cannot observe an in-flight qmm output on its first dispatch.
    const bool gpu_finished_at_ane_done =
        qmm_buffer.status == MTLCommandBufferStatusCompleted;
    [qmm_buffer waitUntilCompleted];
    if (qmm_buffer.status == MTLCommandBufferStatusError) {
      const std::string error = error_text(
          @"Dual ANE hybrid GPU suffix command buffer failed",
          qmm_buffer.error);
      [qmm_buffer release];
      throw std::runtime_error(error);
    }
    [qmm_buffer release];
    if (profiling) {
      const uint64_t done = profile_now_ns();
      profile_add(profile_category, kAneRegionNs, done - launch);
      g_previous_ane_done_ns.store(done, std::memory_order_relaxed);
      if (gpu_finished_at_ane_done) {
        profile_add(profile_category, kAneLast, 1);
      } else {
        profile_add(profile_category, kGpuLast, 1);
      }
    }

    auto merge = device.get_kernel(
        std::string(cpu_weight
                        ? (fuse_swiglu_
                               ? "qwen35_ane_merge_dual_cpu_swiglu_output_"
                               : "qwen35_ane_merge_dual_cpu_output_")
                        : (fuse_swiglu_
                               ? "qwen35_ane_merge_dual_swiglu_output_"
                               : "qwen35_ane_merge_dual_output_")) +
            metal_type_name(x.dtype()),
        library);
    encoder.set_compute_pipeline_state(merge);
    encoder.set_buffer(model0_->output_buffer(), 0);
    encoder.set_buffer(model1_->output_buffer(), 1);
    if (cpu_weight) {
      encoder.set_input_array(*cpu_output, 2);
      encoder.set_input_array(gpu_output, 3);
      encoder.set_output_array(output, 4);
      encoder.set_bytes(M, 5);
    } else {
      encoder.set_input_array(gpu_output, 2);
      encoder.set_output_array(output, 3);
      encoder.set_bytes(M, 4);
    }
    const int output_n = fuse_swiglu_ ? (ane0_n + ane1_n + cpu_n + gpu_n) / 2
                                      : ane0_n + ane1_n + cpu_n + gpu_n;
    const int merge_ane0_n = fuse_swiglu_ ? ane0_n / 2 : ane0_n;
    const int merge_ane1_n = fuse_swiglu_ ? ane1_n / 2 : ane1_n;
    const int merge_gpu_n = fuse_swiglu_ ? gpu_n / 2 : gpu_n;
    if (cpu_weight) {
      const int merge_cpu_n = fuse_swiglu_ ? cpu_n / 2 : cpu_n;
      encoder.set_bytes(merge_ane0_n, 6);
      encoder.set_bytes(merge_ane1_n, 7);
      encoder.set_bytes(merge_cpu_n, 8);
      encoder.set_bytes(merge_gpu_n, 9);
    } else {
      encoder.set_bytes(merge_ane0_n, 5);
      encoder.set_bytes(merge_ane1_n, 6);
      encoder.set_bytes(merge_gpu_n, 7);
    }
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
    if (cpu_input) {
      encoder.add_temporary(std::move(*cpu_input));
    }
    if (cpu_output) {
      encoder.add_temporary(std::move(*cpu_output));
    }
  }

  DEFINE_NAME(Qwen35DualAneHybrid)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs = static_cast<const DualAneHybridPrimitive &>(other);
    return model0_.get() == rhs.model0_.get() &&
           model1_.get() == rhs.model1_.get() && bits_ == rhs.bits_ &&
           variant_ == rhs.variant_ && group_size_ == rhs.group_size_ &&
           fuse_swiglu_ == rhs.fuse_swiglu_ &&
           profile_category_ == rhs.profile_category_ &&
           cpu_fp16_ == rhs.cpu_fp16_ && cpu_threads_ == rhs.cpu_threads_ &&
           cpu_shared_resource_ == rhs.cpu_shared_resource_;
  }
  auto state() const {
    return std::make_tuple(reinterpret_cast<uintptr_t>(model0_.get()),
                           reinterpret_cast<uintptr_t>(model1_.get()), bits_,
                           variant_, group_size_, fuse_swiglu_,
                           profile_category_, cpu_fp16_, cpu_threads_,
                           cpu_shared_resource_);
  }

private:
  std::shared_ptr<AneLinearModel> model0_;
  std::shared_ptr<AneLinearModel> model1_;
  int variant_;
  int bits_;
  int group_size_;
  bool fuse_swiglu_;
  int profile_category_;
  bool cpu_fp16_;
  int cpu_threads_;
  bool cpu_shared_resource_;
};

class AneHybridQ4SwiGLUDownPrimitive : public Primitive {
public:
  AneHybridQ4SwiGLUDownPrimitive(Stream stream,
                                 std::shared_ptr<AneLinearModel> model,
                                 int variant, int group_size,
                                 std::shared_ptr<AneLinearModel> model1 = nullptr,
                                 bool cpu_fp16 = false, int cpu_threads = 0,
                                 bool cpu_shared_resource = false)
      : Primitive(stream), model_(std::move(model)), model1_(std::move(model1)),
        variant_(variant), group_size_(group_size), cpu_fp16_(cpu_fp16),
        cpu_threads_(cpu_threads),
        cpu_shared_resource_(cpu_shared_resource) {}

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
    const array *cpu_gate_up_weight = cpu_fp16_ ? &inputs[7] : nullptr;
    const array *cpu_down_weight = cpu_fp16_ ? &inputs[8] : nullptr;
    auto &output = outputs[0];

    const int input_dim = static_cast<int>(x.shape(-1));
    const int M = static_cast<int>(x.size() / input_dim);
    const int gpu_hidden = static_cast<int>(gate_up_weight.shape(0)) / 2;
    const int cpu_hidden =
        cpu_gate_up_weight
            ? static_cast<int>(cpu_gate_up_weight->shape(0)) / 2
            : 0;
    const int output_dim = model_->output_dim();
    const bool profiling = ane_profile_enabled();
    constexpr int profile_category = 0;
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
    array gate_up({M, 2 * gpu_hidden}, x.dtype(), nullptr, {});
    array activation({M, gpu_hidden}, x.dtype(), nullptr, {});
    array gpu_output({M, output_dim}, x.dtype(), nullptr, {});
    gate_up.set_data(allocator::malloc(gate_up.nbytes()));
    activation.set_data(allocator::malloc(activation.nbytes()));
    gpu_output.set_data(allocator::malloc(gpu_output.nbytes()));
    std::unique_ptr<array> cpu_input;
    std::unique_ptr<array> cpu_gate_up;
    std::unique_ptr<array> cpu_activation;
    std::unique_ptr<array> cpu_output;
    if (cpu_fp16_) {
      if (x.dtype() != float16) {
        cpu_input = std::make_unique<array>(Shape{M, input_dim}, float16,
                                            nullptr, std::vector<array>{});
        cpu_input->set_data(allocator::malloc(cpu_input->nbytes()));
      }
      cpu_gate_up = std::make_unique<array>(Shape{M, 2 * cpu_hidden}, float16,
                                            nullptr, std::vector<array>{});
      cpu_activation = std::make_unique<array>(Shape{M, cpu_hidden}, float16,
                                               nullptr, std::vector<array>{});
      cpu_output = std::make_unique<array>(Shape{M, output_dim}, float16,
                                           nullptr, std::vector<array>{});
      cpu_gate_up->set_data(allocator::malloc(cpu_gate_up->nbytes()));
      cpu_activation->set_data(allocator::malloc(cpu_activation->nbytes()));
      cpu_output->set_data(allocator::malloc(cpu_output->nbytes()));
    }

    auto library =
        device.get_library("omlx_qwen35_prefill_kernels", binary_dir());
    auto pack = device.get_kernel(
        std::string(model1_ ? "qwen35_ane_pack_input_dual_"
                            : "qwen35_ane_pack_input_") +
            metal_type_name(x.dtype()),
        library);
    encoder.set_compute_pipeline_state(pack);
    encoder.set_input_array(x, 0);
    encoder.set_buffer(model_->input_buffer(), 1);
    if (model1_) {
      encoder.set_buffer(model1_->input_buffer(), 2);
      encoder.set_bytes(M, 3);
      encoder.set_bytes(input_dim, 4);
    } else {
      encoder.set_bytes(M, 2);
      encoder.set_bytes(input_dim, 3);
    }
    encoder.dispatch_threads(MTL::Size(input_dim, M, 1),
                             MTL::Size(16, 16, 1));
    encoder.end_encoding();
    if (cpu_input) {
      auto cpu_pack = device.get_kernel(
          "qwen35_cpu_pack_input_" + metal_type_name(x.dtype()), library);
      encoder.set_compute_pipeline_state(cpu_pack);
      encoder.set_input_array(x, 0);
      encoder.set_output_array(*cpu_input, 1);
      encoder.set_bytes(M, 2);
      encoder.set_bytes(input_dim, 3);
      encoder.dispatch_threads(MTL::Size(input_dim, M, 1),
                               MTL::Size(16, 16, 1));
      encoder.end_encoding();
    }

    auto *producer_buffer = encoder.get_command_buffer();
    producer_buffer->retain();
    AneLinearModel::Ticket ticket{};
    AneLinearModel::Ticket ticket1{};
    try {
      if (model1_) {
        const auto tickets =
            begin_ane_ticket_pair(model_, model1_, producer_buffer);
        ticket = tickets.first;
        ticket1 = tickets.second;
      } else {
        ticket = model_->begin(producer_buffer);
      }
    } catch (...) {
      producer_buffer->release();
      throw;
    }
    // Cancel the tickets and free the buffer if anything below throws before
    // the detached evaluation threads take ownership (finding C1).
    AneDispatchGuard ane_guard(producer_buffer, model_, ticket, model1_,
                               ticket1);
    encoder.commit();
    producer_buffer->waitUntilCompleted();
    if (pack_buffer_failed(producer_buffer)) {
      // ane_guard releases the buffer and cancels the tickets on unwind.
      throw std::runtime_error(pack_buffer_error(producer_buffer));
    }
    producer_buffer->release();
    ane_guard.producer_released();
    if (profiling) {
      profile_add(profile_category, kPackNs,
                  profile_now_ns() - operation_start);
    }

    const std::string qmm_type = metal_type_name(x.dtype());
    auto gate_up_qmm = hybrid_qmm_kernel(
        device, library, 4, group_size_, qmm_type, input_dim,
        2 * gpu_hidden);
    encoder.set_compute_pipeline_state(gate_up_qmm);
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

    auto down_qmm = hybrid_qmm_kernel(
        device, library, 4, group_size_, qmm_type, gpu_hidden, output_dim);
    encoder.set_compute_pipeline_state(down_qmm);
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

    const uint64_t launch = profiling ? profile_now_ns() : 0;
    id<MTLCommandBuffer> suffix_buffer =
        (__bridge id<MTLCommandBuffer>)(static_cast<void *>(
            encoder.get_command_buffer()));
    if (profiling) {
      [suffix_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        const double duration = completed.GPUEndTime - completed.GPUStartTime;
        if (duration > 0) {
          profile_add(profile_category, kGpuQmmNs,
                      static_cast<uint64_t>(duration * 1e9));
        }
        profile_add(profile_category, kGpuCompletionNs,
                    profile_now_ns() - launch);
      }];
    }
    [suffix_buffer retain];
    encoder.commit();
    auto model = model_;
    std::thread([model = std::move(model), ticket, profiling, launch] {
      const uint64_t start = profiling ? profile_now_ns() : 0;
      model->execute(ticket);
      if (profiling) {
        const uint64_t end = profile_now_ns();
        profile_add(profile_category, kAne0LaunchNs, start - launch);
        profile_add(profile_category, kAne0EvalNs, end - start);
      }
    }).detach();
    ane_guard.transfer_ticket0();
    if (model1_) {
      auto model1 = model1_;
      std::thread([model1 = std::move(model1), ticket1, profiling, launch] {
        const uint64_t start = profiling ? profile_now_ns() : 0;
        model1->execute(ticket1);
        if (profiling) {
          const uint64_t end = profile_now_ns();
          profile_add(profile_category, kAne1LaunchNs, start - launch);
          profile_add(profile_category, kAne1EvalNs, end - start);
        }
      }).detach();
      ane_guard.transfer_ticket1();
    }
    ane_guard.disarm();
    try {
      if (cpu_fp16_) {
        const uint64_t cpu_start = profiling ? profile_now_ns() : 0;
        cpu_fp16_matmul(cpu_input ? *cpu_input : x, *cpu_gate_up_weight,
                        *cpu_gate_up, M, 2 * cpu_hidden, input_dim,
                        cpu_threads_, cpu_shared_resource_);
        cpu_fp16_swiglu(*cpu_gate_up, *cpu_activation, M, cpu_hidden,
                        cpu_threads_, cpu_shared_resource_);
        cpu_fp16_matmul(*cpu_activation, *cpu_down_weight, *cpu_output, M,
                        output_dim, cpu_hidden, cpu_threads_,
                        cpu_shared_resource_);
        if (profiling) {
          const uint64_t cpu_done = profile_now_ns();
          profile_add(profile_category, kCpuMatmulNs, cpu_done - cpu_start);
          profile_add(profile_category, kCpuCompletionNs, cpu_done - launch);
        }
      }
      model_->wait(ticket);
      if (model1_) {
        model1_->wait(ticket1);
      }
    } catch (...) {
      // Same unwind hazard as the shared linear paths: drain the committed
      // suffix before this frame's MLX arrays can be recycled. The detached
      // ANE threads hold their own model references.
      [suffix_buffer waitUntilCompleted];
      [suffix_buffer release];
      throw;
    }
    [suffix_buffer waitUntilCompleted];
    [suffix_buffer release];
    if (profiling) {
      const uint64_t done = profile_now_ns();
      profile_add(profile_category, kAneRegionNs, done - launch);
      g_previous_ane_done_ns.store(done, std::memory_order_relaxed);
    }

    auto sum = device.get_kernel(
        std::string(cpu_fp16_ ? "qwen35_ane_sum_dual_cpu_output_"
                              : model1_ ? "qwen35_ane_sum_dual_output_"
                            : "qwen35_ane_sum_output_") +
            metal_type_name(x.dtype()),
        library);
    encoder.set_compute_pipeline_state(sum);
    encoder.set_buffer(model_->output_buffer(), 0);
    if (cpu_fp16_) {
      encoder.set_buffer(model1_->output_buffer(), 1);
      encoder.set_input_array(*cpu_output, 2);
      encoder.set_input_array(gpu_output, 3);
      encoder.set_output_array(output, 4);
      encoder.set_bytes(M, 5);
      encoder.set_bytes(output_dim, 6);
    } else if (model1_) {
      encoder.set_buffer(model1_->output_buffer(), 1);
      encoder.set_input_array(gpu_output, 2);
      encoder.set_output_array(output, 3);
      encoder.set_bytes(M, 4);
      encoder.set_bytes(output_dim, 5);
    } else {
      encoder.set_input_array(gpu_output, 1);
      encoder.set_output_array(output, 2);
      encoder.set_bytes(M, 3);
      encoder.set_bytes(output_dim, 4);
    }
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
    if (cpu_input) {
      encoder.add_temporary(std::move(*cpu_input));
    }
    if (cpu_gate_up) {
      encoder.add_temporary(std::move(*cpu_gate_up));
      encoder.add_temporary(std::move(*cpu_activation));
      encoder.add_temporary(std::move(*cpu_output));
    }
  }

  DEFINE_NAME(Qwen35AneHybridQ4SwiGLUDown)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs =
        static_cast<const AneHybridQ4SwiGLUDownPrimitive &>(other);
    return model_.get() == rhs.model_.get() &&
           model1_.get() == rhs.model1_.get() && variant_ == rhs.variant_ &&
           group_size_ == rhs.group_size_ && cpu_fp16_ == rhs.cpu_fp16_ &&
           cpu_threads_ == rhs.cpu_threads_ &&
           cpu_shared_resource_ == rhs.cpu_shared_resource_;
  }
  auto state() const {
    return std::make_tuple(reinterpret_cast<uintptr_t>(model_.get()),
                           reinterpret_cast<uintptr_t>(model1_.get()), variant_,
                           group_size_, cpu_fp16_, cpu_threads_,
                           cpu_shared_resource_);
  }

private:
  std::shared_ptr<AneLinearModel> model_;
  std::shared_ptr<AneLinearModel> model1_;
  int variant_;
  int group_size_;
  bool cpu_fp16_;
  int cpu_threads_;
  bool cpu_shared_resource_;
};

} // namespace

bool qwen35_ane_hybrid_nax_enabled() { return hybrid_nax_enabled(); }

bool qwen35_cpu_shared_resource_available() {
  return cpu_shared_resource_policy_available();
}

array qwen35_cpu_fp16_affine_qmm_t(
    const array &x, const array &cpu_weight, const array &gpu_weight,
    const array &gpu_scales, const array &gpu_biases, int bits, int variant,
    int group_size, int cpu_threads, bool cpu_shared_resource,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  if (stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || cpu_weight.dtype() != float16 ||
      cpu_weight.ndim() != 2 || !row_contiguous(cpu_weight) ||
      gpu_weight.dtype() != mlx::core::uint32 || gpu_weight.ndim() != 2 ||
      !row_contiguous(gpu_weight) || gpu_scales.dtype() != x.dtype() ||
      gpu_biases.dtype() != x.dtype() || !row_contiguous(gpu_scales) ||
      !row_contiguous(gpu_biases) || gpu_scales.shape() != gpu_biases.shape() ||
      (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (group_size != 64 && group_size != 128) || variant != 8 ||
      cpu_threads < 0 || cpu_threads > 64) {
    throw std::invalid_argument("Unsupported CPU/GPU fp16 hybrid qmm.");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / K);
  const int cpu_n = static_cast<int>(cpu_weight.shape(0));
  const int gpu_n = static_cast<int>(gpu_weight.shape(0));
  if (M <= 0 || cpu_n <= 0 || gpu_n <= 0 || cpu_n % 64 || gpu_n % 64 ||
      cpu_weight.shape(1) != K || gpu_weight.shape(1) * 32 != K * bits ||
      gpu_scales.shape(0) != gpu_n ||
      gpu_scales.shape(1) != K / group_size) {
    throw std::invalid_argument("CPU/GPU fp16 hybrid qmm shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = cpu_n + gpu_n;
  return array(std::move(shape), x.dtype(),
               std::make_shared<CpuFp16HybridPrimitive>(
                   stream, bits, variant, group_size, cpu_threads,
                   cpu_shared_resource),
               std::vector<array>{x, cpu_weight, gpu_weight, gpu_scales,
                                  gpu_biases});
}

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

array qwen35_ane_cpu_fp16_affine_qmm_t(
    const array &x, const array &cpu_weight, const array &gpu_weight,
    const array &gpu_scales, const array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int bits, int variant,
    int group_size, int profile_category, int cpu_threads,
    bool cpu_shared_resource, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!ane_model || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || cpu_weight.dtype() != float16 ||
      cpu_weight.ndim() != 2 || !row_contiguous(cpu_weight) ||
      gpu_weight.dtype() != mlx::core::uint32 || gpu_weight.ndim() != 2 ||
      !row_contiguous(gpu_weight) || gpu_scales.dtype() != x.dtype() ||
      gpu_biases.dtype() != x.dtype() ||
      gpu_scales.shape() != gpu_biases.shape() || !row_contiguous(gpu_scales) ||
      !row_contiguous(gpu_biases) ||
      (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (group_size != 64 && group_size != 128) || variant != 8 ||
      cpu_threads < 0 || cpu_threads > 64) {
    throw std::invalid_argument("Unsupported ANE/CPU/GPU affine qmm.");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / K);
  const int cpu_n = static_cast<int>(cpu_weight.shape(0));
  const int gpu_n = static_cast<int>(gpu_weight.shape(0));
  if (K != ane_model->input_dim() || M != ane_model->sequence_length() ||
      cpu_n <= 0 || cpu_n % 64 || gpu_n <= 0 || gpu_n % 64 ||
      cpu_weight.shape(1) != K || K % group_size != 0 ||
      gpu_weight.shape(1) * 32 != K * bits || gpu_scales.shape(0) != gpu_n ||
      gpu_scales.shape(1) != K / group_size) {
    throw std::invalid_argument("ANE/CPU/GPU affine qmm shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = ane_model->output_dim() + cpu_n + gpu_n;
  return array(
      std::move(shape), x.dtype(),
      std::make_shared<AneHybridQ4Primitive>(
          stream, ane_model, bits, variant, group_size,
          /* fuse_swiglu */ false, profile_category,
          /* cpu_fp16 */ true, cpu_threads, cpu_shared_resource),
      std::vector<array>{x, gpu_weight, gpu_scales, gpu_biases, cpu_weight});
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

array qwen35_ane_cpu_fp16_swiglu_t(
    const array &x, const array &cpu_weight, const array &gpu_weight,
    const array &gpu_scales, const array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int bits, int variant,
    int group_size, int cpu_threads, bool cpu_shared_resource,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!ane_model || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || cpu_weight.dtype() != float16 ||
      cpu_weight.ndim() != 2 || !row_contiguous(cpu_weight) ||
      gpu_weight.dtype() != mlx::core::uint32 || gpu_weight.ndim() != 2 ||
      !row_contiguous(gpu_weight) || gpu_scales.dtype() != x.dtype() ||
      gpu_biases.dtype() != x.dtype() ||
      gpu_scales.shape() != gpu_biases.shape() || !row_contiguous(gpu_scales) ||
      !row_contiguous(gpu_biases) ||
      (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (group_size != 64 && group_size != 128) || variant != 8 ||
      cpu_threads < 0 || cpu_threads > 64) {
    throw std::invalid_argument("Unsupported ANE/CPU/GPU SwiGLU.");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / K);
  const int cpu_n = static_cast<int>(cpu_weight.shape(0));
  const int gpu_n = static_cast<int>(gpu_weight.shape(0));
  const int ane_n = ane_model->output_dim();
  if (K != ane_model->input_dim() || M != ane_model->sequence_length() ||
      ane_n <= 0 || ane_n % 2 || cpu_n <= 0 || cpu_n % 128 || gpu_n <= 0 ||
      gpu_n % 128 || cpu_weight.shape(1) != K || K % group_size != 0 ||
      gpu_weight.shape(1) * 32 != K * bits || gpu_scales.shape(0) != gpu_n ||
      gpu_scales.shape(1) != K / group_size) {
    throw std::invalid_argument("ANE/CPU/GPU SwiGLU shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = (ane_n + cpu_n + gpu_n) / 2;
  return array(
      std::move(shape), x.dtype(),
      std::make_shared<AneHybridQ4Primitive>(
          stream, ane_model, bits, variant, group_size,
          /* fuse_swiglu */ true, /* profile_category */ 0,
          /* cpu_fp16 */ true, cpu_threads, cpu_shared_resource),
      std::vector<array>{x, gpu_weight, gpu_scales, gpu_biases, cpu_weight});
}

array qwen35_ane_cpu_fp16_q4_swiglu_t(
    const array &x, const array &cpu_weight, const array &gpu_weight,
    const array &gpu_scales, const array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int variant,
    int group_size, int cpu_threads, bool cpu_shared_resource,
    StreamOrDevice s) {
  return qwen35_ane_cpu_fp16_swiglu_t(
      x, cpu_weight, gpu_weight, gpu_scales, gpu_biases, ane_model, 4, variant,
      group_size, cpu_threads, cpu_shared_resource, s);
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

array qwen35_ane_dual_cpu_fp16_affine_qmm_t(
    const array &x, const array &cpu_weight, const array &gpu_weight,
    const array &gpu_scales, const array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int bits, int variant,
    int group_size, int profile_category, int cpu_threads,
    bool cpu_shared_resource, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!ane_model0 || !ane_model1 || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || cpu_weight.dtype() != float16 ||
      cpu_weight.ndim() != 2 || !row_contiguous(cpu_weight) ||
      gpu_weight.dtype() != mlx::core::uint32 || gpu_weight.ndim() != 2 ||
      !row_contiguous(gpu_weight) || gpu_scales.dtype() != x.dtype() ||
      gpu_biases.dtype() != x.dtype() || gpu_scales.shape() != gpu_biases.shape() ||
      !row_contiguous(gpu_scales) || !row_contiguous(gpu_biases) ||
      (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (group_size != 64 && group_size != 128) || variant != 8 ||
      cpu_threads < 0 || cpu_threads > 64) {
    throw std::invalid_argument("Unsupported dual ANE/CPU/GPU affine qmm.");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / K);
  const int cpu_n = static_cast<int>(cpu_weight.shape(0));
  const int gpu_n = static_cast<int>(gpu_weight.shape(0));
  if (K != ane_model0->input_dim() || K != ane_model1->input_dim() ||
      M != ane_model0->sequence_length() || M != ane_model1->sequence_length() ||
      cpu_n <= 0 || cpu_n % 64 || gpu_n <= 0 || gpu_n % 64 ||
      cpu_weight.shape(1) != K || K % group_size != 0 ||
      gpu_weight.shape(1) * 32 != K * bits ||
      gpu_scales.shape(0) != gpu_n ||
      gpu_scales.shape(1) != K / group_size) {
    throw std::invalid_argument("Dual ANE/CPU/GPU affine qmm shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = ane_model0->output_dim() + ane_model1->output_dim() +
                 cpu_n + gpu_n;
  return array(
      std::move(shape), x.dtype(),
      std::make_shared<DualAneHybridPrimitive>(
          stream, ane_model0, ane_model1, bits, variant, group_size,
          /* fuse_swiglu */ false, profile_category,
          /* cpu_fp16 */ true, cpu_threads, cpu_shared_resource),
      std::vector<array>{x, gpu_weight, gpu_scales, gpu_biases, cpu_weight});
}

array qwen35_ane_dual_cpu_fp16_swiglu_t(
    const array &x, const array &cpu_weight, const array &gpu_weight,
    const array &gpu_scales, const array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int bits, int variant,
    int group_size, int cpu_threads, bool cpu_shared_resource,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!ane_model0 || !ane_model1 || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || cpu_weight.dtype() != float16 ||
      cpu_weight.ndim() != 2 || !row_contiguous(cpu_weight) ||
      gpu_weight.dtype() != mlx::core::uint32 || gpu_weight.ndim() != 2 ||
      !row_contiguous(gpu_weight) || gpu_scales.dtype() != x.dtype() ||
      gpu_biases.dtype() != x.dtype() ||
      gpu_scales.shape() != gpu_biases.shape() ||
      !row_contiguous(gpu_scales) || !row_contiguous(gpu_biases) ||
      (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
      (group_size != 64 && group_size != 128) || variant != 8 ||
      cpu_threads < 0 || cpu_threads > 64) {
    throw std::invalid_argument("Unsupported dual ANE/CPU/GPU SwiGLU.");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / K);
  const int cpu_n = static_cast<int>(cpu_weight.shape(0));
  const int gpu_n = static_cast<int>(gpu_weight.shape(0));
  const int ane0_n = ane_model0->output_dim();
  const int ane1_n = ane_model1->output_dim();
  if (K != ane_model0->input_dim() || K != ane_model1->input_dim() ||
      M != ane_model0->sequence_length() ||
      M != ane_model1->sequence_length() || ane0_n <= 0 || ane1_n <= 0 ||
      ane0_n % 2 || ane1_n % 2 || cpu_n <= 0 || cpu_n % 128 ||
      gpu_n <= 0 || gpu_n % 128 || cpu_weight.shape(1) != K ||
      K % group_size != 0 || gpu_weight.shape(1) * 32 != K * bits ||
      gpu_scales.shape(0) != gpu_n ||
      gpu_scales.shape(1) != K / group_size) {
    throw std::invalid_argument("Dual ANE/CPU/GPU SwiGLU shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = (ane0_n + ane1_n + cpu_n + gpu_n) / 2;
  return array(
      std::move(shape), x.dtype(),
      std::make_shared<DualAneHybridPrimitive>(
          stream, ane_model0, ane_model1, bits, variant, group_size,
          /* fuse_swiglu */ true, /* profile_category */ 0,
          /* cpu_fp16 */ true, cpu_threads, cpu_shared_resource),
      std::vector<array>{x, gpu_weight, gpu_scales, gpu_biases, cpu_weight});
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

array qwen35_ane_dual_cpu_fp16_q4_swiglu_t(
    const array &x, const array &cpu_weight, const array &gpu_weight,
    const array &gpu_scales, const array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int variant,
    int group_size, int cpu_threads, bool cpu_shared_resource,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!ane_model0 || !ane_model1 || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || cpu_weight.dtype() != float16 ||
      cpu_weight.ndim() != 2 || !row_contiguous(cpu_weight) ||
      gpu_weight.dtype() != mlx::core::uint32 || gpu_weight.ndim() != 2 ||
      !row_contiguous(gpu_weight) || gpu_scales.dtype() != x.dtype() ||
      gpu_biases.dtype() != x.dtype() || gpu_scales.shape() != gpu_biases.shape() ||
      !row_contiguous(gpu_scales) || !row_contiguous(gpu_biases) ||
      (group_size != 64 && group_size != 128) || variant != 8 ||
      cpu_threads < 0 || cpu_threads > 64) {
    throw std::invalid_argument("Unsupported dual ANE/CPU/GPU q4 SwiGLU.");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / K);
  const int cpu_n = static_cast<int>(cpu_weight.shape(0));
  const int gpu_n = static_cast<int>(gpu_weight.shape(0));
  const int ane0_n = ane_model0->output_dim();
  const int ane1_n = ane_model1->output_dim();
  if (K != ane_model0->input_dim() || K != ane_model1->input_dim() ||
      M != ane_model0->sequence_length() || M != ane_model1->sequence_length() ||
      ane0_n <= 0 || ane1_n <= 0 || ane0_n % 2 || ane1_n % 2 ||
      cpu_n <= 0 || cpu_n % 128 || gpu_n <= 0 || gpu_n % 128 ||
      cpu_weight.shape(1) != K || gpu_weight.shape(1) * 8 != K ||
      gpu_scales.shape(0) != gpu_n ||
      gpu_scales.shape(1) != K / group_size) {
    throw std::invalid_argument("Dual ANE/CPU/GPU q4 SwiGLU shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = (ane0_n + ane1_n + cpu_n + gpu_n) / 2;
  return array(
      std::move(shape), x.dtype(),
      std::make_shared<DualAneHybridPrimitive>(
          stream, ane_model0, ane_model1, 4, variant, group_size,
          /* fuse_swiglu */ true, /* profile_category */ 0,
          /* cpu_fp16 */ true, cpu_threads, cpu_shared_resource),
      std::vector<array>{x, gpu_weight, gpu_scales, gpu_biases, cpu_weight});
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

array qwen35_ane_dual_q4_swiglu_down_t(
    const array &x, const array &gpu_gate_up_weight,
    const array &gpu_gate_up_scales, const array &gpu_gate_up_biases,
    const array &gpu_down_weight, const array &gpu_down_scales,
    const array &gpu_down_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int variant,
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
  if (!ane_model0 || !ane_model1 || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || group_size != 128 || variant != 8) {
    throw std::invalid_argument(
        "Unsupported dual-ANE hybrid SwiGLU/down configuration.");
  }
  const int input_dim = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / input_dim);
  const int gate_up_n = static_cast<int>(gpu_gate_up_weight.shape(0));
  const int gpu_hidden = gate_up_n / 2;
  const int output_dim = static_cast<int>(gpu_down_weight.shape(0));
  if (input_dim != ane_model0->input_dim() ||
      input_dim != ane_model1->input_dim() ||
      output_dim != ane_model0->output_dim() ||
      output_dim != ane_model1->output_dim() ||
      M != ane_model0->sequence_length() ||
      M != ane_model1->sequence_length() || gate_up_n <= 0 ||
      gate_up_n % 128 != 0 || gpu_hidden % 128 != 0 ||
      output_dim % 64 != 0 ||
      !q4_valid(gpu_gate_up_weight, gpu_gate_up_scales, gpu_gate_up_biases,
                input_dim) ||
      !q4_valid(gpu_down_weight, gpu_down_scales, gpu_down_biases,
                gpu_hidden)) {
    throw std::invalid_argument(
        "Dual-ANE hybrid SwiGLU/down shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = output_dim;
  return array(
      std::move(shape), x.dtype(),
      std::make_shared<AneHybridQ4SwiGLUDownPrimitive>(
          stream, ane_model0, variant, group_size, ane_model1),
      std::vector<array>{x, gpu_gate_up_weight, gpu_gate_up_scales,
                         gpu_gate_up_biases, gpu_down_weight, gpu_down_scales,
                         gpu_down_biases});
}

array qwen35_ane_dual_cpu_fp16_q4_swiglu_down_t(
    const array &x, const array &cpu_gate_up_weight,
    const array &cpu_down_weight, const array &gpu_gate_up_weight,
    const array &gpu_gate_up_scales, const array &gpu_gate_up_biases,
    const array &gpu_down_weight, const array &gpu_down_scales,
    const array &gpu_down_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int variant,
    int group_size, int cpu_threads, bool cpu_shared_resource,
    StreamOrDevice s) {
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
  if (!ane_model0 || !ane_model1 || stream.device == Device::cpu ||
      (x.dtype() != float16 && x.dtype() != bfloat16) || x.ndim() < 2 ||
      !row_contiguous(x) || group_size != 128 || variant != 8 ||
      cpu_threads < 0 || cpu_threads > 64 ||
      cpu_gate_up_weight.dtype() != float16 ||
      cpu_down_weight.dtype() != float16 || cpu_gate_up_weight.ndim() != 2 ||
      cpu_down_weight.ndim() != 2 || !row_contiguous(cpu_gate_up_weight) ||
      !row_contiguous(cpu_down_weight)) {
    throw std::invalid_argument(
        "Unsupported dual-ANE/CPU fused SwiGLU/down configuration.");
  }
  const int input_dim = static_cast<int>(x.shape(-1));
  const int M = static_cast<int>(x.size() / input_dim);
  const int gate_up_n = static_cast<int>(gpu_gate_up_weight.shape(0));
  const int gpu_hidden = gate_up_n / 2;
  const int cpu_gate_up_n = static_cast<int>(cpu_gate_up_weight.shape(0));
  const int cpu_hidden = cpu_gate_up_n / 2;
  const int output_dim = static_cast<int>(gpu_down_weight.shape(0));
  if (input_dim != ane_model0->input_dim() ||
      input_dim != ane_model1->input_dim() ||
      output_dim != ane_model0->output_dim() ||
      output_dim != ane_model1->output_dim() ||
      M != ane_model0->sequence_length() ||
      M != ane_model1->sequence_length() || gate_up_n <= 0 ||
      gate_up_n % 128 != 0 || gpu_hidden % 128 != 0 ||
      cpu_gate_up_n <= 0 || cpu_gate_up_n % 128 != 0 ||
      cpu_hidden % 64 != 0 || output_dim % 64 != 0 ||
      cpu_gate_up_weight.shape(1) != input_dim ||
      cpu_down_weight.shape(0) != output_dim ||
      cpu_down_weight.shape(1) != cpu_hidden ||
      !q4_valid(gpu_gate_up_weight, gpu_gate_up_scales, gpu_gate_up_biases,
                input_dim) ||
      !q4_valid(gpu_down_weight, gpu_down_scales, gpu_down_biases,
                gpu_hidden)) {
    throw std::invalid_argument(
        "Dual-ANE/CPU fused SwiGLU/down shape mismatch.");
  }
  Shape shape = x.shape();
  shape.back() = output_dim;
  return array(
      std::move(shape), x.dtype(),
      std::make_shared<AneHybridQ4SwiGLUDownPrimitive>(
          stream, ane_model0, variant, group_size, ane_model1,
          /* cpu_fp16 */ true, cpu_threads, cpu_shared_resource),
      std::vector<array>{x, gpu_gate_up_weight, gpu_gate_up_scales,
                         gpu_gate_up_biases, gpu_down_weight, gpu_down_scales,
                         gpu_down_biases, cpu_gate_up_weight,
                         cpu_down_weight});
}

} // namespace omlx::qwen35_prefill_kernels
