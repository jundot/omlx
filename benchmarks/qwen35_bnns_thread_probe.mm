#include <Accelerate/Accelerate.h>
#include <mach/mach.h>
#include <mach/thread_info.h>
#include <mach/thread_policy.h>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct ThreadTime {
  uint64_t id = 0;
  double seconds = 0.0;
};

struct CpuLocation {
  size_t cpu = 0;
  uint32_t cluster = 0;
};

CpuLocation current_cpu_location() {
  uint64_t tpidr = 0;
#if defined(__aarch64__)
  __asm__ volatile("mrs %0, tpidr_el0" : "=r"(tpidr));
#endif
  return {static_cast<size_t>(tpidr & 0xfff),
          static_cast<uint32_t>((tpidr >> 12) & 0xff)};
}

double time_value_seconds(const time_value_t &value) {
  return static_cast<double>(value.seconds) +
         static_cast<double>(value.microseconds) * 1e-6;
}

std::map<uint64_t, double> task_thread_times() {
  thread_act_array_t threads = nullptr;
  mach_msg_type_number_t count = 0;
  if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS) {
    throw std::runtime_error("task_threads failed");
  }
  std::map<uint64_t, double> result;
  for (mach_msg_type_number_t i = 0; i < count; ++i) {
    thread_identifier_info_data_t identifier{};
    mach_msg_type_number_t identifier_count = THREAD_IDENTIFIER_INFO_COUNT;
    thread_basic_info_data_t basic{};
    mach_msg_type_number_t basic_count = THREAD_BASIC_INFO_COUNT;
    if (thread_info(threads[i], THREAD_IDENTIFIER_INFO,
                    reinterpret_cast<thread_info_t>(&identifier),
                    &identifier_count) == KERN_SUCCESS &&
        thread_info(threads[i], THREAD_BASIC_INFO,
                    reinterpret_cast<thread_info_t>(&basic),
                    &basic_count) == KERN_SUCCESS) {
      result[identifier.thread_id] = time_value_seconds(basic.user_time) +
                                     time_value_seconds(basic.system_time);
    }
    mach_port_deallocate(mach_task_self(), threads[i]);
  }
  vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads),
                count * sizeof(thread_t));
  return result;
}

ThreadTime current_thread_time() {
  const thread_t thread = mach_thread_self();
  thread_identifier_info_data_t identifier{};
  mach_msg_type_number_t identifier_count = THREAD_IDENTIFIER_INFO_COUNT;
  thread_basic_info_data_t basic{};
  mach_msg_type_number_t basic_count = THREAD_BASIC_INFO_COUNT;
  if (thread_info(thread, THREAD_IDENTIFIER_INFO,
                  reinterpret_cast<thread_info_t>(&identifier),
                  &identifier_count) != KERN_SUCCESS ||
      thread_info(thread, THREAD_BASIC_INFO,
                  reinterpret_cast<thread_info_t>(&basic),
                  &basic_count) != KERN_SUCCESS) {
    mach_port_deallocate(mach_task_self(), thread);
    throw std::runtime_error("thread_info failed");
  }
  mach_port_deallocate(mach_task_self(), thread);
  return {identifier.thread_id, time_value_seconds(basic.user_time) +
                                    time_value_seconds(basic.system_time)};
}

BNNSNDArrayDescriptor matrix_descriptor(void *data, int rows, int cols) {
  BNNSNDArrayDescriptor descriptor{};
  descriptor.layout = BNNSDataLayoutRowMajorMatrix;
  descriptor.size[0] = static_cast<size_t>(cols);
  descriptor.size[1] = static_cast<size_t>(rows);
  descriptor.data = data;
  descriptor.data_type = BNNSDataTypeFloat16;
  descriptor.data_scale = 1.0f;
  return descriptor;
}

struct MatMul {
  BNNSNDArrayDescriptor input{};
  BNNSNDArrayDescriptor weight{};
  BNNSNDArrayDescriptor output{};
  BNNSFilterParameters parameters{};
  std::vector<uint8_t> workspace;

  MatMul(_Float16 *input_data, _Float16 *weight_data, _Float16 *output_data,
         int rows, int columns, int inner, int threads)
      : input(matrix_descriptor(input_data, rows, inner)),
        weight(matrix_descriptor(weight_data, columns, inner)),
        output(matrix_descriptor(output_data, rows, columns)) {
    parameters.n_threads = static_cast<size_t>(std::max(threads, 0));
    const ssize_t size = BNNSMatMulWorkspaceSize(
        false, true, 1.0f, &input, &weight, &output, &parameters);
    if (size < 0) {
      throw std::runtime_error("BNNSMatMulWorkspaceSize rejected the shape");
    }
    workspace.resize(static_cast<size_t>(size));
  }

  void run() {
    if (BNNSMatMul(false, true, 1.0f, &input, &weight, &output,
                   workspace.empty() ? nullptr : workspace.data(),
                   &parameters) != 0) {
      throw std::runtime_error("BNNSMatMul failed");
    }
  }
};

std::vector<ThreadTime>
thread_deltas(const std::map<uint64_t, double> &before,
              const std::map<uint64_t, double> &after) {
  std::vector<ThreadTime> result;
  for (const auto &[id, end] : after) {
    auto it = before.find(id);
    const double start = it == before.end() ? 0.0 : it->second;
    if (end - start >= 0.0005) {
      result.push_back({id, end - start});
    }
  }
  std::sort(result.begin(), result.end(),
            [](const auto &a, const auto &b) { return a.seconds > b.seconds; });
  return result;
}

void print_result(const std::string &mode, int requested, int repeats,
                  double elapsed, std::vector<ThreadTime> workers) {
  const double per_call_ms = elapsed * 1000.0 / repeats;
  double cpu_seconds = 0.0;
  for (const auto &worker : workers) {
    cpu_seconds += worker.seconds;
  }
  std::cout << mode << ',' << requested << ',' << std::fixed
            << std::setprecision(3) << per_call_ms << ',' << workers.size()
            << ',' << std::setprecision(3) << cpu_seconds / elapsed << ',';
  for (size_t i = 0; i < workers.size(); ++i) {
    if (i) {
      std::cout << ';';
    }
    std::cout << workers[i].id << ':' << std::setprecision(1)
              << workers[i].seconds * 1000.0;
  }
  std::cout << '\n';
}

void probe_direct(_Float16 *input, _Float16 *weight, _Float16 *output, int M,
                  int N, int K, int requested, int repeats) {
  MatMul matmul(input, weight, output, M, N, K, requested);
  for (int i = 0; i < 3; ++i) {
    matmul.run();
  }
  const auto before = task_thread_times();
  const auto start = Clock::now();
  for (int i = 0; i < repeats; ++i) {
    matmul.run();
  }
  const auto end = Clock::now();
  const auto after = task_thread_times();
  print_result("direct", requested, repeats,
               std::chrono::duration<double>(end - start).count(),
               thread_deltas(before, after));
}

enum class WorkerPlacement {
  None,
  PairAffinity,
  PairPerformance,
  PairEfficiency,
  PairMixed,
  SharedResourceRoundRobin,
};

void set_shared_resource_round_robin(int worker_index, bool enabled) {
  using BsdThreadCtl = int (*)(uint64_t, uint64_t, uint64_t, uint64_t);
  static const auto control = reinterpret_cast<BsdThreadCtl>(
      dlsym(RTLD_DEFAULT, "__bsdthread_ctl"));
  if (!control) {
    throw std::runtime_error("__bsdthread_ctl was not found");
  }
  constexpr uint64_t command = 0x2000;
  constexpr uint64_t set_flag = 0x1;
  constexpr uint64_t clear_flag = 0x2;
  if (control(command, enabled ? set_flag : clear_flag,
              static_cast<uint64_t>(worker_index), 2) != 0) {
    throw std::runtime_error("shared-resource scheduler policy was rejected");
  }
}

void apply_worker_placement(int shard, WorkerPlacement placement) {
  if (placement == WorkerPlacement::None) {
    return;
  }
  if (placement == WorkerPlacement::SharedResourceRoundRobin) {
    set_shared_resource_round_robin(shard, true);
    return;
  }
  const thread_t thread = mach_thread_self();
  thread_affinity_policy_data_t affinity{shard / 2 + 1};
  const kern_return_t affinity_result = thread_policy_set(
      thread, THREAD_AFFINITY_POLICY,
      reinterpret_cast<thread_policy_t>(&affinity),
      THREAD_AFFINITY_POLICY_COUNT);
  mach_port_deallocate(mach_task_self(), thread);
  if (affinity_result != KERN_SUCCESS) {
    static std::atomic<int> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) == 0) {
      std::cerr << "THREAD_AFFINITY_POLICY rejected: "
                << mach_error_string(affinity_result) << " ("
                << affinity_result << ")\n";
    }
  }
  qos_class_t qos = QOS_CLASS_UNSPECIFIED;
  switch (placement) {
  case WorkerPlacement::PairPerformance:
    qos = QOS_CLASS_USER_INTERACTIVE;
    break;
  case WorkerPlacement::PairEfficiency:
    qos = QOS_CLASS_BACKGROUND;
    break;
  case WorkerPlacement::PairMixed:
    qos = shard % 2 == 0 ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_BACKGROUND;
    break;
  default:
    break;
  }
  if (qos != QOS_CLASS_UNSPECIFIED &&
      pthread_set_qos_class_self_np(qos, 0) != 0) {
    throw std::runtime_error("pthread QoS placement was rejected");
  }
}

void probe_manual(_Float16 *input, _Float16 *weight, _Float16 *output, int M,
                  int N, int K, int shards, int repeats,
                  bool align_rows = false,
                  WorkerPlacement placement = WorkerPlacement::None) {
  std::vector<std::unique_ptr<MatMul>> matmuls;
  matmuls.reserve(shards);
  for (int shard = 0; shard < shards; ++shard) {
    const int row_blocks = M / 64;
    const bool can_align = align_rows && M % 64 == 0 && row_blocks >= shards;
    const int begin = can_align
                          ? static_cast<int>((static_cast<int64_t>(row_blocks) *
                                              shard) /
                                             shards) *
                                64
                          : static_cast<int>((static_cast<int64_t>(M) * shard) /
                                             shards);
    const int end = can_align
                        ? static_cast<int>((static_cast<int64_t>(row_blocks) *
                                            (shard + 1)) /
                                           shards) *
                              64
                        : static_cast<int>((static_cast<int64_t>(M) *
                                            (shard + 1)) /
                                           shards);
    matmuls.push_back(std::make_unique<MatMul>(
        input + static_cast<size_t>(begin) * K, weight,
        output + static_cast<size_t>(begin) * N, end - begin, N, K, 1));
    matmuls.back()->run();
  }

  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<ThreadTime> worker_deltas(static_cast<size_t>(shards));
  std::vector<CpuLocation> worker_starts(static_cast<size_t>(shards));
  std::vector<CpuLocation> worker_ends(static_cast<size_t>(shards));
  std::vector<std::array<uint32_t, 256>> cluster_samples(
      static_cast<size_t>(shards));
  std::vector<std::thread> workers;
  workers.reserve(shards);
  const auto start = Clock::now();
  for (int shard = 0; shard < shards; ++shard) {
    workers.emplace_back([&, shard] {
      apply_worker_placement(shard, placement);
      const ThreadTime before = current_thread_time();
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      worker_starts[static_cast<size_t>(shard)] = current_cpu_location();
      for (int i = 0; i < repeats; ++i) {
        ++cluster_samples[static_cast<size_t>(shard)]
                         [current_cpu_location().cluster];
        matmuls[static_cast<size_t>(shard)]->run();
        ++cluster_samples[static_cast<size_t>(shard)]
                         [current_cpu_location().cluster];
      }
      worker_ends[static_cast<size_t>(shard)] = current_cpu_location();
      const ThreadTime after = current_thread_time();
      worker_deltas[static_cast<size_t>(shard)] = {
          before.id, after.seconds - before.seconds};
      if (placement == WorkerPlacement::SharedResourceRoundRobin) {
        set_shared_resource_round_robin(shard, false);
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != shards) {
    std::this_thread::yield();
  }
  const auto synchronized_start = Clock::now();
  go.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  const auto end = Clock::now();
  (void)start;
  std::sort(worker_deltas.begin(), worker_deltas.end(),
            [](const auto &a, const auto &b) { return a.seconds > b.seconds; });
  std::string mode = align_rows ? "manual_aligned" : "manual";
  switch (placement) {
  case WorkerPlacement::PairAffinity:
    mode += "_pair";
    break;
  case WorkerPlacement::PairPerformance:
    mode += "_pair_p";
    break;
  case WorkerPlacement::PairEfficiency:
    mode += "_pair_e";
    break;
  case WorkerPlacement::PairMixed:
    mode += "_pair_mixed";
    break;
  case WorkerPlacement::SharedResourceRoundRobin:
    mode += "_shared_rr";
    break;
  default:
    break;
  }
  print_result(mode, shards, repeats,
               std::chrono::duration<double>(end - synchronized_start).count(),
               std::move(worker_deltas));
  if (placement == WorkerPlacement::SharedResourceRoundRobin) {
    std::array<uint64_t, 256> totals{};
    for (const auto &samples : cluster_samples) {
      for (size_t cluster = 0; cluster < totals.size(); ++cluster) {
        totals[cluster] += samples[cluster];
      }
    }
    std::cout << "# shared_rr_clusters=";
    bool first = true;
    for (size_t cluster = 0; cluster < totals.size(); ++cluster) {
      if (totals[cluster] == 0) {
        continue;
      }
      if (!first) {
        std::cout << ';';
      }
      first = false;
      std::cout << cluster << ':' << totals[cluster];
    }
    std::cout << " starts=";
    for (size_t i = 0; i < worker_starts.size(); ++i) {
      if (i) {
        std::cout << ';';
      }
      std::cout << worker_starts[i].cpu << ':' << worker_starts[i].cluster;
    }
    std::cout << " ends=";
    for (size_t i = 0; i < worker_ends.size(); ++i) {
      if (i) {
        std::cout << ';';
      }
      std::cout << worker_ends[i].cpu << ':' << worker_ends[i].cluster;
    }
    std::cout << '\n';
  }
}

void probe_bruteforce_clusters(_Float16 *input, _Float16 *weight,
                               _Float16 *output, int M, int N, int K,
                               const std::vector<uint32_t> &clusters,
                               int workers_per_cluster, int repeats,
                               const std::string &label) {
  const int worker_count =
      static_cast<int>(clusters.size()) * workers_per_cluster;
  const int candidate_count = 28;
  if (M % 64 != 0 || M / 64 < worker_count) {
    throw std::runtime_error("cluster probe requires at least one 64-row block per worker");
  }
  std::array<int, 256> targets{};
  std::array<int, 256> bases{};
  std::array<std::atomic<int>, 256> claimed{};
  for (auto &value : claimed) {
    value.store(0, std::memory_order_relaxed);
  }
  for (size_t index = 0; index < clusters.size(); ++index) {
    targets[clusters[index]] = workers_per_cluster;
    bases[clusters[index]] = static_cast<int>(index) * workers_per_cluster;
  }
  std::vector<std::unique_ptr<MatMul>> matmuls;
  matmuls.reserve(static_cast<size_t>(worker_count));
  const int row_blocks = M / 64;
  for (int slot = 0; slot < worker_count; ++slot) {
    const int begin = static_cast<int>(
                          (static_cast<int64_t>(row_blocks) * slot) /
                          worker_count) *
                      64;
    const int end = static_cast<int>(
                        (static_cast<int64_t>(row_blocks) * (slot + 1)) /
                        worker_count) *
                    64;
    matmuls.push_back(std::make_unique<MatMul>(
        input + static_cast<size_t>(begin) * K, weight,
        output + static_cast<size_t>(begin) * N, end - begin, N, K, 1));
    matmuls.back()->run();
  }

  std::atomic<int> selected{0};
  std::atomic<bool> go{false};
  std::atomic<int> mismatches{0};
  std::vector<ThreadTime> worker_deltas(static_cast<size_t>(worker_count));
  std::vector<CpuLocation> starts(static_cast<size_t>(worker_count));
  std::vector<CpuLocation> ends(static_cast<size_t>(worker_count));
  std::vector<std::thread> candidates;
  candidates.reserve(candidate_count);
  for (int candidate = 0; candidate < candidate_count; ++candidate) {
    candidates.emplace_back([&, candidate] {
      int slot = -1;
      uint32_t target_cluster = 0;
      while (!go.load(std::memory_order_acquire)) {
        const CpuLocation location = current_cpu_location();
        if (targets[location.cluster] > 0) {
          const int local = claimed[location.cluster].fetch_add(
              1, std::memory_order_acq_rel);
          if (local < targets[location.cluster]) {
            target_cluster = location.cluster;
            slot = bases[location.cluster] + local;
            starts[static_cast<size_t>(slot)] = location;
            selected.fetch_add(1, std::memory_order_release);
            break;
          }
          claimed[location.cluster].fetch_sub(1, std::memory_order_acq_rel);
        }
        std::this_thread::yield();
      }
      if (slot < 0) {
        return;
      }
      const ThreadTime before = current_thread_time();
      while (!go.load(std::memory_order_acquire)) {
        if (current_cpu_location().cluster != target_cluster) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
      }
      for (int repeat = 0; repeat < repeats; ++repeat) {
        if (current_cpu_location().cluster != target_cluster) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
        matmuls[static_cast<size_t>(slot)]->run();
        if (current_cpu_location().cluster != target_cluster) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
      }
      const ThreadTime after = current_thread_time();
      ends[static_cast<size_t>(slot)] = current_cpu_location();
      worker_deltas[static_cast<size_t>(slot)] = {
          before.id, after.seconds - before.seconds};
    });
  }
  const auto deadline = Clock::now() + std::chrono::seconds(3);
  while (selected.load(std::memory_order_acquire) != worker_count &&
         Clock::now() < deadline) {
    std::this_thread::yield();
  }
  if (selected.load(std::memory_order_acquire) != worker_count) {
    go.store(true, std::memory_order_release);
    for (auto &candidate : candidates) {
      candidate.join();
    }
    throw std::runtime_error("could not retain the requested cluster workers");
  }
  const auto start = Clock::now();
  go.store(true, std::memory_order_release);
  for (auto &candidate : candidates) {
    candidate.join();
  }
  const auto end = Clock::now();
  std::sort(worker_deltas.begin(), worker_deltas.end(),
            [](const auto &a, const auto &b) { return a.seconds > b.seconds; });
  print_result("manual_bruteforce_" + label, worker_count, repeats,
               std::chrono::duration<double>(end - start).count(),
               std::move(worker_deltas));
  std::cout << "# cluster_mismatches=" << mismatches.load() << " starts=";
  for (size_t i = 0; i < starts.size(); ++i) {
    if (i) {
      std::cout << ';';
    }
    std::cout << starts[i].cpu << ':' << starts[i].cluster;
  }
  std::cout << " ends=";
  for (size_t i = 0; i < ends.size(); ++i) {
    if (i) {
      std::cout << ';';
    }
    std::cout << ends[i].cpu << ':' << ends[i].cluster;
  }
  std::cout << '\n';
}

void probe_private_dispatch(int iterations, bool set_parallelism,
                            intptr_t parallelism = -1,
                            uint64_t parallelism_flags = 1) {
  using AttrStorage = uint64_t[8];
  using AttrInit = void (*)(AttrStorage *);
  using AttrDestroy = void (*)(AttrStorage *);
  using AttrSetParallelism = void (*)(AttrStorage *, intptr_t, uint64_t);
  using AttrQuery = size_t (*)(AttrStorage *, uint64_t, uint64_t);
  using ApplyWithAttr = void (*)(size_t, AttrStorage *, void (^)(size_t));
  const auto init = reinterpret_cast<AttrInit>(
      dlsym(RTLD_DEFAULT, "dispatch_apply_attr_init"));
  const auto destroy = reinterpret_cast<AttrDestroy>(
      dlsym(RTLD_DEFAULT, "dispatch_apply_attr_destroy"));
  const auto apply = reinterpret_cast<ApplyWithAttr>(
      dlsym(RTLD_DEFAULT, "dispatch_apply_with_attr"));
  const auto set = reinterpret_cast<AttrSetParallelism>(
      dlsym(RTLD_DEFAULT, "dispatch_apply_attr_set_parallelism"));
  const auto query = reinterpret_cast<AttrQuery>(
      dlsym(RTLD_DEFAULT, "dispatch_apply_attr_query"));
  if (!init || !destroy || !apply || !set || !query) {
    throw std::runtime_error("private dispatch apply APIs were not found");
  }
  AttrStorage attributes{};
  init(&attributes);
  if (set_parallelism) {
    set(&attributes, parallelism, parallelism_flags);
  }
  const size_t queried_parallelism = query(&attributes, 1, 1);
  std::atomic<int> active{0};
  std::atomic<int> maximum{0};
  std::mutex ids_mutex;
  std::vector<uint64_t> ids;
  std::vector<CpuLocation> locations;
  auto *active_ptr = &active;
  auto *maximum_ptr = &maximum;
  auto *ids_mutex_ptr = &ids_mutex;
  auto *ids_ptr = &ids;
  auto *locations_ptr = &locations;
  apply(static_cast<size_t>(iterations), &attributes, ^(size_t) {
    const ThreadTime thread = current_thread_time();
    const CpuLocation location = current_cpu_location();
    {
      std::lock_guard<std::mutex> lock(*ids_mutex_ptr);
      if (std::find(ids_ptr->begin(), ids_ptr->end(), thread.id) ==
          ids_ptr->end()) {
        ids_ptr->push_back(thread.id);
      }
      if (std::find_if(locations_ptr->begin(), locations_ptr->end(),
                       [&](const CpuLocation &entry) {
                         return entry.cpu == location.cpu &&
                                entry.cluster == location.cluster;
                       }) == locations_ptr->end()) {
        locations_ptr->push_back(location);
      }
    }
    const int now = active_ptr->fetch_add(1, std::memory_order_acq_rel) + 1;
    int observed = maximum_ptr->load(std::memory_order_relaxed);
    while (observed < now &&
           !maximum_ptr->compare_exchange_weak(observed, now,
                                               std::memory_order_relaxed)) {
    }
    const auto deadline = Clock::now() + std::chrono::milliseconds(20);
    volatile uint64_t value = thread.id;
    while (Clock::now() < deadline) {
      value = value * 6364136223846793005ULL + 1;
    }
    (void)value;
    active_ptr->fetch_sub(1, std::memory_order_acq_rel);
  });
  destroy(&attributes);
  std::cout << (set_parallelism ? "private_dispatch_parallelism" :
                                   "private_dispatch")
            << ',' << iterations << ",max_concurrent="
            << maximum.load() << ",unique_threads=" << ids.size()
            << ",query=" << queried_parallelism;
  if (set_parallelism) {
    std::cout << ",set=" << parallelism << ",flags=" << parallelism_flags;
  }
  std::sort(locations.begin(), locations.end(),
            [](const auto &a, const auto &b) { return a.cpu < b.cpu; });
  std::cout << ",locations=";
  for (size_t i = 0; i < locations.size(); ++i) {
    if (i) {
      std::cout << ';';
    }
    std::cout << locations[i].cpu << ':' << locations[i].cluster;
  }
  std::cout << '\n';
}

} // namespace

int main(int argc, char **argv) {
  const int M = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int N = argc > 2 ? std::atoi(argv[2]) : 4608;
  const int K = argc > 3 ? std::atoi(argv[3]) : 5120;
  const int repeats = argc > 4 ? std::atoi(argv[4]) : 8;
  const std::string selector = argc > 5 ? argv[5] : "";
  if (M <= 0 || N <= 0 || K <= 0 || repeats <= 0) {
    std::cerr << "usage: qwen35_bnns_thread_probe [M N K repeats]\n";
    return 2;
  }

  if (!selector.empty() && (selector[0] == 'a' || selector[0] == 'b')) {
    try {
      const bool set_parallelism = selector[0] == 'b' || argc > 6;
      const intptr_t parallelism = argc > 6 ? std::atoll(argv[6]) : -1;
      const uint64_t flags = argc > 7 ? std::strtoull(argv[7], nullptr, 0) : 1;
      probe_private_dispatch(std::atoi(selector.c_str() + 1),
                             set_parallelism, parallelism, flags);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "probe failed: " << error.what() << '\n';
      return 1;
    }
  }

  std::vector<_Float16> input(static_cast<size_t>(M) * K, 0.0078125f);
  std::vector<_Float16> weight(static_cast<size_t>(N) * K, 0.00390625f);
  std::vector<_Float16> output(static_cast<size_t>(M) * N);

  std::cout << "# shape=" << M << 'x' << K << " times " << N << 'x' << K
            << "^T repeats=" << repeats << '\n';
  std::cout << "mode,requested,wall_ms_per_call,active_threads,cpu_over_wall,"
               "thread_cpu_ms\n";
  try {
    if (!selector.empty() && selector[0] == 'd') {
      const int only_direct = std::atoi(selector.c_str() + 1);
      probe_direct(input.data(), weight.data(), output.data(), M, N, K,
                   only_direct, repeats);
      return 0;
    }
    if (!selector.empty() && selector[0] == 'm') {
      const int only_shards = std::atoi(selector.c_str() + 1);
      probe_manual(input.data(), weight.data(), output.data(), M, N, K,
                   only_shards, repeats);
      return 0;
    }
    if (!selector.empty() &&
        (selector[0] == 'x' || selector[0] == 'p' || selector[0] == 'u' ||
         selector[0] == 'e' || selector[0] == 'h' || selector[0] == 'z')) {
      const int only_shards = std::atoi(selector.c_str() + 1);
      WorkerPlacement placement = WorkerPlacement::None;
      if (selector[0] == 'p') {
        placement = WorkerPlacement::PairAffinity;
      } else if (selector[0] == 'u') {
        placement = WorkerPlacement::PairPerformance;
      } else if (selector[0] == 'e') {
        placement = WorkerPlacement::PairEfficiency;
      } else if (selector[0] == 'h') {
        placement = WorkerPlacement::PairMixed;
      } else if (selector[0] == 'z') {
        placement = WorkerPlacement::SharedResourceRoundRobin;
      }
      probe_manual(input.data(), weight.data(), output.data(), M, N, K,
                   only_shards, repeats, true, placement);
      return 0;
    }
    if (selector == "r12") {
      probe_bruteforce_clusters(input.data(), weight.data(), output.data(), M,
                                N, K, {0, 1, 2, 3, 4, 5}, 2, repeats,
                                "2_each_cluster");
      return 0;
    }
    if (selector == "r8p") {
      probe_bruteforce_clusters(input.data(), weight.data(), output.data(), M,
                                N, K, {1, 2, 4, 5}, 2, repeats,
                                "2_each_p_cluster");
      return 0;
    }
    if (selector == "r4e") {
      probe_bruteforce_clusters(input.data(), weight.data(), output.data(), M,
                                N, K, {0, 3}, 2, repeats,
                                "2_each_e_cluster");
      return 0;
    }
    for (int requested : {1, 2, 4, 8, 12, 16, 20, 28, 0}) {
      probe_direct(input.data(), weight.data(), output.data(), M, N, K,
                   requested, repeats);
    }
    for (int shards : {2, 4, 6, 8, 12, 16, 20}) {
      probe_manual(input.data(), weight.data(), output.data(), M, N, K, shards,
                   repeats);
    }
  } catch (const std::exception &error) {
    std::cerr << "probe failed: " << error.what() << '\n';
    return 1;
  }
  volatile _Float16 sink = output[0];
  (void)sink;
  return 0;
}
