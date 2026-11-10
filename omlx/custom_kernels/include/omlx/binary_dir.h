// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <dlfcn.h>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace omlx {
namespace {

// Directory of the extension that includes this header, where its metallibs
// live. The unnamed namespace gives each extension its own copy for dladdr.
std::string binary_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) {
      throw std::runtime_error("Unable to get the oMLX kernel binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return dir;
}

} // namespace
} // namespace omlx
