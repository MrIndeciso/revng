#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <atomic>
#include <memory>

/// A copyable shared atomic flag. Starts out cleared and can be tested or set.
/// Can be used to ensure one-time-only entry into a critical section.
class SharedOnceFlag {
  struct FlagType {
    std::atomic_flag Flag = {};
  };

public:
  SharedOnceFlag() : Flag(std::make_shared<FlagType>()) {}

  [[nodiscard]] bool test() const { return Flag->Flag.test(); }
  [[nodiscard]] bool testAndSet() const { return Flag->Flag.test_and_set(); }

private:
  std::shared_ptr<FlagType> Flag;
};
