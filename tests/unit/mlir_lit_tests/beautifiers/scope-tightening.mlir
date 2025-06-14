//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --tighten-variable-scopes | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

clift.func @f<!f>() attributes {
  handle = "/function/0x40001001:Code_x86_64"
} {
  // CHECK: %0 = clift.local !int32_t
  %0 = clift.local !int32_t
  %1 = clift.local !int32_t
  %2 = clift.local !int32_t
  %3 = clift.local !int32_t

  clift.if {
    %4 = clift.undef : !int32_t
    clift.yield %4 : !int32_t
  } {
    clift.expr {
      clift.yield %0 : !int32_t
    }
  } else {
    clift.expr {
      clift.yield %0 : !int32_t
    }
    // CHECK: %2 = clift.local !int32_t
    clift.expr {
      // CHECK: clift.yield %2 : !int32_t
      clift.yield %1 : !int32_t
    }
    clift.if {
      %5 = clift.undef : !int32_t
      clift.yield %5 : !int32_t
    } {
      // CHECK: %3 = clift.local !int32_t
      clift.expr {
        // CHECK: clift.yield %3 : !int32_t
        clift.yield %2 : !int32_t
      }
    }
  }
  // CHECK: %1 = clift.local !int32_t
  clift.expr {
    // CHECK: clift.yield %1 : !int32_t
    clift.yield %3 : !int32_t
  }
}
