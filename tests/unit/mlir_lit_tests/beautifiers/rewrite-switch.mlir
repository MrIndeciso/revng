//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --switch-case-rewrite | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %0 = clift.local !int32_t

    // CHECK: clift.if
    clift.switch {
      // CHECK: %1 = clift.imm 1 : !int32_t
      // CHECK: %2 = clift.eq %0, %1 : !int32_t
      clift.yield %0: !int32_t
      // CHECK: clift.yield %2 : !int32_t
    } case 1 {
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
    } default {
      // CHECK: } else {
      clift.expr {
        %1 = clift.imm 2 : !int32_t
        clift.yield %1 : !int32_t
      }
    }

    // Different condition region
    // CHECK: clift.if
    clift.switch {
      %1 = clift.imm 1 : !int32_t
      // CHECK: %2 = clift.imm 3 : !int32_t
      // CHECK: %3 = clift.eq %1, %2 : !int32_t
      clift.yield %1: !int32_t
      // CHECK: clift.yield %3 : !int32_t
    } case 3 {
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
    } default {
      // CHECK: } else {
      clift.expr {
        %1 = clift.imm 2 : !int32_t
        clift.yield %1 : !int32_t
      }
    }

    // This shouldn't get rewritten
    // CHECK: clift.switch
    clift.switch {
      clift.yield %0: !int32_t
    } case 1 {
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
    } case 2 {
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
    } default {
      clift.expr {
        %1 = clift.imm 2 : !int32_t
        clift.yield %1 : !int32_t
      }
    }
  }
  // CHECK: }
}