//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --switch-case-rewrite --emit-c="tagless model=%S/../backend/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

clift.module {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: int32_t _var_0;
    %x = clift.local  !int32_t "x"

    // CHECK: if (_var_0 == 1)
    clift.switch {
      clift.yield %x: !int32_t
    } case 1 {
      // CHECK: 1;
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
      clift.switch_break
    } default {
      // CHECK: else
      // CHECK: 2;
      clift.expr {
        %1 = clift.imm 2 : !int32_t
        clift.yield %1 : !int32_t
      }
      clift.switch_break
    }

    // No switch_break at the end of the default case
    // CHECK: if (_var_0 == 3)
    clift.switch {
      clift.yield %x: !int32_t
    } case 3 {
      // CHECK: 1;
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
      clift.switch_break
    } default {
      // CHECK: else
      // CHECK: 2;
      clift.expr {
        %1 = clift.imm 2 : !int32_t
        clift.yield %1 : !int32_t
      }
    }

    // Different condition region
    // CHECK if (1 == 3)
    clift.switch {
      %y = clift.imm 1 : !int32_t
      clift.yield %y: !int32_t
    } case 3 {
      // CHECK: 1;
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
      clift.switch_break
    } default {
      // CHECK: else
      // CHECK: 2;
      clift.expr {
        %1 = clift.imm 2 : !int32_t
        clift.yield %1 : !int32_t
      }
    }

    // More than one case
    // CHECK: switch (_var_0)
    clift.switch {
      clift.yield %x: !int32_t
    } case 1 {
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
      clift.switch_break
    } case 2 {
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
      clift.switch_break
    } default {
      clift.expr {
        %1 = clift.imm 2 : !int32_t
        clift.yield %1 : !int32_t
      }
      clift.switch_break
    }
  }
  // CHECK: }
}