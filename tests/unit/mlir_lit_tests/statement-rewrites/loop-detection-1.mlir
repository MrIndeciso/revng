//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --loop-detection | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void(!int32_t)>

// CHECK: clift.module {
clift.module {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) attributes {
    handle = "/function/0x40001004:Code_x86_64"
  } {
    %label_0 = clift.make_label "label_0"
    %label_1 = clift.make_label "label_1"

    clift.assign_label %label_0

    clift.expr {
      %f = clift.use @f : !f
      %0 = clift.imm 0 : !int32_t
      %r = clift.call %f(%0) : !f
      clift.yield %r : !void
    }

    clift.assign_label %label_1

    clift.expr {
      %f = clift.use @f : !f
      %1 = clift.imm 1 : !int32_t
      %r = clift.call %f(%1) : !f
      clift.yield %r : !void
    }

    clift.if {
      clift.yield %arg0 : !int32_t
    } {
      clift.expr {
        %f = clift.use @f : !f
        %2 = clift.imm 2 : !int32_t
        %r = clift.call %f(%2) : !f
        clift.yield %r : !void
      }

      clift.if {
        clift.yield %arg0 : !int32_t
      } {
        clift.expr {
          %f = clift.use @f : !f
          %3 = clift.imm 3 : !int32_t
          %r = clift.call %f(%3) : !f
          clift.yield %r : !void
        }

      } else {
        clift.goto %label_0
      }
    } else {
      clift.goto %label_1
    }
  }
// CHECK: }
}
