//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t)>

// CHECK: clift.module {
clift.module {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    %label = clift.make_label "label"

    // CHECK: clift.while {
    clift.while {
      clift.yield %arg0 : !int32_t
    // CHECK: } {
    } {
      // CHECK-NOT: clift.loop_continue
      clift.loop_continue
    // CHECK: }
    }

    // CHECK: clift.switch {
    clift.switch {
      clift.yield %arg0 : !int32_t
    // CHECK: } case 0 {
    } case 0 {
      // CHECK: clift.expr
      clift.expr {
        clift.yield %arg0 : !int32_t
      // CHECK: }
      }

      // CHECK-NOT: clift.switch_break
      clift.switch_break
    // CHECK: }
    }

    // CHECK: clift.if {
    clift.if {
      clift.yield %arg0 : !int32_t
    // CHECK: } {
    } {
      // CHECK: clift.expr
      clift.expr {
        clift.yield %arg0 : !int32_t
      // CHECK: }
      }

      // CHECK-NOT: clift.goto
      clift.goto %label
    // CHECK: }
    }
    // CHECK: clift.assign_label
    clift.assign_label %label

    // CHECK: clift.if {
    clift.if {
      clift.yield %arg0 : !int32_t
    // CHECK: } {
    } {
      // CHECK: clift.expr
      clift.expr {
        clift.yield %arg0 : !int32_t
      // CHECK: }
      }

      // CHECK-NOT: clift.return
      clift.return {}
    // CHECK: }
    }

  // CHECK: }
  }
// CHECK: }
}
