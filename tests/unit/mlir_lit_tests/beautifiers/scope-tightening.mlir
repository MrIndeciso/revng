//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --tighten-variable-scopes --emit-c="tagless model=%S/../backend/model.yml" -o /dev/null | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

clift.module {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    %v1 = clift.local !int32_t "v1"
    %v2 = clift.local !int32_t "v2"
    %v3 = clift.local !int32_t "v3"
    %v4 = clift.local !int32_t "v4"

    // CHECK: int32_t _var_0;
    clift.if {
      %c1 = clift.undef : !int32_t
      clift.yield %c1 : !int32_t
    } {
      // CHECK: _var_0;
      clift.expr {
        clift.yield %v1 : !int32_t
      }
    } else {
      clift.expr {
        clift.yield %v1 : !int32_t
      }
      // CHECK: int32_t _var_1;
      clift.expr {
        clift.yield %v2 : !int32_t
      }
      clift.if {
        %c2 = clift.undef : !int32_t
        clift.yield %c2 : !int32_t
      } {
        // CHECK: int32_t _var_2;
        clift.expr {
          clift.yield %v3 : !int32_t
        }
      }
    }
    // CHECK: int32_t _var_3;
    clift.expr {
      clift.yield %v4 : !int32_t
    }
  }
}
