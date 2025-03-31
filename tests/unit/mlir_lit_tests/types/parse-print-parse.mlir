//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
// 1. Parse this source file and print it as textual assembly.
// 2. Parse the textual assembly from step 1 and emit it as bytecode.
// 3. Parse the bytecode from step 2 and emit it as textual assembly.
// 4. Parse the textual assembly from step 3.
//
// RUN: %revngcliftopt %s | %revngcliftopt --emit-bytecode | %revngcliftopt | %revngcliftopt

!int32_t = !clift.primitive<signed 4>
!uint32_t = !clift.primitive<unsigned 8>

!my_enum = !clift.enum<
  "/type-definition/1001-EnumDefinition" as "my_enum" : !uint32_t {
    20 as "my_enum_20",
    21 as "my_enum_21"
  }
>

!my_typedef = !clift.typedef<
  "/type-definition/1002-TypedefDefinition" as "my_typedef" : !clift.const<!int32_t>
>

!my_struct$const = !clift.const<!clift.struct<
  "/type-definition/1003-StructDefinition" as "my_struct" : size(40) {
    offset(10) as "my_struct_10" : !clift.const<!clift.primitive<signed 4>>,
    offset(20) as "my_struct_20" : !clift.primitive<signed 4>
  }
>>

!my_union$const = !clift.const<!clift.union<
  "/type-definition/1004-UnionDefinition" as "my_union" : {
    "my_union_10" : !clift.const<!clift.primitive<signed 4>>,
    "my_union_20" : !clift.primitive<signed 4>
  }
>>

!my_function = !clift.func<
  "/type-definition/1005-CABIFunctionDefinition" as "my_function" : !my_struct$const(!clift.const<!int32_t>)
>

!my_recursive_union = !clift.const<!clift.union<
  "/type-definition/1006-UnionDefinition" as "my_recursive_union" : {
    "my_recursive_union_10" : !clift.const<!clift.primitive<signed 4>>,
    "my_recursive_union_20" : !clift.ptr<8 to !clift.const<!clift.union<"/type-definition/1006-UnionDefinition">>>
  }
>>

!my_recursive_struct_2 = !clift.struct<
  "/type-definition/1007-UnionDefinition" : size(8) {
    offset(0) : !clift.struct<"/type-definition/1008-UnionDefinition">
  }
>

!my_recursive_struct_1 = !clift.struct<
  "/type-definition/1008-UnionDefinition" : size(8) {
    offset(0) : !clift.ptr<8 to !my_recursive_struct_2>
  }
>

clift.undef : !clift.const<!int32_t>
clift.undef : !clift.const<!uint32_t>
clift.undef : !my_enum
clift.undef : !my_typedef
clift.undef : !my_struct$const
clift.undef : !my_union$const
clift.undef : !my_function
clift.undef : !my_recursive_union
clift.undef : !my_recursive_struct_1
