ä
ç¼
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8À¿

garm_embedding/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬*&
shared_namegarm_embedding/kernel

)garm_embedding/kernel/Read/ReadVariableOpReadVariableOpgarm_embedding/kernel* 
_output_shapes
:
¬*
dtype0

garm_embedding/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*$
shared_namegarm_embedding/bias
x
'garm_embedding/bias/Read/ReadVariableOpReadVariableOpgarm_embedding/bias*
_output_shapes	
:¬*
dtype0

NoOpNoOp
è
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*£
valueB B

layer-0
layer_with_weights-0
layer-1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
	bias

	variables
regularization_losses
trainable_variables
	keras_api

0
	1
 

0
	1
­
layer_metrics
layer_regularization_losses
	variables
metrics

layers
non_trainable_variables
regularization_losses
trainable_variables
 
a_
VARIABLE_VALUEgarm_embedding/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgarm_embedding/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
	1
 

0
	1
­
layer_metrics
layer_regularization_losses

	variables
metrics

layers
non_trainable_variables
regularization_losses
trainable_variables
 
 
 

0
1
 
 
 
 
 
 
|
serving_default_input_2Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ã
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2garm_embedding/kernelgarm_embedding/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_821678
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ñ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)garm_embedding/kernel/Read/ReadVariableOp'garm_embedding/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_821764
Ä
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegarm_embedding/kernelgarm_embedding/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_821780Â©
ó

2__inference_garm_vector_model_layer_call_fn_821716

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_8216602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
z
$__inference_signature_wrapper_821678
input_2
unknown
	unknown_0
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_8215902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ó

2__inference_garm_vector_model_layer_call_fn_821707

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_8216422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö

2__inference_garm_vector_model_layer_call_fn_821649
input_2
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_8216422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
«
Ì
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821621
input_2
garm_embedding_821615
garm_embedding_821617
identity¢&garm_embedding/StatefulPartitionedCall´
&garm_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_2garm_embedding_821615garm_embedding_821617*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_garm_embedding_layer_call_and_return_conditional_losses_8216042(
&garm_embedding/StatefulPartitionedCall­
IdentityIdentity/garm_embedding/StatefulPartitionedCall:output:0'^garm_embedding/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2P
&garm_embedding/StatefulPartitionedCall&garm_embedding/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2

¢
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821688

inputs1
-garm_embedding_matmul_readvariableop_resource2
.garm_embedding_biasadd_readvariableop_resource
identity¢%garm_embedding/BiasAdd/ReadVariableOp¢$garm_embedding/MatMul/ReadVariableOp¼
$garm_embedding/MatMul/ReadVariableOpReadVariableOp-garm_embedding_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02&
$garm_embedding/MatMul/ReadVariableOp¡
garm_embedding/MatMulMatMulinputs,garm_embedding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
garm_embedding/MatMulº
%garm_embedding/BiasAdd/ReadVariableOpReadVariableOp.garm_embedding_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02'
%garm_embedding/BiasAdd/ReadVariableOp¾
garm_embedding/BiasAddBiasAddgarm_embedding/MatMul:product:0-garm_embedding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
garm_embedding/BiasAddÃ
IdentityIdentitygarm_embedding/BiasAdd:output:0&^garm_embedding/BiasAdd/ReadVariableOp%^garm_embedding/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2N
%garm_embedding/BiasAdd/ReadVariableOp%garm_embedding/BiasAdd/ReadVariableOp2L
$garm_embedding/MatMul/ReadVariableOp$garm_embedding/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
Ë
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821660

inputs
garm_embedding_821654
garm_embedding_821656
identity¢&garm_embedding/StatefulPartitionedCall³
&garm_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsgarm_embedding_821654garm_embedding_821656*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_garm_embedding_layer_call_and_return_conditional_losses_8216042(
&garm_embedding/StatefulPartitionedCall­
IdentityIdentity/garm_embedding/StatefulPartitionedCall:output:0'^garm_embedding/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2P
&garm_embedding/StatefulPartitionedCall&garm_embedding/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

È
"__inference__traced_restore_821780
file_prefix*
&assignvariableop_garm_embedding_kernel*
&assignvariableop_1_garm_embedding_bias

identity_3¢AssignVariableOp¢AssignVariableOp_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¡
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
RestoreV2/shape_and_slicesº
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¥
AssignVariableOpAssignVariableOp&assignvariableop_garm_embedding_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1«
AssignVariableOp_1AssignVariableOp&assignvariableop_1_garm_embedding_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_2

Identity_3IdentityIdentity_2:output:0^AssignVariableOp^AssignVariableOp_1*
T0*
_output_shapes
: 2

Identity_3"!

identity_3Identity_3:output:0*
_input_shapes

: ::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¨
Ë
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821642

inputs
garm_embedding_821636
garm_embedding_821638
identity¢&garm_embedding/StatefulPartitionedCall³
&garm_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsgarm_embedding_821636garm_embedding_821638*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_garm_embedding_layer_call_and_return_conditional_losses_8216042(
&garm_embedding/StatefulPartitionedCall­
IdentityIdentity/garm_embedding/StatefulPartitionedCall:output:0'^garm_embedding/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2P
&garm_embedding/StatefulPartitionedCall&garm_embedding/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö

2__inference_garm_vector_model_layer_call_fn_821667
input_2
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_8216602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2

¢
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821698

inputs1
-garm_embedding_matmul_readvariableop_resource2
.garm_embedding_biasadd_readvariableop_resource
identity¢%garm_embedding/BiasAdd/ReadVariableOp¢$garm_embedding/MatMul/ReadVariableOp¼
$garm_embedding/MatMul/ReadVariableOpReadVariableOp-garm_embedding_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02&
$garm_embedding/MatMul/ReadVariableOp¡
garm_embedding/MatMulMatMulinputs,garm_embedding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
garm_embedding/MatMulº
%garm_embedding/BiasAdd/ReadVariableOpReadVariableOp.garm_embedding_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02'
%garm_embedding/BiasAdd/ReadVariableOp¾
garm_embedding/BiasAddBiasAddgarm_embedding/MatMul:product:0-garm_embedding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
garm_embedding/BiasAddÃ
IdentityIdentitygarm_embedding/BiasAdd:output:0&^garm_embedding/BiasAdd/ReadVariableOp%^garm_embedding/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2N
%garm_embedding/BiasAdd/ReadVariableOp%garm_embedding/BiasAdd/ReadVariableOp2L
$garm_embedding/MatMul/ReadVariableOp$garm_embedding/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 	
ã
J__inference_garm_embedding_layer_call_and_return_conditional_losses_821726

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

/__inference_garm_embedding_layer_call_fn_821735

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_garm_embedding_layer_call_and_return_conditional_losses_8216042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
Ö
__inference__traced_save_821764
file_prefix4
0savev2_garm_embedding_kernel_read_readvariableop2
.savev2_garm_embedding_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¡
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_garm_embedding_kernel_read_readvariableop.savev2_garm_embedding_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
: :
¬:¬: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
¬:!

_output_shapes	
:¬:

_output_shapes
: 
 	
ã
J__inference_garm_embedding_layer_call_and_return_conditional_losses_821604

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
Ì
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821630
input_2
garm_embedding_821624
garm_embedding_821626
identity¢&garm_embedding/StatefulPartitionedCall´
&garm_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_2garm_embedding_821624garm_embedding_821626*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_garm_embedding_layer_call_and_return_conditional_losses_8216042(
&garm_embedding/StatefulPartitionedCall­
IdentityIdentity/garm_embedding/StatefulPartitionedCall:output:0'^garm_embedding/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2P
&garm_embedding/StatefulPartitionedCall&garm_embedding/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2

¿
!__inference__wrapped_model_821590
input_2C
?garm_vector_model_garm_embedding_matmul_readvariableop_resourceD
@garm_vector_model_garm_embedding_biasadd_readvariableop_resource
identity¢7garm_vector_model/garm_embedding/BiasAdd/ReadVariableOp¢6garm_vector_model/garm_embedding/MatMul/ReadVariableOpò
6garm_vector_model/garm_embedding/MatMul/ReadVariableOpReadVariableOp?garm_vector_model_garm_embedding_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype028
6garm_vector_model/garm_embedding/MatMul/ReadVariableOpØ
'garm_vector_model/garm_embedding/MatMulMatMulinput_2>garm_vector_model/garm_embedding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'garm_vector_model/garm_embedding/MatMulð
7garm_vector_model/garm_embedding/BiasAdd/ReadVariableOpReadVariableOp@garm_vector_model_garm_embedding_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype029
7garm_vector_model/garm_embedding/BiasAdd/ReadVariableOp
(garm_vector_model/garm_embedding/BiasAddBiasAdd1garm_vector_model/garm_embedding/MatMul:product:0?garm_vector_model/garm_embedding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2*
(garm_vector_model/garm_embedding/BiasAddù
IdentityIdentity1garm_vector_model/garm_embedding/BiasAdd:output:08^garm_vector_model/garm_embedding/BiasAdd/ReadVariableOp7^garm_vector_model/garm_embedding/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2r
7garm_vector_model/garm_embedding/BiasAdd/ReadVariableOp7garm_vector_model/garm_embedding/BiasAdd/ReadVariableOp2p
6garm_vector_model/garm_embedding/MatMul/ReadVariableOp6garm_vector_model/garm_embedding/MatMul/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
<
input_21
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿC
garm_embedding1
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ¬tensorflow/serving/predict:ÎG

layer-0
layer_with_weights-0
layer-1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
__call__
_default_save_signature
*&call_and_return_all_conditional_losses"¤
_tf_keras_network{"class_name": "Functional", "name": "garm_vector_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "garm_vector_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2560]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "garm_embedding", "trainable": true, "dtype": "float32", "units": 300, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "garm_embedding", "inbound_nodes": [[["input_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["garm_embedding", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2560]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2560]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "garm_vector_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2560]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "garm_embedding", "trainable": true, "dtype": "float32", "units": 300, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "garm_embedding", "inbound_nodes": [[["input_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["garm_embedding", 0, 0]]}}}
ï"ì
_tf_keras_input_layerÌ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2560]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2560]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}


kernel
	bias

	variables
regularization_losses
trainable_variables
	keras_api
__call__
*&call_and_return_all_conditional_losses"à
_tf_keras_layerÆ{"class_name": "Dense", "name": "garm_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "garm_embedding", "trainable": true, "dtype": "float32", "units": 300, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2560}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2560]}}
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
Ê
layer_metrics
layer_regularization_losses
	variables
metrics

layers
non_trainable_variables
regularization_losses
trainable_variables
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
):'
¬2garm_embedding/kernel
": ¬2garm_embedding/bias
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
­
layer_metrics
layer_regularization_losses

	variables
metrics

layers
non_trainable_variables
regularization_losses
trainable_variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
2__inference_garm_vector_model_layer_call_fn_821707
2__inference_garm_vector_model_layer_call_fn_821649
2__inference_garm_vector_model_layer_call_fn_821716
2__inference_garm_vector_model_layer_call_fn_821667À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
!__inference__wrapped_model_821590·
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *'¢$
"
input_2ÿÿÿÿÿÿÿÿÿ
2ÿ
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821698
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821688
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821630
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821621À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ù2Ö
/__inference_garm_embedding_layer_call_fn_821735¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_garm_embedding_layer_call_and_return_conditional_losses_821726¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ËBÈ
$__inference_signature_wrapper_821678input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!__inference__wrapped_model_821590y	1¢.
'¢$
"
input_2ÿÿÿÿÿÿÿÿÿ
ª "@ª=
;
garm_embedding)&
garm_embeddingÿÿÿÿÿÿÿÿÿ¬¬
J__inference_garm_embedding_layer_call_and_return_conditional_losses_821726^	0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¬
 
/__inference_garm_embedding_layer_call_fn_821735Q	0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬¸
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821621g	9¢6
/¢,
"
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¬
 ¸
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821630g	9¢6
/¢,
"
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¬
 ·
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821688f	8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¬
 ·
M__inference_garm_vector_model_layer_call_and_return_conditional_losses_821698f	8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¬
 
2__inference_garm_vector_model_layer_call_fn_821649Z	9¢6
/¢,
"
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¬
2__inference_garm_vector_model_layer_call_fn_821667Z	9¢6
/¢,
"
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¬
2__inference_garm_vector_model_layer_call_fn_821707Y	8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¬
2__inference_garm_vector_model_layer_call_fn_821716Y	8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¬­
$__inference_signature_wrapper_821678	<¢9
¢ 
2ª/
-
input_2"
input_2ÿÿÿÿÿÿÿÿÿ"@ª=
;
garm_embedding)&
garm_embeddingÿÿÿÿÿÿÿÿÿ¬