ри
Р
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
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
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
О
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
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.4.12v2.4.0-49-g85c8b2a817f8М

embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ш*'
shared_nameembedding_1/embeddings

*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings* 
_output_shapes
:
ш*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ш*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
ш*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ш*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:ш*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
Ј
%simple_rnn_1/simple_rnn_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%simple_rnn_1/simple_rnn_cell_1/kernel
Ё
9simple_rnn_1/simple_rnn_cell_1/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_1/simple_rnn_cell_1/kernel* 
_output_shapes
:
*
dtype0
М
/simple_rnn_1/simple_rnn_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel
Е
Csimple_rnn_1/simple_rnn_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel* 
_output_shapes
:
*
dtype0

#simple_rnn_1/simple_rnn_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#simple_rnn_1/simple_rnn_cell_1/bias

7simple_rnn_1/simple_rnn_cell_1/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_1/simple_rnn_cell_1/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ш*.
shared_nameAdam/embedding_1/embeddings/m

1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m* 
_output_shapes
:
ш*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ш*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
ш*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ш*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:ш*
dtype0
Ж
,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m
Џ
@Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m* 
_output_shapes
:
*
dtype0
Ъ
6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*G
shared_name86Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m
У
JAdam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m* 
_output_shapes
:
*
dtype0
­
*Adam/simple_rnn_1/simple_rnn_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/simple_rnn_1/simple_rnn_cell_1/bias/m
І
>Adam/simple_rnn_1/simple_rnn_cell_1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_1/simple_rnn_cell_1/bias/m*
_output_shapes	
:*
dtype0

Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ш*.
shared_nameAdam/embedding_1/embeddings/v

1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v* 
_output_shapes
:
ш*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ш*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
ш*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ш*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:ш*
dtype0
Ж
,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v
Џ
@Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v* 
_output_shapes
:
*
dtype0
Ъ
6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*G
shared_name86Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v
У
JAdam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v* 
_output_shapes
:
*
dtype0
­
*Adam/simple_rnn_1/simple_rnn_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/simple_rnn_1/simple_rnn_cell_1/bias/v
І
>Adam/simple_rnn_1/simple_rnn_cell_1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_1/simple_rnn_cell_1/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
Ќ)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ч(
valueн(Bк( Bг(
ц
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
b


embeddings
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Ќ
iter

beta_1

beta_2
	decay
learning_rate
mLmMmN mO!mP"mQ
vRvSvT vU!vV"vW
*

0
 1
!2
"3
4
5
*

0
 1
!2
"3
4
5
 
­
	variables
#non_trainable_variables

$layers
%layer_metrics
trainable_variables
regularization_losses
&layer_regularization_losses
'metrics
 
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE


0


0
 
­
	variables
(non_trainable_variables

)layers
*layer_metrics
trainable_variables
regularization_losses
+layer_regularization_losses
,metrics
~

 kernel
!recurrent_kernel
"bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
 

 0
!1
"2

 0
!1
"2
 
Й
	variables

1states

2layers
3non_trainable_variables
4layer_metrics
trainable_variables
regularization_losses
5layer_regularization_losses
6metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
	variables
7non_trainable_variables

8layers
9layer_metrics
trainable_variables
regularization_losses
:layer_regularization_losses
;metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_1/simple_rnn_cell_1/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_1/simple_rnn_cell_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
 
 

<0
=1
 
 
 
 
 

 0
!1
"2

 0
!1
"2
 
­
-	variables
>non_trainable_variables

?layers
@layer_metrics
.trainable_variables
/regularization_losses
Alayer_regularization_losses
Bmetrics
 

0
 
 
 
 
 
 
 
 
 
4
	Ctotal
	Dcount
E	variables
F	keras_api
D
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

E	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

J	variables

VARIABLE_VALUEAdam/embedding_1/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/simple_rnn_1/simple_rnn_cell_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/embedding_1/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/simple_rnn_1/simple_rnn_cell_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

!serving_default_embedding_1_inputPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
ћ
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_1_inputembedding_1/embeddings%simple_rnn_1/simple_rnn_cell_1/kernel#simple_rnn_1/simple_rnn_cell_1/bias/simple_rnn_1/simple_rnn_cell_1/recurrent_kerneldense_1/kerneldense_1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_14764
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ѓ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9simple_rnn_1/simple_rnn_cell_1/kernel/Read/ReadVariableOpCsimple_rnn_1/simple_rnn_cell_1/recurrent_kernel/Read/ReadVariableOp7simple_rnn_1/simple_rnn_cell_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp@Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_1/simple_rnn_cell_1/bias/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp@Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_1/simple_rnn_cell_1/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_15743

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%simple_rnn_1/simple_rnn_cell_1/kernel/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel#simple_rnn_1/simple_rnn_cell_1/biastotalcounttotal_1count_1Adam/embedding_1/embeddings/mAdam/dense_1/kernel/mAdam/dense_1/bias/m,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m*Adam/simple_rnn_1/simple_rnn_cell_1/bias/mAdam/embedding_1/embeddings/vAdam/dense_1/kernel/vAdam/dense_1/bias/v,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v*Adam/simple_rnn_1/simple_rnn_cell_1/bias/v*'
Tin 
2*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_15834д
к
У
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_15594

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdd
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
TanhЈ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

IdentityЌ

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*G
_input_shapes6
4:џџџџџџџџџ:џџџџџџџџџ:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0

Ш
,__inference_sequential_1_layer_call_fn_14737
embedding_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_147222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameembedding_1_input
іG

G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15177

inputs4
0simple_rnn_cell_1_matmul_readvariableop_resource5
1simple_rnn_cell_1_biasadd_readvariableop_resource6
2simple_rnn_cell_1_matmul_1_readvariableop_resource
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Х
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOpМ
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMulУ
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOpЪ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/BiasAddЫ
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOpИ
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMul_1Д
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/add
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_15111*
condR
while_cond_15110*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permІ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
transpose_1і
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ:::2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш


simple_rnn_1_while_cond_148156
2simple_rnn_1_while_simple_rnn_1_while_loop_counter<
8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations"
simple_rnn_1_while_placeholder$
 simple_rnn_1_while_placeholder_1$
 simple_rnn_1_while_placeholder_28
4simple_rnn_1_while_less_simple_rnn_1_strided_slice_1M
Isimple_rnn_1_while_simple_rnn_1_while_cond_14815___redundant_placeholder0M
Isimple_rnn_1_while_simple_rnn_1_while_cond_14815___redundant_placeholder1M
Isimple_rnn_1_while_simple_rnn_1_while_cond_14815___redundant_placeholder2M
Isimple_rnn_1_while_simple_rnn_1_while_cond_14815___redundant_placeholder3
simple_rnn_1_while_identity
Б
simple_rnn_1/while/LessLesssimple_rnn_1_while_placeholder4simple_rnn_1_while_less_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_1/while/Less
simple_rnn_1/while/IdentityIdentitysimple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_1/while/Identity"C
simple_rnn_1_while_identity$simple_rnn_1/while/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ч3
ћ
while_body_14409
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_1_matmul_readvariableop_resource;
7while_simple_rnn_cell_1_biasadd_readvariableop_resource<
8while_simple_rnn_cell_1_matmul_1_readvariableop_resourceЂ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemй
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOpц
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
while/simple_rnn_cell_1/MatMulз
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpт
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
while/simple_rnn_cell_1/BiasAddп
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpЯ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 while/simple_rnn_cell_1/MatMul_1Ь
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/add
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/Tanhф
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ё
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ѓ
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ї

,__inference_simple_rnn_1_layer_call_fn_15557
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_143292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
д
С
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_13892

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdd
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
TanhЈ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

IdentityЌ

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*G
_input_shapes6
4:џџџџџџџџџ:џџџџџџџџџ:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
х
Н
,__inference_sequential_1_layer_call_fn_15048

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_147222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
эu
м
!__inference__traced_restore_15834
file_prefix+
'assignvariableop_embedding_1_embeddings%
!assignvariableop_1_dense_1_kernel#
assignvariableop_2_dense_1_bias 
assignvariableop_3_adam_iter"
assignvariableop_4_adam_beta_1"
assignvariableop_5_adam_beta_2!
assignvariableop_6_adam_decay)
%assignvariableop_7_adam_learning_rate<
8assignvariableop_8_simple_rnn_1_simple_rnn_cell_1_kernelF
Bassignvariableop_9_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel;
7assignvariableop_10_simple_rnn_1_simple_rnn_cell_1_bias
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_15
1assignvariableop_15_adam_embedding_1_embeddings_m-
)assignvariableop_16_adam_dense_1_kernel_m+
'assignvariableop_17_adam_dense_1_bias_mD
@assignvariableop_18_adam_simple_rnn_1_simple_rnn_cell_1_kernel_mN
Jassignvariableop_19_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_mB
>assignvariableop_20_adam_simple_rnn_1_simple_rnn_cell_1_bias_m5
1assignvariableop_21_adam_embedding_1_embeddings_v-
)assignvariableop_22_adam_dense_1_kernel_v+
'assignvariableop_23_adam_dense_1_bias_vD
@assignvariableop_24_adam_simple_rnn_1_simple_rnn_cell_1_kernel_vN
Jassignvariableop_25_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_vB
>assignvariableop_26_adam_simple_rnn_1_simple_rnn_cell_1_bias_v
identity_28ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*І
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЦ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesИ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityІ
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Є
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_1_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_3Ё
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ѓ
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѓ
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ђ
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Њ
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Н
AssignVariableOp_8AssignVariableOp8assignvariableop_8_simple_rnn_1_simple_rnn_cell_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ч
AssignVariableOp_9AssignVariableOpBassignvariableop_9_simple_rnn_1_simple_rnn_cell_1_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10П
AssignVariableOp_10AssignVariableOp7assignvariableop_10_simple_rnn_1_simple_rnn_cell_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ё
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ё
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ѓ
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ѓ
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Й
AssignVariableOp_15AssignVariableOp1assignvariableop_15_adam_embedding_1_embeddings_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Б
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Џ
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_1_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ш
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_simple_rnn_1_simple_rnn_cell_1_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19в
AssignVariableOp_19AssignVariableOpJassignvariableop_19_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ц
AssignVariableOp_20AssignVariableOp>assignvariableop_20_adam_simple_rnn_1_simple_rnn_cell_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Й
AssignVariableOp_21AssignVariableOp1assignvariableop_21_adam_embedding_1_embeddings_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Б
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_1_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Џ
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_1_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ш
AssignVariableOp_24AssignVariableOp@assignvariableop_24_adam_simple_rnn_1_simple_rnn_cell_1_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25в
AssignVariableOp_25AssignVariableOpJassignvariableop_25_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ц
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_simple_rnn_1_simple_rnn_cell_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpА
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27Ѓ
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*
_input_shapesp
n: :::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ї

,__inference_simple_rnn_1_layer_call_fn_15546
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_142122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
іG

G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_14475

inputs4
0simple_rnn_cell_1_matmul_readvariableop_resource5
1simple_rnn_cell_1_biasadd_readvariableop_resource6
2simple_rnn_cell_1_matmul_1_readvariableop_resource
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Х
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOpМ
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMulУ
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOpЪ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/BiasAddЫ
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOpИ
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMul_1Д
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/add
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_14409*
condR
while_cond_14408*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permІ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
transpose_1і
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ:::2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
іG

G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_14587

inputs4
0simple_rnn_cell_1_matmul_readvariableop_resource5
1simple_rnn_cell_1_biasadd_readvariableop_resource6
2simple_rnn_cell_1_matmul_1_readvariableop_resource
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Х
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOpМ
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMulУ
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOpЪ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/BiasAddЫ
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOpИ
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMul_1Д
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/add
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_14521*
condR
while_cond_14520*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permІ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
transpose_1і
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ:::2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч3
ћ
while_body_15469
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_1_matmul_readvariableop_resource;
7while_simple_rnn_cell_1_biasadd_readvariableop_resource<
8while_simple_rnn_cell_1_matmul_1_readvariableop_resourceЂ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemй
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOpц
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
while/simple_rnn_cell_1/MatMulз
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpт
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
while/simple_rnn_cell_1/BiasAddп
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpЯ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 while/simple_rnn_cell_1/MatMul_1Ь
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/add
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/Tanhф
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ё
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ѓ
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Э
Ѕ
while_cond_15222
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_15222___redundant_placeholder03
/while_while_cond_15222___redundant_placeholder13
/while_while_cond_15222___redundant_placeholder23
/while_while_cond_15222___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

к
G__inference_sequential_1_layer_call_and_return_conditional_losses_14686

inputs
embedding_1_14670
simple_rnn_1_14673
simple_rnn_1_14675
simple_rnn_1_14677
dense_1_14680
dense_1_14682
identityЂdense_1/StatefulPartitionedCallЂ#embedding_1/StatefulPartitionedCallЂ$simple_rnn_1/StatefulPartitionedCall
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_14670*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_143522%
#embedding_1/StatefulPartitionedCallт
$simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0simple_rnn_1_14673simple_rnn_1_14675simple_rnn_1_14677*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_144752&
$simple_rnn_1/StatefulPartitionedCallД
dense_1/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_1/StatefulPartitionedCall:output:0dense_1_14680dense_1_14682*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_146282!
dense_1/StatefulPartitionedCallь
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall%^simple_rnn_1/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2L
$simple_rnn_1/StatefulPartitionedCall$simple_rnn_1/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э
Ѕ
while_cond_14520
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_14520___redundant_placeholder03
/while_while_cond_14520___redundant_placeholder13
/while_while_cond_14520___redundant_placeholder23
/while_while_cond_14520___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ъ
q
+__inference_embedding_1_layer_call_fn_15065

inputs
unknown
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_143522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о	

F__inference_embedding_1_layer_call_and_return_conditional_losses_14352

inputs
embedding_lookup_14346
identityЂembedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
Castќ
embedding_lookupResourceGatherembedding_lookup_14346Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/14346*,
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupэ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/14346*,
_output_shapes
:џџџџџџџџџ2
embedding_lookup/IdentityЁ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч3
ћ
while_body_14521
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_1_matmul_readvariableop_resource;
7while_simple_rnn_cell_1_biasadd_readvariableop_resource<
8while_simple_rnn_cell_1_matmul_1_readvariableop_resourceЂ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemй
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOpц
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
while/simple_rnn_cell_1/MatMulз
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpт
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
while/simple_rnn_cell_1/BiasAddп
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpЯ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 while/simple_rnn_cell_1/MatMul_1Ь
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/add
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/Tanhф
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ё
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ѓ
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ЌH

G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15535
inputs_04
0simple_rnn_cell_1_matmul_readvariableop_resource5
1simple_rnn_cell_1_biasadd_readvariableop_resource6
2simple_rnn_cell_1_matmul_1_readvariableop_resource
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Х
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOpМ
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMulУ
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOpЪ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/BiasAddЫ
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOpИ
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMul_1Д
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/add
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_15469*
condR
while_cond_15468*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeђ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЏ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ2
transpose_1і
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџ:::2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
д
С
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_13875

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdd
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
TanhЈ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

IdentityЌ

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*G
_input_shapes6
4:џџџџџџџџџ:џџџџџџџџџ:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
Э
Ѕ
while_cond_15110
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_15110___redundant_placeholder03
/while_while_cond_15110___redundant_placeholder13
/while_while_cond_15110___redundant_placeholder23
/while_while_cond_15110___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
§	
л
B__inference_dense_1_layer_call_and_return_conditional_losses_14628

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџш2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ш*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџш2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџш2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№	
Г
1__inference_simple_rnn_cell_1_layer_call_fn_15639

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_138922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*G
_input_shapes6
4:џџџџџџџџџ:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ЩB

simple_rnn_1_while_body_149416
2simple_rnn_1_while_simple_rnn_1_while_loop_counter<
8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations"
simple_rnn_1_while_placeholder$
 simple_rnn_1_while_placeholder_1$
 simple_rnn_1_while_placeholder_25
1simple_rnn_1_while_simple_rnn_1_strided_slice_1_0q
msimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0J
Fsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0K
Gsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0
simple_rnn_1_while_identity!
simple_rnn_1_while_identity_1!
simple_rnn_1_while_identity_2!
simple_rnn_1_while_identity_3!
simple_rnn_1_while_identity_43
/simple_rnn_1_while_simple_rnn_1_strided_slice_1o
ksimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resourceH
Dsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resourceI
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceЂ;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpн
Dsimple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2F
Dsimple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeЂ
6simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_1_while_placeholderMsimple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype028
6simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem
:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02<
:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp
+simple_rnn_1/while/simple_rnn_cell_1/MatMulMatMul=simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2-
+simple_rnn_1/while/simple_rnn_cell_1/MatMulў
;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02=
;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp
,simple_rnn_1/while/simple_rnn_cell_1/BiasAddBiasAdd5simple_rnn_1/while/simple_rnn_cell_1/MatMul:product:0Csimple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2.
,simple_rnn_1/while/simple_rnn_cell_1/BiasAdd
<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02>
<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp
-simple_rnn_1/while/simple_rnn_cell_1/MatMul_1MatMul simple_rnn_1_while_placeholder_2Dsimple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2/
-simple_rnn_1/while/simple_rnn_cell_1/MatMul_1
(simple_rnn_1/while/simple_rnn_cell_1/addAddV25simple_rnn_1/while/simple_rnn_cell_1/BiasAdd:output:07simple_rnn_1/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2*
(simple_rnn_1/while/simple_rnn_cell_1/addП
)simple_rnn_1/while/simple_rnn_cell_1/TanhTanh,simple_rnn_1/while/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2+
)simple_rnn_1/while/simple_rnn_cell_1/TanhЅ
7simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_1_while_placeholder_1simple_rnn_1_while_placeholder-simple_rnn_1/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_1/while/add/y
simple_rnn_1/while/addAddV2simple_rnn_1_while_placeholder!simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_1/while/addz
simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_1/while/add_1/yЗ
simple_rnn_1/while/add_1AddV22simple_rnn_1_while_simple_rnn_1_while_loop_counter#simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_1/while/add_1П
simple_rnn_1/while/IdentityIdentitysimple_rnn_1/while/add_1:z:0<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_1/while/Identityп
simple_rnn_1/while/Identity_1Identity8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_1/while/Identity_1С
simple_rnn_1/while/Identity_2Identitysimple_rnn_1/while/add:z:0<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_1/while/Identity_2ю
simple_rnn_1/while/Identity_3IdentityGsimple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_1/while/Identity_3ц
simple_rnn_1/while/Identity_4Identity-simple_rnn_1/while/simple_rnn_cell_1/Tanh:y:0<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_1/while/Identity_4"C
simple_rnn_1_while_identity$simple_rnn_1/while/Identity:output:0"G
simple_rnn_1_while_identity_1&simple_rnn_1/while/Identity_1:output:0"G
simple_rnn_1_while_identity_2&simple_rnn_1/while/Identity_2:output:0"G
simple_rnn_1_while_identity_3&simple_rnn_1/while/Identity_3:output:0"G
simple_rnn_1_while_identity_4&simple_rnn_1/while/Identity_4:output:0"d
/simple_rnn_1_while_simple_rnn_1_strided_slice_11simple_rnn_1_while_simple_rnn_1_strided_slice_1_0"
Dsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resourceFsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceGsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"
Csimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resourceEsimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"м
ksimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensormsimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2z
;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2x
:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp2|
<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ц<
ё
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_14212

inputs
simple_rnn_cell_1_14137
simple_rnn_cell_1_14139
simple_rnn_cell_1_14141
identityЂ)simple_rnn_cell_1/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
)simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_1_14137simple_rnn_cell_1_14139simple_rnn_cell_1_14141*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_138752+
)simple_rnn_cell_1/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterі
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_1_14137simple_rnn_cell_1_14139simple_rnn_cell_1_14141*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_14149*
condR
while_cond_14148*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeђ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЏ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ2
transpose_1Ё
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_1/StatefulPartitionedCall^while*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџ:::2V
)simple_rnn_cell_1/StatefulPartitionedCall)simple_rnn_cell_1/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж
П
#__inference_signature_wrapper_14764
embedding_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_138262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameembedding_1_input
ГQ

*sequential_1_simple_rnn_1_while_body_13753P
Lsequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_while_loop_counterV
Rsequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_while_maximum_iterations/
+sequential_1_simple_rnn_1_while_placeholder1
-sequential_1_simple_rnn_1_while_placeholder_11
-sequential_1_simple_rnn_1_while_placeholder_2O
Ksequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_strided_slice_1_0
sequential_1_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0V
Rsequential_1_simple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0W
Ssequential_1_simple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0X
Tsequential_1_simple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0,
(sequential_1_simple_rnn_1_while_identity.
*sequential_1_simple_rnn_1_while_identity_1.
*sequential_1_simple_rnn_1_while_identity_2.
*sequential_1_simple_rnn_1_while_identity_3.
*sequential_1_simple_rnn_1_while_identity_4M
Isequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_strided_slice_1
sequential_1_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorT
Psequential_1_simple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resourceU
Qsequential_1_simple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resourceV
Rsequential_1_simple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceЂHsequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂGsequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpЂIsequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpї
Qsequential_1/simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2S
Qsequential_1/simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeё
Csequential_1/simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_1_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0+sequential_1_simple_rnn_1_while_placeholderZsequential_1/simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype02E
Csequential_1/simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemЇ
Gsequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpRsequential_1_simple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02I
Gsequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpЮ
8sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMulMatMulJsequential_1/simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2:
8sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMulЅ
Hsequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpSsequential_1_simple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02J
Hsequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЪ
9sequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAddBiasAddBsequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul:product:0Psequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2;
9sequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd­
Isequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpTsequential_1_simple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02K
Isequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpЗ
:sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1MatMul-sequential_1_simple_rnn_1_while_placeholder_2Qsequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1Д
5sequential_1/simple_rnn_1/while/simple_rnn_cell_1/addAddV2Bsequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd:output:0Dsequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ27
5sequential_1/simple_rnn_1/while/simple_rnn_cell_1/addц
6sequential_1/simple_rnn_1/while/simple_rnn_cell_1/TanhTanh9sequential_1/simple_rnn_1/while/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ28
6sequential_1/simple_rnn_1/while/simple_rnn_cell_1/Tanhц
Dsequential_1/simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_1_simple_rnn_1_while_placeholder_1+sequential_1_simple_rnn_1_while_placeholder:sequential_1/simple_rnn_1/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02F
Dsequential_1/simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem
%sequential_1/simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/simple_rnn_1/while/add/yб
#sequential_1/simple_rnn_1/while/addAddV2+sequential_1_simple_rnn_1_while_placeholder.sequential_1/simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/simple_rnn_1/while/add
'sequential_1/simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/simple_rnn_1/while/add_1/yј
%sequential_1/simple_rnn_1/while/add_1AddV2Lsequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_while_loop_counter0sequential_1/simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/simple_rnn_1/while/add_1
(sequential_1/simple_rnn_1/while/IdentityIdentity)sequential_1/simple_rnn_1/while/add_1:z:0I^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(sequential_1/simple_rnn_1/while/IdentityК
*sequential_1/simple_rnn_1/while/Identity_1IdentityRsequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_while_maximum_iterationsI^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_1/simple_rnn_1/while/Identity_1
*sequential_1/simple_rnn_1/while/Identity_2Identity'sequential_1/simple_rnn_1/while/add:z:0I^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_1/simple_rnn_1/while/Identity_2М
*sequential_1/simple_rnn_1/while/Identity_3IdentityTsequential_1/simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0I^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_1/simple_rnn_1/while/Identity_3Д
*sequential_1/simple_rnn_1/while/Identity_4Identity:sequential_1/simple_rnn_1/while/simple_rnn_cell_1/Tanh:y:0I^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2,
*sequential_1/simple_rnn_1/while/Identity_4"]
(sequential_1_simple_rnn_1_while_identity1sequential_1/simple_rnn_1/while/Identity:output:0"a
*sequential_1_simple_rnn_1_while_identity_13sequential_1/simple_rnn_1/while/Identity_1:output:0"a
*sequential_1_simple_rnn_1_while_identity_23sequential_1/simple_rnn_1/while/Identity_2:output:0"a
*sequential_1_simple_rnn_1_while_identity_33sequential_1/simple_rnn_1/while/Identity_3:output:0"a
*sequential_1_simple_rnn_1_while_identity_43sequential_1/simple_rnn_1/while/Identity_4:output:0"
Isequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_strided_slice_1Ksequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_strided_slice_1_0"Ј
Qsequential_1_simple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resourceSsequential_1_simple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"Њ
Rsequential_1_simple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceTsequential_1_simple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"І
Psequential_1_simple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resourceRsequential_1_simple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"
sequential_1_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorsequential_1_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2
Hsequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpHsequential_1/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2
Gsequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpGsequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp2
Isequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpIsequential_1/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ЩB

simple_rnn_1_while_body_148166
2simple_rnn_1_while_simple_rnn_1_while_loop_counter<
8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations"
simple_rnn_1_while_placeholder$
 simple_rnn_1_while_placeholder_1$
 simple_rnn_1_while_placeholder_25
1simple_rnn_1_while_simple_rnn_1_strided_slice_1_0q
msimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0J
Fsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0K
Gsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0
simple_rnn_1_while_identity!
simple_rnn_1_while_identity_1!
simple_rnn_1_while_identity_2!
simple_rnn_1_while_identity_3!
simple_rnn_1_while_identity_43
/simple_rnn_1_while_simple_rnn_1_strided_slice_1o
ksimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resourceH
Dsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resourceI
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceЂ;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpн
Dsimple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2F
Dsimple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeЂ
6simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_1_while_placeholderMsimple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype028
6simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem
:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02<
:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp
+simple_rnn_1/while/simple_rnn_cell_1/MatMulMatMul=simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2-
+simple_rnn_1/while/simple_rnn_cell_1/MatMulў
;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02=
;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp
,simple_rnn_1/while/simple_rnn_cell_1/BiasAddBiasAdd5simple_rnn_1/while/simple_rnn_cell_1/MatMul:product:0Csimple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2.
,simple_rnn_1/while/simple_rnn_cell_1/BiasAdd
<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02>
<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp
-simple_rnn_1/while/simple_rnn_cell_1/MatMul_1MatMul simple_rnn_1_while_placeholder_2Dsimple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2/
-simple_rnn_1/while/simple_rnn_cell_1/MatMul_1
(simple_rnn_1/while/simple_rnn_cell_1/addAddV25simple_rnn_1/while/simple_rnn_cell_1/BiasAdd:output:07simple_rnn_1/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2*
(simple_rnn_1/while/simple_rnn_cell_1/addП
)simple_rnn_1/while/simple_rnn_cell_1/TanhTanh,simple_rnn_1/while/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2+
)simple_rnn_1/while/simple_rnn_cell_1/TanhЅ
7simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_1_while_placeholder_1simple_rnn_1_while_placeholder-simple_rnn_1/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_1/while/add/y
simple_rnn_1/while/addAddV2simple_rnn_1_while_placeholder!simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_1/while/addz
simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_1/while/add_1/yЗ
simple_rnn_1/while/add_1AddV22simple_rnn_1_while_simple_rnn_1_while_loop_counter#simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_1/while/add_1П
simple_rnn_1/while/IdentityIdentitysimple_rnn_1/while/add_1:z:0<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_1/while/Identityп
simple_rnn_1/while/Identity_1Identity8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_1/while/Identity_1С
simple_rnn_1/while/Identity_2Identitysimple_rnn_1/while/add:z:0<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_1/while/Identity_2ю
simple_rnn_1/while/Identity_3IdentityGsimple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_1/while/Identity_3ц
simple_rnn_1/while/Identity_4Identity-simple_rnn_1/while/simple_rnn_cell_1/Tanh:y:0<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_1/while/Identity_4"C
simple_rnn_1_while_identity$simple_rnn_1/while/Identity:output:0"G
simple_rnn_1_while_identity_1&simple_rnn_1/while/Identity_1:output:0"G
simple_rnn_1_while_identity_2&simple_rnn_1/while/Identity_2:output:0"G
simple_rnn_1_while_identity_3&simple_rnn_1/while/Identity_3:output:0"G
simple_rnn_1_while_identity_4&simple_rnn_1/while/Identity_4:output:0"d
/simple_rnn_1_while_simple_rnn_1_strided_slice_11simple_rnn_1_while_simple_rnn_1_strided_slice_1_0"
Dsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resourceFsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceGsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"
Csimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resourceEsimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"м
ksimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensormsimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2z
;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2x
:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp2|
<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
м
|
'__inference_dense_1_layer_call_fn_15577

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_146282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч3
ћ
while_body_15357
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_1_matmul_readvariableop_resource;
7while_simple_rnn_cell_1_biasadd_readvariableop_resource<
8while_simple_rnn_cell_1_matmul_1_readvariableop_resourceЂ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemй
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOpц
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
while/simple_rnn_cell_1/MatMulз
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpт
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
while/simple_rnn_cell_1/BiasAddп
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpЯ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 while/simple_rnn_cell_1/MatMul_1Ь
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/add
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/Tanhф
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ё
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ѓ
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
к
У
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_15611

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdd
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
TanhЈ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

IdentityЌ

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*G
_input_shapes6
4:џџџџџџџџџ:џџџџџџџџџ:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
Дl
а
G__inference_sequential_1_layer_call_and_return_conditional_losses_14889

inputs&
"embedding_1_embedding_lookup_14768A
=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resourceB
>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resourceC
?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂembedding_1/embedding_lookupЂ5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpЂ6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpЂsimple_rnn_1/whileu
embedding_1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
embedding_1/CastИ
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_14768embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/14768*,
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_1/embedding_lookup
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/14768*,
_output_shapes
:џџџџџџџџџ2'
%embedding_1/embedding_lookup/IdentityХ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2)
'embedding_1/embedding_lookup/Identity_1
simple_rnn_1/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
simple_rnn_1/Shape
 simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_1/strided_slice/stack
"simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_1/strided_slice/stack_1
"simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_1/strided_slice/stack_2А
simple_rnn_1/strided_sliceStridedSlicesimple_rnn_1/Shape:output:0)simple_rnn_1/strided_slice/stack:output:0+simple_rnn_1/strided_slice/stack_1:output:0+simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_1/strided_slicew
simple_rnn_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
simple_rnn_1/zeros/mul/y 
simple_rnn_1/zeros/mulMul#simple_rnn_1/strided_slice:output:0!simple_rnn_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_1/zeros/muly
simple_rnn_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
simple_rnn_1/zeros/Less/y
simple_rnn_1/zeros/LessLesssimple_rnn_1/zeros/mul:z:0"simple_rnn_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_1/zeros/Less}
simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
simple_rnn_1/zeros/packed/1З
simple_rnn_1/zeros/packedPack#simple_rnn_1/strided_slice:output:0$simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_1/zeros/packedy
simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_1/zeros/ConstЊ
simple_rnn_1/zerosFill"simple_rnn_1/zeros/packed:output:0!simple_rnn_1/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_1/zeros
simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_1/transpose/permЬ
simple_rnn_1/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0$simple_rnn_1/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
simple_rnn_1/transposev
simple_rnn_1/Shape_1Shapesimple_rnn_1/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_1/Shape_1
"simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_1/strided_slice_1/stack
$simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_1/strided_slice_1/stack_1
$simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_1/strided_slice_1/stack_2М
simple_rnn_1/strided_slice_1StridedSlicesimple_rnn_1/Shape_1:output:0+simple_rnn_1/strided_slice_1/stack:output:0-simple_rnn_1/strided_slice_1/stack_1:output:0-simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_1/strided_slice_1
(simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(simple_rnn_1/TensorArrayV2/element_shapeц
simple_rnn_1/TensorArrayV2TensorListReserve1simple_rnn_1/TensorArrayV2/element_shape:output:0%simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_1/TensorArrayV2й
Bsimple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2D
Bsimple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeЌ
4simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_1/transpose:y:0Ksimple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_1/TensorArrayUnstack/TensorListFromTensor
"simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_1/strided_slice_2/stack
$simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_1/strided_slice_2/stack_1
$simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_1/strided_slice_2/stack_2Ы
simple_rnn_1/strided_slice_2StridedSlicesimple_rnn_1/transpose:y:0+simple_rnn_1/strided_slice_2/stack:output:0-simple_rnn_1/strided_slice_2/stack_1:output:0-simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
simple_rnn_1/strided_slice_2ь
4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype026
4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp№
%simple_rnn_1/simple_rnn_cell_1/MatMulMatMul%simple_rnn_1/strided_slice_2:output:0<simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2'
%simple_rnn_1/simple_rnn_cell_1/MatMulъ
5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpў
&simple_rnn_1/simple_rnn_cell_1/BiasAddBiasAdd/simple_rnn_1/simple_rnn_cell_1/MatMul:product:0=simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2(
&simple_rnn_1/simple_rnn_cell_1/BiasAddђ
6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype028
6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpь
'simple_rnn_1/simple_rnn_cell_1/MatMul_1MatMulsimple_rnn_1/zeros:output:0>simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2)
'simple_rnn_1/simple_rnn_cell_1/MatMul_1ш
"simple_rnn_1/simple_rnn_cell_1/addAddV2/simple_rnn_1/simple_rnn_cell_1/BiasAdd:output:01simple_rnn_1/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2$
"simple_rnn_1/simple_rnn_cell_1/add­
#simple_rnn_1/simple_rnn_cell_1/TanhTanh&simple_rnn_1/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2%
#simple_rnn_1/simple_rnn_cell_1/TanhЉ
*simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2,
*simple_rnn_1/TensorArrayV2_1/element_shapeь
simple_rnn_1/TensorArrayV2_1TensorListReserve3simple_rnn_1/TensorArrayV2_1/element_shape:output:0%simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_1/TensorArrayV2_1h
simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_1/time
%simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%simple_rnn_1/while/maximum_iterations
simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_1/while/loop_counterњ
simple_rnn_1/whileWhile(simple_rnn_1/while/loop_counter:output:0.simple_rnn_1/while/maximum_iterations:output:0simple_rnn_1/time:output:0%simple_rnn_1/TensorArrayV2_1:handle:0simple_rnn_1/zeros:output:0%simple_rnn_1/strided_slice_1:output:0Dsimple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_1_while_body_14816*)
cond!R
simple_rnn_1_while_cond_14815*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
simple_rnn_1/whileЯ
=simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape
/simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_1/while:output:3Fsimple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype021
/simple_rnn_1/TensorArrayV2Stack/TensorListStack
"simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2$
"simple_rnn_1/strided_slice_3/stack
$simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_1/strided_slice_3/stack_1
$simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_1/strided_slice_3/stack_2щ
simple_rnn_1/strided_slice_3StridedSlice8simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_1/strided_slice_3/stack:output:0-simple_rnn_1/strided_slice_3/stack_1:output:0-simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
simple_rnn_1/strided_slice_3
simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_1/transpose_1/permк
simple_rnn_1/transpose_1	Transpose8simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
simple_rnn_1/transpose_1Ї
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ш*
dtype02
dense_1/MatMul/ReadVariableOpЋ
dense_1/MatMulMatMul%simple_rnn_1/strided_slice_3:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџш2
dense_1/MatMulЅ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ш*
dtype02 
dense_1/BiasAdd/ReadVariableOpЂ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџш2
dense_1/BiasAddz
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџш2
dense_1/Softmax
IdentityIdentitydense_1/Softmax:softmax:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding_1/embedding_lookup6^simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp5^simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp7^simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp^simple_rnn_1/while*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2n
5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp2l
4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp2p
6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp2(
simple_rnn_1/whilesimple_rnn_1/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
#
ў
while_body_14149
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_1_14171_0#
while_simple_rnn_cell_1_14173_0#
while_simple_rnn_cell_1_14175_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_1_14171!
while_simple_rnn_cell_1_14173!
while_simple_rnn_cell_1_14175Ђ/while/simple_rnn_cell_1/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemг
/while/simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_1_14171_0while_simple_rnn_cell_1_14173_0while_simple_rnn_cell_1_14175_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_1387521
/while/simple_rnn_cell_1/StatefulPartitionedCallќ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/IdentityЃ
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2П
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Я
while/Identity_4Identity8while/simple_rnn_cell_1/StatefulPartitionedCall:output:10^while/simple_rnn_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_1_14171while_simple_rnn_cell_1_14171_0"@
while_simple_rnn_cell_1_14173while_simple_rnn_cell_1_14173_0"@
while_simple_rnn_cell_1_14175while_simple_rnn_cell_1_14175_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2b
/while/simple_rnn_cell_1/StatefulPartitionedCall/while/simple_rnn_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ч3
ћ
while_body_15223
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_1_matmul_readvariableop_resource;
7while_simple_rnn_cell_1_biasadd_readvariableop_resource<
8while_simple_rnn_cell_1_matmul_1_readvariableop_resourceЂ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemй
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOpц
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
while/simple_rnn_cell_1/MatMulз
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpт
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
while/simple_rnn_cell_1/BiasAddп
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpЯ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 while/simple_rnn_cell_1/MatMul_1Ь
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/add
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/Tanhф
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ё
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ѓ
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ч3
ћ
while_body_15111
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_1_matmul_readvariableop_resource;
7while_simple_rnn_cell_1_biasadd_readvariableop_resource<
8while_simple_rnn_cell_1_matmul_1_readvariableop_resourceЂ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemй
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOpц
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
while/simple_rnn_cell_1/MatMulз
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpт
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
while/simple_rnn_cell_1/BiasAddп
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpЯ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 while/simple_rnn_cell_1/MatMul_1Ь
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/add
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/simple_rnn_cell_1/Tanhф
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ё
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ѓ
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Дl
а
G__inference_sequential_1_layer_call_and_return_conditional_losses_15014

inputs&
"embedding_1_embedding_lookup_14893A
=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resourceB
>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resourceC
?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂembedding_1/embedding_lookupЂ5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpЂ6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpЂsimple_rnn_1/whileu
embedding_1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
embedding_1/CastИ
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_14893embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/14893*,
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_1/embedding_lookup
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/14893*,
_output_shapes
:џџџџџџџџџ2'
%embedding_1/embedding_lookup/IdentityХ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2)
'embedding_1/embedding_lookup/Identity_1
simple_rnn_1/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
simple_rnn_1/Shape
 simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_1/strided_slice/stack
"simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_1/strided_slice/stack_1
"simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_1/strided_slice/stack_2А
simple_rnn_1/strided_sliceStridedSlicesimple_rnn_1/Shape:output:0)simple_rnn_1/strided_slice/stack:output:0+simple_rnn_1/strided_slice/stack_1:output:0+simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_1/strided_slicew
simple_rnn_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
simple_rnn_1/zeros/mul/y 
simple_rnn_1/zeros/mulMul#simple_rnn_1/strided_slice:output:0!simple_rnn_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_1/zeros/muly
simple_rnn_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
simple_rnn_1/zeros/Less/y
simple_rnn_1/zeros/LessLesssimple_rnn_1/zeros/mul:z:0"simple_rnn_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_1/zeros/Less}
simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
simple_rnn_1/zeros/packed/1З
simple_rnn_1/zeros/packedPack#simple_rnn_1/strided_slice:output:0$simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_1/zeros/packedy
simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_1/zeros/ConstЊ
simple_rnn_1/zerosFill"simple_rnn_1/zeros/packed:output:0!simple_rnn_1/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_1/zeros
simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_1/transpose/permЬ
simple_rnn_1/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0$simple_rnn_1/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
simple_rnn_1/transposev
simple_rnn_1/Shape_1Shapesimple_rnn_1/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_1/Shape_1
"simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_1/strided_slice_1/stack
$simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_1/strided_slice_1/stack_1
$simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_1/strided_slice_1/stack_2М
simple_rnn_1/strided_slice_1StridedSlicesimple_rnn_1/Shape_1:output:0+simple_rnn_1/strided_slice_1/stack:output:0-simple_rnn_1/strided_slice_1/stack_1:output:0-simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_1/strided_slice_1
(simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(simple_rnn_1/TensorArrayV2/element_shapeц
simple_rnn_1/TensorArrayV2TensorListReserve1simple_rnn_1/TensorArrayV2/element_shape:output:0%simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_1/TensorArrayV2й
Bsimple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2D
Bsimple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeЌ
4simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_1/transpose:y:0Ksimple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_1/TensorArrayUnstack/TensorListFromTensor
"simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_1/strided_slice_2/stack
$simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_1/strided_slice_2/stack_1
$simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_1/strided_slice_2/stack_2Ы
simple_rnn_1/strided_slice_2StridedSlicesimple_rnn_1/transpose:y:0+simple_rnn_1/strided_slice_2/stack:output:0-simple_rnn_1/strided_slice_2/stack_1:output:0-simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
simple_rnn_1/strided_slice_2ь
4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype026
4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp№
%simple_rnn_1/simple_rnn_cell_1/MatMulMatMul%simple_rnn_1/strided_slice_2:output:0<simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2'
%simple_rnn_1/simple_rnn_cell_1/MatMulъ
5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpў
&simple_rnn_1/simple_rnn_cell_1/BiasAddBiasAdd/simple_rnn_1/simple_rnn_cell_1/MatMul:product:0=simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2(
&simple_rnn_1/simple_rnn_cell_1/BiasAddђ
6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype028
6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpь
'simple_rnn_1/simple_rnn_cell_1/MatMul_1MatMulsimple_rnn_1/zeros:output:0>simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2)
'simple_rnn_1/simple_rnn_cell_1/MatMul_1ш
"simple_rnn_1/simple_rnn_cell_1/addAddV2/simple_rnn_1/simple_rnn_cell_1/BiasAdd:output:01simple_rnn_1/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2$
"simple_rnn_1/simple_rnn_cell_1/add­
#simple_rnn_1/simple_rnn_cell_1/TanhTanh&simple_rnn_1/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2%
#simple_rnn_1/simple_rnn_cell_1/TanhЉ
*simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2,
*simple_rnn_1/TensorArrayV2_1/element_shapeь
simple_rnn_1/TensorArrayV2_1TensorListReserve3simple_rnn_1/TensorArrayV2_1/element_shape:output:0%simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_1/TensorArrayV2_1h
simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_1/time
%simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%simple_rnn_1/while/maximum_iterations
simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_1/while/loop_counterњ
simple_rnn_1/whileWhile(simple_rnn_1/while/loop_counter:output:0.simple_rnn_1/while/maximum_iterations:output:0simple_rnn_1/time:output:0%simple_rnn_1/TensorArrayV2_1:handle:0simple_rnn_1/zeros:output:0%simple_rnn_1/strided_slice_1:output:0Dsimple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_1_while_body_14941*)
cond!R
simple_rnn_1_while_cond_14940*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
simple_rnn_1/whileЯ
=simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape
/simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_1/while:output:3Fsimple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype021
/simple_rnn_1/TensorArrayV2Stack/TensorListStack
"simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2$
"simple_rnn_1/strided_slice_3/stack
$simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_1/strided_slice_3/stack_1
$simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_1/strided_slice_3/stack_2щ
simple_rnn_1/strided_slice_3StridedSlice8simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_1/strided_slice_3/stack:output:0-simple_rnn_1/strided_slice_3/stack_1:output:0-simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
simple_rnn_1/strided_slice_3
simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_1/transpose_1/permк
simple_rnn_1/transpose_1	Transpose8simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
simple_rnn_1/transpose_1Ї
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ш*
dtype02
dense_1/MatMul/ReadVariableOpЋ
dense_1/MatMulMatMul%simple_rnn_1/strided_slice_3:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџш2
dense_1/MatMulЅ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ш*
dtype02 
dense_1/BiasAdd/ReadVariableOpЂ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџш2
dense_1/BiasAddz
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџш2
dense_1/Softmax
IdentityIdentitydense_1/Softmax:softmax:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding_1/embedding_lookup6^simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp5^simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp7^simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp^simple_rnn_1/while*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2n
5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp2l
4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp2p
6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp2(
simple_rnn_1/whilesimple_rnn_1/while:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЌH

G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15423
inputs_04
0simple_rnn_cell_1_matmul_readvariableop_resource5
1simple_rnn_cell_1_biasadd_readvariableop_resource6
2simple_rnn_cell_1_matmul_1_readvariableop_resource
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Х
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOpМ
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMulУ
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOpЪ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/BiasAddЫ
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOpИ
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMul_1Д
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/add
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_15357*
condR
while_cond_15356*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeђ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЏ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ2
transpose_1і
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџ:::2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
С

*sequential_1_simple_rnn_1_while_cond_13752P
Lsequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_while_loop_counterV
Rsequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_while_maximum_iterations/
+sequential_1_simple_rnn_1_while_placeholder1
-sequential_1_simple_rnn_1_while_placeholder_11
-sequential_1_simple_rnn_1_while_placeholder_2R
Nsequential_1_simple_rnn_1_while_less_sequential_1_simple_rnn_1_strided_slice_1g
csequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_while_cond_13752___redundant_placeholder0g
csequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_while_cond_13752___redundant_placeholder1g
csequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_while_cond_13752___redundant_placeholder2g
csequential_1_simple_rnn_1_while_sequential_1_simple_rnn_1_while_cond_13752___redundant_placeholder3,
(sequential_1_simple_rnn_1_while_identity
ђ
$sequential_1/simple_rnn_1/while/LessLess+sequential_1_simple_rnn_1_while_placeholderNsequential_1_simple_rnn_1_while_less_sequential_1_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: 2&
$sequential_1/simple_rnn_1/while/LessЋ
(sequential_1/simple_rnn_1/while/IdentityIdentity(sequential_1/simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: 2*
(sequential_1/simple_rnn_1/while/Identity"]
(sequential_1_simple_rnn_1_while_identity1sequential_1/simple_rnn_1/while/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Э
Ѕ
while_cond_15356
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_15356___redundant_placeholder03
/while_while_cond_15356___redundant_placeholder13
/while_while_cond_15356___redundant_placeholder23
/while_while_cond_15356___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ш


simple_rnn_1_while_cond_149406
2simple_rnn_1_while_simple_rnn_1_while_loop_counter<
8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations"
simple_rnn_1_while_placeholder$
 simple_rnn_1_while_placeholder_1$
 simple_rnn_1_while_placeholder_28
4simple_rnn_1_while_less_simple_rnn_1_strided_slice_1M
Isimple_rnn_1_while_simple_rnn_1_while_cond_14940___redundant_placeholder0M
Isimple_rnn_1_while_simple_rnn_1_while_cond_14940___redundant_placeholder1M
Isimple_rnn_1_while_simple_rnn_1_while_cond_14940___redundant_placeholder2M
Isimple_rnn_1_while_simple_rnn_1_while_cond_14940___redundant_placeholder3
simple_rnn_1_while_identity
Б
simple_rnn_1/while/LessLesssimple_rnn_1_while_placeholder4simple_rnn_1_while_less_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_1/while/Less
simple_rnn_1/while/IdentityIdentitysimple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_1/while/Identity"C
simple_rnn_1_while_identity$simple_rnn_1/while/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
э@

__inference__traced_save_15743
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_simple_rnn_1_simple_rnn_cell_1_kernel_read_readvariableopN
Jsavev2_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_1_simple_rnn_cell_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_1_simple_rnn_cell_1_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_1_simple_rnn_cell_1_bias_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_1_simple_rnn_cell_1_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_1_simple_rnn_cell_1_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*І
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_simple_rnn_1_simple_rnn_cell_1_kernel_read_readvariableopJsavev2_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_read_readvariableop>savev2_simple_rnn_1_simple_rnn_cell_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableopGsavev2_adam_simple_rnn_1_simple_rnn_cell_1_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_1_simple_rnn_cell_1_bias_m_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopGsavev2_adam_simple_rnn_1_simple_rnn_cell_1_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_1_simple_rnn_cell_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*х
_input_shapesг
а: :
ш:
ш:ш: : : : : :
:
:: : : : :
ш:
ш:ш:
:
::
ш:
ш:ш:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ш:&"
 
_output_shapes
:
ш:!

_output_shapes	
:ш:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&	"
 
_output_shapes
:
:&
"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
ш:&"
 
_output_shapes
:
ш:!

_output_shapes	
:ш:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
ш:&"
 
_output_shapes
:
ш:!

_output_shapes	
:ш:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 
х
Н
,__inference_sequential_1_layer_call_fn_15031

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_146862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э
Ѕ
while_cond_15468
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_15468___redundant_placeholder03
/while_while_cond_15468___redundant_placeholder13
/while_while_cond_15468___redundant_placeholder23
/while_while_cond_15468___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
іG

G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15289

inputs4
0simple_rnn_cell_1_matmul_readvariableop_resource5
1simple_rnn_cell_1_biasadd_readvariableop_resource6
2simple_rnn_cell_1_matmul_1_readvariableop_resource
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Х
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOpМ
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMulУ
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOpЪ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/BiasAddЫ
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOpИ
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/MatMul_1Д
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/add
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/Tanh
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_15223*
condR
while_cond_15222*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permІ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
transpose_1і
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ:::2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц<
ё
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_14329

inputs
simple_rnn_cell_1_14254
simple_rnn_cell_1_14256
simple_rnn_cell_1_14258
identityЂ)simple_rnn_cell_1/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
)simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_1_14254simple_rnn_cell_1_14256simple_rnn_cell_1_14258*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_138922+
)simple_rnn_cell_1/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterі
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_1_14254simple_rnn_cell_1_14256simple_rnn_cell_1_14258*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_14266*
condR
while_cond_14265*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeђ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЏ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ2
transpose_1Ё
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_1/StatefulPartitionedCall^while*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџ:::2V
)simple_rnn_cell_1/StatefulPartitionedCall)simple_rnn_cell_1/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ш
,__inference_sequential_1_layer_call_fn_14701
embedding_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_146862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameembedding_1_input
§	
л
B__inference_dense_1_layer_call_and_return_conditional_losses_15568

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџш2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ш*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџш2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџш2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э
Ѕ
while_cond_14148
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_14148___redundant_placeholder03
/while_while_cond_14148___redundant_placeholder13
/while_while_cond_14148___redundant_placeholder23
/while_while_cond_14148___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Њ
х
G__inference_sequential_1_layer_call_and_return_conditional_losses_14664
embedding_1_input
embedding_1_14648
simple_rnn_1_14651
simple_rnn_1_14653
simple_rnn_1_14655
dense_1_14658
dense_1_14660
identityЂdense_1/StatefulPartitionedCallЂ#embedding_1/StatefulPartitionedCallЂ$simple_rnn_1/StatefulPartitionedCall
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_14648*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_143522%
#embedding_1/StatefulPartitionedCallт
$simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0simple_rnn_1_14651simple_rnn_1_14653simple_rnn_1_14655*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_145872&
$simple_rnn_1/StatefulPartitionedCallД
dense_1/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_1/StatefulPartitionedCall:output:0dense_1_14658dense_1_14660*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_146282!
dense_1/StatefulPartitionedCallь
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall%^simple_rnn_1/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2L
$simple_rnn_1/StatefulPartitionedCall$simple_rnn_1/StatefulPartitionedCall:Z V
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameembedding_1_input
Њ
х
G__inference_sequential_1_layer_call_and_return_conditional_losses_14645
embedding_1_input
embedding_1_14361
simple_rnn_1_14610
simple_rnn_1_14612
simple_rnn_1_14614
dense_1_14639
dense_1_14641
identityЂdense_1/StatefulPartitionedCallЂ#embedding_1/StatefulPartitionedCallЂ$simple_rnn_1/StatefulPartitionedCall
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_14361*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_143522%
#embedding_1/StatefulPartitionedCallт
$simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0simple_rnn_1_14610simple_rnn_1_14612simple_rnn_1_14614*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_144752&
$simple_rnn_1/StatefulPartitionedCallД
dense_1/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_1/StatefulPartitionedCall:output:0dense_1_14639dense_1_14641*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_146282!
dense_1/StatefulPartitionedCallь
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall%^simple_rnn_1/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2L
$simple_rnn_1/StatefulPartitionedCall$simple_rnn_1/StatefulPartitionedCall:Z V
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameembedding_1_input


,__inference_simple_rnn_1_layer_call_fn_15300

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_144752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о	

F__inference_embedding_1_layer_call_and_return_conditional_losses_15058

inputs
embedding_lookup_15052
identityЂembedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
Castќ
embedding_lookupResourceGatherembedding_lookup_15052Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/15052*,
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupэ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/15052*,
_output_shapes
:џџџџџџџџџ2
embedding_lookup/IdentityЁ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э
Ѕ
while_cond_14265
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_14265___redundant_placeholder03
/while_while_cond_14265___redundant_placeholder13
/while_while_cond_14265___redundant_placeholder23
/while_while_cond_14265___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

к
G__inference_sequential_1_layer_call_and_return_conditional_losses_14722

inputs
embedding_1_14706
simple_rnn_1_14709
simple_rnn_1_14711
simple_rnn_1_14713
dense_1_14716
dense_1_14718
identityЂdense_1/StatefulPartitionedCallЂ#embedding_1/StatefulPartitionedCallЂ$simple_rnn_1/StatefulPartitionedCall
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_14706*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_143522%
#embedding_1/StatefulPartitionedCallт
$simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0simple_rnn_1_14709simple_rnn_1_14711simple_rnn_1_14713*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_145872&
$simple_rnn_1/StatefulPartitionedCallД
dense_1/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_1/StatefulPartitionedCall:output:0dense_1_14716dense_1_14718*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџш*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_146282!
dense_1/StatefulPartitionedCallь
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall%^simple_rnn_1/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2L
$simple_rnn_1/StatefulPartitionedCall$simple_rnn_1/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№	
Г
1__inference_simple_rnn_cell_1_layer_call_fn_15625

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_138752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*G
_input_shapes6
4:џџџџџџџџџ:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0


,__inference_simple_rnn_1_layer_call_fn_15311

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_145872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э
Ѕ
while_cond_14408
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_14408___redundant_placeholder03
/while_while_cond_14408___redundant_placeholder13
/while_while_cond_14408___redundant_placeholder23
/while_while_cond_14408___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
#
ў
while_body_14266
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_1_14288_0#
while_simple_rnn_cell_1_14290_0#
while_simple_rnn_cell_1_14292_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_1_14288!
while_simple_rnn_cell_1_14290!
while_simple_rnn_cell_1_14292Ђ/while/simple_rnn_cell_1/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemг
/while/simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_1_14288_0while_simple_rnn_cell_1_14290_0while_simple_rnn_cell_1_14292_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_1389221
/while/simple_rnn_cell_1/StatefulPartitionedCallќ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/IdentityЃ
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2П
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Я
while/Identity_4Identity8while/simple_rnn_cell_1/StatefulPartitionedCall:output:10^while/simple_rnn_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_1_14288while_simple_rnn_cell_1_14288_0"@
while_simple_rnn_cell_1_14290while_simple_rnn_cell_1_14290_0"@
while_simple_rnn_cell_1_14292while_simple_rnn_cell_1_14292_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :џџџџџџџџџ: : :::2b
/while/simple_rnn_cell_1/StatefulPartitionedCall/while/simple_rnn_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Э
н
 __inference__wrapped_model_13826
embedding_1_input3
/sequential_1_embedding_1_embedding_lookup_13705N
Jsequential_1_simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resourceO
Ksequential_1_simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resourceP
Lsequential_1_simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource7
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identityЂ+sequential_1/dense_1/BiasAdd/ReadVariableOpЂ*sequential_1/dense_1/MatMul/ReadVariableOpЂ)sequential_1/embedding_1/embedding_lookupЂBsequential_1/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂAsequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpЂCsequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpЂsequential_1/simple_rnn_1/while
sequential_1/embedding_1/CastCastembedding_1_input*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
sequential_1/embedding_1/Castљ
)sequential_1/embedding_1/embedding_lookupResourceGather/sequential_1_embedding_1_embedding_lookup_13705!sequential_1/embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@sequential_1/embedding_1/embedding_lookup/13705*,
_output_shapes
:џџџџџџџџџ*
dtype02+
)sequential_1/embedding_1/embedding_lookupб
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@sequential_1/embedding_1/embedding_lookup/13705*,
_output_shapes
:џџџџџџџџџ24
2sequential_1/embedding_1/embedding_lookup/Identityь
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ26
4sequential_1/embedding_1/embedding_lookup/Identity_1Џ
sequential_1/simple_rnn_1/ShapeShape=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2!
sequential_1/simple_rnn_1/ShapeЈ
-sequential_1/simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_1/simple_rnn_1/strided_slice/stackЌ
/sequential_1/simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/simple_rnn_1/strided_slice/stack_1Ќ
/sequential_1/simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/simple_rnn_1/strided_slice/stack_2ў
'sequential_1/simple_rnn_1/strided_sliceStridedSlice(sequential_1/simple_rnn_1/Shape:output:06sequential_1/simple_rnn_1/strided_slice/stack:output:08sequential_1/simple_rnn_1/strided_slice/stack_1:output:08sequential_1/simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_1/simple_rnn_1/strided_slice
%sequential_1/simple_rnn_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2'
%sequential_1/simple_rnn_1/zeros/mul/yд
#sequential_1/simple_rnn_1/zeros/mulMul0sequential_1/simple_rnn_1/strided_slice:output:0.sequential_1/simple_rnn_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/simple_rnn_1/zeros/mul
&sequential_1/simple_rnn_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2(
&sequential_1/simple_rnn_1/zeros/Less/yЯ
$sequential_1/simple_rnn_1/zeros/LessLess'sequential_1/simple_rnn_1/zeros/mul:z:0/sequential_1/simple_rnn_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$sequential_1/simple_rnn_1/zeros/Less
(sequential_1/simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2*
(sequential_1/simple_rnn_1/zeros/packed/1ы
&sequential_1/simple_rnn_1/zeros/packedPack0sequential_1/simple_rnn_1/strided_slice:output:01sequential_1/simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_1/simple_rnn_1/zeros/packed
%sequential_1/simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential_1/simple_rnn_1/zeros/Constо
sequential_1/simple_rnn_1/zerosFill/sequential_1/simple_rnn_1/zeros/packed:output:0.sequential_1/simple_rnn_1/zeros/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
sequential_1/simple_rnn_1/zerosЉ
(sequential_1/simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_1/simple_rnn_1/transpose/perm
#sequential_1/simple_rnn_1/transpose	Transpose=sequential_1/embedding_1/embedding_lookup/Identity_1:output:01sequential_1/simple_rnn_1/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2%
#sequential_1/simple_rnn_1/transpose
!sequential_1/simple_rnn_1/Shape_1Shape'sequential_1/simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:2#
!sequential_1/simple_rnn_1/Shape_1Ќ
/sequential_1/simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_1/simple_rnn_1/strided_slice_1/stackА
1sequential_1/simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_1/strided_slice_1/stack_1А
1sequential_1/simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_1/strided_slice_1/stack_2
)sequential_1/simple_rnn_1/strided_slice_1StridedSlice*sequential_1/simple_rnn_1/Shape_1:output:08sequential_1/simple_rnn_1/strided_slice_1/stack:output:0:sequential_1/simple_rnn_1/strided_slice_1/stack_1:output:0:sequential_1/simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_1/simple_rnn_1/strided_slice_1Й
5sequential_1/simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ27
5sequential_1/simple_rnn_1/TensorArrayV2/element_shape
'sequential_1/simple_rnn_1/TensorArrayV2TensorListReserve>sequential_1/simple_rnn_1/TensorArrayV2/element_shape:output:02sequential_1/simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_1/simple_rnn_1/TensorArrayV2ѓ
Osequential_1/simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2Q
Osequential_1/simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeр
Asequential_1/simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_1/simple_rnn_1/transpose:y:0Xsequential_1/simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_1/simple_rnn_1/TensorArrayUnstack/TensorListFromTensorЌ
/sequential_1/simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_1/simple_rnn_1/strided_slice_2/stackА
1sequential_1/simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_1/strided_slice_2/stack_1А
1sequential_1/simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_1/strided_slice_2/stack_2
)sequential_1/simple_rnn_1/strided_slice_2StridedSlice'sequential_1/simple_rnn_1/transpose:y:08sequential_1/simple_rnn_1/strided_slice_2/stack:output:0:sequential_1/simple_rnn_1/strided_slice_2/stack_1:output:0:sequential_1/simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2+
)sequential_1/simple_rnn_1/strided_slice_2
Asequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpJsequential_1_simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02C
Asequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpЄ
2sequential_1/simple_rnn_1/simple_rnn_cell_1/MatMulMatMul2sequential_1/simple_rnn_1/strided_slice_2:output:0Isequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ24
2sequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul
Bsequential_1/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpKsequential_1_simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bsequential_1/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpВ
3sequential_1/simple_rnn_1/simple_rnn_cell_1/BiasAddBiasAdd<sequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul:product:0Jsequential_1/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ25
3sequential_1/simple_rnn_1/simple_rnn_cell_1/BiasAdd
Csequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpLsequential_1_simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02E
Csequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp 
4sequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul_1MatMul(sequential_1/simple_rnn_1/zeros:output:0Ksequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ26
4sequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul_1
/sequential_1/simple_rnn_1/simple_rnn_cell_1/addAddV2<sequential_1/simple_rnn_1/simple_rnn_cell_1/BiasAdd:output:0>sequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџ21
/sequential_1/simple_rnn_1/simple_rnn_cell_1/addд
0sequential_1/simple_rnn_1/simple_rnn_cell_1/TanhTanh3sequential_1/simple_rnn_1/simple_rnn_cell_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ22
0sequential_1/simple_rnn_1/simple_rnn_cell_1/TanhУ
7sequential_1/simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7sequential_1/simple_rnn_1/TensorArrayV2_1/element_shape 
)sequential_1/simple_rnn_1/TensorArrayV2_1TensorListReserve@sequential_1/simple_rnn_1/TensorArrayV2_1/element_shape:output:02sequential_1/simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_1/simple_rnn_1/TensorArrayV2_1
sequential_1/simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential_1/simple_rnn_1/timeГ
2sequential_1/simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2sequential_1/simple_rnn_1/while/maximum_iterations
,sequential_1/simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/simple_rnn_1/while/loop_counterА
sequential_1/simple_rnn_1/whileWhile5sequential_1/simple_rnn_1/while/loop_counter:output:0;sequential_1/simple_rnn_1/while/maximum_iterations:output:0'sequential_1/simple_rnn_1/time:output:02sequential_1/simple_rnn_1/TensorArrayV2_1:handle:0(sequential_1/simple_rnn_1/zeros:output:02sequential_1/simple_rnn_1/strided_slice_1:output:0Qsequential_1/simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_1_simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resourceKsequential_1_simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resourceLsequential_1_simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*6
body.R,
*sequential_1_simple_rnn_1_while_body_13753*6
cond.R,
*sequential_1_simple_rnn_1_while_cond_13752*9
output_shapes(
&: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2!
sequential_1/simple_rnn_1/whileщ
Jsequential_1/simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2L
Jsequential_1/simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeб
<sequential_1/simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_1/simple_rnn_1/while:output:3Ssequential_1/simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype02>
<sequential_1/simple_rnn_1/TensorArrayV2Stack/TensorListStackЕ
/sequential_1/simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ21
/sequential_1/simple_rnn_1/strided_slice_3/stackА
1sequential_1/simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/simple_rnn_1/strided_slice_3/stack_1А
1sequential_1/simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_1/strided_slice_3/stack_2З
)sequential_1/simple_rnn_1/strided_slice_3StridedSliceEsequential_1/simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:08sequential_1/simple_rnn_1/strided_slice_3/stack:output:0:sequential_1/simple_rnn_1/strided_slice_3/stack_1:output:0:sequential_1/simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2+
)sequential_1/simple_rnn_1/strided_slice_3­
*sequential_1/simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential_1/simple_rnn_1/transpose_1/perm
%sequential_1/simple_rnn_1/transpose_1	TransposeEsequential_1/simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:03sequential_1/simple_rnn_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2'
%sequential_1/simple_rnn_1/transpose_1Ю
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ш*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOpп
sequential_1/dense_1/MatMulMatMul2sequential_1/simple_rnn_1/strided_slice_3:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџш2
sequential_1/dense_1/MatMulЬ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ш*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpж
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџш2
sequential_1/dense_1/BiasAddЁ
sequential_1/dense_1/SoftmaxSoftmax%sequential_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџш2
sequential_1/dense_1/Softmaxѓ
IdentityIdentity&sequential_1/dense_1/Softmax:softmax:0,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*^sequential_1/embedding_1/embedding_lookupC^sequential_1/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpB^sequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpD^sequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp ^sequential_1/simple_rnn_1/while*
T0*(
_output_shapes
:џџџџџџџџџш2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2
Bsequential_1/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpBsequential_1/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp2
Asequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpAsequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp2
Csequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpCsequential_1/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp2B
sequential_1/simple_rnn_1/whilesequential_1/simple_rnn_1/while:Z V
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameembedding_1_input"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*П
serving_defaultЋ
O
embedding_1_input:
#serving_default_embedding_1_input:0џџџџџџџџџ<
dense_11
StatefulPartitionedCall:0џџџџџџџџџшtensorflow/serving/predict:шД
ђ&
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
X__call__
Y_default_save_signature
*Z&call_and_return_all_conditional_losses"В$
_tf_keras_sequential${"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_1_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "input_dim": 1000, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 3}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1000, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_1_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "input_dim": 1000, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 3}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1000, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ѕ


embeddings
	variables
trainable_variables
regularization_losses
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layerь{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "input_dim": 1000, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
ў

cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"е	
_tf_keras_rnn_layerЗ	{"class_name": "SimpleRNN", "name": "simple_rnn_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 256]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 256]}}
ї

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
___call__
*`&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1000, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
П
iter

beta_1

beta_2
	decay
learning_rate
mLmMmN mO!mP"mQ
vRvSvT vU!vV"vW"
	optimizer
J

0
 1
!2
"3
4
5"
trackable_list_wrapper
J

0
 1
!2
"3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
	variables
#non_trainable_variables

$layers
%layer_metrics
trainable_variables
regularization_losses
&layer_regularization_losses
'metrics
X__call__
Y_default_save_signature
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
*:(
ш2embedding_1/embeddings
'

0"
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
(non_trainable_variables

)layers
*layer_metrics
trainable_variables
regularization_losses
+layer_regularization_losses
,metrics
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
ш

 kernel
!recurrent_kernel
"bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
b__call__
*c&call_and_return_all_conditional_losses"­
_tf_keras_layer{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
 "
trackable_list_wrapper
Й
	variables

1states

2layers
3non_trainable_variables
4layer_metrics
trainable_variables
regularization_losses
5layer_regularization_losses
6metrics
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
": 
ш2dense_1/kernel
:ш2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
7non_trainable_variables

8layers
9layer_metrics
trainable_variables
regularization_losses
:layer_regularization_losses
;metrics
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
9:7
2%simple_rnn_1/simple_rnn_cell_1/kernel
C:A
2/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel
2:02#simple_rnn_1/simple_rnn_cell_1/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
-	variables
>non_trainable_variables

?layers
@layer_metrics
.trainable_variables
/regularization_losses
Alayer_regularization_losses
Bmetrics
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Л
	Ctotal
	Dcount
E	variables
F	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
џ
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api"И
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
C0
D1"
trackable_list_wrapper
-
E	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
G0
H1"
trackable_list_wrapper
-
J	variables"
_generic_user_object
/:-
ш2Adam/embedding_1/embeddings/m
':%
ш2Adam/dense_1/kernel/m
 :ш2Adam/dense_1/bias/m
>:<
2,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m
H:F
26Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m
7:52*Adam/simple_rnn_1/simple_rnn_cell_1/bias/m
/:-
ш2Adam/embedding_1/embeddings/v
':%
ш2Adam/dense_1/kernel/v
 :ш2Adam/dense_1/bias/v
>:<
2,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v
H:F
26Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v
7:52*Adam/simple_rnn_1/simple_rnn_cell_1/bias/v
ў2ћ
,__inference_sequential_1_layer_call_fn_14701
,__inference_sequential_1_layer_call_fn_15048
,__inference_sequential_1_layer_call_fn_14737
,__inference_sequential_1_layer_call_fn_15031Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ш2х
 __inference__wrapped_model_13826Р
В
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
annotationsЊ *0Ђ-
+(
embedding_1_inputџџџџџџџџџ
ъ2ч
G__inference_sequential_1_layer_call_and_return_conditional_losses_15014
G__inference_sequential_1_layer_call_and_return_conditional_losses_14664
G__inference_sequential_1_layer_call_and_return_conditional_losses_14889
G__inference_sequential_1_layer_call_and_return_conditional_losses_14645Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
+__inference_embedding_1_layer_call_fn_15065Ђ
В
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
annotationsЊ *
 
№2э
F__inference_embedding_1_layer_call_and_return_conditional_losses_15058Ђ
В
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
annotationsЊ *
 
2
,__inference_simple_rnn_1_layer_call_fn_15300
,__inference_simple_rnn_1_layer_call_fn_15311
,__inference_simple_rnn_1_layer_call_fn_15557
,__inference_simple_rnn_1_layer_call_fn_15546е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
џ2ќ
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15535
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15423
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15289
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15177е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
б2Ю
'__inference_dense_1_layer_call_fn_15577Ђ
В
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
annotationsЊ *
 
ь2щ
B__inference_dense_1_layer_call_and_return_conditional_losses_15568Ђ
В
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
annotationsЊ *
 
дBб
#__inference_signature_wrapper_14764embedding_1_input"
В
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
annotationsЊ *
 
Њ2Ї
1__inference_simple_rnn_cell_1_layer_call_fn_15625
1__inference_simple_rnn_cell_1_layer_call_fn_15639О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
р2н
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_15611
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_15594О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 __inference__wrapped_model_13826x
 "!:Ђ7
0Ђ-
+(
embedding_1_inputџџџџџџџџџ
Њ "2Њ/
-
dense_1"
dense_1џџџџџџџџџшЄ
B__inference_dense_1_layer_call_and_return_conditional_losses_15568^0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџш
 |
'__inference_dense_1_layer_call_fn_15577Q0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџшЊ
F__inference_embedding_1_layer_call_and_return_conditional_losses_15058`
/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "*Ђ'
 
0џџџџџџџџџ
 
+__inference_embedding_1_layer_call_fn_15065S
/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџП
G__inference_sequential_1_layer_call_and_return_conditional_losses_14645t
 "!BЂ?
8Ђ5
+(
embedding_1_inputџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџш
 П
G__inference_sequential_1_layer_call_and_return_conditional_losses_14664t
 "!BЂ?
8Ђ5
+(
embedding_1_inputџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџш
 Д
G__inference_sequential_1_layer_call_and_return_conditional_losses_14889i
 "!7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџш
 Д
G__inference_sequential_1_layer_call_and_return_conditional_losses_15014i
 "!7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџш
 
,__inference_sequential_1_layer_call_fn_14701g
 "!BЂ?
8Ђ5
+(
embedding_1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџш
,__inference_sequential_1_layer_call_fn_14737g
 "!BЂ?
8Ђ5
+(
embedding_1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџш
,__inference_sequential_1_layer_call_fn_15031\
 "!7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџш
,__inference_sequential_1_layer_call_fn_15048\
 "!7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџшЕ
#__inference_signature_wrapper_14764
 "!OЂL
Ђ 
EЊB
@
embedding_1_input+(
embedding_1_inputџџџџџџџџџ"2Њ/
-
dense_1"
dense_1џџџџџџџџџшК
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15177o "!@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ

 
p

 
Њ "&Ђ#

0џџџџџџџџџ
 К
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15289o "!@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ

 
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 Ъ
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15423 "!PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "&Ђ#

0џџџџџџџџџ
 Ъ
G__inference_simple_rnn_1_layer_call_and_return_conditional_losses_15535 "!PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 
,__inference_simple_rnn_1_layer_call_fn_15300b "!@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ
,__inference_simple_rnn_1_layer_call_fn_15311b "!@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџЂ
,__inference_simple_rnn_1_layer_call_fn_15546r "!PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџЂ
,__inference_simple_rnn_1_layer_call_fn_15557r "!PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_15594Л "!^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p
Њ "TЂQ
JЂG

0/0џџџџџџџџџ
%"
 
0/1/0џџџџџџџџџ
 
L__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_15611Л "!^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p 
Њ "TЂQ
JЂG

0/0џџџџџџџџџ
%"
 
0/1/0џџџџџџџџџ
 у
1__inference_simple_rnn_cell_1_layer_call_fn_15625­ "!^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p
Њ "FЂC

0џџџџџџџџџ
# 

1/0џџџџџџџџџу
1__inference_simple_rnn_cell_1_layer_call_fn_15639­ "!^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p 
Њ "FЂC

0џџџџџџџџџ
# 

1/0џџџџџџџџџ