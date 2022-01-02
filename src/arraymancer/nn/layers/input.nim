import
  ../../tensor,
  ../../autograd

# ############################################################
#
#                  Input layer type for nn_dsl TODO: what is nn_dsl?
#
# ############################################################

type # TODO: ask if it makes sense to generalize Tensor[T] to AnyTensor[T], or TT
  Input2 = object
    in_shape: Metadata

proc init*[T](
  ctx: Context[Tensor[T]],
  layer_type: typedesc[Input2],
  in_shape: Metadata
): Input2 =
  result.in_shape = in_shape

proc forward*[T](self: Input2, input: Variable[Tensor[T]]): Variable[Tensor[T]] =
  assert input.shape == self.in_shape
  input


proc out_shape*(self: Input2): Metadata =    
  self.in_shape

proc in_shape*(self: Input2): MetaData =
  self.in_shape