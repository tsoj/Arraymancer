import
  ../../tensor,
  ../../autograd

# ############################################################
#
#                  Flatten layer type for nn_dsl TODO: what is nn_dsl?
#
# ############################################################

type # TODO: ask if it makes sense to generalize Tensor[T] to AnyTensor[T], or TT
  Flatten2 = object
    in_shape: Metadata

proc init*[T](
  ctx: Context[Tensor[T]],
  layer_type: typedesc[Flatten2],
  in_shape: Metadata
): Flatten2 =
  result.in_shape = in_shape

proc forward*[T](self: Flatten2, input: Variable[Tensor[T]]): Variable[Tensor[T]] =
  input.flatten()


proc out_shape*(self: Flatten2): Metadata =    
  result = [1].toMetadata
  for i in self.in_shape:
      result[0] *= i

proc in_shape*(self: Flatten2): MetaData =
  self.in_shape