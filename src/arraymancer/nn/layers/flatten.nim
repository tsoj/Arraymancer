import
  ../../tensor,
  ../../autograd

# ############################################################
#
#                  Flatten layer type for nn_dsl TODO: what is nn_dsl?
#
# ############################################################

type # TODO: ask if it makes sense to generalize Tensor[T] to AnyTensor[T], or TT
  Flatten2*[T] = object
    in_shape: seq[int]

proc init*[T](
  ctx: Context[Tensor[T]],
  layer_type: typedesc[Flatten2[T]],
  in_shape: seq[int]
): Flatten2[T] =
  result.in_shape = in_shape

proc forward*[T](self: Flatten2[T], input: Variable[Tensor[T]]): Variable[Tensor[T]] =
  input.flatten()


proc out_shape*[T](self: Flatten2[T]): seq[int] =    
  result = @[1]
  for i in self.in_shape:
      result[0] *= i

proc in_shape*[T](self: Flatten2[T]): seq[int] =
  self.in_shape