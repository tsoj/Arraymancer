import ../src/arraymancer, random
import macros
import ../src/arraymancer/nn/layers/conv2D
import ../src/arraymancer/nn/layers/maxpool2D
import ../src/arraymancer/nn/layers/flatten
import ../src/arraymancer/nn/layers/linear

# This is an early minimum viable example of handwritten digits recognition.
# It uses convolutional neural networks to achieve high accuracy.
#
# Data files (MNIST) can be downloaded here http://yann.lecun.com/exdb/mnist/
# and must be decompressed in "./build/" (or change the path "build/..." below)
#

# Make the results reproducible by initializing a random seed
randomize(42)

let
  ctx = newContext Tensor[float32] # Autograd/neural network graph
  n = 32                           # Batch size

let
  mnist = load_mnist(cache = true)
  # Training data is 60k 28x28 greyscale images from 0-255,
  # neural net prefers input rescaled to [0, 1] or [-1, 1]
  x_train = mnist.train_images.astype(float32) / 255'f32

  # Change shape from [N, H, W] to [N, C, H, W], with C = 1 (unsqueeze). Convolution expect 4d tensors
  # And store in the context to track operations applied and build a NN graph
  X_train = ctx.variable x_train.unsqueeze(1)

  # Labels are uint8, we must convert them to int
  y_train = mnist.train_labels.astype(int)

  # Idem for testing data (10000 images)
  x_test = mnist.test_images.astype(float32) / 255'f32
  X_test = ctx.variable x_test.unsqueeze(1)
  y_test = mnist.test_labels.astype(int)

# Configuration of the neural network
network ctx, DemoNet:
  layers:
    cv1:        Conv2DLayer2(@[1, 28, 28], 20, (5, 5))
    mp1:        MaxPool2DLayer2(cv1.out_shape, (2,2), (0,0), (2,2))
    cv2:        Conv2DLayer2(mp1.out_shape, 50, (5, 5))
    mp2:        MaxPool2DLayer2(cv2.out_shape, (2,2), (0,0), (2,2))
    fl:         Flatten2(mp2.out_shape)
    hidden:     LinearLayer2(fl.out_shape[0], 500)
    hidden2:    LinearLayer2(hidden.out_shape[0], 200)
    classifier: LinearLayer2(hidden2.out_shape[0], 10)
  forward x:
    x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.hidden2.relu.classifier

#dumptree:
  # network ctx, DemoNet:
  #   layers:
  #     x:          Input([1, 28, 28])
  #     cv1:        Conv2D(x.out_shape, out_channels = 20, 5, 5)
  #     mp1:        MaxPool2D(cv1.out_shape, (2,2), (0,0), (2,2))
  #     cv2:        Conv2D(mp1.out_shape, 50, 5, 5)
  #     mp2:        MaxPool2D(cv2.out_shape, (2,2), (0,0), (2,2))
  #     fl:         Flatten(mp2.out_shape)
  #     hidden:     Linear(fl.out_shape, 500)
  #     hidden2:    Linear(hidden.out_shape, 200)
  #     classifier: Linear(hidden2.out_shape, 10)
  #     classifier2: Linear()
  #   forward x:
  #     x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.hidden2.relu.classifier

  # type DemoNet = object
  #   x: Input
  #   cv1: Conv2DLayer
  #   mp1: MaxPool2D
  #   cv2: Conv2DLayer
  #   mp2: MaxPool2D
  #   fl: Flatten
  #   hidden: LinearLayer
  #   classifier: LinearLayer
  # proc init(ctx: Context[Tensor[float32]], model_type: typedesc[DemoNet]): DemoNet =
  #   template x(): auto = result.x
  #   template cv1(): auto = result.cv1
  #   template mp1 = result.mp1
  #   template cv2 = result.cv2
  #   template mp2 = result.mp2
  #   template fl = result.fl
  #   template hidden = result.hidden
  #   template classifier = result.classifier

  #   x = init(ctx, Input2, [1, 28, 28])
  #   cv1 = init(ctx, Conv2DLayer2, x.out_shape, 20, 5, 5)
  #   mp1 = ctx.init(MaxPool2DLayer2, cv1.out_shape, (2,2), (0,0), (2,2))
  #   cv2 = ctx.init(Conv2DLayer2, mp1.out_shape, 50, 5, 5)
  #   mp2 = ctx.init(MaxPool2DLayer2, cv2.out_shape, (2,2), (0,0), (2,2))
  #   fl = ctx.init(Flatten2, mp2.out_shape)
  #   hidden = ctx.init(LinearLayer2, fl.out_shape, 500)
  #   classifier = ctx.init(LinearLayer2, 500, 10)
  # proc forward(self: DemoNet; x: Variable[Tensor[float32]]): Variable[Tensor[float32]] =

  #   template hidden(x: Variable): Variable =
  #     forward(self.hidden, x)

  #   template fl(x: Variable): Variable =
  #     forward(self.fl, x)

  #   x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.hidden2.relu.classifier
  # forward x:
  #   x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.hidden2.relu.classifier

let model = ctx.init(DemoNet)

# Stochastic Gradient Descent (API will change)
let optim = model.optimizerSGD(learning_rate = 0.01'f32)

# Learning loop
for epoch in 0 ..< 5:
  for batch_id in 0 ..< X_train.value.shape[0] div n: # some at the end may be missing, oh well ...
    # minibatch offset in the Tensor
    let offset = batch_id * n
    let x = X_train[offset ..< offset + n, _]
    let target = y_train[offset ..< offset + n]

    # Running through the network and computing loss
    let clf = model.forward(x)
    let loss = clf.sparse_softmax_cross_entropy(target)

    if batch_id mod 200 == 0:
      # Print status every 200 batches
      echo "Epoch is: " & $epoch
      echo "Batch id: " & $batch_id
      echo "Loss is:  " & $loss.value[0]

    # Compute the gradient (i.e. contribution of each parameter to the loss)
    loss.backprop()

    # Correct the weights now that we have the gradient information
    optim.update()

  # Validation (checking the accuracy/generalization of our model on unseen data)
  ctx.no_grad_mode:
    echo "\nEpoch #" & $epoch & " done. Testing accuracy"

    # To avoid using too much memory we will compute accuracy in 10 batches of 1000 images
    # instead of loading 10 000 images at once
    var score = 0.0
    var loss = 0.0
    for i in 0 ..< 10:
      let y_pred = model.forward(X_test[i*1000 ..< (i+1)*1000, _]).value.softmax.argmax(axis = 1).squeeze
      score += y_pred.accuracy_score(y_test[i*1000 ..< (i+1)*1000])

      loss += model.forward(X_test[i*1000 ..< (i+1)*1000, _]).sparse_softmax_cross_entropy(y_test[i*1000 ..< (i+1)*1000]).value.unsafe_raw_offset[0]
    score /= 10
    loss /= 10
    echo "Accuracy: " & $(score * 100) & "%"
    echo "Loss:     " & $loss
    echo "\n"


############# Output ############

# Epoch is: 0
# Batch id: 0
# Loss is:  2.83383584022522
# Epoch is: 0
# Batch id: 200
# Loss is:  0.2911527752876282
# Epoch is: 0
# Batch id: 400
# Loss is:  0.1666509807109833
# Epoch is: 0
# Batch id: 600
# Loss is:  0.2486120313405991
# Epoch is: 0
# Batch id: 800
# Loss is:  0.165436714887619
# Epoch is: 0
# Batch id: 1000
# Loss is:  0.210975781083107
# Epoch is: 0
# Batch id: 1200
# Loss is:  0.1667802333831787
# Epoch is: 0
# Batch id: 1400
# Loss is:  0.08688683807849884
# Epoch is: 0
# Batch id: 1600
# Loss is:  0.07058585435152054
# Epoch is: 0
# Batch id: 1800
# Loss is:  0.2075864225625992

# Epoch #0 done. Testing accuracy
# Accuracy: 96.70999999999998%
# Loss:     0.1007537815719843


# Epoch is: 1
# Batch id: 0
# Loss is:  0.04121939837932587
# Epoch is: 1
# Batch id: 200
# Loss is:  0.02066932618618011
# Epoch is: 1
# Batch id: 400
# Loss is:  0.08200274407863617
# Epoch is: 1
# Batch id: 600
# Loss is:  0.05399921536445618
# Epoch is: 1
# Batch id: 800
# Loss is:  0.06251053512096405
# Epoch is: 1
# Batch id: 1000
# Loss is:  0.1627875566482544
# Epoch is: 1
# Batch id: 1200
# Loss is:  0.1231627687811852
# Epoch is: 1
# Batch id: 1400
# Loss is:  0.04727928340435028
# Epoch is: 1
# Batch id: 1600
# Loss is:  0.04230280220508575
# Epoch is: 1
# Batch id: 1800
# Loss is:  0.1406233310699463

# Epoch #1 done. Testing accuracy
# Accuracy: 97.76000000000001%
# Loss:     0.06900692787021398


# Epoch is: 2
# Batch id: 0
# Loss is:  0.01567600667476654
# Epoch is: 2
# Batch id: 200
# Loss is:  0.01009216904640198
# Epoch is: 2
# Batch id: 400
# Loss is:  0.07341829687356949
# Epoch is: 2
# Batch id: 600
# Loss is:  0.02835254371166229
# Epoch is: 2
# Batch id: 800
# Loss is:  0.03980693221092224
# Epoch is: 2
# Batch id: 1000
# Loss is:  0.1551663875579834
# Epoch is: 2
# Batch id: 1200
# Loss is:  0.1222885027527809
# Epoch is: 2
# Batch id: 1400
# Loss is:  0.02436092495918274
# Epoch is: 2
# Batch id: 1600
# Loss is:  0.0285358726978302
# Epoch is: 2
# Batch id: 1800
# Loss is:  0.1016025543212891

# Epoch #2 done. Testing accuracy
# Accuracy: 98.11000000000001%
# Loss:     0.05803158972412348


# Epoch is: 3
# Batch id: 0
# Loss is:  0.01016336679458618
# Epoch is: 3
# Batch id: 200
# Loss is:  0.008497849106788635
# Epoch is: 3
# Batch id: 400
# Loss is:  0.06492133438587189
# Epoch is: 3
# Batch id: 600
# Loss is:  0.01684235036373138
# Epoch is: 3
# Batch id: 800
# Loss is:  0.03516259789466858
# Epoch is: 3
# Batch id: 1000
# Loss is:  0.12979856133461
# Epoch is: 3
# Batch id: 1200
# Loss is:  0.1137113943696022
# Epoch is: 3
# Batch id: 1400
# Loss is:  0.01397424936294556
# Epoch is: 3
# Batch id: 1600
# Loss is:  0.02437949180603027
# Epoch is: 3
# Batch id: 1800
# Loss is:  0.07698698341846466

# Epoch #3 done. Testing accuracy
# Accuracy: 98.22999999999999%
# Loss:     0.05075375782325864


# Epoch is: 4
# Batch id: 0
# Loss is:  0.006512492895126343
# Epoch is: 4
# Batch id: 200
# Loss is:  0.008149772882461548
# Epoch is: 4
# Batch id: 400
# Loss is:  0.05767479538917542
# Epoch is: 4
# Batch id: 600
# Loss is:  0.01081934571266174
# Epoch is: 4
# Batch id: 800
# Loss is:  0.02772468328475952
# Epoch is: 4
# Batch id: 1000
# Loss is:  0.1089183539152145
# Epoch is: 4
# Batch id: 1200
# Loss is:  0.1075025796890259
# Epoch is: 4
# Batch id: 1400
# Loss is:  0.009760886430740356
# Epoch is: 4
# Batch id: 1600
# Loss is:  0.0264655202627182
# Epoch is: 4
# Batch id: 1800
# Loss is:  0.06295807659626007

# Epoch #4 done. Testing accuracy
# Accuracy: 98.33999999999999%
# Loss:     0.04616197692230344