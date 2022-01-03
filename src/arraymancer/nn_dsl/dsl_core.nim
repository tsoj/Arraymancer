# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros, tables,
  ./dsl_types, ./dsl_initialization, ./dsl_utils, ./dsl_topology, ./dsl_forwardsugar,
  ../autograd

proc splitSections(config: NimNode): NetworkSections =
  template unknown =
    error:
      lineInfo(section) &
        ": unknown neural network configuration section \"" &
        $section[0] & "\""

  for section in config:
    if section.kind == nnkCall:
      if eqIdent(section[0], "layers"):
        result.layers = section[1]
      else:
        unknown()
    elif section.kind == nnkCommand:
      if eqIdent(section[0], "forward"):
        # For forward we copy everything.
        # We have to deal with forward with multiple inputs like "forward x, y, z:"
        # and we will do that later.
        result.forward = section
      else:
        unknown()
    else:
        unknown()

proc genModelType(self: Neuromancer, model_name: string) =

  var records = nnkRecList.newTree

  for record in self.trainparams:
    let (field_name, field_type, _) = record
    records.add nnkIdentDefs.newTree(
      newIdentNode($field_name),
      field_type,
      newEmptyNode()
    )

  self.type_section = nnkStmtList.newTree(
    nnkTypeSection.newTree(
      nnkTypeDef.newTree(
        newIdentNode(model_name),
        newEmptyNode(), # Generic params get here
        nnkObjectTy.newTree(
          newEmptyNode(),
          newEmptyNode(),
          records
        )
      )
    )
  )

proc genInitProc(self: Neuromancer, model_name: string) =

  self.init_proc = newStmtList()

  let
    subtype = self.subtype
    modelType = newIdentNode(model_name)
    procBody = newStmtList()

  for record in self.trainparams:
    let (_, _, initStmt) = record
    procBody.add initStmt

  self.init_proc.add quote do:
    proc init(ctx: Context[`subtype`], model_type: typedesc[`modelType`]): `modelType` =
      `procBody`

proc genForwardProc(self: Neuromancer, model_name: string, forward: NimNode) =

  forward.expectKind(nnkCommand)
  assert eqIdent(forward[0], "forward")
  forward[^1].expectKind(nnkStmtList)
  # forward x:
  #   x.cv1.relu.mp1.flatten.classifier
  # -----------------------------------
  # Command
  #   Ident "forward"
  #   Ident "x"
  #   StmtList
  #     DotExpr
  #       DotExpr
  #         DotExpr
  #           DotExpr
  #             DotExpr
  #               Ident "x"
  #               Ident "cv1"
  #             Ident "relu"
  #           Ident "mp1"
  #         Ident "flatten"
  #       Ident "classifier"

  # 0. Prepare type information and the raw proc body
  let
    ModelType = newIdentNode(model_name)
    InOutType = nnkBracketExpr.newTree(
      newIdentNode("Variable"), self.subtype
    )
    procBody = forward[^1]

  # 1. Create the input variables with their type
  var inputVars = nnkIdentDefs.newTree()

  for varIndex in 1..forward.len-2:
    inputVars.add newIdentNode($forward[varIndex])

  inputVars.add InOutType
  inputVars.add newEmptyNode() # Default Value

  # 2. Add the shortut syntax templates
  var shortcutTemplates = newStmtList()

  for shortcut in self.forward_templates:
    shortcutTemplates.add shortcut

  # 3. Create the forward proc
  self.forward_proc = nnkProcDef.newTree(
    newIdentNode("forward"), newEmptyNode(), newEmptyNode(),
    nnkFormalParams.newTree(
      # Result type
      InOutType,
      # Model
      nnkIdentDefs.newTree(newIdentNode("self"), ModelType, newEmptyNode()),
      # Variables
      inputVars
    ),
    newEmptyNode(), newEmptyNode(),
    nnkStmtlist.newTree(
      # TODO asserts
      shortcutTemplates,
      procBody,
    )
  )


proc splitSections2(config: NimNode): tuple[layers, forward: NimNode] =
  template unknown =
    error:
      lineInfo(section) &
        ": unknown neural network configuration section \"" &
        $section[0] & "\""

  for section in config:
    if section.kind == nnkCall:
      if eqIdent(section[0], "layers"):
        result.layers = section[1]
      else:
        unknown()
    elif section.kind == nnkCommand:
      if eqIdent(section[0], "forward"):
        # For forward we copy everything.
        # We have to deal with forward with multiple inputs like "forward x, y, z:"
        # and we will do that later.
        result.forward = section
      else:
        unknown()
    else:
        unknown()

type
  LayerInfo = object
    name: NimNode
    typeName: NimNode
    arguments: seq[NimNode]

func createLayerInfo(layers: NimNode): seq[LayerInfo] =
  discard
  # debugEcho ">KKSKADOSOIADHSID"
  # debugEcho treeRepr layers
  # debugEcho ">-----------------<"
  for layer in layers:

    doAssert layer.kind == nnkCall
    doAssert layer.len == 2
    doAssert layer[0].kind == nnkIdent
    doAssert layer[1].kind == nnkStmtList
    doAssert layer[1].len == 1
    doAssert layer[1][0].kind == nnkCall
    doAssert layer[1][0].len >= 1
    doAssert layer[1][0][0].kind == nnkIdent
    result.add LayerInfo(
      name: layer[0],
      typeName: layer[1][0][0]
    )
    if layer[1][0].len >= 2:
      result[^1].arguments = layer[1][0][1..^1]
  #   debugEcho treeRepr result[^1].name
  #   debugEcho treeRepr result[^1].typeName
  #   for arg in result[^1].arguments:
  #     debugEcho treeRepr arg
  # debugEcho "<KKSKADOSOIADHSID"

func createModelType(layerInfos: seq[LayerInfo], modelName: NimNode): NimNode =
  var recList = newNimNode(nnkRecList)
  for layerInfo in layerInfos:
    doAssert layerInfo.name.kind == nnkIdent
    doAssert layerInfo.typeName.kind == nnkIdent
    recList.add newIdentDefs(layerInfo.name, layerInfo.typeName)
  
  doAssert modelName.kind == nnkIdent
  result = newNimNode(nnkStmtList).add(
    newNimNode(nnkTypeSection).add(
      newNimNode(nnkTypeDef).add(
        modelName,
        newEmptyNode(),
        newNimNode(nnkObjectTy).add(
          newEmptyNode(),
          newEmptyNode(),
          recList
        )
      )
    )
  )
  
func createInitProc(layerInfos: seq[LayerInfo], modelName, ctxSubtype: NimNode): NimNode =
  doAssert modelName.kind == nnkIdent
  doAssert ctxSubtype.kind == nnkBracketExpr

  var body = newNimNode(nnkStmtList)
  for layerInfo in layerInfos:
    body.add(
      newNimNode(nnkTemplateDef).add(
        layerInfo.name,
        newEmptyNode(),
        newEmptyNode(),
        newNimNode(nnkFormalParams).add ident"auto",
        newEmptyNode(),
        newEmptyNode(),
        newStmtList(
          newDotExpr(
            ident"result",
            layerInfo.name
          )
        )
      )
    )
  for layerInfo in layerInfos:
    body.add(
      newAssignment(
        layerInfo.name,
        newCall(
          ident"init",
          ident"ctx",
          layerInfo.typeName
        ).add(layerInfo.arguments)
      )
    )
    
  result = newProc(
    name = ident"init",
    params = @[
      modelName,
      newIdentDefs(
        ident"ctx",
        newNimNode(nnkBracketExpr).add(
          ident"Context",
          ctxSubtype
        )
      ),
      newIdentDefs(
        ident"model_type",
        newNimNode(nnkBracketExpr).add(
          ident"typedesc",
          modelName
        )
      )
    ],
    body = body
  )


macro network*(ctx: Context, model_name: untyped, config: untyped): untyped =
  ## Declare a neural network.
  ##
  ## Example usage:
  ##    .. code:: nim
  ##         network ctx, DemoNet:
  ##           layers:
  ##             x:          Input([1, 28, 28])
  ##             cv1:        Conv2D(x.out_shape, 20, 5, 5)
  ##             mp1:        MaxPool2D(cv1.out_shape, (2,2), (0,0), (2,2))
  ##             cv2:        Conv2D(mp1.out_shape, 50, 5, 5)
  ##             mp2:        MaxPool2D(cv2.out_shape, (2,2), (0,0), (2,2))
  ##             fl:         Flatten(mp2.out_shape)
  ##             hidden:     Linear(fl.out_shape, 500)
  ##             classifier: Linear(500, 10)
  ##           forward x:
  ##             x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.classifier

  # example should be expanded to:
  # type
  #   DemoNet = object
  #     x: Input
  #     cv1: Conv2DLayer
  #     mp1: MaxPool2D
  #     cv2: Conv2DLayer
  #     mp2: MaxPool2D
  #     fl: Flatten
  #     hidden: LinearLayer
  #     classifier: LinearLayer
  
  # proc init(ctx: Context[Tensor[float32]], model_type: typedesc[DemoNet]): DemoNet =

  #   result.x = ctx.init(Input2, [1, 28, 28])
  #   result.cv1 = ctx.init(Conv2DLayer2, self.x.out_shape, 20, 5, 5)
  #   result.mp1 = ctx.init(MaxPool2DLayer2, self.cv1.out_shape, (2,2), (0,0), (2,2))
  #   result.cv2 = ctx.init(Conv2DLayer2, self.mp1.out_shape, 50, 5, 5)
  #   result.mp2 = ctx.init(MaxPool2DLayer2, self.cv2.out_shape, (2,2), (0,0), (2,2))
  #   result.fl = ctx.init(Flatten2, self.mp2.out_shape)
  #   result.hidden = ctx.init(LinearLayer2, self.fl.out_shape, 500)
  #   result.classifier = ctx.init(LinearLayer2, 500, 10)
    
  # # TODO: make Tensor[float32] as TT parameter configurable
  # proc forward(self: DemoNet, x: Variable[Tensor[float32]]): Variable[Tensor[float32]] =

  # TODO: add in_shape, out_shape functions

  # TODO better doc

  # 0. - Separate the configuration into layers and forward part
  #    - get the subtype of the model (Tensor[float32], CudaTensor[float64], ...)
  let sections = config.splitSections2()


  # 1. create layer info
  let layerInfos = sections.layers.createLayerInfo()

  # 2. create model type
  let modelType = createModelType(layerInfos, model_name)

  # 3. create init proc
  let ctxSubtype = getAST(ctxSubtype(ctx))
  let initProc = createInitProc(layerInfos, model_name, ctxSubtype)

  #[-----------------------------------------------------]#

  # 1. Initialize the VM to analyse the neural network Graph.
  #    - Get the input shapes
  #    - Get the layers
  let vm = new Neuromancer
  vm.context = ctx
  vm.subtype = ctxSubtype
  debugEcho "!!!!!!"
  debugEcho treeRepr ctxSubtype
  vm.topoTable = initTable[NimNode, LayerTopology]()#TODO: maybe don't need this anymore
  vm.topoTable.topoFromLayers(sections.layers)

  # 2. Generate the model fields, initialization and template synctactic sugar
  vm.genModelFieldInit()
  vm.genTemplateShortcuts()

  # 3. Generate the type section
  vm.genModelType($model_name)
  vm.genInitProc($model_name)
  vm.genForwardProc($model_name, sections.forward)

  # 4. Output the result: type + init proc + forward proc
  result = newStmtList()
  result.add vm.type_section
  result.add vm.init_proc
  result.add vm.forward_proc

  echo toStrLit(result)

proc forward() =
  template hidden(x: Variable): Variable =
    x.linear(self.hidden.weight, self.hidden.bias)

  template fl(x: Variable): Variable =
    x.flatten

  template classifier(x: Variable): Variable =
    x.linear(self.classifier.weight, self.classifier.bias)

  template cv1(x: Variable): Variable =
    x.conv2d(self.cv1.weight, self.cv1.bias)

  template mp1(x: Variable): Variable =
    x.maxpool2D((2, 2), (0, 0), (2, 2))

  template mp2(x: Variable): Variable =
    x.maxpool2D((2, 2), (0, 0), (2, 2))

  template cv2(x: Variable): Variable =
    x.conv2d(self.cv2.weight, self.cv2.bias)