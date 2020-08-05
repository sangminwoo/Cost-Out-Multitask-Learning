from yacs.config import CfgNode

_C = CfgNode()

"""======================================="""
_C.DATASET = CfgNode()
_C.DATASET.DATA1 = "cifar10" # 'mnist, cifar-10, cifar-100, imagenet'
_C.DATASET.NUM_CLASSES1 = 10 # 10, 10, 100, 1000
_C.DATASET.DATA2 = "cifar100" # 'mnist, cifar-10, cifar-100, imagenet'
_C.DATASET.NUM_CLASSES2 = 100 # 10, 10, 100, 1000
_C.DATASET.PATH = "data/"
_C.DATASET.TRAIN_BATCH_SIZE = 64
_C.DATASET.TEST_BATCH_SIZE = 64

"""======================================="""
_C.MODEL = CfgNode()
_C.MODEL.BASE_MODEL = "resnet18" # 'mlp, resnet18, 50, 101'
_C.MODEL.CHECKPOINT = "checkpoint_99.pth.tar" # model to resume
_C.MODEL.CONVERGENCE_THRESHOLD = 1e-4

"""======================================="""
_C.SOLVER = CfgNode()
_C.SOLVER.EPOCH = 100
_C.SOLVER.LR = 1e-3
_C.SOLVER.OPTIMIZER = "SGD"
_C.SOLVER.MOMENTUM = 0
_C.SOLVER.DROPOUT = 0
_C.SOLVER.BN_MOMENTUM = 1e-2 # batch norm momentum
_C.SOLVER.WEIGHT_DECAY = 1e-5
_C.SOLVER.CHECKPOINT_PERIOD = 10

"""======================================="""
_C.ETC = CfgNode()
_C.ETC.WORKERS = 16 # data loading workers
_C.ETC.VERBOSE = True # print progress
_C.ETC.SEED = 0 # random seed

"""======================================="""