"""TruckScenes loader for drivestudio. The processed folders have exactly the nuScenes layout
(datasets/truckscenes/truckscenes_preprocess.py), so this reuses the nuScenes loaders and only adds
the class names nuScenes does not have."""
from datasets.base.scene_dataset import ModelType
from datasets.nuscenes import nuscenes_sourceloader as nus
from datasets.nuscenes.nuscenes_sourceloader import NuScenesPixelSource, NuScenesLiDARSource

# TruckScenes-only classes (vehicle.ego_trailer, our own trailer, is never exported as an object)
nus.OBJECT_CLASS_NODE_MAPPING.update({
    "vehicle.other": ModelType.RigidNodes,
    "vehicle.train": ModelType.RigidNodes,
})


class TruckScenesPixelSource(NuScenesPixelSource):
    pass


class TruckScenesLiDARSource(NuScenesLiDARSource):
    pass
