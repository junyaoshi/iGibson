import os
import gibson2
import pybullet as p
from gibson2.objects.articulated_object import ArticulatedObject


class IGisbonObject(ArticulatedObject):
    """
    RBO object from assets/models/rbo
    Reference: https://tu-rbo.github.io/articulated-objects/
    """

    def __init__(self, name, scale=1):
        dirname = os.path.join(gibson2.ig_dataset_path, 'objects', name)
        object_dir = [f.path for f in os.scandir(dirname) if f.is_dir()][0]
        object_name = os.path.basename(object_dir)
        filename = os.path.join(object_dir, f'{object_name}.urdf')
        super(IGisbonObject, self).__init__(filename, scale)

    def load(self):
        """
        Load the object into pybullet.
        _load() will be implemented in the subclasses
        """
        if self.loaded:
            return self.body_id
        self.body_id = self._load()
        self.loaded = True

        return self.body_id