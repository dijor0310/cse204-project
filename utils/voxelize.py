import open3d as o3d
import numpy as np

class Voxelizer(object):

    def __call__(self, pcd_np):
        
        pcd = self.normalize(pcd_np)

        return self.voxelize(pcd)

    def normalize(self, pcd_np):
        assert len(pcd_np.shape)==2
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)

        # fit to unit cube
        pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
          center=pcd.get_center())

        return  pcd
    
    def voxelize(self, pcd):
        # returns 32 x 32 x 32 voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.03125)
        return voxel_grid