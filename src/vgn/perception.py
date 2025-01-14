from math import cos, sin
import time

import numpy as np
import open3d as o3d

from vgn.utils.transform import Transform


class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution, color_type=None):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        if color_type is None:
            color = o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        elif color_type == "rgb":
            color = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        else:
            raise ValueError("Unknown color type: {}".format(color_type))
        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=color,
        )

    def integrate(self, depth_img, intrinsic, extrinsic, rgb_img=None):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        if rgb_img is None:
            rgb_img = np.empty_like(depth_img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_img),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        extrinsic = extrinsic.as_matrix()

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        # TODO(mbreyer) very slow (~35 ms / 50 ms of the whole pipeline)
        shape = (1, self.resolution, self.resolution, self.resolution)
        tsdf_grid = np.zeros(shape, dtype=np.float32)
        voxels = self._volume.extract_voxel_grid().get_voxels()
        for voxel in voxels:
            i, j, k = voxel.grid_index
            tsdf_grid[0, i, j, k] = voxel.color[0]
        return tsdf_grid
    
    def get_mesh(self):
        return self._volume.extract_triangle_mesh()

    def get_cloud(self):
        return self._volume.extract_point_cloud()


class ScalableTSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution, color_type=None, voxel_size=None):
        self.size = size # for cropped voxel grid 
        self.resolution = resolution # for cropped voxel grid 
        self.voxel_size = voxel_size
        if voxel_size is None:
            self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        if color_type is None:
            color = o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        elif color_type == "rgb":
            color = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        else:
            raise ValueError("Unknown color type: {}".format(color_type))

        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            # resolution=self.resolution, # this is not supported by ScalableTSDFVolume
            sdf_trunc=self.sdf_trunc,
            color_type=color,
        )

        self.cropped_cloud:o3d.geometry.PointCloud = None
        self.cropped_mesh:o3d.geometry.TriangleMesh = None
        self.cropped_grid:o3d.geometry.VoxelGrid = None


    def integrate(self, depth_img, intrinsic, extrinsic, rgb_img=None, mask_img=None):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        if rgb_img is None:
            rgb_img = np.zeros([depth_img.shape[0], depth_img.shape[1], 3], dtype=np.uint8)
        
        # mask out the points out of the region of interest
        if mask_img is not None:
            depth_img[mask_img == 0] = np.inf
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_img),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        extrinsic = extrinsic.as_matrix()

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def set_region_of_interest(self, center, size):
        """Set the region of interest for the TSDF volume to generate voxel grid as network input.

        Args:
            center: The center of the region of interest.
            size: The size of the region of interest.
        """
        self.crop_center = center
        self.crop_size = size

        # create data for cropped region
        self.cropped_cloud = self.crop_cloud(self.crop_center, self.crop_size)
        self.cropped_mesh = self.crop_mesh(self.crop_center, self.crop_size)
        self.cropped_grid = self.crop_grid(self.crop_center, self.crop_size)

    def get_grid(self):
        assert self.cropped_grid is not None, "Please set the region of interest first."
        shape = (1, self.resolution, self.resolution, self.resolution)
        tsdf_grid = np.zeros(shape, dtype=np.float32)
        voxels = self.cropped_grid.get_voxels()
        for voxel in voxels:
            i, j, k = voxel.grid_index
            tsdf_grid[0, i, j, k] = voxel.color[0]

        return tsdf_grid

    def crop_grid(self, crop_center, crop_size):
        # crop voxel grid 
        voxel_cloud = self._volume.extract_voxel_point_cloud()
        # crop point cloud 
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=crop_center - crop_size / 2,
            max_bound=crop_center + crop_size / 2,
        )
        cropped_voxel_cloud = voxel_cloud.crop(bounding_box)
        cropped_voxel_size = self.size / self.resolution
        min_bound = crop_center - crop_size / 2
        max_bound = crop_center + crop_size / 2
        cropped_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            cropped_voxel_cloud, voxel_size=cropped_voxel_size, min_bound=min_bound, max_bound=max_bound)
        return cropped_grid
    
    def get_mesh(self):
        if self.cropped_mesh:
            # return cropped mesh if region of interest is set
            return self.cropped_mesh
        else:
            # return full mesh if region of interest is not set
            mesh = self._volume.extract_triangle_mesh()
            return mesh
    
    def crop_mesh(self, crop_center, crop_size):
        mesh = self._volume.extract_triangle_mesh()
        # crop mesh
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=crop_center - crop_size / 2,
            max_bound=crop_center + crop_size / 2,
        )
        cropped_mesh = mesh.crop(bounding_box)
        return cropped_mesh

    def get_cloud(self):
        if self.cropped_cloud:
            # return cropped cloud if region of interest is set
            return self.cropped_cloud
        else:
            # return full cloud if region of interest is not set
            point_cloud = self._volume.extract_point_cloud()
            return point_cloud

    def crop_cloud(self, crop_center, crop_size):
        point_cloud = self._volume.extract_point_cloud()
        # crop point cloud 
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=crop_center - crop_size / 2,
            max_bound=crop_center + crop_size / 2,
        )
        cropped_point_cloud = point_cloud.crop(bounding_box)
        return cropped_point_cloud


def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics):
    tsdf = TSDFVolume(size, resolution)
    for i in range(depth_imgs.shape[0]):
        extrinsic = Transform.from_list(extrinsics[i])
        tsdf.integrate(depth_imgs[i], intrinsic, extrinsic)
    return tsdf


def camera_on_sphere(origin, radius, theta, phi):
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()
