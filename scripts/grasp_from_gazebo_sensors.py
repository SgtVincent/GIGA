"""
This script is used to predict grasp pose by:
- Generate TSDF from RGB-Depth images of gazebo camera sets.  
- Predict grasp pose from TSDF with GIGA model.
"""

import os 
from typing import List, Tuple, Dict
import argparse
import numpy as np
import open3d as o3d
import genpy
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque    

# ROS related
import rospy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped
import ros_numpy

from vgn.detection_implicit import VGNImplicit
from vgn.experiments.clutter_removal import State
from vgn.perception import TSDFVolume, ScalableTSDFVolume, CameraIntrinsic
from vgn.utils.transform import Transform, Rotation
from vgn.utils.comm import receive_msg, send_msg
from vgn.utils.visual import grasp2mesh, plot_voxel_as_cloud, plot_tsdf_with_grasps

class GazeboRGBDCamera:
    """
    This class is used to buffer gazebo RGBD camera data and provide easy IO interface.
    """
    def __init__(self, camera_name, namespace="", buffer_size=10, sub_pcl=False) -> None:
        self.ns = namespace
        self.camera_name = camera_name
        self.rgb_camera_info:CameraInfo  = None
        self.depth_camera_info:CameraInfo = None
        
        # subscribe to camera topics
        self.rgb_image_sub = rospy.Subscriber(self.ns + '/' + camera_name + "/color/image_raw", Image, self.rgb_image_callback)
        self.depth_image_sub = rospy.Subscriber(self.ns + '/' + camera_name + "/depth/image_raw", Image, self.depth_image_callback)
        self.rgb_camera_info_sub = rospy.Subscriber(self.ns + '/' + camera_name + "/color/camera_info", CameraInfo, self.rgb_camera_info_callback)
        self.depth_camera_info_sub = rospy.Subscriber(self.ns + '/' + camera_name + "/depth/camera_info", CameraInfo, self.depth_camera_info_callback)
        
        # point cloud subscriber 
        self.sub_pcl = None
        if sub_pcl:
            self.pcl_sub = rospy.Subscriber(self.ns + '/' + camera_name + "/depth/color/points", PointCloud2, self.pcl_callback)


        self.bridge = CvBridge()
        # buffer for data synchronization
        self.rgb_buffer = deque(maxlen=buffer_size)
        self.depth_buffer = deque(maxlen=buffer_size)
        self.pcl_buffer = deque(maxlen=buffer_size)

        # get camera info from ros
        self._init_with_camera_info()

    def _init_with_camera_info(self):
        """
        Get camera intrinsic and frame ids from camera info
        """
        while self.rgb_camera_info is None:
            rospy.loginfo(f"{self.camera_name}: Waiting for {self.camera_name}/color/camera_info...")
            rospy.sleep(0.5)
        rospy.loginfo(f"{self.camera_name}: {self.camera_name}/color/camera_info received!")

        # parse rgb camera intrinsics from camera info
        self.rgb_camera_intrinsic = CameraIntrinsic(
            width=self.rgb_camera_info.width,
            height=self.rgb_camera_info.height,
            fx=self.rgb_camera_info.K[0],   
            fy=self.rgb_camera_info.K[4],
            cx=self.rgb_camera_info.K[2],
            cy=self.rgb_camera_info.K[5],
        )
        # parse rgb camera frame id from camera info
        self.rgb_camera_frame_id = self.rgb_camera_info.header.frame_id

        while self.depth_camera_info is None:
            rospy.loginfo(f"{self.camera_name}: Waiting for {self.camera_name}/depth/camera_info...")
            rospy.sleep(0.5)
        rospy.loginfo(f"{self.camera_name}: {self.camera_name}/depth/camera_info received!")

        # parse depth camera intrinsics from camera info
        self.depth_camera_intrinsic = CameraIntrinsic(
            width=self.depth_camera_info.width,
            height=self.depth_camera_info.height,
            fx=self.depth_camera_info.K[0],   
            fy=self.depth_camera_info.K[4],
            cx=self.depth_camera_info.K[2],
            cy=self.depth_camera_info.K[5],
        )

        # parse depth camera frame id from camera info
        self.depth_camera_frame_id = self.depth_camera_info.header.frame_id
        

    def rgb_image_callback(self, data):
        try:
            # save all raw data for synchronization and processing
            self.rgb_buffer.append(data)
        except CvBridgeError as e:
            rospy.logerr(e)
    
    def depth_image_callback(self, data):
        try:
            # save all raw data for synchronization and processing
            self.depth_buffer.append(data)
        except CvBridgeError as e:
            rospy.logerr(e)

    def pcl_callback(self, data):
        try:
            # save all raw data for synchronization and processing
            self.pcl_buffer.append(data)
        except CvBridgeError as e:
            rospy.logerr(e)

    def rgb_camera_info_callback(self, data):
        self.rgb_camera_info = data

    def depth_camera_info_callback(self, data):
        self.depth_camera_info = data 
        
    def get_data_by_timestamp(self, timestamp:genpy.Time)->Tuple[np.ndarray, np.ndarray]:
        """
        Get synchronized rgb and depth image from buffer by its closest timestamp
        """
        rgb_image = self.rgb_buffer[-1]
        depth_image = self.depth_buffer[-1]
        rgb_timestamp = None
        depth_timestamp = None
        for rgb_image in self.rgb_buffer:
            rgb_timestamp = rgb_image.header.stamp
            if rgb_timestamp > timestamp:
                break
        for depth_image in self.depth_buffer:
            depth_timestamp = depth_image.header.stamp
            if depth_timestamp > timestamp:
                break
        # TODO: do we need to check if rgb_timestamp == depth_timestamp?
        try:
            rgb_image_np = self.bridge.imgmsg_to_cv2(rgb_image, "rgb8")
            depth_image_np = self.bridge.imgmsg_to_cv2(depth_image, "16UC1")
            # realsense depth image is uint16 in millimeter
            depth_image_np = depth_image_np.astype(np.float32) / 1000.0
        except CvBridgeError as e:
            rospy.logerr(e)
            return None, None
        
        return rgb_image_np, depth_image_np

    def get_pcl_by_timestamp(self, timestamp:genpy.Time)->np.ndarray:
        """
        Get synchronized point cloud from buffer by its closest timestamp
        """
        pcl:PointCloud2 = self.pcl_buffer[-1]
        pcl_timestamp = None
        for pcl in self.pcl_buffer:
            pcl_timestamp = pcl.header.stamp
            if pcl_timestamp > timestamp:
                break
        
        # return processed point cloud as numpy array
        pcl_numpy = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcl)
        return pcl_numpy
        

    def get_last_timestamp(self)->Tuple[genpy.Time, genpy.Time]:
        """
        Get last synchronized timestamp of rgb and depth image
        """
        if len(self.rgb_buffer) > 0 and len(self.depth_buffer) > 0:
            rgb_image = self.rgb_buffer[-1]
            depth_image = self.depth_buffer[-1]
            return rgb_image.header.stamp, depth_image.header.stamp
        else:
            return None, None
        
    def get_camera_instrinsics_and_frames(self)->Tuple[CameraIntrinsic, str, CameraIntrinsic, str]:
        """
        Get camera instrinsics and frames
        """
        return self.rgb_camera_intrinsic, self.rgb_camera_frame_id, self.depth_camera_intrinsic, self.depth_camera_frame_id

class GazeboRGBDCameraSet:
    """
    This class is used to buffer gazebo RGBD camera set data and provide easy IO interface.
    """
    def __init__(self, cameras_list: List[str], namespace="", buffer_size=10, sub_pcl=False) -> None:
        
        self.ns = namespace
        self.buffer_size = buffer_size
        self.cameras_list = cameras_list
        self.sub_pcl = sub_pcl
        self.cameras: Dict[str, GazeboRGBDCamera] = {}
        for camera_name in cameras_list:
            self.cameras[camera_name] = GazeboRGBDCamera(camera_name, namespace=namespace, buffer_size=buffer_size, sub_pcl=sub_pcl)

        # initialize tf listener 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # waitForTransform function not needed for tf2
        # _, first_rgb_frame, _, _ = self.cameras[self.cameras_list[0]].get_camera_instrinsics_and_frames()
        # self.tf_listener.waitForTransform("world", first_rgb_frame, rospy.Time(), rospy.Duration(4.0))


    def query_extrinsic(self, time:genpy.Time, camera_frame:str, source_frame:str="world")->Transform:
        """
        Query extrinsic transform from camera to target frame
        """
        try:
            transform:TransformStamped = self.tf_buffer.lookup_transform(
                source_frame=source_frame,
                target_frame=camera_frame,
                time=time,
                timeout=rospy.Duration(1.0)
            )
            rotation = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            translation = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            
            return Transform.from_dict({
                'rotation': np.array(rotation),
                'translation': np.array(translation)
            })
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(e)
            return None
        

    def get_latest_data(self):
        """
        Get latest synchronized rgb and depth image from all cameras
        """
        # first get min timestamp within all lastest timestamps from cameras
        timestamp_list = []
        for camera_name in self.cameras_list:
            rgb_timestamp, depth_timestamp = self.cameras[camera_name].get_last_timestamp()
            if rgb_timestamp is None or depth_timestamp is None:
                rospy.logwarn(f"{camera_name}: No data received yet!")
                return None, None
            else:
                timestamp_list.append(rgb_timestamp)
                timestamp_list.append(depth_timestamp)
        timestamp = min(timestamp_list)

        # then get synchronized rgb and depth image from all cameras
        data = {
            'camera_names': [],
            'rgb_image_list': [],
            'rgb_camera_intrinsic_list': [],
            'rgb_camera_frame_list': [],
            'rgb_camera_extrinsic_list': [],
            'depth_image_list': [],
            'depth_camera_intrinsic_list': [],
            'depth_camera_frame_list': [],
            'depth_camera_extrinsic_list': []
        }
        if self.sub_pcl:
            data['points'] = []

        for camera_name in self.cameras_list:
            rgb_image, depth_image = self.cameras[camera_name].get_data_by_timestamp(timestamp)
            if rgb_image is None or depth_image is None:
                rospy.logwarn(f"{camera_name}: No data received yet!")
                return None, None
            else:
                # load data from camera 
                rgb_camera_intrinsic, rgb_camera_frame, depth_camera_intrinsic, depth_camera_frame = \
                    self.cameras[camera_name].get_camera_instrinsics_and_frames()
                data['camera_names'].append(camera_name)
                data['rgb_image_list'].append(rgb_image)
                data['rgb_camera_intrinsic_list'].append(rgb_camera_intrinsic)
                data['rgb_camera_frame_list'].append(rgb_camera_frame)
                data['depth_image_list'].append(depth_image)
                data['depth_camera_intrinsic_list'].append(depth_camera_intrinsic)
                data['depth_camera_frame_list'].append(depth_camera_frame)

                # query extrinsic transform from camera to world
                rgb_camera_extrinsic = self.query_extrinsic(timestamp, rgb_camera_frame)
                depth_camera_extrinsic = self.query_extrinsic(timestamp, depth_camera_frame)
                data['rgb_camera_extrinsic_list'].append(rgb_camera_extrinsic)
                data['depth_camera_extrinsic_list'].append(depth_camera_extrinsic)

                if self.sub_pcl:
                    points = self.cameras[camera_name].get_pcl_by_timestamp(timestamp)
                    data['points'].append(points)

        return data
    

def get_mask_from_3D_bbox(bbox_center:np.ndarray, bbox_size:np.ndarray, depth_image:np.ndarray, 
                          intrinsic:CameraIntrinsic, extrinsic:Transform)->np.ndarray:
    """
    Get 2D mask image of same size as depth image by projecting 3D bounding box onto the depth image 
    """
    # get the 8 corners of the 3D bounding box
    corners = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1]
    ]) * bbox_size / 2 + bbox_center

    # project 3D bounding box onto the depth image
    corners = extrinsic.transform_point(corners)
    K = intrinsic.K 
    corners = K.dot(corners.T).T
    corners = corners[:, :2] / corners[:, 2:]
    corner_pixels = corners.astype(np.int32)
    
    # calculate the 2D bounding box of the 8 corner_pixels
    min_x = np.min(corner_pixels[:, 0])
    max_x = np.max(corner_pixels[:, 0])
    min_y = np.min(corner_pixels[:, 1])
    max_y = np.max(corner_pixels[:, 1])

    # create mask image
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    mask[min_y:max_y, min_x:max_x] = 1
    return mask
     
def predict_grasp(args, planner, data: Dict):
    # DO NOT use color since realsense D435 camera has different color and depth image resolution & optical center
    if args.volume_type == "uniform":
        tsdf = TSDFVolume(args.size, args.resolution)
    elif args.volume_type == "scalable":
        tsdf = ScalableTSDFVolume(args.size, args.resolution)

    for i, camera in enumerate(data['camera_names']):

        if args.color_type == "rgb":
            rgb = data['rgb_image_list'][i]
        else:
            rgb = None
        depth = data['depth_image_list'][i]
        intrinsics: CameraIntrinsic = data['depth_camera_intrinsic_list'][i]
        extrinsics: Transform = data['depth_camera_extrinsic_list'][i]
    
        # intrinsics = CameraIntrinsic.from_dict(intrinsics)
        # extrinsics = Transform.from_matrix(extrinsics)

        # calculate mask image from 3D bounding box 
        mask = None
        if args.use_depth_mask:
            mask = get_mask_from_3D_bbox(np.array(args.object_center), np.array(args.object_size), depth, intrinsics, extrinsics)

        tsdf.integrate(depth, intrinsics, extrinsics, rgb_img=rgb, mask_img=mask)

    if args.volume_type == "scalable":
        tsdf.set_region_of_interest(np.array(args.object_center), np.array(args.object_size))

    pc = tsdf.get_cloud()
    state = State(tsdf, pc)
    grasps, scores, _ = planner(state)
    print(len(grasps))

    # visualize grasps
    # plt.ion()
    if len(grasps) > 0:
        fig = plot_tsdf_with_grasps(tsdf.get_grid()[0], [grasps[0]])
        print(scores)
    else:
        fig = plot_voxel_as_cloud(tsdf.get_grid()[0])
    fig.show()
    # while True:
    #     if plt.waitforbuttonpress():
    #         break
    # plt.close(fig)


    # grasp_meshes = [
    #     grasp2mesh(grasps[idx], 1).as_open3d for idx in range(len(grasps))
    # ]
    # geometries = [pc] + grasp_meshes

    # from copy import deepcopy
    # grasp_bck = deepcopy(grasps[0])
    # grasp_mesh_bck = grasp2mesh(grasp_bck, 1).as_open3d
    # grasp_mesh_bck.paint_uniform_color([0, 0.8, 0])

    # pos = grasps[0].pose.translation
    # # pos[2] += 0.05
    # angle = grasps[0].pose.rotation.as_euler('xyz')
    # print(pos, angle)
    # if angle[2] > np.pi / 2 or angle[2] < - np.pi / 2:
    #     reflect = Transform(Rotation.from_euler('xyz', (0, 0, np.pi)), np.zeros((3)))
    #     grasps[0].pose = grasps[0].pose * reflect
    # pos = grasps[0].pose.translation
    # angle = grasps[0].pose.rotation.as_euler('xyz')
    # print(pos, angle)
    # # grasps[0].pose = Transform(Rotation.from_euler('xyz', (angle[0], angle[1], angle[2])), pos)
    # grasp_mesh = grasp2mesh(grasps[0], 1).as_open3d
    # grasp_mesh.paint_uniform_color([0.8, 0, 0])
    # geometries = [tsdf.get_mesh(), grasp_mesh, grasp_mesh_bck]

    # shift grasp translation by tsdf origin
    tsdf_origin = tsdf.cropped_grid.origin
    for i, grasp in enumerate(grasps):
        grasps[i].pose.translation = grasp.pose.translation + tsdf_origin

    if len(grasps) == 0:
        # return [], [], [tsdf.get_mesh()]
        return [], [], [tsdf.get_cloud()]
    pos = grasps[0].pose.translation
    # pos[2] += 0.05
    angle = grasps[0].pose.rotation.as_euler('xyz')
    print(pos, angle)
    if angle[2] > np.pi / 2 or angle[2] < - np.pi / 2:
        reflect = Transform(Rotation.from_euler('xyz', (0, 0, np.pi)), np.zeros((3)))
        grasps[0].pose = grasps[0].pose * reflect
    pos = grasps[0].pose.translation
    angle = grasps[0].pose.rotation.as_euler('xyz')
    print(pos, angle)
    # grasps[0].pose = Transform(Rotation.from_euler('xyz', (angle[0], angle[1], angle[2])), pos)
    grasp_mesh = grasp2mesh(grasps[0], 1).as_open3d
    grasp_mesh.paint_uniform_color([0, 0.8, 0])
    # geometries = [tsdf.get_mesh(), grasp_mesh]
    geometries = [tsdf.get_cloud(), grasp_mesh]
    #exit(0)
    return grasps, scores, geometries
    
def visualize_point_clouds(args, data: Dict, use_mask=False):
    """
    This function is used to visualize all point clouds from all depth images 
    """
    gt_pcl_list = []
    pcl_list = []
    for i, camera in enumerate(data['camera_names']):
        depth = data['depth_image_list'][i]
        intrinsic: CameraIntrinsic = data['depth_camera_intrinsic_list'][i]
        extrinsic: Transform = data['depth_camera_extrinsic_list'][i]
        gt_points = data['points'][i]
        # reconstruct point cloud from depth image
        
        if use_mask:
            mask = get_mask_from_3D_bbox(np.array(args.object_center), np.array(args.object_size), depth, intrinsic, extrinsic)
            depth = depth * mask
        
        o3d_pcl = o3d.geometry.PointCloud.create_from_depth_image(
            depth = o3d.geometry.Image(depth),
            intrinsic= o3d.camera.PinholeCameraIntrinsic(
                width=intrinsic.width,
                height=intrinsic.height,
                fx=intrinsic.fx,
                fy=intrinsic.fy,
                cx=intrinsic.cx,
                cy=intrinsic.cy,
            ),
            extrinsic=extrinsic.as_matrix(),
            depth_scale=1.0,
            depth_trunc=3.0,
        )
        pcl_list.append(o3d_pcl)
        
        # create GT point cloud from gt_points
        gt_pcl = o3d.geometry.PointCloud()
        gt_pcl.points = o3d.utility.Vector3dVector(gt_points)
        gt_pcl_list.append(gt_pcl)

        # visualize two point clouds
    o3d.visualization.draw_geometries(pcl_list)

def compute_projection_error(point_cloud, depth_image, intrinsic, extrinsic, visualize=False):
    """
    This function is used to compute reprojection error of a point cloud
    """
    # Project 3D points to 2D image coordinates
    homogeneous_coords = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    projected_points = np.dot(extrinsic, homogeneous_coords.T)
    projected_points = projected_points[:3, :] / projected_points[3, :]
    projected_points = np.dot(intrinsic, projected_points)
    projected_points = projected_points[:2, :]

    # Ensure the projected points are within the image bounds
    height, width = depth_image.shape
    projected_points[0, :] = np.clip(projected_points[0, :], 0, width - 1)
    projected_points[1, :] = np.clip(projected_points[1, :], 0, height - 1)

    # Compute the projected depth image
    projected_depth = depth_image[projected_points[1, :].astype(np.int), projected_points[0, :].astype(np.int)]

    if visualize:
        # visualize depth images on the fly for debugging
        plt.subplot(1, 2, 1)
        plt.imshow(depth_image)
        plt.subplot(1, 2, 2)
        plt.imshow(projected_depth.reshape(depth_image.shape))
        plt.show()

    error = projected_depth - point_cloud[:, 2]
    # Compute the mean and root mean square error
    mean_error = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))

    print("Mean Projection Error:", mean_error)
    print("Root Mean Square Error:", rmse)

    return mean_error


def main(args):

    # initialize ros node
    rospy.init_node('grasp_from_gazebo_sensors', anonymous=False)

    # initialize grasp perdiction model
    planner = VGNImplicit(args.model,
                        args.type,
                        best=True,
                        qual_th=0.8,
                        rviz=False,
                        force_detection=True,
                        out_th=0.1,
                        resolution=args.resolution)

    # initialize gazebo camera set
    # camera_set = GazeboRGBDCameraSet(cameras_list=args.camera_names, namespace=args.namespace)
    camera_set = GazeboRGBDCameraSet(cameras_list=args.camera_names, namespace=args.namespace, sub_pcl=True)

    while True:
        data = camera_set.get_latest_data()

        # visualize_point_clouds(args, data, use_mask=True)

        # compute reprojection error
        # for i, camera in enumerate(data['camera_names']):
        #     depth = data['depth_image_list'][i]
        #     intrinsic: CameraIntrinsic = data['depth_camera_intrinsic_list'][i]
        #     extrinsic: Transform = data['depth_camera_extrinsic_list'][i]
        #     gt_points = data['points'][i]
        #     error = compute_projection_error(gt_points, depth, intrinsic.K, extrinsic.as_matrix(), visualize=True)

        grasps, scores, geometries = predict_grasp(args, planner, data)

        #o3d.visualization.draw_geometries(geometries, zoom=1.0, front=[0, -1, 0], up=[0, 0, 1], lookat=[0.15, 0.15, 0.05], mesh_show_back_face=True)
        # o3d.visualization.draw_geometries(geometries)
        o3d.visualization.draw_geometries(geometries)
        # output_path = input_path.replace('.pkl', 'grasps.pkl')
        # with open(output_path, 'wb') as f:
        #     pickle.dump({'grasps': grasps, 'scores': scores}, f)
        # send_msg(['output', output_path])
        pass


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=Path, default="data/models/pile_convonet_v_cat_occ.pt")
    parser.add_argument("--model", type=Path, default="data/models/giga_pile.pt")
    parser.add_argument("--type", type=str, default="giga")
    parser.add_argument("--color_type", type=str, default="depth", choices=["rgb", "depth"])
    parser.add_argument("--volume_type", type=str, default="scalable", choices=["uniform", "scalable"]) # do not use scalable for now
    parser.add_argument("--use_depth_mask", action="store_true")
    parser.add_argument("--size", type=float, default=0.3)
    parser.add_argument("--resolution", type=float, default=40)
    parser.add_argument("--lower", type=float, nargs=3, default=[0.02, 0.02, 0.005])
    parser.add_argument("--upper", type=float, nargs=3, default=[0.28, 0.28, 0.3])
    parser.add_argument("--object_center", type=float, nargs=3, default=[0.244, -0.383, 1.014])
    parser.add_argument("--object_size", type=float, nargs=3, default=[0.3, 0.3, 0.3]) # best choice: size, 0.3
    # parser.add_argument("--object_center", type=float, nargs=3, default=[-0.177, -0.44, 1.071])
    # parser.add_argument("--object_size", type=float, nargs=3, default=[0.3, 0.3, 0.3]) 
    parser.add_argument("--camera_names", type=str, nargs='+', default=['camera_left', 'camera_right', 'camera_top'])
    parser.add_argument("--namespace", type=str, default="")
    args = parser.parse_args()

    args.use_depth_mask = False

    main(args)
