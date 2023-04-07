
import os
import open3d as o3d
from src.utilt import *
from pyquaternion import Quaternion as Quat
from scipy.spatial.transform import Rotation as Rot
import time
import torch
from random import choice
from src.grasp_utilt import view_pre_grasp_trajectory
from graspnetAPI import GraspGroup
from graspnetAPI.utils.utils import plot_gripper_pro_max

import rospy
from sensor_msgs.msg import PointCloud2,PointField
from geometry_msgs.msg import PoseArray
from graspnet_pkg.srv import GraspNetList,GraspNetListResponse
from graspnet_pkg.msg import GraspMsg
from msg_srv.srv import GraspAffordance, GraspTrajectory,GraspAffordanceResponse,GraspTrajectoryResponse
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point, Quaternion,Pose

def xyzl_array_to_pointcloud2(points, stamp=None, frame_id=None):
    '''
        Numpy to PointCloud2
        Create a sensor_msgs.PointCloud2 from an array
        of points (x, y, z, l)
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            # PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()
    return msg

def getGraspNetService(pc_msg):
    rospy.wait_for_service("GraspNet")
    try:
        srv_handle = rospy.ServiceProxy("GraspNet", GraspNetList)
        rep = srv_handle(pc_msg)
        gg_list = rep.gg

    except rospy.ServiceException as e:
        print("Service call failed : %s" % e)
        return None
    return gg_list

def convertGraspMsgtoNumpy(gg_list):
    # grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids
    gg_array = []
    for gg in gg_list:
        grasp_score = gg.grasp_score
        grasp_width = gg.grasp_width
        grasp_height = gg.grasp_height
        grasp_depth = gg.grasp_depth
        pre_ = [grasp_score, grasp_width, grasp_height, grasp_depth]
        rotation = gg.rotation
        grasp_center = gg.grasp_center
        obj_ids = gg.obj_ids
        quat = Quat(rotation.w, rotation.x, rotation.y, rotation.z)
        rotation_matrix = quat.rotation_matrix.flatten().tolist()
        grasp_center = [grasp_center.x, grasp_center.y, grasp_center.z]
        gg_array.append(pre_ + rotation_matrix + grasp_center + [obj_ids])
    gg_array = np.array(gg_array)
    return gg_array

def getAffordanceService(pc_msg,gg):
    rospy.wait_for_service("Affordance")
    try:
        srv_handle = rospy.ServiceProxy("Affordance",GraspAffordance)
        result = srv_handle(pointcloud=pc_msg,gg=gg)
    except rospy.ServiceException as e:
        print("Service call failed : %s" % e)
    return result

def getTrajectoryService(pc_msg,gg):
    rospy.wait_for_service("Trajectory")
    try:
        srv_handle = rospy.ServiceProxy("Trajectory",GraspTrajectory)
        trajectory = srv_handle(gg=gg,pointcloud=pc_msg)
    except rospy.ServiceException as e:
        print("Service call failed : %s" % e)
    return trajectory

if __name__ == '__main__':
    # real data
    file_path = "./real_data/microwaves/021/"
    file = np.load("./L515_test/without_train/0000/xyz0.21_remove.npz")
    pointcloud = file["point_cloud"]


    pc_msg = xyzl_array_to_pointcloud2(pointcloud,frame_id='pc_base')
    gg_list = getGraspNetService(pc_msg)
    gg = convertGraspMsgtoNumpy(gg_list)
    grasp_group = GraspGroup(gg)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pointcloud)
    grippers = grasp_group.to_open3d_geometry_list()
    # o3d.visualization.draw_geometries([cloud, axis_pcd, *grippers])

    affordance_result = getAffordanceService(pc_msg,gg_list)
    print(affordance_result)
    for i in range(len(gg_list)):
        if i in affordance_result.result:
            grippers[i].paint_uniform_color([0.0, 1.0, 0.0])
        else:
            grippers[i].paint_uniform_color([1.0, 0.0, 0.0])
    # o3d.visualization.draw_geometries([cloud, axis_pcd, *grippers])

    for _index in affordance_result.result:
        gg = gg_list[_index]
        trajectory = getTrajectoryService(pc_msg, gg)

        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pointcloud)
        cloud.paint_uniform_color([0.5, 0.5, 0.5])
        grasps = []

        grasps.append(plot_gripper_pro_max(center=[gg.grasp_center.x, gg.grasp_center.y, gg.grasp_center.z],
                                           R=np.array(pb.getMatrixFromQuaternion(
                                               [gg.rotation.x, gg.rotation.y, gg.rotation.z, gg.rotation.w])).reshape((3,3)),
                                           width=gg.grasp_width,
                                           depth=gg.grasp_depth,
                                           score=1))

        for _pose in trajectory.trajectory.poses:
            grasps.append(plot_gripper_pro_max(center=[_pose.position.x,_pose.position.y,_pose.position.z],
                                               R=np.array(pb.getMatrixFromQuaternion(
                                                   [_pose.orientation.x,_pose.orientation.y,_pose.orientation.z,_pose.orientation.w])).reshape((3,3)),
                                               width=gg.grasp_width,
                                               depth=gg.grasp_depth,
                                               score=gg.grasp_score))
        o3d.visualization.draw_geometries([cloud, *grasps, axis_pcd])


        # import pdb;pdb.set_trace()