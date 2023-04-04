import ros_numpy
import rospy
import numpy as np
from sensor_msgs.msg import Image,PointCloud2,PointField

from collections import deque
import cv2
from functools import partial
import ros_numpy

import threading


def depth_callback(msg):
    global depth_list
    try:
        cv_image = ros_numpy.numpify(msg)
    except:
      print('convert wrong')
    depth_list.append(cv_image)
    # cv2.imshow('rgb',cv_image)
    # cv2.waitKey(1)

def rgb_callback(msg):
    global rgb_list
    try:
        cv_image = ros_numpy.numpify(msg)
    except:
      print('convert wrong')
    rgb_list.append(cv_image)
    # cv2.imshow('rgb',cv_image)
    # cv2.waitKey(1)


def getXYZRGB(color, depth, depth_scale,mtx,camera_pose):
    '''
    :param color:
    :param depth:
    :param robot_pose: array 4*4
    :param camee_pose: array 4*4
    :param camIntrinsics: array 3*3
    :param inpaint: bool
    :return: xyzrgb
    '''
    camIntrinsics = mtx
    robot_pose = camera_pose
    heightIMG, widthIMG, _ = color.shape
    # heightIMG = 720
    # widthIMG = 1280
    depthImg = depth * depth_scale
    # depthImg = depth

    [pixX, pixY] = np.meshgrid(np.arange(widthIMG), np.arange(heightIMG))
    camX = (pixX - camIntrinsics[0][2]) * depthImg / camIntrinsics[0][0]
    camY = (pixY - camIntrinsics[1][2]) * depthImg / camIntrinsics[1][1]
    camZ = depthImg

    camPts = [camX.reshape(camX.shape + (1,)), camY.reshape(camY.shape + (1,)), camZ.reshape(camZ.shape + (1,))]
    camPts = np.concatenate(camPts, 2)
    camPts = camPts.reshape((camPts.shape[0] * camPts.shape[1], camPts.shape[2]))  # shape = (heightIMG*widthIMG, 3)
    rgb = color.reshape((-1, 3)) / 255.
    worldPts = np.dot(robot_pose[:3, :3], camPts.T) + robot_pose[:3, 3].reshape(3,1)  # shape = (3, heightIMG*widthIMG)
    xyzrgb = np.hstack((worldPts.T, rgb))
    return xyzrgb

def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    assert(points.shape == colors.shape)

    buf = []

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True
    msg.data = xyzrgb.tostring()

    return msg


def publish_pointcloud_ros(color_list,depth_list,xyzrgb_list):
    global depth_scale, intrinsics,handeye
    pointcloud_publisher = rospy.Publisher("l515/pointcloud",PointCloud2,queue_size=1)
    # handeye = np.load(os.path.join(file_dir,'..','conig','handeye.npy'))
    # euler = tf.transformations.euler_from_matrix(handeye,'sxyz')
    # pose = handeye[:3,3].tolist() + euler.tolist() # [0.5,0,0.5,0,np.pi,np.pi/2]
    pose = [0.5,0,0.5,0,np.pi,np.pi/2]
    frame_id = 'pc_base'
    # static_transform(new_frame=frame_id,pose=pose,base_frame='panda_link0')
    rospy.loginfo("Publish pointcloud in /l515/poindcloud in %s frame."%frame_id)
    while len(color_list)==0 or len(depth_list)==0:
        pass
    while not rospy.is_shutdown():
        color, depth = color_list[-1],depth_list[-1]

        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03),cv2.COLORMAP_JET)
        xyzrgb = getXYZRGB(color[:,:,::-1], depth,depth_scale,intrinsics,handeye)
        xyzrgb_list.append(xyzrgb)
        
        stamp = rospy.Time.now()
        pub_pc = xyzrgb_array_to_pointcloud2(xyzrgb[:, :3], xyzrgb[:, 3:6], stamp=stamp,frame_id=frame_id)
        pointcloud_publisher.publish(pub_pc)


if __name__ == "__main__":
    rgb_list = deque(maxlen=2)
    depth_list = deque(maxlen=2)
    xyzrgb_list = deque(maxlen=2)
    rospy.init_node('rgb_receive',anonymous=True)
    rospy.Subscriber('/l515/depth',Image,depth_callback,queue_size=1)
    rospy.Subscriber('/l515/rgb',Image,rgb_callback,queue_size=1)
    
    handeye = np.array(rospy.get_param('handeye')).reshape((4,4))
    intrinsics = np.array(rospy.get_param("intrinsics")).reshape((3,3))
    depth_scale = rospy.get_param("depth_scale")
    
    pc_t = threading.Thread(target=partial(publish_pointcloud_ros,rgb_list,depth_list,xyzrgb_list))
    pc_t.start()
    
    # use rgb_list[-1] and depth_list[-1] as rgb and depth image
    # depth image * depth_scale to convert depth map im meter
    # use xyzrgb_list as poingcloud with color, n*6
    while not rospy.is_shutdown():
        if len(rgb_list)>0:
            rgb = rgb_list[-1]
            depth = depth_list[-1]
            cv2.imshow('rgb',rgb)
            cv2.waitKey(1)
        if len(xyzrgb_list)>0:
            # print(xyzrgb_list[-1].shape)
            pass