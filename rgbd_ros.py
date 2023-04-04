from utilt import *

from collections import deque
from functools import partial
import threading


if __name__ == "__main__":
    rgb_list = deque(maxlen=2)
    depth_list = deque(maxlen=2)
    xyzrgb_list = deque(maxlen=2)
    rospy.init_node('rgb_receive',anonymous=True)
    rospy.Subscriber('/l515/depth',Image,partial(depth_callback,depth_list),queue_size=1)
    rospy.Subscriber('/l515/rgb',Image,partial(rgb_callback,rgb_list),queue_size=1)
    
    handeye = np.array(rospy.get_param('handeye')).reshape((4,4))
    intrinsics = np.array(rospy.get_param("intrinsics")).reshape((3,3))
    depth_scale = rospy.get_param("depth_scale")
    add_info = [depth_scale, intrinsics,handeye]
    pc_t = threading.Thread(target=partial(publish_pointcloud_ros,rgb_list,depth_list,xyzrgb_list,add_info))
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