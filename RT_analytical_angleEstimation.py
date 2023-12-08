import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from std_msgs.msg import Bool

import numpy as np
import time

import _thread as thread

it_max = 1
params = np.zeros((it_max,8))
params[0][1] = 1
params_ref = np.zeros((it_max,8))
params_ref[0][1] = 1

def readTopic(data):
    global params
    rows = int(len(data.data)/8)
    #params = np.zeros((rows,8))

    for i in range(it_max):
        if i < rows:
            for j in range(8):
                params[i][j] = data.data[i*8+j]
        else:
            for j in range(8):
                params[i][j] = 0
    #print(params)
    return

def set_ref_parameters():
    global params_ref
    params_ref = params.copy()
    return

def reset_ref_parameters():
    global params_ref
    params_ref = np.zeros((it_max,8))
    params_ref[0][1] = 1
    return

def manual_recalibration():
    while not rospy.is_shutdown():
        text = input("Press 0 to recalibrate angle\n")
        if text == '0':
            set_ref_parameters()
        else:
            print("input is not 0")
        time.sleep(0.1)
    return

if __name__ == '__main__':
    rospy.init_node('RTAngleEstimation')
    freq = 20
    rate = rospy.Rate(freq)
    rospy.Subscriber('/estimatedParameters',Float64MultiArray,readTopic,queue_size=1)

    #continous angle estimation
    thread.start_new_thread(manual_recalibration,())

    publishing = True
    if publishing:
        pub = rospy.Publisher('angle_from_PR', Float64,queue_size=1)

    while not rospy.is_shutdown():
        if params[0][0] > 5000:
            if np.sum(params_ref) == 1:
                set_ref_parameters()
            
            angle_est = params_ref[0][5] - params[0][5]
            while (angle_est > 90):
                angle_est = angle_est - 180
            while (angle_est < -90):
                angle_est = angle_est + 180

            if publishing:
                pub.publish(angle_est)
            else:
                print(angle_est)
        else:
            reset_ref_parameters()
            if publishing:
                pub.publish(-999)
        
        rate.sleep()