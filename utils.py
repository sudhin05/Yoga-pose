from math import sqrt, acos, pi, atan2 
import torch
import torch.nn as nn

class Tools(nn.Module):
    def __init__(self):
        super(Tools,self).__init__()

    def angle_lines(self,c_point,point1,point2):
        """
            Function to calculate angle between two lines and a common point
        """
        num = (point1[0] - c_point[0])*(point2[0] - c_point[0]) + (point1[1] - c_point[1])*(point2[1] - c_point[1])
        denom = (sqrt((point1[0] - c_point[0])**2 + (point1[1] - c_point[1])**2)*sqrt((point2[0] - c_point[0])**2 + (point2[1] - c_point[1])**2))

        if denom == 0:
            raise ValueError("directional vectors zero hai")
        ang = acos(num/denom)
        ang = ang * 180 / pi

        return ang
    
    def angle_horiz(self,point1,point2):
        """
            Function to calc angle bw line and horizontal
        """

        num = point2[1] - point1[1]
        denom = point2[0] - point1[0]

        ang = atan2(num,denom)
        ang = ang*180/pi

        return ang
    
    def dist_points(self,point1,point2):
        """
            Function to calculate the distance between two points
        """
        return sqrt((point2[1]-point1[1])**2 + (point2[0]-point1[0])**2)

    def posn_point(self,point,lpoint1,lpoint2):
        """
            Function to find whether point lies to left or right of the line
        """
        value = (lpoint2[0] - lpoint1[0]) * (point[1] - lpoint1[1]) - (lpoint2[1] - lpoint1[1]) * (point[0] - lpoint1[0])
        if value >= 0:
            return "left"
        return "right"




