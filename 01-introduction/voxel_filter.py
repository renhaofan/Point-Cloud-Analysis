#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:05:44 2021

@author: steve
@description: implemente the voxel filter for point clouds
"""


# import os
import open3d as o3d
import numpy as np
import argparse


def construct_bin(records_array):
    """ voxel filter for point cloud, then display the results
    
    Parameters
    ----------
        records_array: Nx1, numpy.ndarray. 
                       key: point cloud index, value: voxel index
    Return
    ---------
        res: filter object
    """
    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(records_array)

    # sorts records array so all unique elements are together 
    sorted_records_array = records_array[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])

    #filter them with respect to their size, keeping only items occurring more than once
    vals = vals[count > 1]
    res = filter(lambda x: x.size > 1, res)
    
    return res
    
    

def my_voxel_filter(point_cloud_o3d, voxel_size):
    """ voxel filter for point cloud, then display the results
        For point clouds in each voxel, centroid as result
    
    Parameters
    ----------
        point_cloud_o3d: open3d.geometry.PointCloud
        voxel_size: voxel size
        
    Return
    ---------
        point_cloud_filtered: np.ndarray
    """
    points= np.asarray(point_cloud_o3d.points)
    # bounding box size
    mmax = np.max(points, axis=0)
    mmin = np.min(points, axis=0)
    bbs = mmax - mmin
    
    # dimension voxel grid
    [DX,DY,DZ] = np.ceil(bbs / voxel_size).astype(np.int64)
    
    # index for every dimension
    h_xyz = np.floor((points - mmin) / voxel_size).astype(np.int64)
    # final index
    h = np.dot(h_xyz, np.array([1, DX, DX*DY]))
    
    # classifier
    point_cloud_bin = construct_bin(h)
    
    point_cloud_filtered = np.zeros([1, 3])
    for i in point_cloud_bin:
        # i is ndarray
        # key: voxel index, value, point clouds index in the same voxel
        centroid = np.mean(points[i], axis=0)
        point_cloud_filtered = np.vstack(
                (
                    point_cloud_filtered,
                    centroid
                )
            )
    point_cloud_filtered = np.delete(point_cloud_filtered, obj=0, axis=0)
    
    print(f'Voxel dim DX, DY, DZ: {DX,DY,DZ}')
    print('Number of points after voxel filter:', \
                  point_cloud_filtered.shape[0])
    return point_cloud_filtered
    
def main(point_cloud_filename, voxel_size):
    # read point cloud from file
    point_cloud_o3d = o3d.io.read_point_cloud(point_cloud_filename)
    points_ndarray = np.asarray(point_cloud_o3d.points)
    print('------------------------------------' + \
          '------------------------------------')
    print(f'Path: {point_cloud_filename}')
    print('Model name:', point_cloud_filename.split('/')[-1])
    print(f'Total number of points: {points_ndarray.shape[0]}')
    print(f'Voxel size: {voxel_size}')
    print('------------------------------------' + \
          '------------------------------------')
    # voxel filter
    filtered_points_o3d = o3d.geometry.PointCloud()
    filtered_points = my_voxel_filter(point_cloud_o3d, voxel_size)
    filtered_points_o3d.points = o3d.utility.Vector3dVector(filtered_points)
    
    
    # visualization for comparision
    distance = filtered_points - np.mean(filtered_points, axis=0)
    radius = [np.linalg.norm(distance[i]) \
              for i in np.arange(distance.shape[0])]
    max_radius = np.max(radius)
    filtered_points_o3d.translate([max_radius, 0, 0])
    # yellow
    point_cloud_o3d.paint_uniform_color([1, 1, 0])
    # cyan
    filtered_points_o3d.paint_uniform_color([0, 1, 1])
    o3d.visualization.draw_geometries([point_cloud_o3d, filtered_points_o3d])

def get_arguments():
    """
    Get command-line arguments
    """
    # init parser:
    parser = argparse.ArgumentParser\
        ("voxel filter for point clouds")

    # # add required, in case that defatult arg useless
    # parser.add_argument(
    #     "-i", dest="input", help="Input path of point cloud in ply format.",
    #     required=True
    # )
    
    path = r'/home/steve/dataset/ModelNet40/ply_data/airplane/' + \
        r'train/airplane_0001.ply'
        
    parser.add_argument(
        '-i', dest='input', help='Input path of point cloud in ply format.',
        default=path
    )
    parser.add_argument(
        '-vs', dest='voxel_size', help='voxel size for filter',
        default=10.0
    )

    parser.add_argument(
        '-m', dest='method', default='my', choices=['official', 'my'])

    # parse arguments:
    return parser.parse_args()
    
if __name__ == '__main__':
    arguments = get_arguments()
    path = arguments.input
    main(path, arguments.voxel_size)
    # if arguments.method == 'my':
    #     my_estimate_normals(path)
    # else:
    #     offical_estimate_normals(path)
