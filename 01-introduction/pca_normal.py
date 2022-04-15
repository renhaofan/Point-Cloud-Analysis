#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:05:44 2021

@author: steve
@description: implemente the surface normals estimation by PCA with 
search param KNN-5. However, the estimated surface normals is
equal to minus surface normals by offical implementation.
"""

# import time
# import os
# import copy
from tqdm import tqdm
import argparse
import open3d as o3d 
import numpy as np
from numpy.testing import suppress_warnings

def covariance(point_cloud_filename):
    """ Calculate covariance matrix for given point cloud Nx3
    Parameters
    ----------
        point_cloud_filename: point cloud path
    
    Returns
    ----------
        H: my implementation, 3x3, numpy.ndarray
        officaial implementation: 3x3, numpy.ndarray
    """
    point_cloud_o3d = o3d.io.read_point_cloud(point_cloud_filename)
    data = point_cloud_o3d.points
    
    # format as numpy 
    X = np.asarray(data)
    N = X.shape[0]

    # normalize by center:
    mu = np.mean(X, axis=0)
    X_normalized = X - mu
    
    # my implementation
    H = np.matmul(X_normalized.transpose(), X_normalized) / N
    # official implementation
    
    return H, np.cov(X_normalized, rowvar=False, bias=True)

def PCA(data, correlation=False, sort=True):
    """ Calculate PCA for given point cloud
    
    Parameters
    ----------
        data: Nx3 matrix, np.array
        correlation: distinguish np.cov and np.corrcoef
        sort: sort eigenvalues for else function usage, descend by default
    
    Returns
    ----------
        eigenvalues: 1x3, numpy.ndarray, in descending order by default
        eigenvectors:3x3, numpy.ndarray, in descending order by default
                      column vector is eigenvector
        
    Example
    ----------
        (array([3498.53766092, 2643.36463854,  418.03693071]),
         array([[ 0.66178944,  0.73077414, -0.16734364],
                [ 0.24402867,  0.00107942,  0.96976742],
                [-0.70886158,  0.68261848,  0.17761549]]))
    """
    # format as numpy, if data is Nx3 matrix, open3d.utility.Vector3dVector
    # X = np.asarray(data)
    # N = X.shape[0]
    
    X = data
    # normalize by center:
    mu = np.mean(X, axis=0)
    X_normalized = X - mu

    # print(f'X_normalized shape: {X_normalized.shape}')
    # compute variance
    func = np.cov if not correlation else np.corrcoef
    H = func(X_normalized, rowvar=False, bias=True) # 3x3
    
    # get eigen pairs:
    eigenvalues, eigenvectors = np.linalg.eig(H)

    # print(eigenvalues, eigenvectors)
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

def get_eigenvectors_geometry_structure(w, v, points):
    """ get eigenvectors geometry structure (and eigenvalues) \
    scale should be the same with w[0],w[1],w[2], if wanna match vector length
    
    Parameters
    ----------
        w: eigenvalues in descending order
        v: eigenvectors in descending order
        points: Nx3, np.ndarray
    
    Returns
    ----------
        line_set: o3d lineset for eigenvectors visualization
    """
    centroid = points.mean(axis=0)
    projection = np.dot(points, v[:, 0])
    scale = projection.max() - projection.min()
    
    # frame 4 points
    frame = centroid + np.vstack(
        (
            np.asarray([0.0, 0.0, 0.0]),
            scale * v.T
        )
    )
    frame = frame.tolist()
    
    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    # from the largest to the smallest: RGB
    colors = np.identity(3).tolist()
    # build pca line set:
    pca_frame_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(frame),
        lines=o3d.utility.Vector2iVector(lines),
    )
    pca_frame_lineset.colors = o3d.utility.Vector3dVector(colors)
    
    return pca_frame_lineset

def get_surface_normal(pcd, knn=5):
    """ calculate surface normal for each points by knn neighbors
    
    Parameters
    ----------
        pcd: open3d.geometry.PointCloud
        knn: the number of search neighbors
        
    Returns
    ----------
        normals: Nx3 numpy.ndarray
    """
    # create neighbor search tree
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points_np = np.asarray(pcd.points)
    
    # init
    N = points_np.shape[0]
    normals = []
    
    
    # for i in range(N):            
    #     # find knn:
    #     [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)
    #     # get normal:
    #     w, v = PCA(points_np[idx,:])
    #     normals.append(v[:, -1])
    
    # add process bar
    print("Start to compute surface normals:")
    for i in tqdm(range(N)):
        # find knn:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)
        # get normal:
        w, v = PCA(points_np[idx,:])
        normals.append(v[:, -1])
    print('------------------------------------' + \
          '------------------------------------')

    # fix ComplexWarning:  
    # Casting complex values to real discards the imaginary part
    # caused complex complex128 data type in PCA solving eigenvectors
    # (-1.266728529997394048e-01-1.516960254733458469e-01j)
    # (7.960678716452213033e-01-0.000000000000000000e+00j)
    # (-1.825200864563703840e-01-5.421297500992759977e-01j)
    with suppress_warnings() as sup:
        sup.filter(np.ComplexWarning)
        normals = np.array(normals, dtype=np.float64)
        
    return normals
   
def get_surface_normal_geometry_structure(normals_ndarray, points_ndarray, \
                                          scale=2):
    """ get surface normal lineset for each points
    
    Parameters
    ----------
        normals_ndarray: open3d.geometry.PointCloud
        points_ndarray: the number of search neighbors
        scale: float, the length of each surface normal vector
        
    Returns
    ----------
        normals: o3d lineset for surface normal visualization
    """
    N = points_ndarray.shape[0];
    
    points = np.vstack(
        (
            points_ndarray,
            points_ndarray + scale * normals_ndarray
        )
    )
    lines = [[i, i+N] for i in range(N)]
    colors = np.zeros((N, 3)).tolist()
    
    # construct suface normal lineset
    surface_normals_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    surface_normals_lineset.colors = o3d.utility.Vector3dVector(colors)
    
    return surface_normals_lineset

def my_estimate_normals(point_cloud_filename):
    """ calculate surface normal for each points by knn neighbors and PCA
    but my (-normals_ndarray) is exactly equal to the offical implementation
    Parameters
    ----------
        point_cloud_filename: path       
    """
    
    # read point cloud from file
    point_cloud_o3d = o3d.io.read_point_cloud(point_cloud_filename)
    points_ndarray = np.asarray(point_cloud_o3d.points)
    print('------------------------------------' + \
          '------------------------------------')
    print(f'Path: {point_cloud_filename}')
    print('Model name:', point_cloud_filename.split('/')[-1])
    print(f'Total number of points: {points_ndarray.shape[0]}')
    print('------------------------------------' + \
          '------------------------------------')
    
    # PCA for point clouds
    w, v = PCA(points_ndarray)
    print(f'Eigenvalues: {w}')
    print(f'Eigenvectors:\n {v}')
    print('------------------------------------' + \
          '------------------------------------')
    print(f'center by np.ndarray: {points_ndarray.mean(axis=0)}')
    print(f'center by open3d: {point_cloud_o3d.get_center()}')
    print('------------------------------------' + \
          '------------------------------------')

    # generate scaled eigenvectors lineset
    pca_o3d = get_eigenvectors_geometry_structure \
                        (w, v, points_ndarray)
   
    # calculate surface normal
    normals_ndarray = get_surface_normal(point_cloud_o3d)
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals_ndarray)
    
    # visualization
    point_cloud_o3d.paint_uniform_color([1, 1, 0])
    o3d.visualization.draw_geometries([point_cloud_o3d, \
                                      pca_o3d], \
                                      point_show_normal=True)
    
    # generate surface normal lineset
    # surface_normals_o3d = get_surface_normal_geometry_structure \
    #                    (normals_ndarray, points_ndarray)
    # o3d.visualization.draw_geometries([point_cloud_o3d, \
    #                                   pca_o3d, \
    #                                   surface_normals_o3d])

def offical_estimate_normals(point_cloud_filename):
    """
    KNN, also you cloud utilize the default search param - Hybrid 
    it looks better than KNN.
    `search_param=o3d.geometry.KDTreeSearchParamHybrid\ 
        (radius=0.1, max_nn=30))`
    """
    point_cloud_o3d = o3d.io.read_point_cloud(point_cloud_filename)
    point_cloud_o3d.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(5))
    o3d.visualization.draw_geometries([point_cloud_o3d],
                                      point_show_normal=True)


def get_arguments():
    """
    Get command-line arguments
    """
    # init parser:
    parser = argparse.ArgumentParser\
        ("Get PCA and surface normals for given point cloud.")

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
        '-m', dest='method', default='my', choices=['official', 'my'])

    # parse arguments:
    return parser.parse_args()
    
if __name__ == '__main__':
    arguments = get_arguments()
    path = arguments.input
    if arguments.method == 'my':
        my_estimate_normals(path)
    else:
        offical_estimate_normals(path)
    
    
    
    
    