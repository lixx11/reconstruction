'''Simulate diffraction patterns for coreshell particle.
Usage:
    simulate_diffraction_patterns.py [options]

Options:
    -h --help                                       Show this screen.
    --rotation-param=rotation_param                 rotation parameters in zxz Euler format: alpha_min, alpha_max, alpha_step, beta_min, beta_max, beta_step, gamma_min, gamma_max, gamma_step [default: 0,30,10,0,30,10,0,30,10].
    --output-dir=output_dir                         Output directory [default: output].
'''

from docopt import docopt
import numpy as np
from numpy import fft
from math import cos, sin
import scipy.interpolate as spint
import logging
import os
from mpi4py import MPI
import time
import h5py


def slice2D(data, euler_angle, grid_size, pattern_size):
    '''
    Calculate single diffraction/scattering pattern of given euler angle by slicing
     3D Fourier space.
    '''
    euler_angle = np.deg2rad(euler_angle) #Notice, deg2rad
    model_size = data.shape[0]
    center = int(round((model_size-1.)/2.))
    # sampling plane
    sp_spacing = np.linspace(-(grid_size*(pattern_size-1.)/2.), (grid_size*(pattern_size-1.)/2.), pattern_size)
    sp_grid_x, sp_grid_y = np.meshgrid(sp_spacing, sp_spacing)
    sp_grid_z = np.zeros_like(sp_grid_x)
    x, y, z = sp_grid_x.reshape(sp_grid_x.size), sp_grid_y.reshape(sp_grid_y.size), sp_grid_z.reshape(sp_grid_z.size)
    sp_points = np.array([x, y, z])

    Rx = lambda t: np.array([[1, 0, 0], [0, cos(t), -sin(t)], [0, sin(t), cos(t)]], dtype=float)
    Rz = lambda t: np.array([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]], dtype=float)
    # print('before dot')
    sp_points_rotated = Rz(euler_angle[0]).dot(Rx(euler_angle[1])).dot(Rz(euler_angle[2])).dot(sp_points)
    # print('after dot')
    RGI = spint.RegularGridInterpolator
    model_spacing = np.arange(0, model_size) - (model_size-1.)/2.
    rgi = RGI(points=[model_spacing]*3, values=data, bounds_error=False, fill_value=0)
    return rgi(sp_points_rotated.T).reshape([pattern_size, pattern_size])


def cal_grid_size(particle_size=52E-9, det_ps=110E-6, det_dist=0.565, det_size=401, src_wavelength=2.06E-10, oversampling_ratio=7):
    """
    Calculate interp grid size
    """
    fourier_size = (det_size * det_ps) / (det_dist * src_wavelength)
    fourier_pixel_size = fourier_size / det_size 
    fourier_grid_size = 1. / (particle_size * oversampling_ratio)
    grid_size = fourier_pixel_size / fourier_grid_size
    return grid_size


class Coreshell(object):
    """docstring for Particle"""
    def __init__(self, model_size=35, oversampling_ratio=7, core_value=79, shell_value=46, space_value=0):
        self.model_size = model_size
        self.oversampling_ratio = oversampling_ratio
        self.core_value = core_value
        self.shell_value = shell_value
        self.space_value = space_value
        self.real_space, self.inverse_space = self.make_coreshell()

    def make_coreshell(self):
        model_size = self.model_size
        oversampling_ratio = self.oversampling_ratio
        space_value = self.space_value
        shell_value = self.shell_value
        core_value = self.core_value
        space_size = model_size * oversampling_ratio
        particle = np.ones((model_size, model_size, model_size), dtype=float)
        real_space = np.ones((space_size, space_size, space_size), dtype=float) * space_value
        particle_center = round((model_size - 1) / 2.)   

        # Points
        up_center = np.array([particle_center, particle_center, model_size-1], dtype=float)
        bottom_center = np.array([particle_center, particle_center, 0], dtype=float)
        left_center = np.array([0, particle_center, particle_center], dtype=float)
        right_center = np.array([model_size-1, particle_center, particle_center], dtype=float)
        front_center = np.array([particle_center, 0, particle_center], dtype=float)
        back_center = np.array([particle_center, model_size-1, particle_center], dtype=float)    

        # Lines
        up_left = up_center - left_center
        up_right = up_center - right_center
        up_front = up_center - front_center
        up_back = up_center - back_center   

        # Planes
        # calculate normal verctors and constant for each plane
        n_up_left_front = np.cross(up_left,up_front)
        n_up_right_front = np.cross(up_right,up_front)
        n_up_left_back = np.cross(up_left,up_back)
        n_up_right_back = np.cross(up_right,up_back)
        c_up_left_front = -np.dot(n_up_left_front,up_center)
        c_up_right_front = -np.dot(n_up_right_front,up_center)
        c_up_left_back = -np.dot(n_up_left_back,up_center)
        c_up_right_back = -np.dot(n_up_right_back,up_center)
        c_bottom_left_front = -np.dot(n_up_left_front,bottom_center)
        c_bottom_right_front = -np.dot(n_up_right_front,bottom_center)
        c_bottom_left_back = -np.dot(n_up_left_back,bottom_center)
        c_bottom_right_back = -np.dot(n_up_right_back,bottom_center)    

        # make core
        for i in range(model_size):
            for j in range(model_size):
                for k in range(model_size):
                    this_point = np.array([i, j, k])
                    t1 = np.dot(this_point, n_up_left_front)
                    t2 = np.dot(this_point, n_up_right_front);
                    t3 = np.dot(this_point, n_up_left_back);
                    t4 = np.dot(this_point, n_up_right_back);
                    judge1 = (t1+c_up_left_front)*(t1+c_bottom_left_front);
                    judge2 = (t2+c_up_right_front)*(t2+c_bottom_right_front);
                    judge3 = (t3+c_up_left_back)*(t3+c_bottom_left_back);
                    judge4 = (t4+c_up_right_back)*(t4+c_bottom_right_back);
                    if judge1<=0 and judge2<=0 and judge3<=0 and judge4<=0: #core
                        particle[i,j,k] = core_value
                    else: # shell
                        particle[i,j,k] = shell_value   

        s1 = int(round((space_size-1.)/2. - (model_size-1.)/2.))
        s2 = int(round(s1 + model_size))
        real_space[s1:s2, s1:s2, s1:s2] = particle
        inverse_space = fft.fftshift(fft.fftn(real_space))  

        return real_space, inverse_space


if __name__ == '__main__':
    argv = docopt(__doc__)
    rotation_param = argv['--rotation-param']
    output_dir= str(argv['--output-dir'])
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if os.path.isdir(output_dir):
        pass
    else:
        try:
            os.makedirs('%s' %output_dir)
        except Exception as e:
            raise e
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='%s.log' % os.path.join(output_dir, 'debug_%d' % rank))
    logging.info('Input arguments: %s' % str(argv))

    model_size = 35
    oversampling_ratio = 7
    core_value = 79
    shell_value = 46
    particle_size = 52E-9
    det_ps = 110E-6
    det_dist = 0.565
    det_size = 401
    src_wavelength = 2.06E-10

    if rank == 0:  # assign jobs
        rotation_list = rotation_param.split(',')
        alpha_min = float(rotation_list[0])
        alpha_max = float(rotation_list[1])
        alpha_step = float(rotation_list[2])
        beta_min = float(rotation_list[3])
        beta_max = float(rotation_list[4])
        beta_step = float(rotation_list[5])
        gamma_min = float(rotation_list[6])
        gamma_max = float(rotation_list[7])
        gamma_step = float(rotation_list[8])
        alpha = np.linspace(alpha_min, alpha_max, (alpha_max-alpha_min)/alpha_step+1)
        beta = np.linspace(beta_min, beta_max, (beta_max-beta_min)/beta_step+1)
        gamma = np.linspace(gamma_min, gamma_max, (gamma_max-gamma_min)/gamma_step+1)
        mesh_alpha, mesh_beta, mesh_gamma = np.meshgrid(alpha, beta, gamma)

        N_jobs = mesh_alpha.size
        logging.info('Generating %d orientaions for simulation' % N_jobs)
        N_subjobs = N_jobs // size
        for i in range(size):
            if i == (size-1):
                job_alpha = mesh_alpha.flat[i*N_subjobs:]
                job_beta = mesh_beta.flat[i*N_subjobs:]
                job_gamma = mesh_gamma.flat[i*N_subjobs:]
            else:
                job_alpha = mesh_alpha.flat[i*N_subjobs:(i+1)*N_subjobs]
                job_beta = mesh_beta.flat[i*N_subjobs:(i+1)*N_subjobs]
                job_gamma = mesh_gamma.flat[i*N_subjobs:(i+1)*N_subjobs]
            job_euler_angle = np.stack((job_alpha, job_beta, job_gamma), axis=1)
            np.savetxt('%s/job_%d' % (output_dir, i), job_euler_angle, fmt='%.2f %.2f %.2f')

    comm.barrier()

    coreshell = Coreshell(model_size=model_size, oversampling_ratio=oversampling_ratio)
    grid_size = cal_grid_size(particle_size=particle_size, det_ps=det_ps, det_dist=det_dist, det_size=det_size,\
                              src_wavelength=src_wavelength, oversampling_ratio=oversampling_ratio)
    intensity3D = np.abs(coreshell.inverse_space) ** 2.

    job_euler_angle = np.loadtxt('%s/job_%d' % (output_dir, rank))
    job_size = job_euler_angle.shape[0]
    h5f = h5py.File('%s/data_%d.h5' % (output_dir, rank))
    h5f.create_dataset('pattern', shape=(job_size, det_size, det_size))
    h5f.create_dataset('euler_angle', shape=(job_size, 3))
    t0 = time.time()
    for i in range(job_size):
        print('Rank %d processing %d/%d task, ' % (rank, i+1, job_size)),
        if i > 0:
            tp = time.time()
            t_to_finish = (tp - t0) * float(job_size - i) / float(i) / 3600.
            print('will finish in %.2f hours' % t_to_finish)
        euler_angle = job_euler_angle[i]
        pattern = slice2D(intensity3D, euler_angle, grid_size, det_size)
        # write to h5
        h5f['pattern'][i] = pattern
        h5f['euler_angle'][i] = euler_angle
    h5f.close()
