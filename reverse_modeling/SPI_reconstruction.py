#! /opt/local/bin/python2.7

"""
Usage:
    SPI_reconstruction.py [options]

Options:
    -h --help                                       Show this screen.
    --intensity=<intensity_file>                    Input intensity data as reference. Generate intensity from model if not specified.
    --output-dir=<output_dir>                       Output directory storing logs, results. Default is './output/%s' %timestamp.
    --min-iter=<min_iter>                           Minimum iteration of reconstruction [default: 10000].
    --max-iter=<max_iter>                           Maximum iteration of reconstruction [default: auto].
    --show-gui=<show_gui>                           Show GUI window True or False [default: True].
    --model=<model_file>                            Model filename.
    --model-size=<model_size>                       Model size in pixel [default: 45].
    --oversampling-ratio=<oversampling_ratio>       Oversampling ratio when simulating diffraction patter [default: 9].
    --mask-size=<mask_size>                         Mask size in pixel [default: 77].
    --init-model-size=<init_model_size>             Initial model size in pixel [default: 41].
    --model-angle=<model_angle>                     Rotation angle of simulated model in degree (anticlockwise) [default: 0.].
    --init-model-angle=<init_model_angle>           Rotation angle of initial model in degree (anticlockwise) [default: 0.].
    --scale-factor=<scale_factor>                   Scale input intensity by multiply this scale factor [default: 5].
    --init-T=<init_T>                               Init temprature [default: 1].
    --inner-cooling-factor=<inner_cooling_factor>   Annealing parameter [default: 0.99].
    --outer-cooling-factor=<outer_cooling_factor>   Annealing parameter [default: 0.5].
    --batch-size=<batch_size>                       Annealing parameter: Iteration number in a round [default: 1000].
    --init-step-size=<init_step_size>               Init step size [default: 1].
    --step-shringk-factor=<step_shrink_factor>      Annealing parameter [default: 0.99].
    --ignore-negative=<ignore_negative>             Ignore negative values while calculating cost [default: True].
    --timer-interval=<interval>                     Timer interval in ms [default: 0].
    --update-period=<uf>                            Period to update plot  [default: 50].
    --cpuid=<cpuid>                                 Specify cpu id to run, -1 is auto-find free cpu. Only valid on linux platform [default: -1].
    --TV-factor=<TV_factor>                         TV factor when calculating cost [default: 0.].
"""

from docopt import docopt
argv = docopt(__doc__)
import logging
import datetime
import sys
import os
import time
from util import *
time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
output_dir = argv['--output-dir']
if output_dir is None:
    output_dir = 'output/%s' %time_stamp
try:
    os.makedirs('%s' %output_dir)
except Exception as e:
    raise e
print_with_timestamp('logging into %s.log' %os.path.join(output_dir, 'debug'))
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='%s.log' %os.path.join(output_dir, 'debug'))
import platform
import psutil
if platform.system() == 'Linux':
    logging.info('Running on Linux')
    pid = os.getpid()
    cpu_percents = psutil.cpu_percent(percpu=True, interval=1)
    logging.debug('cpu percent: %s' %str(cpu_percents))
    if argv['--cpuid'] != '-1':
        cpuid = int(argv['--cpuid'])
    else:
        cpuid = None
        for i in range(len(cpu_percents)):
            if cpu_percents[i] < 10.:  # find a free cpu
                cpuid = i
                break
        if cpuid is None:
            logging.critical('No free cpu found!')
            sys.exit()
    os.system('taskset -p %s %s' %(str(hex(2**cpuid)), pid))
from annealing import Annealing 
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from scipy.stats import linregress
import random
import math
from scipy.stats import pearsonr
import cv2


class SPIAnnealing(object):
    """docstring for SPIAnnealing"""
    def __init__(self, init_T=1.0E5, inner_cooling_factor=0.99, outer_cooling_factor=0.5, batch_size=1E3, init_solution=None, init_step_size=1.0, step_shrink_factor=0.99,\
                 ref_intensity=None, mask=None, ignore_negative=True, TV_factor=1.):
        super(SPIAnnealing, self).__init__()
        # annealing parameters
        self.init_T = init_T
        self.T = init_T
        self.inner_cooling_factor = inner_cooling_factor
        self.outer_cooling_factor = outer_cooling_factor
        self.batch_size = batch_size
        self.init_solution = init_solution
        self.solution = init_solution 
        self.init_step_size = init_step_size
        self.step_size = init_step_size  # max iteration in a round
        self.step_shrink_factor = step_shrink_factor
        self.total_iter = 0
        # SPI parameters/data
        self.ref_intensity = ref_intensity
        self.ref_intensity_valid = None
        self.cal_intensity = None
        self.cal_intensity_valid = None
        if mask is None:
            self.mask = np.ones_like(ref_intensity)
        else:
            self.mask = mask
        self.ignore_negative = ignore_negative
        self.TV_factor = TV_factor
        self.cost = self.calc_cost(init_solution)
        self.total_iter = 0
        self.best_solution = init_solution
        self.best_cost = self.calc_cost(init_solution)
        self.restart_list = []
        self.debug_data = None

    def neighbor(self, old_solution):
        edge = get_edge(self.solution, width=1, find_edge='both')
        edge_x, edge_y = np.where(edge)
        edge_xy = np.asarray([edge_x, edge_y]).T 
        # self.debug_data = self.solution.copy()
        # self.debug_data[edge_xy[:,0], edge_xy[:,1]] += 0.5
        N_edge = edge_xy.shape[0]
        N_mutant = random.randint(1, max(min(int(self.step_size), N_edge//3), 1))
        logging.debug('%d edge points, %d points to be mutated' %(edge_xy.shape[0], N_mutant))
        mutant_ids = []
        count = 0
        while count < N_mutant:
            rn = np.random.randint(0, edge_xy.shape[0])
            if count == 0:
                mutant_ids.append(rn)
                count += 1
            else:
                ps = edge_xy[mutant_ids, :]
                diff = ps - edge_xy[rn, :]
                dists = np.sqrt(diff[:,0]**2. + diff[:,1]**2.)
                if dists.min() > 1.:
                    mutant_ids.append(rn)
                    count += 1
        new_solution = self.solution.copy()
        for n in xrange(N_mutant):
            x, y = edge_xy[mutant_ids[n], :]
            if self.solution[x, y] > 0:
                new_solution[x, y] = 0
            else:
                new_solution[x, y] = 1
        return new_solution

    def calc_cost(self, solution):
        self.cal_intensity = np.abs(np.fft.fft2(solution))**2.
        if self.ignore_negative:
            mask = np.fft.fftshift(self.mask) * (self.cal_intensity > 0) * (self.ref_intensity > 0)
        else:
            mask = np.fft.fftshift(self.mask)
        self.debug_data = mask
        self.cal_intensity_valid = self.cal_intensity * mask 
        self.ref_intensity_valid = self.ref_intensity * mask 
        cal_intensity_valid_1d = self.cal_intensity_valid[np.where(self.cal_intensity_valid != 0.)]
        ref_intensity_valid_1d = self.ref_intensity_valid[np.where(self.ref_intensity_valid != 0.)]
        logging.debug('valid pixel number: %d/%d' %(ref_intensity_valid_1d.size, self.ref_intensity.size))
        scaling_factor, _, _, _, _ = linregress(cal_intensity_valid_1d, ref_intensity_valid_1d)
        logging.debug('scaling factor: %.3f' %scaling_factor)
        diff = scaling_factor * cal_intensity_valid_1d - ref_intensity_valid_1d
        error = np.linalg.norm(diff)
        grad = np.gradient(solution)
        TV = np.linalg.norm(np.sqrt(grad[0]**2. + grad[1]**2.).reshape(-1), ord=1)
        logging.debug('2-norm of diff: %3e, TV norm: %3e' %(error, TV))
        # score, _ = pearsonr(this_intensity.reshape(-1), self.ref_intensity.reshape(-1))
        # error = 1 - score
        return error + TV * self.TV_factor

    def acceptance_probability(self, old_cost, new_cost, T):
        _exp = ((old_cost - new_cost) / T)
        if _exp > 0:
            return 1.
        else:
            return math.exp(_exp)

    def run(self, N=50):
        """Summary
        
        Args:
            N (int, optional): Run N step in this run.
        
        Returns:
            TYPE: None
        """
        global costs, accepted_costs, best_ids, better_ids, Ts, max_iter
        best_ids = []
        better_ids = []
        for n in xrange(N):
            logging.debug('===================STEP %d===================' %self.total_iter)
            if self.total_iter != 0 and self.total_iter % self.batch_size == 0:  # restart
                self.restart_list.append(self.best_solution)
                self.solution = self.best_solution
                self.cost = self.calc_cost(self.solution)
                self.T = self.init_T * math.pow(self.outer_cooling_factor, len(self.restart_list))
                self.step_size = self.init_step_size
                costs.append(self.cost)
                accepted_costs.append(self.cost)
                Ts.append(self.T)
                self.total_iter += 1
            new_solution = self.neighbor(self.solution)
            new_cost = self.calc_cost(new_solution)
            if new_cost < self.best_cost:
                self.best_cost = new_cost
                self.best_solution = new_solution
                best_ids.append(self.total_iter)
                if not fix_max_iter:
                    max_iter = max(max_iter, 2 * self.total_iter)
            ap = self.acceptance_probability(self.cost, new_cost, self.T)
            if ap >= random.uniform(0, 1):
                self.solution = new_solution
                self.cost = new_cost
                better_ids.append(self.total_iter)
            self.T *= self.inner_cooling_factor
            costs.append(new_cost)
            accepted_costs.append(self.cost)
            Ts.append(self.T)
            self.total_iter += 1
            self.step_size *= self.step_shrink_factor


def update():
    global last_time, spi_annealing, max_iter, app
    now = time.time()
    dt = now - last_time
    last_time = now 
    fps = 1.0 / dt * int(argv['--update-period']) 
    print_with_timestamp('fps: %.2f' %fps),

    # update plot
    spi_annealing.run(N=int(argv['--update-period']))
    im3.setImage(spi_annealing.solution)
    im4.setImage(np.log(np.abs(np.fft.fftshift(spi_annealing.cal_intensity)))+1.)

    curve2.setData(costs)
    line2.setData(np.ones_like(costs) * costs[-1])
    p2.setTitle('<p><font size="4">Searched Costs: %.3E</font></p>' %costs[-1])
    scatter2.addPoints(x=np.asarray(better_ids), y=np.asarray(costs)[better_ids], pen='y')
    curve3.setData(accepted_costs)
    line3.setData(np.ones_like(accepted_costs) * accepted_costs[-1])
    p3.setTitle('<p><font size="4">Accepted Costs: %.3E</font></p>' %accepted_costs[-1])
    scatter3.addPoints(x=np.asarray(best_ids), y=np.asarray(accepted_costs)[best_ids], pen='g')
    curve4.setData(Ts)
    line4.setData(np.ones_like(Ts) * Ts[-1])
    p4.setTitle('<p><font size="4">Temperature: %.3E Step Size: %d</font></p>' %(Ts[-1], spi_annealing.step_size))

    record['current-model'] = spi_annealing.solution
    record['current-intensity'] = spi_annealing.cal_intensity

    if spi_annealing.total_iter >= max_iter:
        app.quit()


# collect data during reconstruction
costs = []
accepted_costs = []
best_ids = []
better_ids = []
Ts = []
record = {}
record['costs'] = costs
record['accepted-costs'] = accepted_costs
record['Ts'] = Ts 

if __name__ == '__main__':
    logging.debug('Start of SPI reconstruction')
    # add signal to enable CTRL-C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    logging.debug('options: %s' %str(argv))

    # generate test data if no given intensity file
    intensity_file = argv['--intensity']
    if intensity_file is None:
        oversampling_ratio = int(argv['--oversampling-ratio'])
        model_size = int(argv['--model-size'])
        space_size = model_size * oversampling_ratio
        mask_size = int(argv['--mask-size'])
        model_file = argv['--model']
        model_angle = float(argv['--model-angle'])
        if model_file is None:  # generate a square model with rotation angle of 30 degree.
            model = make_model(model_size=model_size, space_size=space_size)
            model = imrotate(model, angle=model_angle)
            model[model<0.5] = 0
            model[model>=0.5] = 1
        else:
            model = load_model(model_file, model_size=model_size, space_size=space_size)
        mask = make_square_mask(space_size, mask_size)
        intensity = np.abs(np.fft.fft2(model))**2.
        record['model'] = model 
    else:
        model = None
        intensity = np.fft.fftshift(np.load(intensity_file))
        rows, cols = intensity.shape
        assert rows == cols  # must be a square
        space_size = rows
        mask = np.ones_like(intensity)
    ref_intensity = intensity * np.fft.fftshift(mask)
    record['ref-intensity'] = ref_intensity

    # generate init model
    init_model_size = int(argv['--init-model-size'])
    init_model_angle = float(argv['--init-model-angle'])
    init_model = make_model(model_size=init_model_size, space_size=space_size)
    init_model = imrotate(init_model, angle=init_model_angle)

    scale_factor = float(argv['--scale-factor'])
    init_T = float(argv['--init-T'])
    inner_cooling_factor = float(argv['--inner-cooling-factor'])
    outer_cooling_factor = float(argv['--outer-cooling-factor'])
    batch_size = int(argv['--batch-size'])
    init_step_size = int(argv['--init-step-size'])
    step_shrink_factor = float(argv['--step-shringk-factor'])
    ignore_negative = bool(argv['--ignore-negative'])
    TV_factor = float(argv['--TV-factor'])


    spi_annealing = SPIAnnealing(init_T=init_T, inner_cooling_factor=inner_cooling_factor,\
                                 outer_cooling_factor=outer_cooling_factor, batch_size=batch_size,\
                                 init_step_size=init_step_size, step_shrink_factor=step_shrink_factor,\
                                 ref_intensity=scale_factor*ref_intensity, init_solution=init_model, \
                                 mask=mask, ignore_negative=ignore_negative, TV_factor=TV_factor)

    show_gui = argv['--show-gui']
    min_iter = int(argv['--min-iter'])
    max_iter = argv['--max-iter']
    if max_iter == 'auto':
        max_iter = 2 * min_iter
        fix_max_iter = False
    else:
        max_iter = int(max_iter)
        fix_max_iter = True
    if show_gui == 'True':
        app = QtGui.QApplication([])
        win = pg.GraphicsWindow('SPI Annealing Reconstruction %s' %time_stamp)

        p11 = win.addPlot(title='<p><font size="4">Model</font></p>')
        im1 = pg.ImageItem()
        p11.addItem(im1)
        if model is None:
            pass
        else:
            im1.setImage(model)
            p11.getViewBox().setAspectLocked(True)  

        p12 = win.addPlot(title='<p><font size="4">Intensity</font></p>')
        im2 = pg.ImageItem()
        p12.addItem(im2)
        im2.setImage(np.log(np.abs(np.fft.fftshift(ref_intensity))+1.))
        p12.getViewBox().setAspectLocked(True)  

        p13 = win.addPlot(title='<p><font size="4">Current Model</font></p>')
        im3 = pg.ImageItem()
        p13.addItem(im3)
        im3.setImage(init_model)
        p13.getViewBox().setAspectLocked(True)  

        p14 = win.addPlot(title='<p><font size="4">Current Intensity</font></p>')
        im4 = pg.ImageItem()
        p14.addItem(im4)
        im4.setImage(np.random.rand(100,100))
        p14.getViewBox().setAspectLocked(True)  

        win.nextRow()
        p2 = win.addPlot(title='<p><font size="4">Searched Costs</font></p>', colspan=4)
        curve2 = p2.plot()
        line2 = p2.plot(pen=pg.mkPen('r'))
        scatter2 = pg.ScatterPlotItem()
        p2.addItem(scatter2)
        win.nextRow()
        p3 = win.addPlot(title='<p><font size="4">Accepted Costs</font></p>', colspan=4)
        curve3 = p3.plot()
        line3 = p3.plot(pen=pg.mkPen('r'))
        scatter3 = pg.ScatterPlotItem()
        p3.addItem(scatter3)
        win.nextRow()
        p4 = win.addPlot(title='<p><font size="4">Temperature</font></p>', colspan=4)
        p4.setLogMode(y=True)
        curve4 = p4.plot()
        line4 = p4.plot(pen=pg.mkPen('r'))  

        last_time = time.time()
        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(int(argv['--timer-interval']))  

        win.show()
        app.exec_()
    else:
        while True:
            spi_annealing.run(N=int(argv['--update-period']))
            record['current-model'] = spi_annealing.solution
            record['current-intensity'] = spi_annealing.cal_intensity
            time.sleep(float(argv['--timer-interval']) / 1000)
            print_with_timestamp('step: %d, cost: %.4e' %(spi_annealing.total_iter, spi_annealing.cost))
            if spi_annealing.total_iter >= max_iter:
                break

    record_path = os.path.join(output_dir, 'record.h5')
    dict2h5(record, record_path)
    print_with_timestamp('record saved into %s' %record_path)
    logging.debug('End of programe')