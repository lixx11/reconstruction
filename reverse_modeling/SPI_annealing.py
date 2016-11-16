from annealing import Annealing 
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import sys
import time
from scipy.stats import linregress
import random
import math
from util import *
from scipy.stats import pearsonr


class SPIAnnealing(object):
    """docstring for SPIAnnealing"""
    def __init__(self, init_T=1.0E5, inner_cooling_factor=0.99, outer_cooling_factor=0.5, batch_size=1E3, init_solution=None, init_step_size=1.0, step_shrink_factor=0.5,\
                 intensity_data=None, mask=None, center=None):
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
        self.intensity_data = intensity_data
        self.mask = mask
        self.center = center

        self.cost = self.calc_cost(init_solution)
        self.total_iter = 0
        self.best_solution = init_solution
        self.best_cost = self.calc_cost(init_solution)
        self.restart_list = []

    def neighbor(self, old_solution):
        edge = get_edge(self.solution, width=1, find_edge='both')
        np.save('edge.npy', edge)
        edge_y, edge_x = np.where(edge)
        edge_xy = np.asarray([edge_x, edge_y]).T 
        print edge_xy.shape
        N_mutant = int(self.step_size)
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
                # print dists.min()
                if dists.min() > 1.:
                    mutant_ids.append(rn)
                    count += 1
        new_solution = self.solution.copy()
        for n in xrange(N_mutant):
            y, x = edge_xy[mutant_ids[n], :]
            if self.solution[y, x] > 0:
                new_solution[y, x] = 0
            else:
                new_solution[y, x] = 1
        return new_solution

    def calc_cost(self, solution):
        this_intensity = np.abs(np.fft.fft2(solution))**2.
        scaling_factor, _, _, _, _ = linregress(self.intensity_data.reshape(-1), this_intensity.reshape(-1))
        diff = scaling_factor * this_intensity - self.intensity_data
        # error = np.linalg.norm(diff, ord='fro')
        score, _ = pearsonr(this_intensity.reshape(-1), self.intensity_data.reshape(-1))
        error = 1 - score
        return error

    def acceptance_probability(self, old_cost, new_cost, T):
        _exp = ((old_cost - new_cost) / T)
        if _exp > 0:
            return 1.
        else:
            return math.exp(_exp)
            # return 0.

    def run(self, max_iter=50):
        global costs, accepted_costs, Ts
        for n in xrange(max_iter):
            if self.total_iter % self.batch_size == 0:  # restart
                self.restart_list.append(self.best_solution)
                self.solution = self.best_solution
                self.cost = self.calc_cost(self.solution)
                self.T = self.init_T * math.pow(self.outer_cooling_factor, len(self.restart_list))
                costs.append(self.cost)
                accepted_costs.append(self.cost)
                Ts.append(self.T)
            new_solution = self.neighbor(self.solution)
            new_cost = self.calc_cost(new_solution)
            if new_cost < self.best_cost:
                self.best_cost = new_cost
                self.best_solution = new_solution
            ap = self.acceptance_probability(self.cost, new_cost, self.T)
            if ap > random.uniform(0, 1):
                print("updating at step %d, cost: %.4E -> %.4E"\
                      %(self.total_iter, self.cost, new_cost))
                self.solution = new_solution
                self.cost = new_cost
            self.T *= self.inner_cooling_factor
            costs.append(new_cost)
            accepted_costs.append(self.cost)
            Ts.append(self.T)
            self.total_iter += 1


def update():
    global last_time, spi_annealing
    now = time.time()
    dt = now - last_time
    last_time = now 
    fps = 1.0 / dt 
    print('fps: %.2f' %fps)

    # update plot
    spi_annealing.run(max_iter=10)
    im3.setImage(spi_annealing.solution[model_range[0]:model_range[1], model_range[0]:model_range[1]])
    current_intensity = np.abs(np.fft.fft2(spi_annealing.solution))**2.
    im4.setImage(np.log(np.abs(np.fft.fftshift(current_intensity)+1.)))

    curve2.setData(costs)
    line2.setData(np.ones_like(costs) * costs[-1])
    curve3.setData(accepted_costs)
    line3.setData(np.ones_like(accepted_costs) * accepted_costs[-1])
    curve4.setData(Ts)
    line4.setData(np.ones_like(Ts) * Ts[-1])


costs = []
accepted_costs = []
Ts = []

if __name__ == '__main__':
    # add signal to enable CTRL-C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # generate test data
    oversampling_ratio = 5
    model_size = 45
    space_size = model_size * oversampling_ratio
    mask_size = 0
    model = load_model('data/coreshell-proj.npy', model_size=model_size, space_size=space_size)
    # model = make_model(model_size=model_size, space_size=space_size)
    model_range = [space_size//2 - model_size//2 - 10, space_size//2 + model_size//2 + 10]
    mask = make_square_mask(space_size, mask_size)
    intensity = np.abs(np.fft.fft2(model))**2.

    # generate init model
    init_model_size = 43
    init_model = make_model(model_size=init_model_size, space_size=space_size)

    spi_annealing = SPIAnnealing(init_solution=init_model, intensity_data=intensity, init_step_size=1, init_T=1.0E3)

    app = QtGui.QApplication([])
    win = pg.GraphicsWindow('SPI Annealing')

    p11 = win.addPlot(title='<p><font size="4">Model</font></p>')
    im1 = pg.ImageItem()
    p11.addItem(im1)
    im1.setImage(model[model_range[0]:model_range[1], model_range[0]:model_range[1]])
    p11.getViewBox().setAspectLocked(True)

    p12 = win.addPlot(title='<p><font size="4">Intensity</font></p>')
    im2 = pg.ImageItem()
    p12.addItem(im2)
    im2.setImage(np.log(np.abs(np.fft.fftshift(intensity)+2.)))
    p12.getViewBox().setAspectLocked(True)

    p13 = win.addPlot(title='<p><font size="4">Current Model</font></p>')
    im3 = pg.ImageItem()
    p13.addItem(im3)
    im3.setImage(init_model[model_range[0]:model_range[1], model_range[0]:model_range[1]])
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
    win.nextRow()
    p3 = win.addPlot(title='<p><font size="4">Accepted Costs</font></p>', colspan=4)
    curve3 = p3.plot()
    line3 = p3.plot(pen=pg.mkPen('r'))
    win.nextRow()
    p4 = win.addPlot(title='<p><font size="4">Temperature</font></p>', colspan=4)
    p4.setLogMode(y=True)
    curve4 = p4.plot()
    line4 = p4.plot(pen=pg.mkPen('r'))

    last_time = time.time()
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)

    win.show()
    app.exec_()
