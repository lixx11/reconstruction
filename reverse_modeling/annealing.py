from abc import ABCMeta, abstractmethod
import random
import math
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import time
import sys


class Annealing(object):
    """docstring for Annealing"""
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def neighbor(self):
        pass

    @abstractmethod
    def calc_cost(self):
        pass
        
    @abstractmethod
    def acceptance_probability(self):
        pass

    @abstractmethod
    def run(self):
        pass


class SimpleAnnealing(Annealing):
    """docstring for SimpleAnnealing"""
    def __init__(self, init_T=1.0E3, inner_cooling_factor=0.99, outer_cooling_facter=0.5, batch_size=1E3,
                 init_solution=10., init_step_size=1.0, step_shrink_factor=0.5):
        super(SimpleAnnealing, self).__init__()
        self.init_T = init_T
        self.T = init_T
        self.inner_cooling_factor = inner_cooling_factor
        self.outer_cooling_facter = outer_cooling_facter
        self.init_solution = init_solution
        self.solution = init_solution 
        self.init_step_size = init_step_size
        self.step_size = init_step_size
        self.step_shrink_factor = step_shrink_factor
        self.batch_size = batch_size
        self.cost = self.calc_cost(init_solution)
        self.total_iter = 0
        self.best_solution = init_solution
        self.best_cost = self.calc_cost(init_solution)
        self.restart_list = []

    def neighbor(self, old_solution):
        neighbor = old_solution + random.uniform(-self.step_size, self.step_size)
        return neighbor

    def calc_cost(self, solution):
        cost = -scale_factor * math.cos(2. * math.pi * solution) * math.exp(-abs(solution)/1.)
        return cost

    def acceptance_probability(self, old_cost, new_cost, T):
        _exp = ((old_cost - new_cost) / T)
        if _exp > 0:
            return 1.
        else:
            return math.exp(_exp)

    def init_parm(self):
        self.total_iter = 0
        self.T = self.init_T
        self.solution = self.init_solution
        self.cast = self.calc_cost(self.solution)
        self.best_solution = self.solution
        self.best_cost = self.cost
        self.restart_list = []
        self.step_size = self.init_step_size

    def run(self, max_iter=1E3):
        n_iter = 0
        while n_iter < max_iter:
            if self.total_iter % self.batch_size == 0:   # restart
                self.restart_list.append(self.best_solution)
                self.solution = self.best_solution
                self.cost = self.calc_cost(self.solution)
                self.T = self.init_T * math.pow(self.outer_cooling_facter, len(self.restart_list)-1)
                self.step_size *= self.step_shrink_factor
                global costs, accepted_costs
                costs.append(self.cost)
                accepted_costs.append(self.cost)
                Ts.append(self.T)
                print("\n==============restart at step %d=================" %self.total_iter)
                print("best solution: %.2f with cost of %.2E; step size: %.2f, T: %.2f" \
                      %(self.best_solution, self.best_cost, self.step_size, self.T))
            # print("while loop")
            new_solution = self.neighbor(self.solution)
            new_cost = self.calc_cost(new_solution)
            if new_cost < self.best_cost:
                self.best_cost = new_cost
                self.best_solution = new_solution
            ap = self.acceptance_probability(self.cost, new_cost, self.T)
            # print(ap)
            if ap > random.uniform(0, 1):
                # print("updating at step %d: %.2f -> %.2f with cost: %.4E -> %.4E"\
                #       %(self.total_iter, self.solution, new_solution, self.cost, new_cost))
                self.solution = new_solution
                self.cost = new_cost
            self.T *= self.inner_cooling_factor
            global costs, accepted_costs
            costs.append(new_cost)
            accepted_costs.append(self.cost)
            Ts.append(self.T)
            self.total_iter += 1
            n_iter += 1
            # print(self.__str__())
            print("\r step: %d" %self.total_iter),

    def rerun(self, max_iter=1E3):
        self.init_parm()
        self.run(max_iter=max_iter)

    def __str__(self):
        return "step: %d, solution: %.2f, cost: %.2f, T: %.2E" \
               %(self.total_iter, self.solution, self.cost, self.T)


def update():
    global curvePoint, index, lastTime, fps, SA, costs, curve2
    now = time.time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    print("fps: %d" %fps)
    SA.run(max_iter=10)
    solution, cast = SA.solution, SA.cost
    if solution < x.min():
        _x = x.min()
    elif solution > x.max():
        _x = x.max()
    else:
        _x = solution
    index = (_x - x_start) / x_step
    curvePoint.setPos(float(index)/(len(x)-1))
    text2.setText('[%0.1f, %0.1f]' % (x[index], y[index]))

    # update plot2,3,4
    curve2.setData(np.asarray(costs))
    curve3.setData(np.asarray(accepted_costs))
    curve4.setData(np.asarray(Ts))


costs = []  # store all searched costs 
accepted_costs = []  # store all accepted costs
Ts = []  # store all Ts
if __name__ == '__main__':
    x_start = -10
    x_end = 10
    x_step = 0.01
    scale_factor = 1.
    x = np.arange(x_start, x_end, x_step)
    y = -scale_factor * np.cos(2. * math.pi * x) * np.exp(-abs(x)/1.)

    ## init annealing
    SA = SimpleAnnealing(init_T=1.E2, inner_cooling_factor=0.99, outer_cooling_facter=0.5, batch_size=1E3,
                         init_solution=1.E1, init_step_size = 1.E1, step_shrink_factor=0.5,)

    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.setWindowTitle('pyqtgraph example: Plotting')

    plot1 = win.addPlot(title="Simulated Annealing")
    plot1.setYRange(-1.5*scale_factor, 1.5*scale_factor)
    plot1.setWindowTitle('SimpleAnnealing')
    curve = plot1.plot(x,y)  ## add a single curve

    ## Create text object, use HTML tags to specify color/size
    text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">This is the</span><br><span style="color: #FF0; font-size: 16pt;">global minima</span></div>', anchor=(-0.3,1.3), border='w', fill=(0, 0, 255, 100))
    plot1.addItem(text)
    text.setPos(0, y.min())

    ## Draw an arrowhead next to the text box
    arrow = pg.ArrowItem(pos=(0, y.min()), angle=-45)
    plot1.addItem(arrow)

    ## Set up an animated arrow and text that track the curve
    curvePoint = pg.CurvePoint(curve)
    plot1.addItem(curvePoint)
    text2 = pg.TextItem("test", anchor=(0.5, -1.0))
    text2.setParentItem(curvePoint)
    arrow2 = pg.ArrowItem(angle=90)
    arrow2.setParentItem(curvePoint)

    ## Add plot2,3,4 to win
    win.nextRow()
    plot2 = win.addPlot(title="Searched Costs")
    curve2 = plot2.plot()
    win.nextRow()
    plot3 = win.addPlot(title="Accepted Costs")
    curve3 = plot3.plot()
    win.nextRow()
    plot4 = win.addPlot(title="Temperature")
    curve4 = plot4.plot()
    plot4.setLogMode(y=True)

    ## update position
    index = 0
    lastTime = time.time()
    fps = None

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()