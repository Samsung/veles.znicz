"""
Created on May 17, 2013

@author: Kumok Akim <a.kumok@samsung.com>
"""
import matplotlib
import matplotlib.pyplot
import matplotlib.widgets
import units
import tkinter
import _thread
import queue

matplotlib.pyplot.ioff()


class Graphics:
    """ Class handling all interaction with main graphics window
        NOTE: This class should be created ONLY within one thread
        (preferably main)

    Attributes:
        _instance: instance of MainGraphicsData class. Used for implementing
            Singleton pattern for this class.
        root: TKinter graphics root.
        event_queue: Queue of all pending changes created by other threads.
        run_lock: Lock to determine whether graphics window is running
        figure: Figure containing canvas and axes
        canvas : Canvas for drawing plots
        axes: Axes of plot
        is_initialized: whether this class was already initialized.
    """

    _instance = None
    root = None
    event_queue = None
    run_lock = None
    figure = None
    canvas = None
    axes = None
    is_initialized = False


    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Graphics, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def run(self):
        self.run_lock = _thread.allocate_lock()
        self.run_lock.acquire()
        self.root = tkinter.Tk()
        self.root.wm_title("Znicz")

        # Creating axes for plotting
        self.figure = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
            self.figure, self.root)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP,
                                         fill=tkinter.BOTH,
                                         expand=1)
        self.root.after(100, self.update)
        tkinter.mainloop()  # Wait for user to close the window
        self.run_lock.release()

    def update(self):
        try:
            while True:
                event = self.event_queue.get_nowait()
                self.axes.plot(event, "r")
                self.canvas.show()
        except queue.Empty:
            pass
        self.root.after(100, self.update)

    def initialize(self):
        if not self.is_initialized:
            self.is_initialized = True
            self.event_queue = queue.Queue()
            self.initialize_lock = _thread.allocate_lock()
            _thread.start_new_thread(self.run, ())

    def wait_finish(self):
        print("Waiting for user to close the window...")
        self.run_lock.acquire()
        self.run_lock.release()


class SimplePlotter(units.OpenCLUnit):
    """ Plotter for given values

    Attributes:
        event_queue_: queue of events scheduled for plotting
        values: history of all parameter values given to plotter.
        input: connector to take values from
        input_field: name of field in input we want to plot

    """
    def __init__(self, graphics, device=None, unpickling=0):
        super(SimplePlotter, self).__init__(unpickling=unpickling,
                                            device=device)
        self.event_queue_ = graphics.event_queue
        self.values = list()
        self.input = None  # Connector
        self.input_field = None
        if unpickling:
            return

    def cpu_run(self):
        value = self.input.__dict__[self.input_field]
        self.values.append(value)
        self.event_queue_.put(self.values, block=True)
