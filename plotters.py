"""
Created on May 17, 2013

@author: Kumok Akim <a.kumok@samsung.com>
"""
import matplotlib.pyplot
import units
import tkinter
import _thread
import queue


matplotlib.pyplot.ion()


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
        registered_plotters: List of registered plotters
        is_initialized: whether this class was already initialized.
    """

    _instance = None
    root = None
    event_queue = None
    run_lock = None
    registered_plotters = None
    is_initialized = False

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Graphics, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self.is_initialized:
            self.is_initialized = True
            self.event_queue = queue.Queue()
            self.initialize_lock = _thread.allocate_lock()
            self.registered_plotters = {}
            _thread.start_new_thread(self.run, ())

    def run(self):
        """Creates and runs main graphics window.
        Note that this function should be called only by __init__()
        """
        self.run_lock = _thread.allocate_lock()
        self.run_lock.acquire()
        self.root = tkinter.Tk()
        self.root.withdraw()
        self.root.after(100, self.update)
        tkinter.mainloop()  # Wait for user to close the window
        self.run_lock.release()

    def process_event(self, event):
        """Processes scheduled event.
           Event should be an instance of ScheduledEvent
        """
        if event["event_type"] == "register_plot":
            figure_label = event["figure_label"]
            plotter_id = event["plotter_id"]
            self.registered_plotters[plotter_id] = {
                "figure_label": figure_label,
                "plot_style": event["plot_style"],
                }
        elif event["event_type"] == "update_plot":
            plotter = self.registered_plotters[event["plotter_id"]]
            figure_label = plotter["figure_label"]
            figure = matplotlib.pyplot.figure(figure_label)
            axes = figure.add_subplot(111)  # Main axes
            axes.plot(event["new_values"], plotter["plot_style"])
            figure.show()

    def update(self):
        """Processes all events scheduled for plotting
        """
        try:
            while True:
                event = self.event_queue.get_nowait()
                self.process_event(event)
        except queue.Empty:
            pass
        self.root.after(100, self.update)

    def wait_finish(self):
        """Waits for user to close the window.
        """
        print("Waiting for user to close the window...")
        self.root.destroy()
        self.run_lock.acquire()
        self.run_lock.release()
        print("Done")


class SimplePlotter(units.OpenCLUnit):
    """ Plotter for given values

    Attributes:
        values: history of all parameter values given to plotter.
        input: connector to take values from
        input_field: name of field in input we want to plot
        figure_label: label of figure used for drawing. If two ploters share
                      the same figure_label, their plots will appear together.
        plot_style: Style of lines used for plotting. See
            http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
            for reference.
    """
    def __init__(self, figure_label="num errors",
                plot_style="k-",
                device=None,
                unpickling=0):
        super(SimplePlotter, self).__init__(unpickling=unpickling,
                                            device=device)
        if unpickling:
            register_event = {"event_type": "register_plot",
                          "plotter_id": id(self),
                          "figure_label": self.figure_label,
                          "plot_style": self.plot_style
                          }
            Graphics().event_queue.put(register_event, block=True)
            return
        self.values = list()
        self.input = None  # Connector
        self.input_field = None
        self.figure_label = figure_label
        self.plot_style = plot_style
        register_event = {"event_type": "register_plot",
                          "plotter_id": id(self),
                          "figure_label": self.figure_label,
                          "plot_style": self.plot_style
                          }
        Graphics().event_queue.put(register_event, block=True)

    def cpu_run(self):
        if type(self.input_field) == dict:
            value = self.input.__dict__[self.input_field]
        elif type(self.input_field) == int:
            if self.input_field < 0 or self.input_field >= len(self.input):
                return
            value = self.input[self.input_field]
        else:
            value = self.input
        self.values.append(value)
        update_event = {"event_type": "update_plot",
                        "plotter_id": id(self),
                        "new_values": self.values,
                        }
        Graphics().event_queue.put(update_event, block=True)
