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
            _thread.start_new_thread(self.run, ())

    def run(self):
        """Creates and runs main graphics window.
        Note that this function should be called only by __init__()
        """
        self.run_lock = _thread.allocate_lock()
        self.run_lock.acquire()
        self.root = tkinter.Tk()
        self.root.wm_title("Znicz")

        # Creating figures for plotting
        self.figure = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
        self.registered_plotters = {}
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
            self.figure, self.root)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP,
                                         fill=tkinter.BOTH,
                                         expand=1)
        self.root.after(100, self.update)
        tkinter.mainloop()  # Wait for user to close the window
        self.run_lock.release()

    def process_event(self, event):
        """Processes scheduled event.
           Event should be an instance of ScheduledEvent
        """
        if event["event_type"] == "register_plot":
            axes_label = event["axes_label"]
            plotter_id = event["plotter_id"]
            cur_axes = None
            existing_axes = {}
            for plotter in self.registered_plotters.values():
                existing_axes[plotter["axes_label"]] = plotter["axes"]
            if not axes_label in existing_axes:
                #  First, reposition existing axes
                for axes in existing_axes.values():
                    relative_position = axes.get_geometry()[2];
                    axes.change_geometry(1, len(existing_axes) + 1,
                                         relative_position)

                #  And then add a new one
                existing_axes[axes_label] = self.figure.add_subplot(1,
                    len(existing_axes) + 1, len(existing_axes) + 1)
                existing_axes[axes_label].set_title(axes_label)

            cur_axes = existing_axes[axes_label]
            self.registered_plotters[plotter_id] = {
                "axes_label": axes_label,
                "axes": cur_axes,
                "plot_style": event["plot_style"],
                }
        elif event["event_type"] == "update_plot":
            plotter = self.registered_plotters[event["plotter_id"]]
            axes = plotter["axes"]
            axes.plot(event["new_values"], plotter["plot_style"])
            self.canvas.show()

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
        self.run_lock.acquire()
        self.run_lock.release()
        print("Done")


class SimplePlotter(units.OpenCLUnit):
    """ Plotter for given values

    Attributes:
        values: history of all parameter values given to plotter.
        input: connector to take values from
        input_field: name of field in input we want to plot
        axes_label: label of axes used for drawing. If two ploters share
                    the same axes_label, their plots will appear together.
        plot_style: Style of lines used for plotting. See
                    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
                    for reference.
    """
    def __init__(self, axes_label,
                plot_style="k-",
                device=None,
                unpickling=0):
        super(SimplePlotter, self).__init__(unpickling=unpickling,
                                            device=device)
        self.values = list()
        self.input = None  # Connector
        self.input_field = None
        self.axes_label = axes_label
        self.plot_style = plot_style
        register_event = {"event_type": "register_plot",
                          "plotter_id": id(self),
                          "axes_label": self.axes_label,
                          "plot_style": self.plot_style
                          }
        Graphics().event_queue.put(register_event, block=True)
        if unpickling:
            return

    def cpu_run(self):
        value = self.input.__dict__[self.input_field]
        self.values.append(value)
        update_event = {"event_type": "update_plot",
                        "plotter_id": id(self),
                        "new_values": self.values,
                        }
        Graphics().event_queue.put(update_event, block=True)
