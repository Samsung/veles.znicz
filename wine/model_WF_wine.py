'''
Created on Apr 19, 2013
model use always dataset in the txt format 

(data parameters is  in the  separate configuration file data.
 need to write a module configuration file read data
 need to write a reader txt files in any format.
)

Example wine data set -13 paramets input.  3 class outputs / 178 samples( all dataset and train and test)
Fixed structure NN (5[Tanh]-3[SoftMax])

All train set is in the train-batch BP methods
(simple)  (type of operation (surgery or criterion) must be specified in the configuration file jobs)

Criterion of learning is to remember all. (-"-)
  
Result is in the serialize (-"-)

no testing for test data set. (-"-)

@author: Seresov Denis <d.seresov@samsung.com>

'''
import units
import opencl
import text
import all2all
import evaluator
import gd
import repeater
import units_end_point
import rnd
import numpy

def strf(x):
    return "%.4f" % (x, )

class model_WF_wine(units.SmartPickling):
    """UUseCaseTxt.

    Attributes:
        device_list: list of an OpenCL devices as DeviceList object.
        start_point: Filter.
        end_point: EndPoint.
        t: t.
    """
    def __init__(self,data_set={},param ={},cpu = True, unpickling = 0):
        super(model_WF_wine, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        self.data_set =data_set
        self.param=param
        
        #rnd.default.seed(numpy.fromfile("seed", numpy.integer, 1024))
        numpy.random.seed(numpy.fromfile("seed", numpy.integer))# сделать считывание из конфига 
        dev = None
        if not cpu:
            self.device_list = opencl.DeviceList()
            dev = self.device_list.get_device()

        # Setup notification flow
        self.start_point = units.Unit()

        
        # print(self.config_data_seta)
        # print(self.config_datasa)
        #print(self.data_set)
        t = text.TXTLoader(self.data_set,self.param)
        self.t = t
        #sys.exit()
        print("1")
        print(t)
        t.link_from(self.start_point)
        print("2")

        rpt = repeater.Repeater()
        rpt.link_from(t)

        aa1 = all2all.All2AllTanh(output_shape=[70], device=dev)
        aa1.input = t.output2
        aa1.link_from(rpt)
        
        out = all2all.All2AllSoftmax(output_shape=[3], device=dev)
        out.input = aa1.output
        out.link_from(aa1)


        ev = evaluator.EvaluatorSoftmax2(device=dev )
        ev.y = out.output
        ev.labels = t.labels
        ev.params=t.params
        ev.TrainIndex = t.TrainIndex
        ev.ValidIndex = t.ValidIndex
        ev.TestIndex  = t.TestIndex
        #ev.Index=t.Index
        ev.link_from(out)
        
        """
        ev = evaluator.EvaluatorMSE(device=dev)
        ev.y = out.output
        ev.labels = t.labels
        ev.params=t.params
        ev.TrainIndex = t.TrainIndex
        ev.ValidIndex = t.ValidIndex
        ev.TestIndex  = t.TestIndex
        ev.Index=t.Index
        ev.link_from(out)
        """

        gdsm = gd.GDSM(device=dev )
        gdsm.weights = out.weights
        gdsm.bias = out.bias
        gdsm.h = out.input
        gdsm.y = out.output
        gdsm.L = ev.L
        gdsm.err_y = ev.err_y

        gd1 = gd.GDTanh(device=dev)
        gd1.weights = aa1.weights
        gd1.bias = aa1.bias
        gd1.h = aa1.input
        gd1.y = aa1.output
        gd1.err_y = gdsm.err_h
        gd1.L = ev.L
        gd1.link_from(gdsm)

        rpt.link_from(gd1)

        self.end_point = units_end_point.EndPoint(self, self.do_log, (out, gdsm, gd1))
        self.end_point.status = ev.status
        self.end_point.link_from(ev)
        gdsm.link_from(self.end_point)
        
        self.ev = ev  
        self.sm = out          #?
        self.gdsm = gdsm       #?
        self.gd1 = gd1         #?   
        print("ok_init ")
        
    def do_log(self, out, gdsm, gd1):
        return
        flog = open("logs/out.log", "a")
        flog.write("Iteration %d" % (self.end_point.n_passes, ))
        flog.write("\nSoftMax layer input:\n")
        for sample in out.input.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nSoftMax layer output:\n")
        for sample in out.output.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nSoftMax layer weights:\n")
        for sample in out.weights.v:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nSoftMax layer bias:\n")
        flog.write(" ".join(strf(x) for x in out.bias.v))
        flog.write("\n(min, max)(input, output, weights, bias) = ((%f, %f), (%f, %f), (%f, %f), (%f, %f)\n" % \
                   (out.input.batch.min(), out.input.batch.max(), \
                    out.output.batch.min(), out.output.batch.max(), \
                    out.weights.v.min(), out.weights.v.max(), \
                    out.bias.v.min(), out.bias.v.max()))
        flog.write("\n")
        flog.close()

        flog = open("logs/gdsm.log", "a")
        flog.write("Iteration %d" % (self.end_point.n_passes, ))
        flog.write("\nGD SoftMax err_y:\n")
        for sample in gdsm.err_y.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD SoftMax err_h:\n")
        for sample in gdsm.err_h.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD SoftMax weights:\n")
        for sample in gdsm.weights.v:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD SoftMax bias:\n")
        flog.write(" ".join(strf(x) for x in gdsm.bias.v))
        flog.write("\n(min, max)(err_y, err_h, weights, bias) = ((%f, %f), (%f, %f), (%f, %f), (%f, %f)\n" % \
                   (gdsm.err_y.batch.min(), gdsm.err_y.batch.max(), \
                    gdsm.err_h.batch.min(), gdsm.err_h.batch.max(), \
                    gdsm.weights.v.min(), gdsm.weights.v.max(), \
                    gdsm.bias.v.min(), gdsm.bias.v.max()))
        flog.write("\n")
        flog.close()
       
        
        flog = open("logs/gd1.log", "a")
        flog.write("Iteration %d" % (self.end_point.n_passes, ))
        flog.write("\nGD1 err_y:\n")
        for sample in gd1.err_y.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD1 err_h:\n")
        for sample in gd1.err_h.batch:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD1 weights:\n")
        for sample in gd1.weights.v:
            flog.write(" ".join(strf(x) for x in sample))
            flog.write("\n")
        flog.write("\nGD1 bias:\n")
        flog.write(" ".join(strf(x) for x in gd1.bias.v))
        flog.write("\n(min, max)(err_y, err_h, weights, bias) = ((%f, %f), (%f, %f), (%f, %f), (%f, %f)\n" % \
                   (gd1.err_y.batch.min(), gd1.err_y.batch.max(), \
                    gd1.err_h.batch.min(), gd1.err_h.batch.max(), \
                    gd1.weights.v.min(), gd1.weights.v.max(), \
                    gd1.bias.v.min(), gd1.bias.v.max()))
        flog.write("\n")
        flog.close()
        
          
   
    def run(self):
        
        _t =self.param['train_param']   # считаю что структуру param с параметрами прочли или она сериализована
        print(_t)
        print(" WF WINE RUN START")        
        # Start the process:
        self.sm.threshold = _t['threshold']
        self.sm.threshold_low = _t['threshold_low']
        self.ev.threshold = _t['threshold']
        self.ev.threshold_low = _t['threshold_low']
        self.gdsm.global_alpha = _t['global_alpha']
        self.gdsm.global_lambda = _t['global_lambda']
        self.gd1.global_alpha = _t['global_alpha']
        self.gd1.global_lambda = _t['global_lambda']
        print()
        print("Initializing...")
        self.start_point.initialize_dependent()
        self.end_point.wait()
        #for l in self.t.labels.batch:
        #    print(l)
        #sys.exit()
        print()
        print("Running...")
        self.start_point.run_dependent()
        self.end_point.wait()
