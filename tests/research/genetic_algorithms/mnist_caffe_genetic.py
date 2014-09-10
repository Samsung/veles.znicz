import os
import veles.znicz.tests.research.genetic_algorithms.mnist_genetic as mg


class CaffeChromo(mg.MnistChromo):
    def fit(self, *args):
        folder_num = 0 if len(args) == 0 else args[0]
        self.create_config(folder_num)
        os.system("scripts/velescli.py -v error -s -d 1:0 "
                  "veles/znicz/tests/research/mnist.py " +
                  str(folder_num) + "/new_mnist_config.py "
                  "-r veles/znicz/tests/research/seed:1024:int32")
        min_error = 100.0
        for _root, _dir, files in os.walk(str(folder_num)):
            for file in files:
                if "pt.4.pickle" in file:
                    print(file)
                    error = float(file[len("mnist_caffe_"):
                                       len(file) - len("pt.4.pickle")])
                    os.system("rm " + str(folder_num) + "/" + file)
                    if error < min_error:
                        min_error = error
                elif file == "mnist_caffe_current.4.pickle":
                    os.system("rm " + str(folder_num) + "/" + file)
        if min_error < 0.1:
            l = 0
        return 100 - min_error

    def create_config(self, folder_num):
        try:
            os.mkdir("%s" % str(folder_num))
        except:
            pass
        config = open("../Veles/" + str(folder_num) + "/new_mnist_config.py",
                      "w")
        template = open("mnist_caffe_genetic.template", 'r').read()
        config.write(template % (folder_num,) + tuple(self.numeric))


class CaffeGenetic(mg.MnistGenetic):
    def new_chromo(self, n, min_x, max_x, accuracy, codes,
                   binary=None, numeric=None, *args):
        chromo = CaffeChromo(n, min_x, max_x, accuracy, codes,
                             binary, numeric, *args)
        return chromo

if __name__ == "__main__":
    CaffeGenetic().evolution()
