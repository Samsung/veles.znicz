import units
import os
import sys

class ctask(units.Unit):

    def __init__(self, v, unpickling=0):
        super(ctask, self).__init__(unpickling=unpickling)
        # self.krn_ = None
        if unpickling:
            return
        self.param = v
        # self.__dict__.update(v)


    def test1(self):

        if 'NUMBER' not in self.param:
            self.log().error("in taskhave not NUMBER")
            return -1
        if 'models_train' not in self.param:
            self.log().error("in task have not models_train")
            return -2
        if 'data_set' not in self.param:
            self.log().error("in task have not data_set")
            return -3
        if type(self.param['data_set']) != dict:
            self.log().error("in task data_set is not dict")
            return -4
        if type(self.param['models_train']) != dict:
            self.log().error("in task model_train is not dict")
            return -5
        _t = self.param['models_train']
        if 'model0' not in _t:
            self.log().error("in task have not model_train")
            return -6
        if type(_t['model0']) != dict:
            self.log().error("in task  in model_train  model0 is not dict")
            return -7
        _tt = _t['model0']
        if 'data_model' not in _tt:
            self.log().error("in task have not model_train")
            return -8
        if type(_tt['data_model']) != str:
            self.log().error("in task  in model_train.model0.data_model is not str")
            return -9

        _t = self.param['data_set']
        if 'data' not in _t:
            self.log().error("in task  data_set.data  not")
            return -10
        if type(_t['data']) != str:
            self.log().error("in task  in data_set.data is not str")
            return -11
        return 1


    def read_config_task(self):
        _t = self.param['models_train']
        _tt = _t['model0']
        print(_tt['data_model'])
        self.str_data_model = _tt['data_model']

        self.data_set = self.param['data_set']
        self.str_data = self.data_set['data']

        return 1

    def read_modelWF(self):

        _temp = os.path.split(self.str_data_model)
        # isabs abspath
        name_file = _temp[1].split(".py")
        name = name_file[0]
        directory = _temp[0]
        if directory not in sys.path:
            sys.path.insert(0, directory)
            # print(sys.path)
        self._wf = None
        try:
            # print(name)
            self.nameWF = name
            # print(sys.path)
            self._wf = __import__(name)
            # print(self._wf)
            # print(dir(self._wf))
            if name in self._wf.__dict__:
                self.class_wf = self._wf.__getattribute__(name)
                # print(self.class_wf)
                if not type(self.class_wf) is self.class_wf.__class__:
                    self.log().error("in ", self.str_data_model, " object " , name, " is not class")
                    return -3
            else:
                print("ERRoR!!!!   ", self.str_data_model, " dont have   object " , name, ")\n")
                return -2
        except:
            print("\n don read  model WF in ", self.str_data_model, "\n")
            self._wf = None
            self.error = -10
            return -1

        return 1

    def read_config_data(self):


            _temp = os.path.split(self.str_data)
            # print(self.str_data)
            # print(type(self.data_set))
            # print(self.data_set)
            # isabs abspath
            name_file = _temp[1].split(".py")
            name = name_file[0]
            directory = _temp[0]
            # print(name, " ",directory)
            if directory not in sys.path:
                sys.path.insert(0, directory)
            # print(sys.path)
            # print(name)
            self._data_module = None
            try:
                # print(name)
                self._data_module = __import__(name)
                # print(self._data_module)
                # print(dir(self._data_module))
                if name in self._data_module.__dict__:
                    self.data_set['data_dict'] = {}
                    self.data_set['data_dict'] = self._data_module.__getattribute__(name)
                    if type(self.data_set['data_dict']) != dict:
                        self.log().error("in ", self.str_data, " object " , name, " is not dict")
                        return -3
                else:
                    print("ERRoR!!!!   ", self.str_data, " dont have   object " , name, ")\n")
                    return -2
            except:
                print("\n dont read file with config data ", self.str_data, "\n")
                self._data_module = None
                self.error = -11
                return -1

            return 1

    def run_model_WF(self):

        # print(self.class_wf)
        # print(dir(self.class_wf))
        self.WF = None
        try:
            # print(self.data_set)


            self.WF = self.class_wf(data_set=self.data_set, param=self.param, cpu=True)
            print(self.WF)
        except:
            self.WF = None

            print("ERROR  WF not create")
            raise
            return -1

        print("OK run_model_WF create")

        # global_alpha = 0.9, global_lambda = 0.0, threshold = 1.0,
        # надо добавить чтение параметров обучени из конфигов
        # TODO(v.markovtsev): What the hell?! Russian comments are strongly prohibited!
        self.WF.run()

        return 1

    def run(self):
        print(" RUN task")

        r = self.test1()
        if r < 0:
            print("Error in task = ", r)
            return 1

        r = 0
        r = self.read_config_task()
        if(r < 0):
            self.log().error("in task", self.__getattribute__('NUMBER'))
            return 2

        r = 0
        r = self.read_modelWF()
        print("                    WF ", r)
        if(r < 0):
            self.log().error("not read model WF ")
            return 3

        r = 0
        r = self.read_config_data()
        print("                    RD ", r)
        if(r < 0):
            self.log().error("not read config data")
            return 4

        print("OK task run")

        r = 0;

        r = self.run_model_WF()
        print("                    Run WF ", r)





