'''    
Created on Maj 14, 2013

@author: Seresov Denis <d.seresov@samsung.com>
'''

import sys
import os
import tasks


class Veles(object):
    """Propagates notification if any of the inputs are active.
    """
    def __init__(self):
        self.config_veles=None        
        self.Tasks =tasks.Tasks()
            
    def load_snapshot(self):
        """
        reconstruction of snapshot  
        """
        pass
    
    def reset(self):
        self.config_veles=None
        
        
    
    def read_config(self,config_file_veles):
        self.config_file = config_file_veles 
        
        temp= os.path.split(self.config_file)
        # isabs abspath
        name_file= temp[1].split(".py")
        name=name_file[0]
        directory=temp[0]
        self.error=0
        _t=None
        if directory not in sys.path: 
                        sys.path.insert(0,directory)
        try:
            _t=__import__(name)
        except:
            _t=None
            print("\n read not config module\n")
            self.error =-2000 
            return -2
        if _t != None:
            self.config_veles=None
            print('veles' in dir(_t))
            if 'veles' in dir(_t):        
                self.config_veles=_t.__getattribute__('veles')
            else:
                self.config_veles=None
                print("\n config module have not veles \n")
                self.error =-1000
                return -1
         
        #print("\n read_config ok\n")
        return 1
    
    def validation_config(self):
        self.error=0;
        if 'tasks' not in self.config_veles:
            self.error=-1
            return -1
        """
        next test 
        """
            
        return 1
        
    
    def download_tasks(self):
        #print("download_tasks")
        self.error=0;
        con = self.config_veles['tasks']
        #print(con)
        for k,v in con.items():
            #print("\n*************************************\n")            
            #print(k," & ",v,"\n")
            #print("\n*************************************\n")
            
            if type (v) == str:
                #print("\n      дана строка        \n")
                _temp= os.path.split(v)
                # isabs abspath
                name_file= _temp[1].split(".py")
                name=name_file[0]
                directory=_temp[0]
                if directory not in sys.path: 
                        sys.path.insert(0,directory)
                #print(sys.path)
                _t={}
                try:
                    _t=__import__(name, globals(), locals(), [name], 0)
                    #_t= import name
                except:
                    print("\n config module have not ",name_file,"\n")
                    _t=None
                    self.error =-3000
                    return -3
                    
                #print(k," ||\n",_t.__getattribute__(k),"\n^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
                if _t != None:                   
                    #print(k," ",name ," ===   ", dir(_t)," ==> ",k in dir(_t))
                    if name in dir(_t):                        
                        con[k]=_t.__getattribute__(name)
                        print(" ok ",k,"\n")
                    
                    else:
                        
                        print("\n config module have not ",name," dict \n")
                        self.error =-4000
                        return -6
            
                else:
                    print("not file ",name,"in directory ",directory)
                    self.error = -500
                    return -5   
                         
             
        #print(self.config_veles)             
        return 1
    
    def update_tasks(self):
        #print("update_tasks")
        self.error=0;
        con = self.config_veles['tasks']
        #print(con)
        for k,v in con.items():
           if type (v) == dict:
                #print("\n      дан словарь        \n")
                if 'param_file' in v:
                      
                    _temp= os.path.split(v['param_file'])
                    # isabs abspath
                    name_file= _temp[1].split(".py")
                    name=name_file[0]
                    directory=_temp[0]
                    if directory not in sys.path: 
                        sys.path.insert(0,directory)
                    #print(sys.path)
                    _t=None
                    try:
                        #print(name)
                        _t=__import__(name)
                    except:
                        #print("\n config module have not ",v['param_file'],"\n")
                        _t=None
                        self.error =-6000
                        return -8
                    
                    #print("\n----------------------\n")
                    if _t != None:            
                        if name in dir(_t):
                            tdict={}
                               
                            #if self.config_veles[k]==dict:
                            tdict=con[k].copy()
                            #print(k, " | ",name, "  | |   ",tdict)                                                 
                            #print(_t.__getattribute__(name))
                            con[k]=_t.__getattribute__(name)
                            con[k].update(tdict)
                            #print(" ok ",k,"\n")
                            
                        else:
                            
                            print("\n config module have not ",k," dict \n")
                            self.error =-7000
                            return -9
        return 1
                        
    def collect_continuity_experiments(self):
        #print("collect continuity experiments")
        r=0;
        if type(self.config_veles)==dict:
            r= self.Tasks.create_tasks(self.config_veles)
            
        else:
            r=-1
            
        if r<0:
            print("error config_veles!=dict, ",r)
            return r
        
        r=0;
        r= self.Tasks.connection_task(self.config_veles)
        if r<0:
            print("error in connection_task" ,r)            
            return r
        #print("collect continuity experiments ok")
        return 1
    
    def run(self):
        print("RUN VELES")
        #print()
        #print("Initializing...")
        self.Tasks.start_point.initialize_dependent()
        self.Tasks.end_point.wait()
        #print()
        print("Running...")
        
        
        self.Tasks.start_point.run_dependent()
        self.Tasks.end_point.wait()
        print("VELES OK")