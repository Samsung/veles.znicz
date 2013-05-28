'''
Created on Apr 16, 2013

@author: Seresov Denis <d.seresov@samsung.com>
'''

import units
import formats
import numpy
import os
import rnd

class TXTLoader(units.Unit):
    """Loads Wine data.

    State:
        output: contains Wine training set data
        labels: contains wine training set labels.
    """
    def __init__(self,config_datas={},param={}, unpickling = 0, ):        
        super(TXTLoader, self).__init__(unpickling=unpickling)
        if unpickling:
            return
        print(" TXTLOADER ",config_datas)
        self.config_datass =config_datas
        self.params= param

                        
        
        self.output = formats.Batch()   #x
        self.output2 = formats.Batch()  #normalize x
        self.labels = formats.Labels()  # y 
        
        self.TrainIndex = formats.Labels()
        self.ValidIndex = formats.Labels()
        self.TestIndex  = formats.Labels()
        self.Index = formats.Labels()

        
    def analize_config_file(self):
        print("\n\n analize_config_file \n\n")     
        #print("module_text = ",self.config_data,"|")
        
        print(self.config_datass)
        
        if type(self.config_datass) != dict:
            print("Error!!! config_data have not dict")
            return -1
        
        if 'data_dict' not in self.config_datass:
            return -101
        data_dict=self.config_datass['data_dict']
        self.dataset_size=0
        print( "self.dataset_size =",self.dataset_size)
        if 'count_data' not in data_dict:            
            return -2
        if 'use_all_in_one_struct' not in data_dict:
            return -3
        if 'data_sets' not in data_dict:            
            return -4
        self.data_sets=data_dict['data_sets']
        if type(data_dict['data_sets']) != dict:
            print("Error!!! data_dict.data_sets is not dict")
            return -5
        if len(data_dict['data_sets'])!=data_dict['count_data']:
            print("Error!!! length(data_sets)!= count_data  ")
            return -6
        print(" КОЛИЧЕСТВО пакетов данных - ",data_dict['count_data'])
        
        if 'merge_dataset' in self.config_datass:
            if type(self.config_datass['merge_dataset']) != int: 
                print("self.config_datass['merge_dataset'] != int:")
                return -102
        else:
            print("'merge_dataset' is not in self.config_datass")
            return -103
        
        if 'type1' in self.config_datass:
            if type(self.config_datass['type1']) ==int:
                if self.config_datass['type1']==0:
                    self.count_type_model=1
                    print("только траин. только хардкор")
                elif self.config_datass['type1']==1:
                    self.count_type_model=2
                    print("только траин+валид")
                elif self.config_datass['type1']==2:
                    self.count_type_model=3
                    print("траин+валид+тест")
                elif self.config_datass['type1']==3:
                    self.count_type_model=2
                    print("только траин+тест")
                else:
                    print(" неправильное значение type1")
                    return -104
            else:
                return -105            
        else:
            print(" НЕ УКАЗАНО МОДЕЛЬ РАЗДЕЛЕНИЯ ДАННЫХ \n 0 - вся выборка на обучение 1 -траин+валид 2- траин+валид+тест 3- траин+тест \n  ")
            return -106
        
        if self.config_datass['merge_dataset']==1:
            print(" ДАННЫЕ ЗАГРУЖАТЬ ВСЕ В ОДНУ СТРУКТУРУ.\n позже ОПРЕДЕЛИМ КТО В ТРАИН ВАЛИД ТЕСТ")       
            if 'var1' in  self.config_datass:
                if type(self.config_datass['var1'])==dict:
                    _t =self.config_datass['var1']
                    
                    #if 'use_rand_for_train_data' in _t:
                    #    if type(_t['use_rand_for_train_data'])==int:
                    #        if (_t['use_rand_for_train_data']==0)or(_t['use_rand_for_train_data']==1):
                    #            self.use_rand_for_train_data=_t['use_rand_for_train_data']
                    #        else:
                    #            print('use_rand_for_train_data is not 0 or 1')
                    #            return -118
                    #    else:
                    #        print('use_rand_for_train_data is not int')
                    #        return -114
                    #else:
                    #    print('use_rand_for_train_data is no in var1')
                    #    return -115
                    
                    if self.config_datass['type1'] in [0,1,2,3]:
                        if 'use_rand_for_train_data' in _t:
                            if type(_t['use_rand_for_train_data'])==int:
                                if (_t['use_rand_for_train_data']==0)or(_t['use_rand_for_train_data']==1):
                                    self.use_rand_for_train_data=_t['use_rand_for_train_data']
                                else:
                                    print('use_rand_for_train_data is not 0 or 1')
                                    return -118
                            else:
                                print('use_rand_for_train_data is not int')
                                return -114
                        else:
                            print('use_rand_for_train_data is no in var1')
                            return -115
                    if self.config_datass['type1'] in [1,2]:
                        if 'use_rand_for_valid_data' in _t:
                            if type(_t['use_rand_for_valid_data'])==int:
                                if (_t['use_rand_for_valid_data']==0)or(_t['use_rand_for_valid_data']==1):
                                    self.use_rand_for_valid_data=_t['use_rand_for_valid_data']
                                else:
                                    print('use_rand_for_valid_data is not 0 or 1')
                                    return -118
                            else:
                                print('use_rand_for_valid_data is not int')
                                return -114
                        else:
                            print('use_rand_for_valid_data is no in var1')
                            return -115
                    
                    if self.config_datass['type1'] in [2,3]:
                        if 'use_rand_for_test_data' in _t:
                            if type(_t['use_rand_for_test_data'])==int:
                                if (_t['use_rand_for_test_data']==0)or(_t['use_rand_for_test_data']==1):
                                    self.use_rand_for_test_data=_t['use_rand_for_test_data']
                                else:
                                    print('use_rand_for_test_data is not 0 or 1')
                                    return -118
                            else:
                                print('use_rand_for_test_data is not int')
                                return -114
                        else:
                            print('use_rand_for_test_data is no in var1')
                            return -115
                    
                    if 'divideblock' in _t:  

                        if type(_t['divideblock'])==int:
                            if (_t['divideblock']==0)or(_t['divideblock']==1):
                                self.divideblock=_t['divideblock']
                            else:
                                print('use_rand_for_train_data is not 0 or 1')
                                return -119
                        else:
                            print('divideblock is not int')
                            return -116
                    else:
                        print('divideblock is no in var1')
                        return -117
                    if self.divideblock==0:
                        print(" считываем size_fix_div - вектора диапозонов  - обучащей валидации тестовой части выборки")
                        if 'size_fix_div' in self.config_datass:
                                if self.config_datass['size_fix_div']==dict:
                                    if len(self.config_datass['size_fix_div'])==self.count_type_model:
                                        print("задано правильно количество типов данных  в size_fix_div \n  позже прочитаем")
                                        _t2 = self.config_datass['size_fix_div'] 
                                        """
                                        анализ и  считывание векторов ... 
                                         # 'size_fix_div':{'sizeTR ':range(1,5),'sizeV':range(5,9),'sizeTE':range(9,13)},#               
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         """
                                        print(_t2)
                                else:
                                    print("self.config_datass['size_fix_div']!=dict")
                                    return -120
                        else:
                            print("size_fix_div is not in self.config_datass")
                            return -121
                    elif self.divideblock==1:
                        print(" считываем size_proc_div - вектор % от всей выборки - обучащей валидации тестовой части выборки ")
                        if 'size_proc_div' in _t:
                                if type(_t['size_proc_div'])==dict:
                                    print("задано : в size_proc_div = ",len(_t['size_proc_div'])," self.count_type_model= ",self.count_type_model)
                                    if len(_t['size_proc_div'])==self.count_type_model:
                                        print("задано правильно количество типов данных  в size_proc_div \n  позже прочитаем")
                                        _t2 = _t['size_proc_div']  # 'size_proc_div':{'sizeTR%':0.4,'sizeV%':0.3,'sizeTE%':0.3},#
                                        #print(_t2)
                                        print( 'type1 = ',self.config_datass['type1'])
                                        self.sizeTR_proc=0.0
                                        self.sizeV_proc =0.0
                                        self.sizeTE_proc=0.0
                                        if (self.config_datass['type1']==0  and 'sizeTR%' not in _t2): # O
                                            print('sizeTR%  error3 type1=0');return -132
                                        if  (self.config_datass['type1']==1 and 'sizeTR%' not in _t2 and 'sizeV%' not in _t2): # 1
                                            print('sizeTR%  error3 type1=1');return -133
                                        if  (self.config_datass['type1']==2 and 'sizeTR%' not in _t2 and 'sizeV%' not in _t2 and 'sizeTE%' not in _t2): # 2
                                            print('sizeTR%  error3 type1=2');return -134
                                        if  (self.config_datass['type1']==3 and 'sizeTR%' not in _t2 and 'sizeTE%' not in _t2): # 3
                                            print('sizeTR%  error3 type1=3');return -135
                                        if self.config_datass['type1'] in [0,1,2,3]:
                                            if  'sizeTR%' in _t2:    
                                                if type(_t2['sizeTR%'])==float:
                                                    if (_t2['sizeTR%']>0.0) or (_t2['sizeTR%']<=1.0):
                                                        self.sizeTR_proc=_t2['sizeTR%']
                                                        print("self.sizeTR_proc ",self.sizeTR_proc)
                                                    else: print('sizeTR%  error1 range'); return -126
                                                else:     print('sizeTR%  error2 float');return -127
                                            else: print(" not sizeTR%"); return -164
                                        if self.config_datass['type1'] in [1,2]:
                                            if  'sizeV%' in _t2:
                                                if type(_t2['sizeV%'])==float:
                                                    if (_t2['sizeV%']>0.0) or (_t2['sizeV%']<=1.0):
                                                        self.sizeV_proc=_t2['sizeV%']                                                
                                                        print("self.sizeV_proc ",self.sizeV_proc)
                                                    else:  print('sizeV%  error1 range');return -128
                                                else:      print('sizeV%  error2 float');return -129
                                            else: print(" not sizeV%"); return -165
                                        if self.config_datass['type1'] in [2,3]:
                                            if  'sizeTE%' in _t2:
                                                if type(_t2['sizeTE%'])==float:
                                                    if (_t2['sizeTE%']>0.0) or (_t2['sizeTE%']<=1.0):
                                                        self.sizeTE_proc=_t2['sizeTE%']
                                                        print("self.sizeTE_proc ",self.sizeTE_proc)
                                                    else:  print('sizeTE%  error1 range');return -130
                                                else:   print('sizeTE%  error2 float');return -131
                                            else: print(" not sizeTE%"); return -166                  
                                        print(" =======> ",self.sizeTR_proc, " " ,self.sizeV_proc ," ",self.sizeTE_proc, " || ",self.sizeTR_proc+self.sizeV_proc +self.sizeTE_proc)                                                    
                                        
                                        if self.config_datass['type1']==0 and self.sizeTR_proc!=1.0:
                                            print('self.sizeTR_proc+self.sizeV_proc>1.0  error3 type1=1'); return -1366           
                                            
                                        if self.config_datass['type1']==1 and self.sizeTR_proc+self.sizeV_proc!=1.0:
                                            print('self.sizeTR_proc+self.sizeV_proc>1.0  error3 type1=1'); return -136                                       
                                        
                                        if self.config_datass['type1']==2 and self.sizeTR_proc+self.sizeV_proc +self.sizeTE_proc!=1.0: 
                                            print('self.sizeTR_proc+self.sizeV_proc+self.sizeTE_proc>1.0  error3 type1=1');return -137
                                            
                                        if self.config_datass['type1']==3 and self.sizeTR_proc +self.sizeTE_proc!=1.0:
                                            print('self.sizeTR_proc+self.sizeTE_proc>1.0  error3 type1=1'); return -138
                                        
                                            
                                        print(self.sizeTR_proc, " ",   self.sizeV_proc,'',                      self.sizeTE_proc)
                                        
                                    else:
                                        print("count parameters in size_proc_div != count_type_model_data",len(_t['size_proc_div']),"!=",self.count_type_model)                                        
                                        return-125
                                    
                                else:
                                    print("var1['size_proc_div']!=dict")
                                    print(type(_t['size_proc_div']))
                                    return -122
                        else:
                            print("size_proc_div is not in self.config_datass")
                            return -123
                    
                else:
                    print('var1 - is not dict')
                    return -124
            else:
                print(" not var1")
                return -113
            
        elif self.config_datass['merge_dataset']==0:
            print(" ДАННЫЕ распределенны в пакетах, работать не объединяя.\n позже ОПРЕДЕЛИМ КТО В ТРАИН ВАЛИД ТЕСТ")
            if 'not_merge_type' in self.config_datass:
                if type(self.config_datass['not_merge_type']) != int: 
                    if self.config_datass['not_merge_type']==0:
                        print(" ДОЛЖНЫ БЫТЬ ЗАДАНЫ вектора ('index_file') принадлежности  пакетов к  ТРАИН ВАЛИД ТЕСТ")
                        
                        if 'index_file' in self.config_datass:
                            if self.config_datass['index_file']==dict:
                                if len(self.config_datass['index_file'])==self.count_type_model:
                                    print("задано правильно количество векторов в index_file \n  позже прочитаем")
                                    _t1 = self.config_datass['index_file']
                                    """
                                    'index_file'  проверить и загрузить.... 
                                    # 'index_file':{ 'ind_fileTR':[1], 'ind_fileTV':[0],  'ind_fileTE':[0]   },#
                                    """
                            else:
                                print("self.config_datass['index_file']!=dict")
                                return -109
                        else:
                            print("index_file is not in self.config_datass")
                            return -110
                        
                        
                        
                    elif self.config_datass['not_merge_type']==1:
                        print(" ДОЛЖНЫ БЫТЬ ЗАДАНЫ файлы ('f_index_file') принадлежности  пакетов к  ТРАИН ВАЛИД ТЕСТ")
                        
                        if 'f_index_file' in self.config_datass:
                            if self.config_datass['f_index_file']==dict:
                                if len(self.config_datass['f_index_file'])==self.count_type_model:
                                    print("задано правильно количество файлов в f_index_file \n  позже прочитаем")
                                    
                                    _t1 = self.config_datass['f_index_file']
                                    """
                                    'f_index_file'  проверить и можчет шо прочитать и загрузить....
                                    # 'f_index_file':{'file_ind_fileTR':'xor/indTR.cfg','file_ind_fileTV':'xor/indTV.cfg','file_ind_fileTE':'xor/indTE.cfg'},
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    """
                            else:
                                print("self.config_datass['f_index_file']!=dict")
                                return -111
                        else:
                            print("index_file is not in self.config_datass")
                            return -112
                        
                    else:
                        print(" неправильное значение параметра not_merge_type")
                        return -107
        else:
            print(" неправильное значение параметра merge_dataset")
            return -108            
        
        """   Не проверено еще
                 
            'train_param':{
                           'global_alpha':0.9,
                           'global_lambda':0.0,
                           'threshold':1.0
            }
        """
        #'norm':{'min':-1,'max':1,#'removeconstantrows':0,'mapminmax':1,'mapstd':1 #},#
        if 'use_norm' in self.config_datass:
            if type(self.config_datass['use_norm'])==int:
                    if self.config_datass['use_norm'] in [0,1,2,3,4,5,6]:                    
                        self.use_norm=self.config_datass['use_norm']
                        if self.use_norm==0: print(" НЕ НОРМАЛИЗУЕМ Х")
                        if self.use_norm==1: print(" НОРМАЛИЗУЕМ Х по минмах")
                        if self.use_norm==2: print(" НОРМАЛИЗУЕМ Х по мат/дис")
                        if self.use_norm==3: print(" НОРМАЛИЗУЕМ Х,Y по минмах")
                        if self.use_norm==4: print(" НОРМАЛИЗУЕМ Х,Y по мат/дис")
                        if self.use_norm==5: print(" НОРМАЛИЗУЕМ Х   по мат/дис потом минмах")
                        if self.use_norm==6: print(" НОРМАЛИЗУЕМ Х,Y по мат/дис потом минмах")
                        
                        if 'norm' in self.config_datass:
                            if type(self.config_datass['norm'])==dict:
                                _tf=self.config_datass['norm']
                                if self.use_norm in [1,3,5,6]:
                                    if 'min' in _tf:
                                        if type(_tf['min']) ==float:
                                            self.min=_tf['min']
                                    if 'max' in _tf:
                                        if type(_tf['max']) ==float:
                                            self.max=_tf['max']
                            else: print(" norm !=dict");return -153
                        else: print(" not norm ");return -154
                    else: print( "use_norm == [0;1;2,3,4,5,6]");return -155
            else: print( "use_norm only int");return -156
        else:  print( "use_norm not");return -157
                    
            
        _ttt = self.params
        
        if 'train_param' in _ttt:
            if type(_ttt['train_param'])==dict:
                _tttt=_ttt['train_param']
                if 'type' in _tttt:
                    if type(_tttt['type'])==str:
                        print(" type train =",_tttt['type'])
                        if _tttt['type']=='BP1':
                            if 'global_alpha' in _tttt:
                                if type(_tttt['global_alpha'])==float:
                                    if _tttt['global_alpha']>=0.0 and _tttt['global_alpha']<=1.0:
                                        self.global_alpha=_tttt['global_alpha']
                                    else: print("error .  global_alpha in [0;1]");return -163
                                else: print( 'global_alpha !=float' );return -158
                            else:
                                self.global_alpha=0.9  # из файла по умолчанию бы взять бы
                                
                            if 'global_lambda' in _tttt:
                                if type(_tttt['global_lambda'])==float:
                                    if _tttt['global_lambda']>=0.0 and _tttt['global_lambda']<=1.0:
                                        self.global_lambda=_tttt['global_lambda']
                                    else: print("error .  global_lambda in [0;1]");return -162                                    
                                else: print( 'global_lambda !=float' );return -159
                            else:
                                self.global_lambda=0.0  # из файла по умолчанию бы взять бы
                                
                            if 'threshold' in _tttt:
                                if type(_tttt['threshold'])==float:
                                    if _tttt['threshold']>0.0 and _tttt['threshold']<=1.0:
                                        self.threshold=_tttt['threshold']
                                    else: print("error .  threshold in (0;1]");return -161
                                else: print( 'threshold !=float' );return -160
                            else:
                                self.threshold=0.0  # из файла по умолчанию бы взять бы
                            
                                
                            
                                
                            
                            
                
        if 'use_random' in _ttt:
            if type(_ttt['use_random']) ==int:
                self.use_random=_ttt['use_random']
                if _ttt['use_random']==0:       # Если seed
                    print(" ГЕНЕРАТОР ФИКСИРОВАННЫЙ ")
                    if 'random' in _ttt:
                        if type(_ttt['random'])==dict:
                            _t3=_ttt['random']
                            if 'type' in _t3:
                                if type(_t3['type'])==int:
                                    if _t3['type']==1:  print(" Используем 1 тип генератора")
                                    else:   print(" не знаю такого типа генератора в type"); return -139                                        
                                else: print(" не правильный тип  type"); return -140
                            else: print("не указан тип генератора (type =1)"); return -141                            
                            if 'type_seed' in _t3:
                                if type(_t3['type_seed'])==int:
                                    self.type_seed=_t3['type_seed']
                                    if _t3['type_seed']==1:  print(" Используем 1 тип seed")
                                    else:  print(" не знаю такого типа seed в type"); return -142
                                else: print(" не правильный тип  type_seed"); return -143
                            else: print("не указан тип генератора (type_seed =0)");return -144                            
                            if 'seed' in _t3:
                                    self.seed=_t3['seed']
                                    if type(_t3['seed'])==int:       print(" Используем  seed int")
                                    elif type(_t3['seed'])==list:    print(" Используем  seed list")
                                    elif type(_t3['seed'])==str:     print(" Используем  seed string")
                                    else:      print(" seed не понятного типа") ;return -145
                            else: print("не указан тип генератора (type_seed =0)"); return -146
                            
                            if self.type_seed==1 and not type(_t3['seed'])==str:    print(" задано type_seed но seed не строка");return -147  
                        else: print("random  не dict"); return -148
                    else: print("нет random={}");return -149
                elif  _ttt['use_random']==1:  
                    print(" СЛУЧАЙНЫЙ ГЕНАРТОР")
                    if 'random' in _ttt:
                        if type(_ttt['random'])==dict:
                            _t3=_ttt['random']
                            if 'type' in _t3:
                                if type(_t3['type'])==int:
                                    if _t3['type']==1:  print(" Используем 1 тип генератора")
                                    else:   print(" не знаю такого типа генератора в type") ;return -139
                                else: print(" не правильный тип  type") ;return -140
                            else: print("не указан тип генератора (type =1)"); return -141
                        else: print("random  не dict") ;return -142
                    else: print("нет random={}"); return -143
                else: print(" 'use_random' имеет не правильное значение ,[0,1]"); return -150
            else: print(" 'use_random' !=dict"); return -151
        else:  print(" 'use_random' нет"); return -152
                                
                            
                    
                    
        
        
        
        for _d,_v in data_dict['data_sets'].items():
            print(_d)
            if type(_v) != dict:
                print("Error!!! in data_sets ",_d,"!= dict  ")
                return -7
            if 'type_data' in _v:
                if type(_v['type_data']) != int:
                    #_td['type_data'] =_v['type_data']    
                    return -8
            else:
                return -808
            if _v['type_data']==0:
                                
                if 'file_input' in _v:
                    if type(_v['file_input']) == str:
                        if not os.path.isfile(_v['file_input']):
                            return -18
                    else:
                        return -10
                else:
                    return -1010
                if 'size' in _v:
                    if type(_v['size']) != int:
                        #_td['size'] =_v['size']
                        return -11
                        
                else:
                    return -1111
                print("v size  = ",_v['size'])
                
                self.param_size=_v['count_parametrs']
                
                if 'count_parametrs' in _v:
                    if type(_v['count_parametrs']) != int:
                        #_td['count_parametrs'] =_v['count_parametrs']
                        self.param_size=_v['count_parametrs'];                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 1 raz a ne cikl
                        return -12
                else:
                    return -1212
                if 'type_output' in _v:
                    if type(_v['type_output']) != int:
                        #_td['type_output'] =_v['type_output']
                        return -13
                else:                    
                    return -1313
                if 'count_output' in _v:
                    if type(_v['count_output']) != int:
                        #_td['count_output'] =_v['count_output']
                        self.labels.n_classes = _v['count_output']
                        return -14
                else:
                    return -1414
                if _v['type_output']==2:
                    if 'const_class_y' in _v:
                        if type(_v['const_class_y']) != int:
                            #_td['const_class_y'] =_v['const_class_y']
                            return -15
                        else:
                            return -1515
                if 'file_output' in _v:
                    if type(_v['file_output']) == str:
                        if not os.path.isfile(_v['file_output']):
                            return -17
                    else:
                        return -16
                else:
                    return -1616
                                
                self.dataset_size=self.dataset_size+_v['size'];
                print(" _v['size'] ",_v['size'], "  | ",self.dataset_size)
        print(" in config_file_data  - not error")
        return 1
        
    def read_data_y(self):
        print("read_data_y")
        
        self.labels.batch=numpy.zeros([self.dataset_size], dtype=numpy.float32)
        
       
        i=0
        for _d,_v in self.data_sets.items():
            print(_d)            
            print("Reading ",_d," number ",i)
            x=_v['type_data']
            if  x== 0:
                print(' vector label')
                t1=formats.Labels()
                print(_v['file_output'])
                t1.batch=numpy.loadtxt(_v['file_output'], numpy.int)
                #t1.batch-=1
                #error about size t1.batch!= [self.child_size[i]]
                if i==0:
                    self.labels.batch = t1.batch                    
                    
                    
                else:
                    self.labels.batch =numpy.append(self.labels.batch,t1.batch)    
            elif x == 2: 
                print(' directory \ not ')
                return -1
            else:
                print("not ")
                return -2
            i=i+1   
            
        
            
        print('size all dataset output = ', self.labels.batch.size)
        #print(self.labels.batch)
        print("Read_data output ok")
        return 1
    def read_data_x(self):
        # Resize memory for all dataset self.data [self.dataset_size]
        
        #если данные в разных файлах и надо объединить в одну базу....
        
        i=0
        for _d,_v in self.data_sets.items():
            print(_d)            
            print("Reading ",_d," number ",i)
            x=_v['type_data']
            if  x== 0:
                print(' table -x')
                t1=formats.Batch()
                print(_v['file_input'])
                t1.batch=numpy.loadtxt(_v['file_input'], dtype=numpy.float32).reshape([_v['size'],_v['count_parametrs']])
                    #error about size t1.batch!= [self.child_size[i],self.child_param[i]]
                print(_v['size']," ",_v['count_parametrs'])
                if i==0:
                    self.output.batch = t1.batch    
                else:
                    self.output.batch =numpy.append(self.output.batch,t1.batch).\
                        reshape(self.output.batch.shape[0]+t1.batch.shape[0],_v['count_parametrs'])
            elif x == 2: 
                print(' directory \ not ')
                return -1
            else:
                print("not ")
                return -2
            i=i+1
                
                
        #print('size all dataset input = ', self.output.batch.shape)
        print("Read_data input ok")
        return 1
        
    def load_original(self):
        """ Loads data from original Wine dataset.
            we will read config file with parameters dataset(width,height),
             size(or %) train set,size(or %) validation set,size(or %) test set,
             random or on series or random with seed...
             type normalization dataset(range)
             
             
        """ 
        print("loadoriginal start")
        r=0
        r= self.analize_config_file()
        print("analize_config_file ",r)
        if r<0:
            print("Error in config_data file  ")
            return -1
             
        # if bigggest dataset-> reading piece by piece
        r=0
        r=self.read_data_x()
        if r<0:
            print("error. Read not input files")
            return -2
        r=0
        r=self.read_data_y()
        if r<0:
            print("error. Read not ouptut files")
            return -3
        
        
        
        
        self.distrib()
        self.rand_datas_index()
        self.rand_datas_x()
        self.rand_datas_y()
        self.norm_x_datas()
        print("load_original ok")
        return 1
    
    
    def norm_x_datas(self):
        """ normalization   datas.
            
            result:
            self.outmean - parametrs  normalization with self.use_norm in[5] 
            self.outstd
            self.outmin
            self.outmax
            
            self.output2 - normalize datas all
        """       
        #print("norm_train start",self.use_norm)
        
        print(self.sizeTR)
        if self.use_norm in[5]:  
            self.outmean= numpy.mean(self.output.batch[0:self.sizeTR], axis=0)
            self.outstd= numpy.std(self.output.batch[0:self.sizeTR], axis=0)
            #print(self.dataset_size," ",self.param_size," ",self.sizeTR, " ")
            #print(self.outmean)
            #print(self.outstd)
            for i in range(0,self.param_size):
                self.output2.batch[:,i]=((self.output2.batch[:,i]-self.outmean[i]))/self.outstd[i]
        
            #self.outmin =filters.aligned_zeros([13])
            self.outmin= numpy.min(self.output2.batch[0:self.sizeTR], axis=0)        
            #self.outmax =filters.aligned_zeros([13])
            self.outmax= numpy.max(self.output2.batch[0:self.sizeTR], axis=0)
            
            #print(self.outmin)
            #print(self.outmax)
                   
            for i in range(0,self.param_size):
                self.output2.batch[:,i]=(((((self.output2.batch[:,i]-self.outmin[i]))/(self.outmax[i]-self.outmin[i]))-0.5)*2);         # нормализация на интервал -1 до  1  -0.5)*2);
        else : print(" not use any metod normalize")        
        #del self.output
        #print(self.output2.batch[0,:])
        #print(self.output2.batch[10,:])
        
        print("norm ok")

  
    def rand_datas_index(self):
        print(" rand_datas")
        _t =self.config_datass['var1']
        l=0;
        
        self.Index.batch=numpy.array(range(0,self.dataset_size),dtype = numpy.int)
        print(self.sizeTR," ",self.sizeV," ",self.sizeTE)
        if _t['use_rand_for_train_data']!=0:
            if self.config_datass['type1'] in [0,1,2,3]:
                if _t['use_rand_for_train_data']==1: 
                    l=self.sizeTR
            if self.config_datass['type1'] in [1,2]:
                if _t['use_rand_for_valid_data']==1:
                    l=l+self.sizeV
            if self.config_datass['type1'] in [2]:
                if _t['use_rand_for_test_data']==1:
                    l=l+self.sizeTE
            if self.config_datass['type1'] in [3]:
                if _t['use_rand_for_test_data']==1:
                    l=l+self.sizeTE
            #print(self.config_datass['type1'], " : ",l)
            self.Index.batch[0:l] =numpy.random.permutation(self.Index.batch[0:l])
        else:
            self.Index.batch=numpy.array(range(0,self.dataset_size),dtype = numpy.int)
        
       
        #numpy.random.shuffle(sself.Index[0:l])
        #print(self.Index.batch)
        print(" rand_datas ok")
    
    def rand_datas_x(self): 
        print(" rand_datas_x")
        self.output2.batch =numpy.zeros([self.dataset_size*self.param_size], dtype=numpy.float32).reshape([self.dataset_size,self.param_size])
        self.output2.batch[:]=self.output.batch[self.Index.batch]
        #print(self.output.batch)        
        #print(self.output2.batch)
        
        print(" rand_datas_x ok") 
    
    def rand_datas_y(self):
        print(" rand_datas_y")
        #print(self.labels.batch)
        self.labels.batch[:] =self.labels.batch[self.Index.batch]
        #print(self.labels.batch)

        print(" rand_datas_y ok")
        
    def rand_train(self):
        print(" rand_train")
        _t =self.config_datass['var1']
            
        if _t['use_rand_for_train_data']==1:
            print(" ПЕРЕМЕШАЕМКА")                        
            #numpy.random.shuffle(self.Index)   задать диапозон
            
        
        print(" rand_train ok")

    def distrib(self):       
        #self.labels.batch
        """
          self.config_datass['type1']  [0,1,2,3]
          
        """

        
        self.sizeTR_r=None
        self.sizeV_r=None
        self.sizeTE_r=None
        
        self.sizeTR=0
        self.sizeV=0
        self.sizeTE=0
        if self.config_datass['type1'] in [0,1,2,3]:       
            self.TrainIndex.batch=numpy.zeros(self.dataset_size,dtype = numpy.int)
        if self.config_datass['type1'] in [1,2]:        
            
            self.ValidIndex.batch=numpy.zeros(self.dataset_size,dtype = numpy.int)
        if self.config_datass['type1'] in [2,3]:
            
            self.TestIndex.batch=numpy.zeros(self.dataset_size,dtype = numpy.int)
        
        if self.config_datass['merge_dataset']==1:
            _t =self.config_datass['var1']
            if _t['divideblock']==1:                 
                if self.config_datass['type1'] in [0,1,2,3]: 
                    self.sizeTR = int(numpy.round(self.sizeTR_proc*self.dataset_size))                    
                    self.TrainIndex.batch[0:self.sizeTR+1]=1
                    
                
                if self.config_datass['type1'] in [1,2]:
                    self.sizeV = int(numpy.round(self.sizeV_proc*self.dataset_size))
                    self.ValidIndex.batch[self.sizeTR:self.sizeTR+self.sizeV+1]=1
                    
                    
                if self.config_datass['type1'] in [2]:
                    self.sizeTE =int(numpy.round(self.sizeTE_proc*self.dataset_size))
                    self.TestIndex.batch[self.sizeTR+self.sizeV:self.sizeTR+self.sizeV+self.sizeTE+1]=1
                if self.config_datass['type1'] in [3]:
                    self.sizeTE =int(numpy.round(self.sizeTE_proc*self.dataset_size))
                    self.TestIndex.batch[self.sizeTR:self.sizeTR+self.sizeTE+1]
                    
                print(self.config_datass['type1']," :: ",self.sizeTR," ",self.sizeV, " ",self.sizeTE, " = ",self.dataset_size," ",self.sizeTR+self.sizeV+self.sizeTE)
                #print(self.TrainIndex)
                #print(self.ValidIndex)
                #print(self.TestIndex)
            else: print( "UPS  metoda net poka" ); return -1    
        else: print( "UPS  metoda net poka" ); return -2         
        #self.labels.batch=numpy.zeros([self.dataset_size], dtype=numpy.float32)
             
        
        
        print("distrib ok")
        return 1
    
                 
    def initialize(self):
        """Here we will load Wine data.
        """
        
        #try:
        #    fin = open("cache/Wine-train.pickle", "rb")
        #    self.output.batch,self.output2.batch, self.labels.batch, self.labels.n_classes = pickle.load(fin)
        #    fin.close()
        #except IOError:
        r=0
        r= self.load_original()
        print("self.original",r)
        
        #self.output2.update()
        if r<0:
            print("ERROR!!!! failed result in initialize ->load_original  ")
            return -1
        

        
        
        print("initialize text ok")

    def run(self):
        """Just update an output.
        """
        self.output.update()
        self.output2.update()