#!/usr/bin/python3.3
"""
Created on Mar 11, 2013

Entry point.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import logging
import sys
import numpy
import pickle
import os
import argparse
import veles_demo2

def main():
    logging.debug("Entered")
    
    parser = argparse.ArgumentParser()
    # восстановить по snapshot
    parser.add_argument("-r", type=str, help="resume from snapshot", \
                        default="", dest="resume")
    parser.add_argument("-c", type=str, help="config-file experiments for veles ", \
                        default="/home/dseresov/Projects/Znicz/wine/veles_tasks3.py", dest="config_veles")
    # в конфиг файлы вставить параметры 
        #parser.add_argument("-cpu", action="store_true", help="use numpy only", \
        #                    default=False, dest="cpu")
    # в  конфиг файла параметров метода обучения вставить..
        #parser.add_argument("-global_alpha", type=float, help="global gradient descent speed", \
        #                    default=0.9, dest="global_alpha")
        #parser.add_argument("-global_lambda", type=float, help="global weights regularisation constant", \
        #                    default=0.0, dest="global_lambda")
        #parser.add_argument("-threshold", type=float, help="softmax threshold", \
        #                    default=1.0, dest="threshold")
    # в конфиг файлы вставить параметр ( или  для тестирования отдельная модель (из кода питона), или ключ для тестирования, используя одну модель  
        #parser.add_argument("-t", action="store_true", help="test only", \
        #                    default=False, dest="test_only")
    args = parser.parse_args()
    
    #project. Seresov maj 5, 2013
    #
    #numpy.random.seed ()    # random generic  - это в  эксперимент
    model = None
    if args.resume:        #  snapshot для всего велеса
        try:
            print("Resuming from snapshot...")
            fin = open(args.resume, "rb")
            (model, random_state) = pickle.load(fin)
            numpy.random.set_state(random_state)
            fin.close()
        except IOError:
            print("Could not resume from %s" % (args.resume, ))
            model = None
    if not model:
        model = veles_demo2.Veles()
    print("Launching...")
    
    
    try:
        model.read_config(args.config_veles)
        r=model.validation_config()
        if(r!=1):
            print(" result  validation_config ",r," ",model.error,"|")
            raise Exception("validation_config")
        r=model.download_tasks()
        if(r!=1):
            print(" result  download_tasks ",r," ",model.error,"|")
            raise Exception("download_tasks")
        r=model.update_tasks()
        if(r!=1):
            print(" result  update_tasks ",r," ",model.error,"|")
            raise Exception("update_tasks")
        r=model.collect_continuity_experiments()
        if(r!=1):
            print(" result  collect_continuity_experiments ",r," ",model.error,"|")
            raise Exception("collect_continuity_experiments")      
    except IOError:
        print("error in %s , %s\n",args.config_veles,model.error)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`\n")
    try:
        print("model.run\n")
        model.run()
        print("model.run ok\n")
    except IOError:
        print("error in %d, %s \n", model.num_experiment,model.error)
    print("Done main")

    logging.debug("Finished")


if __name__ == '__main__':
    main()
