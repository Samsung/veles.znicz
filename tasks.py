"""
	Class tasks
	Class task
"""
import units
import units_end_point
import ctask

class Tasks(object):
	
	def __init__(self):
		self.Task={}
		self.start_point = units.Unit()
		self.end_point = units_end_point.EndPointTasks(self)
	
	def create_tasks(self,config_veles):
		
		#print("\n create task\n\n",config_veles)
		if type(config_veles)==dict:
			#print(config_veles)
			for k,v in config_veles['tasks'].items():
				#print("\n			-> \n",k," ",v)
				_t=ctask.ctask(v)
				#_t.__dict__.update(v)
				self.Task[k]=_t
				#print(dir(_t))		
		else:
			print("not dict input param")
			return -1
		#print(self.Task)
		
		return 1

	def connection_task(self,config_veles):
		"""
		прикрепляем стартовую точку к таскам 
		"""
		_list= config_veles['start']
		for _l in _list:
			if _l in  self.Task:
				self.Task[_l].link_from(self.start_point)
			else:
				print( _l, " is not in Tasks\n")
				return -1
		"""
		бегаем по таскам и смотрим кого с кем связать по линку
		"""
		for k,v in config_veles['tasks'].items():
			self.end_point.link_from(self.Task[k])
		for k,v in config_veles['tasks'].items():
			#print("\n			-> ",k," ",self.Task[k])
			_t=self.Task[k]

			#_t1 = _t.param['start']
			
			if 'start' in _t.param:
				_ll=_t.param['start']
				if  type(_ll) is list:
					if len(_ll)>0:
						for _lr in _ll:
							if type(_lr) is str:
								if _lr in self.Task:
									print("\n ==>  ",_lr," ",self.Task[_lr])
									
									self.Task[k].link_from(self.Task[_lr])
									self.end_point.unlink_from(self.Task[_lr])
									
								else:
									print(_lr, "is not in  list tasks")
									return -2
							else:
								print(_lr," - it is not type parameters (not str - name tasks)")
								return -3
					else: 
						print(" start - NULL")
				else:
					print(_ll," - it is not type parameters (not list - name vector names tasks )")
					return -4
			else:
				print(" start -parameters is not")
				return -5				
		#print(self.Task)
		if len(self.end_point.links_from)==0:
			print(" Error!!! End_point have not links")
			return -6
		if len(self.start_point.links_to)==0:
			print(" Error!!! Start_point have not links")
			return -7
		return 1							
