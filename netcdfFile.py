"""
 Clases: 
  Auxiliares:
  netcdfFile : Clase que se encarga de la creacion de archivos netcdf, recibiendo como parametros python diccionarios con las dimensiones
               variables y atributos para su creacion.
               Cuenta con metodos para crear archivo, crear dimensiones, crear variables y salvar datos.

 by Favio Medrano
 Ultima modificacion 20-05-14
"""

import os
import logging as log
import datetime as dt
import netCDF4 as nc 
import numpy as np

class netcdfFile():
        """
         Clase netcdfFile
         Se encarga de crear rapidamente archivos netcdf, enviandole como parametros datos 
         de dimensiones y variables en formato de python dictionary.                
        """
        fileHandler = None 
        fileName = None 

        def __del__(self):
            # Nos aseguramos que el archivo se cierre correctamente.
            self.closeFile()

        def readFile(self,filename,path=''):
            """
             Funcion que lee el contenido de un archivo netCDF, y lo regresa en formato <python dict>
             Limitamos el tamano del archivo a leer a MB
             Formato salida:
              { 'dimensions' : { 'dim1' : value1 , 'dim2' : value2 ... } 
                'variables'  : {}
              }
            """
            if self.fileHandler != None:
                log.warning('readFile: Actualmente se encuentre un archivo netcdf abierto : ' + self.fileName)
                return None 
            if os.path.getsize(os.path.join(path,filename)) > 100000000L:
                log.warning('readFile: Archivo que se quiere leer es demasiado grande, para leerse completo.')
                return None 
            
            try:
                self.fileHandler = nc.Dataset(os.path.join(path,filename),'r')
            except Exception, e:
                log.warning('readFile: Se detecto un error al leer el archivo ' + filename)
                log.warning('readFile: ' + str(e))
                return None 
            self.fileName = os.path.join(path,filename) 
            self.closeFile()
            

        def createFile(self,filename,path='',filetype='NETCDF4'):
            """
             Recibe como diccionario los datos de las dimensiones
             Formato: {'dim' : value , 'dim2' : value2 .... }
            """            
            self.fileName = os.path.join(path,filename) 
            try:
                self.fileHandler = nc.Dataset(filename,'w',filetype)   
            except Exception, e:
                log.warning('Se detecto un error al crear el archivo: ' + filename )
                log.warning(str(e))
                return -1
            
        def closeFile(self):
            """
             Funcion que cierra el archivo netcdf, si es que ya se creo.
             Agrega un atributo global "description" donde indica la fecha de creacion.
            """
            if self.fileHandler == None:
                return -1
            self.fileHandler.description = 'File created ' + dt.datetime.today().strftime('%Y-%m-%d %I:%M:%S %p') + '.'
            self.fileHandler.close() 
            self.fileHandler = None
            self.fileName = None 
            return 0 
                     
        def createDims(self,dimDict):
            """
             Funcion para crear dimensiones del archivo netcdf
             Recibe como parametro dimDict tipo <python dic>, con el siguiente formato:
              {'dimension' : 10 , 'dimension2' : 40 , 'dimension3' : None } 
             Donde "None" hace a una dimension "unlimited"
            """
            if self.fileHandler == None:
                log.warning('Primero es necesario crear el archivo, con el metodo .create')
                return -1
            if dimDict != None:
                for d in dimDict.keys():
                    self.fileHandler.createDimension(d,dimDict[d]) 
            return 0
        

        def createVars(self,varsDict):
            """
             Recibe como <python dict> los datos de las variables, nombres y atributos
             Formato: { 'varName1' : { 'dimensions' : ['dim1','dim2'...] , 'attributes' : {'atribute1':value,'atribute2':'value2'}, 'dataType' : value }  , 
                        'varName2' : { 'dimensions' : ['dim1','dim2'...] , 'attributes' : {'atribute3':value,'atribute4':'value2'}, 'dataType' : value }  ..... }

             "La llave 'dimensions' y 'dataType' son obligatorias para crear la variable(s)" 
            """            
            def cleanVar(v):
                if type(v) == type('str'):
                    return v.strip() 
                else:
                    return (v) 
            
            if self.fileHandler == None:
                log.warning('createVars: Primero es necesario crear el archivo, con el metodo .create')
                return -1
            if varsDict != None:
                for v in varsDict.keys():
                    log.info('createVars: procesando variable: ' + v)
                    dimtuple = tuple(varsDict[v]['dimensions'])
                    try: 
                        # Crear variable
                        try: 
                            fillv = cleanVar(varsDict[v]['attributes']['_FillValue'])
                        except:
                            fillv = None
                        varH = self.fileHandler.createVariable(v.strip(),varsDict[v]['dataType'],dimtuple,fill_value=fillv)
                        if 'attributes' in varsDict[v]:
                            # Agregar los atributos
                            for att in varsDict[v]['attributes'].keys():
                                if att == 'units':
                                    varH.units = cleanVar(varsDict[v]['attributes']['units'])
                                elif att == 'long_name':
                                    varH.long_name =  cleanVar(varsDict[v]['attributes']['long_name'])
                                elif att == 'time_origin':
                                    varH.time_origin =  cleanVar(varsDict[v]['attributes']['time_origin'])  
                                elif att == 'missing_value':
                                    varH.missing_value =  (cleanVar(varsDict[v]['attributes']['missing_value'])) 
                                elif att == 'add_offset':
                                    varH.add_offset = (cleanVar(varsDict[v]['attributes']['add_offset']))
                                elif att == 'calendar':
                                    varH.calendar = cleanVar(varsDict[v]['attributes']['calendar'])
                                elif att == '_FillValue':
                                    pass
                                else:
                                    log.warning('crateVars: Atributo ' + att + ', no es valido')
                        log.info('createVars: Variable ' + v + ' , creada con todos sus atributos')
                    except Exception, e:
                        log.warning('createVars: Fallo al crear la variable : ' + v)
                        log.warning('createVars: Archivo netcdf: ' + self.fileName)
                        log.warning(str(e))
                        return -1 
            return 0
        

        def saveData(self,varDataDict):
            """
             Se encarga de guardar los arreglos de datos a sus variables correspondientes en el archivo netcdf.
             Formato: 
              {'varname1' : np.arrray[:,:,:] , 'varname2' : np.arrray[:,:,:] , ... }
            """            
            if self.fileHandler == None:
                log.warning('saveData: Primero es necesario crear el archivo, con el metodo .create')
                return -1
            if varDataDict != None:
                for v in varDataDict.keys():
                    log.info('saveData: Intento de salvar datos de variable : ' + v)
                    try:
                        varH = self.fileHandler.variables[v] 
                        varH[:] = varDataDict[v][:] 
                        log.info('saveData: OK')        
                    except Exception, e:
                        log.warning('saveData: Fallo al intentar salvar datos en variable: ' + v)
                        log.warning('saveData: ' + str(e))
                        return -1   
            
            return 0
        
        def saveDataS(self,varName,data,indexs):
            """
             Se encarga de guardar los datos "data" en la variable "varName" en los indices marcados por "indexs"
             
            """        
            if self.fileHandler == None:
                log.warning('saveData: Primero es necesario crear el archivo, con el metodo .create')
                return -1
            # La variable varName existe ?    
            try: 
                log.info('saveData: Intento de salvar datos de variable : ' + varName)
                varH = self.fileHandler.variables[varName] 
                varH[indexs] = data 
                log.info('saveDataS: OK')
            except Exception, e:
                log.warning('saveDataS: Fallo al intentar salvar datos en variable: ' + varName)
                log.warning('saveDataS: ' + str(e))
                return -1
                
            return 0
                        
      
     