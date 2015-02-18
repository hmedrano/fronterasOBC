"""
 Script para crear condiciones de frontera OBC a partir de datos descargados 
 de mercator a las configuraciones GOLFO12 y GOLFO24. 


 by Favio Medrano
 
 Aug-6-2014 : Se salvan los obc en archivos anuales o mensuales. 
              Los archivos de salida, pueden tener un prefijo elejido por el usuario
 Aug-7-2014 : Se modifico el codigo para que los datos de entrada no necesariamente contengan
              dominios espaciales (Lat,Lon) iguales, ahora recortamos la rebanada necesaria para 
              la interpolacion tomando la region contenida en el archivo mascara.
            : Parametro en el metodo 'crearFronterasEsteSur' para seleccionar el tamano de los 
              archivos, 'yearly' o 'monthly'.
            : Parametro en el metodo 'crearFronterasEsteSur' para seleccionar la salida, solo datos
              o datos en estructura mensual o anual.

 Jan-14-2015 : Primer cambio 2015
"""

import numpy as np
import netCDF4 as nc 
import logging as log 
import datetime as dt 
from scipy import interpolate
# own libs
import netcdfFile

def dateToNemoCalendar(data, ctype='gregorian',give='full'):
    """ 
     Codigo tomado del codigo de nemo IOIPSL/src/calendar.f90 para construir el valor de la variable temporal en modo ordinal.
     segun el calendario con que se prepare la configuracion de nemo. Estos pueden ser:
      gregorian, noleap, all_leap, 360_day, julian

     El parametro 'give' se utiliza para escojer el valor que regresa la funcion:
      full (default) : Regresa el valor ordinal al que corresponde la fecha 'data' (datetime) segun
                       el calendario quese haya seleccionado en 'ctype' 
      monthLen       : Regresa los dias que tiene el mes contenido en la fecha 'data' (datetime), segun
                       el calendario que se haya seleccionado en 'ctype'
      yearLen        : Regresa los dias que contiene el ano, segun el calendario que se haya seleccionado
                       en 'ctype'

     Se utiliza como fecha epoch 1-1-1950 
    """
    epochy = 1950
    epochm = 1
    epochd = 1
    if ctype == 'gregorian':
        oneyear = 365.2425
        ml = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    elif ctype == 'noleap':
        oneyear = 365 
        ml = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    elif ctype == 'all_leap': # 366_day
        oneyear = 366.0
        ml = np.array([31,29,31,30,31,30,31,31,30,31,30,31])
    elif ctype == '360_day':
        oneyear = 360.0 
        ml = np.array([30,30,30,30,30,30,30,30,30,30,30,30])
    elif ctype == 'julian':
        oneyear = 365.25
        ml = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    
    if give == 'yearLen':
        return oneyear

    if give == 'monthLen':
        return ml[data.month -1] 

    if (not isinstance(data,np.ndarray)):
        data = np.array([data])

    newc = np.zeros((len(data)),float)
    for idx,v in enumerate(data):
        y = v.year 
        m = v.month - 1
        d = (v.day-1) + (v.hour / 24.0) 
        nnumdate = (y - epochy) * oneyear 
        for nm in range(0,m):
            nnumdate = nnumdate + (ml[nm])
        nnumdate = nnumdate + d 
        newc[idx] = nnumdate 

    return np.squeeze(newc) 


def interpIrregularGridToRegular(xgrid, ygrid, zdata, xgridnew, ygridnew, imethod='linear'):
    """
     Funcion para interpolar una seccion 2D zdata[:,:] con puntos validos en 
     el conjunto de coordenadas xgrid[:],ygrid[:] a una nueva malla 2D con 
     dimensiones xgridnew, ygridnew.
     Se utiliza el metodo de numpy griddata para realizar la operacion.
    """
    # Obtener los puntos de la malla de zdata donde los valores 
    # no tengan mascara, es decir solo oceano.
    xdata_valid = np.array([])
    ydata_valid = np.array([]) 
    pointdata   = np.array([])
    for i in range(xgrid.size):
        for j in range(ygrid.size):
            if (not (zdata[i,j] is np.ma.masked)):
                xdata_valid = np.append(xdata_valid,xgrid[i]) # 
                ydata_valid = np.append(ydata_valid,ygrid[j]) #  
                pointdata = np.append(pointdata,zdata[i,j]) 
    # Crear la malla regular, con el metodo griddata, que a partir de la lista de puntos pointdata (X=xdata_valid,Y=ydata_valid) , va a generar la malla 
    # con X = ygridnew[None,:] Y = xgridnew[:,None] 
    newZdata = interpolate.griddata((ydata_valid,xdata_valid), pointdata, (ygridnew[None,:],xgridnew[:,None]), method=imethod) 
    return newZdata 


def applyMask(zData,mask):
    """
     Funcion para regresar los datos "zData" con la mascara que contiene "mask"
     con los datos enmascarados llenados con "fill_value" 
    """
    zDataMasked = np.ma.masked_where(mask==0,zData)  
    np.ma.set_fill_value(zDataMasked,0)
    return zDataMasked.filled()  


def crearFronterasEsteSur(dataSourceFile,sMaskFile,iEastIndex=-1,iSouthIndex=1, fileOutPrefix = 'obc_', saveMethod = 1, sFilesSize = 'yearly'):  
    """
     Script para crear archivos OBC - Entrada simulacion NEMO-OPA 
     En especifico para archivos frontera Este y Sur.
     Mallas 24 y 12 de grado.

     sMaskFile , variables requeridas:
      nav_lat : latitudes en formato curvilineo
      nav_lon : longitudes en formato curvilineo
      nav_lat : profundidades.
      tmask : Variable mascara de la batimetria, dimensiones (t,z,y,x) 

     saveMethod : 
       1 : Salva los datos interpolados en archivos netcdf. 
       2 : Salva los datos interpolados en archivos netcdf con estructura anual o mensual.

    """
    # Configuration paths, indices fronteras,
    # sMaskFile   - Archivo de mascara de batimetria, con nav_lon,nav_lat,nav_lev de la malla
    # iEastIndex  - Indice x (longitudes) para la frontera este
    # iSouthIndex - Indice y (latitudes) para la frontera sur 

    log.info('Proceso para generacion de archivos de fronteras - NEMO')
    log.info('Archivo fuente: ' + dataSourceFile) 
    log.info('Archivo de mascara: ' + sMaskFile)
    sCalendarType = 'noleap'
    ##
    # Cargar datos de la mascara GOLFO24 malla T 
    ##
    ncMask = nc.Dataset(sMaskFile,'r')
    ncMaskLat = ncMask.variables['nav_lat'][:]
    ncMaskLon = ncMask.variables['nav_lon'][:]
    ncMaskDepth = ncMask.variables['nav_lev'][:]


    # Mascara de rebanadas en frontera este y sur.
    ncMaskT = ncMask.variables['tmask'][0][0][:][:]
    ncMaskEast = ncMask.variables['tmask'][0,:,:,iEastIndex] 
    ncMaskSouth = ncMask.variables['tmask'][0,:,iSouthIndex,:] 

    ncMaskEastLon = ncMaskLon[:,iEastIndex]
    ncMaskEastLat = ncMaskLat[:,iEastIndex]

    ncMaskSouthLon = ncMaskLon[iSouthIndex,:]
    ncMaskSouthLat = ncMaskLat[iSouthIndex,:]
    ncMask.close() 

    ##
    # Cargar datos del archivo de mercator
    ##
    ncMer = nc.Dataset(dataSourceFile,'r') 
    ncMerLat = ncMer.variables['latitude'][:]
    ncMerLon = ncMer.variables['longitude'][:] 
    ncMerDepth = ncMer.variables['depth'][:] 
    ncMerTime = ncMer.variables['time_counter'][:] 
    ncMerTime_units = ncMer.variables['time_counter'].units 
    ncMerTime_calendar = ncMer.variables['time_counter'].calendar

    # Obtener el indice mas cercano al requerido para la frontera este y sur
    # haciendo la diferencia mas pequena de las posiciones en la mascara con los datos de mercator
    iMerEastIndex = np.argmin(np.abs(ncMerLon - np.max(ncMaskEastLon))) 
    iMerSouthIndex = np.argmin(np.abs(ncMerLat - np.min(ncMaskSouthLat)))

    iMaxMaskLatInData = np.argmin(np.abs(ncMerLat - np.max(ncMaskLat)))
    iMinMaskLatInData = np.argmin(np.abs(ncMerLat - np.min(ncMaskLat)))
    #print 'MaskLatInData Max, Min ' + str(iMaxMaskLatInData) + ' , ' + str(iMinMaskLatInData)

    iMaxMaskLonInData = np.argmin(np.abs(ncMerLon - np.max(ncMaskLon)))
    iMinMaskLonInData = np.argmin(np.abs(ncMerLon - np.min(ncMaskLon)))   
    #print 'MaskLonInData Max, Min ' + str(iMaxMaskLonInData) + ' , ' + str(iMinMaskLonInData)

    ##
    # Arreglos donde descargar datos interpolados, reservando memoria.
    ##
    nEastTempGrid = np.zeros((ncMerTime.size,ncMaskDepth.size,ncMaskEastLat.size), float)
    nEastSalGrid  = np.zeros((ncMerTime.size,ncMaskDepth.size,ncMaskEastLat.size), float) 
    nEastUcompGrid = np.zeros((ncMerTime.size,ncMaskDepth.size,ncMaskEastLat.size), float) 
    nEastVcompGrid = np.zeros((ncMerTime.size,ncMaskDepth.size,ncMaskEastLat.size), float) 

    nSouthTempGrid = np.zeros((ncMerTime.size,ncMaskDepth.size,ncMaskSouthLon.size), float)
    nSouthSalGrid  = np.zeros((ncMerTime.size,ncMaskDepth.size,ncMaskSouthLon.size), float) 
    nSouthUcompGrid = np.zeros((ncMerTime.size,ncMaskDepth.size,ncMaskSouthLon.size), float) 
    nSouthVcompGrid = np.zeros((ncMerTime.size,ncMaskDepth.size,ncMaskSouthLon.size), float) 
    ##
    # Ciclar en rango de la variable temporal del archivo dataSourceFile
    ##
    for idx,merTime in enumerate(ncMerTime):
        log.info('Proceso de interpolacion, indice: ' + str(idx) + '  Tiempo: ' + str(merTime))
        #________________
        ## TEMPERATURA
        # Frontera Este
        EastSlice = ncMer.variables['temperature'][idx,:,iMinMaskLatInData:iMaxMaskLatInData,iMerEastIndex]
        EastSlice[EastSlice > 0] = EastSlice[EastSlice > 0] - 272.15
        
        # interpolar 
        nEastTempGrid[idx,:,:] = applyMask(interpIrregularGridToRegular(ncMerDepth,ncMerLat[iMinMaskLatInData:iMaxMaskLatInData],EastSlice,ncMaskDepth,ncMaskEastLat,'nearest'), ncMaskEast)
        # Frontera Sur
        SouthSlice = ncMer.variables['temperature'][idx,:,iMerSouthIndex,iMinMaskLonInData:iMaxMaskLonInData]
        SouthSlice[SouthSlice > 0] = SouthSlice[SouthSlice > 0] - 272.15
        # interpolar 
        nSouthTempGrid[idx,:,:] = applyMask(interpIrregularGridToRegular(ncMerDepth,ncMerLon[iMinMaskLonInData:iMaxMaskLonInData],SouthSlice,ncMaskDepth,ncMaskSouthLon,'nearest'), ncMaskSouth)

        #________________
        ## SALINIDAD
        # Frontera Este
        EastSlice = ncMer.variables['salinity'][idx,:,iMinMaskLatInData:iMaxMaskLatInData,iMerEastIndex]
        # interpolar 
        nEastSalGrid[idx,:,:] = applyMask(interpIrregularGridToRegular(ncMerDepth,ncMerLat[iMinMaskLatInData:iMaxMaskLatInData],EastSlice,ncMaskDepth,ncMaskEastLat,'nearest'), ncMaskEast )
        # Frontera Sur
        SouthSlice = ncMer.variables['salinity'][idx,:,iMerSouthIndex,iMinMaskLonInData:iMaxMaskLonInData]
        # interpolar 
        nSouthSalGrid[idx,:,:] = applyMask(interpIrregularGridToRegular(ncMerDepth,ncMerLon[iMinMaskLonInData:iMaxMaskLonInData],SouthSlice,ncMaskDepth,ncMaskSouthLon,'nearest'), ncMaskSouth )   

        #________________
        ## Componente de Velocidad U
        # Frontera Este
        EastSlice = ncMer.variables['u'][idx,:,iMinMaskLatInData:iMaxMaskLatInData,iMerEastIndex]
        # interpolar 
        nEastUcompGrid[idx,:,:] = applyMask(interpIrregularGridToRegular(ncMerDepth,ncMerLat[iMinMaskLatInData:iMaxMaskLatInData],EastSlice,ncMaskDepth,ncMaskEastLat,'nearest'), ncMaskEast )
        # Frontera Sur
        SouthSlice = ncMer.variables['u'][idx,:,iMerSouthIndex,iMinMaskLonInData:iMaxMaskLonInData]
        # interpolar 
        nSouthUcompGrid[idx,:,:] = applyMask(interpIrregularGridToRegular(ncMerDepth,ncMerLon[iMinMaskLonInData:iMaxMaskLonInData],SouthSlice,ncMaskDepth,ncMaskSouthLon,'nearest'), ncMaskSouth ) 

        #________________
        ## Componente de Velocidad V
        # Frontera Este
        EastSlice = ncMer.variables['v'][idx,:,iMinMaskLatInData:iMaxMaskLatInData,iMerEastIndex]
        # interpolar 
        nEastVcompGrid[idx,:,:] = applyMask(interpIrregularGridToRegular(ncMerDepth,ncMerLat[iMinMaskLatInData:iMaxMaskLatInData],EastSlice,ncMaskDepth,ncMaskEastLat,'nearest'), ncMaskEast )
        # Frontera Sur
        SouthSlice = ncMer.variables['v'][idx,:,iMerSouthIndex,iMinMaskLonInData:iMaxMaskLonInData]
        # interpolar 
        nSouthVcompGrid[idx,:,:] = applyMask(interpIrregularGridToRegular(ncMerDepth,ncMerLon[iMinMaskLonInData:iMaxMaskLonInData],SouthSlice,ncMaskDepth,ncMaskSouthLon,'nearest'), ncMaskSouth )                 

    ncMer.close()
    
    # Salvar estos datos en un archivo netcdf
    # Method =  1 - Salvar sin estructura mensual-anual, solo los datos de entrada
    # Method =  2 - Salvar datos con estructura mensual o anual

    if saveMethod==1:
        # EAST Files
        dDims = {'time_counter':None , 'depth' : ncMaskDepth.size , 'y' : ncMaskEastLat.size} 
        dVars = {'dimensions' : ['time_counter','depth','y'] , 'attributes' : {'_FillValue':0}, 'dataType' : 'f4' } 
        dVarsTS = {} 
        dVarsTS['temp'] = dVars 
        dVarsTS['salinity'] = dVars 
        ncOutFile = netcdfFile.netcdfFile()
        ncOutFile.createFile('EastTS_OBC.nc') 
        ncOutFile.createDims(dDims)
        ncOutFile.createVars(dVarsTS)
        ncOutFile.saveData({'temp' : nEastTempGrid , 'salinity' : nEastSalGrid})
        ncOutFile.closeFile()


        ncOutFile = netcdfFile.netcdfFile()
        ncOutFile.createFile('EastU_OBC.nc') 
        ncOutFile.createDims(dDims)
        ncOutFile.createVars( {'u' : dVars} )
        ncOutFile.saveData( {'u' : nEastUcompGrid } )
        ncOutFile.closeFile()   

        ncOutFile = netcdfFile.netcdfFile()
        ncOutFile.createFile('EastV_OBC.nc') 
        ncOutFile.createDims(dDims)
        ncOutFile.createVars( {'v' : dVars} )
        ncOutFile.saveData( {'v' : nEastVcompGrid } )
        ncOutFile.closeFile()   

        # SOUTH Files
        dDims = {'time_counter':None , 'depth' : ncMaskDepth.size , 'x': ncMaskSouthLon.size} 
        dVars = {'dimensions' : ['time_counter','depth','x'] , 'attributes' : {'_FillValue':0}, 'dataType' : 'f4' } 
        dVarsTS = {} 
        dVarsTS['temp'] = dVars 
        dVarsTS['salinity'] = dVars 
        ncOutFile = netcdfFile.netcdfFile()
        ncOutFile.createFile('SouthTS_OBC.nc') 
        ncOutFile.createDims(dDims)
        ncOutFile.createVars(dVarsTS)
        ncOutFile.saveData({'temp' : nSouthTempGrid , 'salinity' : nSouthSalGrid})
        ncOutFile.closeFile()


        ncOutFile = netcdfFile.netcdfFile()
        ncOutFile.createFile('SouthtU_OBC.nc') 
        ncOutFile.createDims(dDims)
        ncOutFile.createVars( {'u' : dVars} )
        ncOutFile.saveData( {'u' : nSouthUcompGrid } )
        ncOutFile.closeFile()   

        ncOutFile = netcdfFile.netcdfFile()
        ncOutFile.createFile('SouthV_OBC.nc') 
        ncOutFile.createDims(dDims)
        ncOutFile.createVars( {'v' : dVars} )
        ncOutFile.saveData( {'v' : nSouthVcompGrid } )
        ncOutFile.closeFile()   

    elif saveMethod == 2:
        """
         Metodo para salvar los datos interpolados en archivos anuales o mensuales
        """        
        # sFilesSize = 'yearly' # Or 'monthly'

        # Correr el ciclo con los valores de la dimension temporal: ncMerTime 
        # dependiendo de su mes y ano generamos un nuevo archivo mensual o anual segun sFilesSize
        # para almacenar la informacion
        log.info('Interpolacion lista, salvando datos en formato para NEMO. Archivos: ' + sFilesSize)
        # Variable del archivo donde se estan almacenando los datos durante el ciclo.
        currentTimeFile = None 
        for idx_tval, tval in enumerate(ncMerTime):
            # Convertir el tval a <python datetime>
            tval_datetime = nc.num2date(tval,ncMerTime_units,ncMerTime_calendar) 
            log.info('Salvando indice: ' + str(idx_tval) + ' Tiempo: ' + str(tval_datetime))

            compareVarDummyForFileSize = tval_datetime.year if (sFilesSize == 'yearly') else tval_datetime.month 

            if currentTimeFile == None or currentTimeFile != compareVarDummyForFileSize:
                # Crear el archivo(s) para el periodo mensual o anual segun corresponda
                if sFilesSize == 'yearly':
                    currentTimeFile = tval_datetime.year
                else:
                    currentTimeFile = tval_datetime.month                

                # Empezar por la creacion de la variable de dimension temporal.
                # ds es el primer dia del (mes o ano) menos un dia 
                if sFilesSize == 'yearly':
                    ds = dt.datetime(tval_datetime.year,1,1,tval_datetime.hour) - dt.timedelta(days=1)
                else:
                    ds = dt.datetime(tval_datetime.year,tval_datetime.month,1,tval_datetime.hour) - dt.timedelta(days=1)
                # nm es el numero de mes, del archivo actual
                # ny es el numero de ano, del archivo actual 
                nm = tval_datetime.month
                ny = tval_datetime.year                
           
                # Tamano de la dimension temporal de este periodo, segun el calendario que se utilize.
                if sFilesSize == 'yearly':
                    sFileTDimSize = dateToNemoCalendar(tval_datetime,sCalendarType,'yearLen') + 2
                else:
                    sFileTDimSize = dateToNemoCalendar(tval_datetime,sCalendarType,'monthLen') + 2

                # timeVD es la variable de la dimension temporal, contiene el tamano del periodo actual segun el calendario
                # que se utilize (gregoriam, noleap, 366day, 360day) mas 1 dia atras y 1 dia adelante. 
                timeVD = [] 
                for i in range(0,sFileTDimSize):
                    # timeVD.append(nc.date2num(ds + dt.timedelta(days=1*i) , ncMerTime_units, ncMerTime_calendar ))
                    timeVD.append(dateToNemoCalendar(ds + dt.timedelta(days=1*i) , sCalendarType)) 

                ncOutFiles = {}
                # Creamos los archivos netcdf para descargar datos.
                # Dimensiones, variables y atributos para archivos ESTE y SUR (TS,U,V)
                dDimEastT = {'time_counter':None , 'deptht' : ncMaskDepth.size , 'y' : ncMaskEastLat.size} 
                dDimVarsT = {'time_counter': {'dimensions':['time_counter']  , 'dataType' : 'f4' } , 'deptht' : {'dimensions':['deptht'],'dataType':'f4'} }
                dVarPropertiesEastT = {'dimensions' : ['time_counter','deptht','y'] , 'attributes' : {'_FillValue':0} , 'dataType' : 'f4' } 
                dDimSouthT = {'time_counter':None , 'deptht' : ncMaskDepth.size , 'x' : ncMaskSouthLon.size} 
                dVarPropertiesSouthT = {'dimensions' : ['time_counter','deptht','x'] , 'attributes' : {'_FillValue':0}, 'dataType' : 'f4' }

                dDimEastU = {'time_counter':None , 'depthu' : ncMaskDepth.size , 'y' : ncMaskEastLat.size} 
                dDimVarsU = {'time_counter': {'dimensions':['time_counter']  , 'dataType' : 'f4' } , 'depthu' : {'dimensions':['depthu'],'dataType':'f4'} } 
                dVarPropertiesEastU = {'dimensions' : ['time_counter','depthu','y'] , 'attributes' : {'_FillValue':0}, 'dataType' : 'f4' } 
                dDimSouthU = {'time_counter':None , 'depthu' : ncMaskDepth.size , 'x' : ncMaskSouthLon.size} 
                dVarPropertiesSouthU = {'dimensions' : ['time_counter','depthu','x'] , 'attributes' : {'_FillValue':0}, 'dataType' : 'f4' }

                dDimEastV = {'time_counter':None , 'depthv' : ncMaskDepth.size , 'y' : ncMaskEastLat.size} 
                dDimVarsV = {'time_counter': {'dimensions':['time_counter'] , 'dataType' : 'f4' } , 'depthv' : {'dimensions':['depthv'],'dataType':'f4'} } 
                dVarPropertiesEastV = {'dimensions' : ['time_counter','depthv','y'] , 'attributes' : {'_FillValue':0}, 'dataType' : 'f4' } 
                dDimSouthV = {'time_counter':None , 'depthv' : ncMaskDepth.size , 'x' : ncMaskSouthLon.size} 
                dVarPropertiesSouthV = {'dimensions' : ['time_counter','depthv','x'] , 'attributes' : {'_FillValue':0}, 'dataType' : 'f4' }

                #############################
                # Archivos de Frontera ESTE #
                #############################
                
                if sFilesSize == 'yearly':
                    sFileOutSuffix = 'y' + str(ny) + 'm00.nc'
                else:
                    sFileOutSuffix = 'y' + str(ny) + 'm' + ("%02d"%nm) + '.nc'

                obcFileName = fileOutPrefix + '_east_TS_' + sFileOutSuffix
                ncOutFiles['eastTS'] = netcdfFile.netcdfFile() 
                ncOutFiles['eastTS'].createFile(obcFileName)
                ncOutFiles['eastTS'].createDims(dDimEastT) 
                ncOutFiles['eastTS'].createVars(dDimVarsT) 
                ncOutFiles['eastTS'].createVars({'votemper' : dVarPropertiesEastT , 'vosaline' : dVarPropertiesEastT }) 
                ncOutFiles['eastTS'].saveData({'time_counter' : timeVD , 'deptht' : ncMaskDepth[:]})

                obcFileName = fileOutPrefix + '_east_U_' + sFileOutSuffix
                ncOutFiles['eastU'] = netcdfFile.netcdfFile()
                ncOutFiles['eastU'].createFile(obcFileName)
                ncOutFiles['eastU'].createDims(dDimEastU) 
                ncOutFiles['eastU'].createVars(dDimVarsU) 
                ncOutFiles['eastU'].createVars({'vozocrtx' : dVarPropertiesEastU}) 
                ncOutFiles['eastU'].saveData({'time_counter' : timeVD , 'depthu' : ncMaskDepth[:]})

                obcFileName = fileOutPrefix + '_east_V_' + sFileOutSuffix
                ncOutFiles['eastV'] = netcdfFile.netcdfFile()
                ncOutFiles['eastV'].createFile(obcFileName)
                ncOutFiles['eastV'].createDims(dDimEastV) 
                ncOutFiles['eastV'].createVars(dDimVarsV) 
                ncOutFiles['eastV'].createVars({'vomecrty' : dVarPropertiesEastV}) 
                ncOutFiles['eastV'].saveData({'time_counter' : timeVD , 'depthv' : ncMaskDepth[:]})

                ############################
                # Archivos de Frontera SUR #   
                ############################                
                obcFileName = fileOutPrefix + '_south_TS_' + sFileOutSuffix
                ncOutFiles['southTS'] = netcdfFile.netcdfFile()
                ncOutFiles['southTS'].createFile(obcFileName)
                ncOutFiles['southTS'].createDims(dDimSouthT) 
                ncOutFiles['southTS'].createVars(dDimVarsT) 
                ncOutFiles['southTS'].createVars({'votemper' : dVarPropertiesSouthT , 'vosaline' : dVarPropertiesSouthT }) 
                ncOutFiles['southTS'].saveData({'time_counter' : timeVD , 'deptht' : ncMaskDepth[:]})

                obcFileName = fileOutPrefix + '_south_U_' + sFileOutSuffix
                ncOutFiles['southU'] = netcdfFile.netcdfFile()
                ncOutFiles['southU'].createFile(obcFileName)
                ncOutFiles['southU'].createDims(dDimSouthU) 
                ncOutFiles['southU'].createVars(dDimVarsU) 
                ncOutFiles['southU'].createVars({'vozocrtx' : dVarPropertiesSouthU}) 
                ncOutFiles['southU'].saveData({'time_counter' : timeVD , 'depthu' : ncMaskDepth[:]})
                
                obcFileName = fileOutPrefix + '_south_V_' + sFileOutSuffix
                ncOutFiles['southV'] = netcdfFile.netcdfFile()
                ncOutFiles['southV'].createFile(obcFileName)
                ncOutFiles['southV'].createDims(dDimSouthV) 
                ncOutFiles['southV'].createVars(dDimVarsV) 
                ncOutFiles['southV'].createVars({'vomecrty' : dVarPropertiesSouthV}) 
                ncOutFiles['southV'].saveData({'time_counter' : timeVD , 'depthv' : ncMaskDepth[:]})

            #############################################
            # FIN BLOQUE Creacion de archivos frontera. #
            ############################################# 

            # Salvar cada dato en su archivo correspondiente.
            
            # Primero localizar el indice donde pondremos el dato: 
            idx = (np.abs(timeVD - dateToNemoCalendar(tval_datetime,sCalendarType,))).argmin()
            log.info('Salvando en el archivo, con indice: ' + str(idx))

            ncOutFiles['eastTS'].saveDataS('votemper' , nEastTempGrid[idx_tval,:,:] , (idx))
            ncOutFiles['eastTS'].saveDataS('vosaline' , nEastSalGrid[idx_tval,:,:] , (idx))
            ncOutFiles['eastU'].saveDataS('vozocrtx' , nEastUcompGrid[idx_tval,:,:] , (idx))
            ncOutFiles['eastV'].saveDataS('vomecrty' , nEastVcompGrid[idx_tval,:,:] , (idx)) 

            ncOutFiles['southTS'].saveDataS('votemper' , nSouthTempGrid[idx_tval,:,:] , (idx))
            ncOutFiles['southTS'].saveDataS('vosaline' , nSouthSalGrid[idx_tval,:,:] , (idx))
            ncOutFiles['southU'].saveDataS('vozocrtx' , nSouthUcompGrid[idx_tval,:,:] , (idx))
            ncOutFiles['southV'].saveDataS('vomecrty' , nSouthVcompGrid[idx_tval,:,:] , (idx)) 

            # Llenar el ultimo valor, en los archivos, para lograr "permanencia."
            if (idx_tval == (ncMerTime.size-1)):
	            ncOutFiles['eastTS'].saveDataS('votemper' , nEastTempGrid[idx_tval,:,:] , slice(idx,timeVD.size) )
	            ncOutFiles['eastTS'].saveDataS('vosaline' , nEastSalGrid[idx_tval,:,:] , slice(idx,timeVD.size) )
	            ncOutFiles['eastU'].saveDataS('vozocrtx' , nEastUcompGrid[idx_tval,:,:] , slice(idx,timeVD.size) )
	            ncOutFiles['eastV'].saveDataS('vomecrty' , nEastVcompGrid[idx_tval,:,:] , slice(idx,timeVD.size) ) 

	            ncOutFiles['southTS'].saveDataS('votemper' , nSouthTempGrid[idx_tval,:,:] , slice(idx,timeVD.size) )
	            ncOutFiles['southTS'].saveDataS('vosaline' , nSouthSalGrid[idx_tval,:,:] , slice(idx,timeVD.size) )
	            ncOutFiles['southU'].saveDataS('vozocrtx' , nSouthUcompGrid[idx_tval,:,:] , slice(idx,timeVD.size) )
	            ncOutFiles['southV'].saveDataS('vomecrty' , nSouthVcompGrid[idx_tval,:,:] , slice(idx,timeVD.size) )            	


            # Lidiar con los indices -1 y +1 del periodo temporal. 
            if sFilesSize == 'yearly':
                conditionLessOne = (tval_datetime.month == 1 and tval_datetime.day == 1 and idx_tval > 0)
            else:
                conditionLessOne = (tval_datetime.day == 1 and idx_tval > 0)

            if conditionLessOne:
                ncOutFiles['eastTS'].saveDataS('votemper' , nEastTempGrid[idx_tval-1,:,:] , (0))
                ncOutFiles['eastTS'].saveDataS('vosaline' , nEastSalGrid[idx_tval-1,:,:] , (0))
                ncOutFiles['eastU'].saveDataS('vozocrtx' , nEastUcompGrid[idx_tval-1,:,:] , (0))
                ncOutFiles['eastV'].saveDataS('vomecrty' , nEastVcompGrid[idx_tval-1,:,:] , (0)) 

                ncOutFiles['southTS'].saveDataS('votemper' , nSouthTempGrid[idx_tval-1,:,:] , (0))
                ncOutFiles['southTS'].saveDataS('vosaline' , nSouthSalGrid[idx_tval-1,:,:] , (0))
                ncOutFiles['southU'].saveDataS('vozocrtx' , nSouthUcompGrid[idx_tval-1,:,:] , (0))
                ncOutFiles['southV'].saveDataS('vomecrty' , nSouthVcompGrid[idx_tval-1,:,:] , (0))  

            if sFilesSize == 'yearly':
                conditionPlusOne = (idx_tval < (ncMerTime.size-1) and nc.num2date(ncMerTime[idx_tval+1],ncMerTime_units,ncMerTime_calendar).year != currentTimeFile)
            else:
                conditionPlusOne = (idx_tval < (ncMerTime.size-1) and nc.num2date(ncMerTime[idx_tval+1],ncMerTime_units,ncMerTime_calendar).month != currentTimeFile)
            if conditionPlusOne:
                ncOutFiles['eastTS'].saveDataS('votemper' , nEastTempGrid[idx_tval+1,:,:] , (sFileTDimSize-1))
                ncOutFiles['eastTS'].saveDataS('vosaline' , nEastSalGrid[idx_tval+1,:,:] , (sFileTDimSize-1))
                ncOutFiles['eastU'].saveDataS('vozocrtx' , nEastUcompGrid[idx_tval+1,:,:] , (sFileTDimSize-1))
                ncOutFiles['eastV'].saveDataS('vomecrty' , nEastVcompGrid[idx_tval+1,:,:] , (sFileTDimSize-1)) 

                ncOutFiles['southTS'].saveDataS('votemper' , nSouthTempGrid[idx_tval+1,:,:] , (sFileTDimSize-1))
                ncOutFiles['southTS'].saveDataS('vosaline' , nSouthSalGrid[idx_tval+1,:,:] , (sFileTDimSize-1))
                ncOutFiles['southU'].saveDataS('vozocrtx' , nSouthUcompGrid[idx_tval+1,:,:] , (sFileTDimSize-1))
                ncOutFiles['southV'].saveDataS('vomecrty' , nSouthVcompGrid[idx_tval+1,:,:] , (sFileTDimSize-1))                            



    log.info('Archivos de frontera creados.')
    log.info('OK')





def main():
    log.getLogger().setLevel(10)
    # Demo de como crear fronteras OBC (NEMO-READY) con estructura mensual.
    # crearFronterasEsteSur('downloaded_mercator_query_20140724-20140812.nc','GOLFO24_mask.nc', 430-2 , 1, 'obc_GOLFO24', 2, 'monthly')
    # Demo de como crear fronteras OBC (NEMO-READY) con estructura anual.
    crearFronterasEsteSur('downloaded_mercator_query_20140724-20140812.nc','GOLFO12_mask.nc', 203 , 1, 'obc_GOLFO12', 2, 'yearly')

if __name__ == "__main__":
    main()
