# fronterasOBC
script para generar archivos de fronteras formato OBC para el modelo nemo-opa

Metodo en py

def crearFronterasEsteSur (dataSourceFile,sMaskFile,iEastIndex=-1,iSouthIndex=1, fileOutPrefix = 'obc_', saveMethod = 1, sFilesSize = 'yearly')

     saveMethod : 
       1 : Salva los datos interpolados en archivos netcdf. 
       2 : Salva los datos interpolados en archivos netcdf con estructura anual o mensual.
