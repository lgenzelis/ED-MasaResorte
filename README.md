# Simulador de sistemas masa resorte

El sistema puede incluir (o no): 
- Amortiguamiento
- Fuerza externa

## Instrucciones de uso

Editar el archivo `main.py` para cargar los parámetros específicos del sistema que deseen simular. Luego, ejecutar con:

```
$ python3 main.py
```

## Instrucciones de instalación

Antes que nada, deben clonar este repositorio (o descargarlo como un zip y descomprimirlo). 

Lo mejor siempre es trabajar con entornos virtuales de python (`virtualenv`). Si no van a usar un entorno virtual, pueden saltearse los dos primeros pasos.
 ```
$ virtualenv mi_entorno_virtual
$ source mi_entorno_virtual/bin/activate
```
Ahora, ya sea que estén o no trabajando con entornos virtuales:
```
$ pip3 install -r requirements.txt
```
