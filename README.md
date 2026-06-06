**Demo YOLO — Instalación en Windows (entorno virtual)**

Este repositorio contiene demos y scripts para ejecutar modelos YOLO en Windows. Este README describe cómo crear y usar un entorno virtual en Windows y cómo instalar las dependencias.

**Requisitos:**
- Python 3.8 o superior instalado y accesible como `python`.
- Conexión a Internet para instalar paquetes.

**Pasos de instalación (PowerShell)**

1. Abre PowerShell y sitúate en la carpeta del proyecto (donde está este README).

```powershell
cd D:\UCALDAS\Vision Artificial\codigos\demo_yolo
```

2. Crea el entorno virtual:

```powershell
python -m venv venv
```

3. (Opcional) Permite la ejecución de scripts en PowerShell si recibes un error al activar:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

4. Activa el entorno virtual (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

Si usas CMD en lugar de PowerShell:

```cmd
venv\Scripts\activate.bat
```

5. Actualiza `pip` y instala las dependencias:

```powershell
python -m pip install --upgrade pip
pip install -r [requirements.txt](requirements.txt#L1)
```

6. Verifica que los modelos están en la carpeta `models/`. Este repositorio incluye:
- [models/yolov8n.pt](models/yolov8n.pt#L1)
- [models/yolov8n-pose.pt](models/yolov8n-pose.pt#L1)

7. Ejecuta el script de ejemplo:

```powershell
python [yolo_demo_v01.py](yolo_demo_v01.py#L1)
```

**Comandos útiles**
- Desactivar el entorno virtual:

```powershell
deactivate
```

**Notas y solución de problemas**
- Si `python` no está disponible, usa la ruta completa al ejecutable de Python o añade Python al PATH.
- Si la instalación de paquetes falla por compilación de ruedas, instala las dependencias del sistema necesarias o busca ruedas precompiladas para Windows.
- Si PowerShell bloquea la activación por política de ejecución, ejecuta `Set-ExecutionPolicy` como se indica arriba.

**Archivos relevantes**
- Dependencias: [requirements.txt](requirements.txt#L1)
- Demo principal: [yolo_demo_v01.py](1_yolo_demo_v01.py#L1)
