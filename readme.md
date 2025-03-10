-------------------------------------
BLOQUE 1: PREPROCESAMIENTO DE DATOS
-------------------------------------

### **1. Agrupación de Variables:**
- **`market_segment`:**
  - **Direct**: Agrupar las reservas hechas directamente por los clientes.
  - **Business**: Agrupar las reservas para viajes de negocios (incluyendo los valores **`Corporate`** y **`Aviation`**).
  - **Leisure**: Agrupar las reservas de turistas o clientes individuales (incluyendo los valores **`Online TA`**, **`Offline TA/TO`** y **`Groups`**).
  - **Event**: Agrupar las reservas para grupos grandes o eventos.
  - **Complementary**: Agrupar las reservas sin costo o promocionales.

- **`distribution_channel`:**
  - **Direct**: Reservas hechas directamente con el hotel.
  - **Agency**: Reservas realizadas a través de agencias de viajes u operadores turísticos (incluyendo los valores **`OTA`** y **`Offline TA/TO`**).
  - **GDS**: Reservas realizadas mediante un sistema global de distribución.
  - **Corporate**: Reservas gestionadas a través de canales corporativos.

### **2. Modificación de Columnas:**
- Eliminar la columna **`country_x`**.
- Eliminar la columna **`hotel_id`**.
- Reconfigurar la columna **`special_requests`**: Asignar valor **0** para "Ninguna" solicitud especial y valor **1** para "Una o más" solicitudes especiales.
- Eliminar las filas de la columna **`reservation_status`** donde el valor sea **'No-Show'**.
- Renombrar la columna **`country_y`** a **`country`**.
- Cambiar el tipo de dato de las columnas **`parking`**, **`restaurant`** y **`pool_and_spa`** a binario (0 y 1).
- Categorizar los valores en la columna **`room_type`**: Asignar **A** al valor original **'A'** y **X** para el resto de los valores.
  - Aplicar la transformación para que **`room_type`** sea 1 si es **"A"**, de lo contrario 0.
- Crear la columna **`hotel_type`**: Asignar **1** para **"City Hotel"** y **0** para otros valores.

### **3. Validación del Dataset:**
- Crear un dataset de validación extrayendo las filas donde la columna **`reservation_status`** tenga el valor **'Booked'**.

### **4. Nueva Columna Objetivo:**
- Calcular la diferencia de fechas restando la columna **`arrival_date`** de **`reservation_status_date`** para obtener la diferencia en días.
- Filtrar las reservas donde **`reservation_status`** sea **'Canceled'** y cuya diferencia de fechas calculada sea menor o igual a 30 días.
- Crear la columna binaria **`cancelled_last_30_days`**: Asignar valor **1** si la reserva fue cancelada dentro de los últimos 30 días, y valor **0** para todas las demás reservas (no canceladas o canceladas con más de 30 días de antelación).
- Eliminar las columnas **`reservation_status_date`**, **`booking_date`** y **`arrival_date`** una vez creada la columna **`cancelled_last_30_days`**.

### **5. Manejo de Outliers:**
- Para la columna **`rate`**, calcular los cuartiles y el rango intercuartílico (IQR) y eliminar las filas fuera del rango [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
- Para la columna **`total_guests`**, calcular los cuartiles y el IQR, y eliminar las filas fuera del rango [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].

### **6. Filtrado de Fechas:**
- Eliminar filas con **`booking_date`** posterior a 2017-06-30. Convertir la columna **`booking_date`** a tipo datetime y filtrar las filas con fechas anteriores o iguales a 2017-06-30.
- Para el dataset de validación, filtrar las filas donde la columna **`booking_date`** no sea posterior a 2017-06-30.

### **7. Nueva Columna `advance_reservation`:**
- Calcular la diferencia entre **`booking_date`** y **`arrival_date`** en ambos DataFrames (de entrenamiento y de validación). Esta diferencia representará el número de días de antelación con el que se hizo la reserva, y se almacenará en la nueva columna **`advance_reservation_days`**.

### **8. One-Hot Encoding:**
- Realizar One-Hot Encoding para las columnas **`board`**, **`distribution_channel`**, **`country`** y **`market_segment`** después de la preparación de los datos.

### **9. Normalización de las columnas numéricas:**
- Realizar una normalización de las columnas numéricas, excluyendo las variables binarias (como **`parking`**, **`restaurant`**, **`pool_and_spa`**, **`room_type`**, **`hotel_type`**, etc.) que ya están en formato binario.
- Aplicar **`StandardScaler`** a las columnas numéricas restantes para escalarlas.

### **10. Filtrar Fechas y Evitar Data Leakage:**
- Filtrar las fechas según el valor de **`booking_date`**, asegurándose de que no haya **data leakage** entre el conjunto de entrenamiento y el conjunto de validación. El conjunto de entrenamiento debe contener datos con **`booking_date`** hasta el **2017-06-30**, y el conjunto de validación debe contener datos con **`booking_date`** posterior a esta fecha.

### **11. Eliminación de Columnas y Filas Innecesarias:**
- Eliminar las filas donde la columna **`reservation_status`** tenga el valor **'No Show'**.
- Eliminar las columnas innecesarias: **`reservation_status_date`**, **`booking_date`**, **`arrival_date`**, **`days_diff`**, **`reservation_status`**, **`country_x`** del conjunto de entrenamiento.
- Eliminar las columnas **`hotel_id`**, **`country_x`**, **`reservation_status_date`** del conjunto de validación.
- Eliminar las columnas **`stay_nights`**, **`distribution_channel_GDS`**, **`total_rooms`**, **`board_Undefined`**, **`board_HB`**, **`distribution_channel_Undefined`** del conjunto de entrenamiento.

-------------------------------------