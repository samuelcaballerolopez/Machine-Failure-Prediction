import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#data url: https://www.kaggle.com/datasets/srinivasanusuri/optimization-of-machine-downtime

data = pd.read_csv('C:/Users/samue/OneDrive/Desktop/PROG/machine_downtime.csv', sep=',')
data.head(20)
print(data.columns)
print(data.dtypes)
print(data['Machine_ID'].unique())
print(data['Assembly_Line_No'].unique())
print(data['Downtime'].unique())
print(data['Date'].unique())
print(data.isna().sum())

#data transformation/cleaning
data['Machine_ID'] = data['Machine_ID'].map({'Makino-L1-Unit1-2013':1, 'Makino-L3-Unit1-2015':3, 'Makino-L2-Unit1-2015':2})
data['Assembly_Line_No'] = data['Assembly_Line_No'].map({'Shopfloor-L1':1, 'Shopfloor-L2':2, 'Shopfloor-L3':3})
data['Downtime'] = data['Downtime'].map({'Machine_Failure':0, 'No_Machine_Failure':1})

data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['Date'] = (data['Date'] - pd.Timestamp('2021-12-08')).dt.days
data.head()

data = data.fillna(0)


#classification
y = data['Downtime']
x = data.drop(columns=['Downtime'])


#random forest
#train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_fitted = rf_model.fit(x_train, y_train)

#prediction
y_pred = rf_model_fitted.predict(x_test)

#evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))    #98.2%
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))   #2 errors in each class
print("Classification Report:\n", classification_report(y_test, y_pred))   #high values of precision, recall and f1-score



#interface
from tkinter import Tk, Label, Entry, Button
root = Tk()
root.title('failure prediction')

Label(root, text="Fecha (dd-mm-aaaa):").grid(row=0, column=0)
entry_date = Entry(root)
entry_date.grid(row=0, column=1)

Label(root, text="ID Máquina (ingrese solo el nº. Ej.: L1=1):").grid(row=1, column=0)
entry_machine_id = Entry(root)
entry_machine_id.grid(row=1, column=1)

Label(root, text="Nº Línea de Ensamblaje (ingrese solo el nº. Ej.:L2=2):").grid(row=2, column=0)
entry_assembly_line = Entry(root)
entry_assembly_line.grid(row=2, column=1)

Label(root, text="Presión Hidráulica (bar):").grid(row=3, column=0)
entry_hydraulic_pressure = Entry(root)
entry_hydraulic_pressure.grid(row=3, column=1)

Label(root, text="Presión del Refrigerante (bar):").grid(row=4, column=0)
entry_coolant_pressure = Entry(root)
entry_coolant_pressure.grid(row=4, column=1)

Label(root, text="Presión del Sistema de Aire (bar):").grid(row=5, column=0)
entry_air_system_pressure = Entry(root)
entry_air_system_pressure.grid(row=5, column=1)

Label(root, text="Temperatura del Refrigerante:").grid(row=6, column=0)
entry_coolant_temperature = Entry(root)
entry_coolant_temperature.grid(row=6, column=1)

Label(root, text="Temperatura del Aceite Hidráulico (°C):").grid(row=7, column=0)
entry_hydraulic_oil_temperature = Entry(root)
entry_hydraulic_oil_temperature.grid(row=7, column=1)

Label(root, text="Temperatura del Rodamiento del Husillo (°C):").grid(row=8, column=0)
entry_spindle_bearing_temperature = Entry(root)
entry_spindle_bearing_temperature.grid(row=8, column=1)

Label(root, text="Vibración del Husillo (µm):").grid(row=9, column=0)
entry_spindle_vibration = Entry(root)
entry_spindle_vibration.grid(row=9, column=1)

Label(root, text="Vibración de la Herramienta (µm):").grid(row=10, column=0)
entry_tool_vibration = Entry(root)
entry_tool_vibration.grid(row=10, column=1)

Label(root, text="Velocidad del Husillo (RPM):").grid(row=11, column=0)
entry_spindle_speed = Entry(root)
entry_spindle_speed.grid(row=11, column=1)

Label(root, text="Voltaje (volts):").grid(row=12, column=0)
entry_voltage = Entry(root)
entry_voltage.grid(row=12, column=1)

Label(root, text="Torque (Nm):").grid(row=13, column=0)
entry_torque = Entry(root)
entry_torque.grid(row=13, column=1)

Label(root, text="Fuerza de Corte (kN):").grid(row=14, column=0)
entry_cutting = Entry(root)
entry_cutting.grid(row=14, column=1)

label_result = Label(root, text="")
label_result.grid(row=16, column=0, columnspan=2)


def predict_failure():
    try:
        date_input_str = entry_date.get()
        date_input = datetime.strptime(date_input_str, '%d-%m-%Y')
        reference_date = datetime.strptime('08-12-2021', '%d-%m-%Y')
        date_input_days = (date_input - reference_date).days

        new_data = pd.DataFrame({
            'Date': [date_input_days],
            'Machine_ID': [int(entry_machine_id.get())],
            'Assembly_Line_No': [int(entry_assembly_line.get())],
            'Hydraulic_Pressure(bar)': [float(entry_hydraulic_pressure.get())],
            'Coolant_Pressure(bar)': [float(entry_coolant_pressure.get())],
            'Air_System_Pressure(bar)': [float(entry_air_system_pressure.get())],
            'Coolant_Temperature': [float(entry_coolant_temperature.get())],
            'Hydraulic_Oil_Temperature(?C)': [float(entry_hydraulic_oil_temperature.get())],
            'Spindle_Bearing_Temperature(?C)': [float(entry_spindle_bearing_temperature.get())],
            'Spindle_Vibration(?m)': [float(entry_spindle_vibration.get())],
            'Tool_Vibration(?m)': [float(entry_tool_vibration.get())],
            'Spindle_Speed(RPM)': [float(entry_spindle_speed.get())],
            'Voltage(volts)': [float(entry_voltage.get())],
            'Torque(Nm)': [float(entry_torque.get())],
            'Cutting(kN)': [float(entry_cutting.get())]
        })

        new_data = new_data.astype({
            'Date': 'int64',
            'Machine_ID': 'int64',
            'Assembly_Line_No': 'int64',
            'Hydraulic_Pressure(bar)': 'float64',
            'Coolant_Pressure(bar)': 'float64',
            'Air_System_Pressure(bar)': 'float64',
            'Coolant_Temperature': 'float64',
            'Hydraulic_Oil_Temperature(?C)': 'float64',
            'Spindle_Bearing_Temperature(?C)': 'float64',
            'Spindle_Vibration(?m)': 'float64',
            'Tool_Vibration(?m)': 'float64',
            'Spindle_Speed(RPM)': 'float64',
            'Voltage(volts)': 'float64',
            'Torque(Nm)': 'float64',
            'Cutting(kN)': 'float64'
        })

        predicted_failure = rf_model_fitted.predict(new_data)
        result = 'fallo' if predicted_failure [0] == 0 else 'no fallo'
        label_result.config(text=f'Predicción de fallo: {result}')
    except Exception as e:
        label_result.config(text=f'Error: {e}')

Button(root, text='Predecir fallo', command=predict_failure).grid(row=15, column=0, columnspan=2)

root.mainloop()