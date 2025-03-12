def split_train_validation(df, train_cutoff_date, validation_output_filename):
    """
    Divide el DataFrame en conjuntos de entrenamiento y validación y guarda el DataFrame de validación en un archivo CSV.
    """
    df_validation = df[df["reservation_status"] == "Booked"].copy()
    df_train = df[df["reservation_status"] != "Booked"].copy()

    df_train = df_train[df_train['booking_date'] <= train_cutoff_date].copy()
    df_validation = df_validation[df_validation['booking_date'] > train_cutoff_date].copy()

    # Guardar df_validation en un archivo CSV
    df_validation.to_csv(validation_output_filename, index=False)
    print(f"El DataFrame df_validation se ha guardado en '{validation_output_filename}'.")

    print("Se han creado los DataFrames df_train y df_validation.")

    return df_train, df_validation