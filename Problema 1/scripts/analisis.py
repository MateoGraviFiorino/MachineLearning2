import streamlit as st
def exploratoryDataAnalysis():
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Configuración de la aplicación
    st.title("Análisis Exploratorio de Datos")
    st.write("## Introducción:")
    st.write('Insertar breve introducción...')

    # Cargar el DataFrame
    # Asegúrate de reemplazar 'Student_Performance.csv' con la ruta a tu archivo CSV
    df = pd.read_csv('Student_Performance.csv')

    # Mostrar las primeras filas del DataFrame
    st.subheader("Primeras filas del DataFrame")
    st.dataframe(df.head())
    
    # Resumen estadístico
    st.subheader("Resumen estadístico")
    st.write(df.describe())
    st.write("""
    Gracias a este resumen estadístico podemos intuir que:
        1 - Las variables parecen ser simétricas (mediana y media aproximadamente iguales).
        2 - Los datos de cada variable se encuentran en una escala distinta.
        3 - No parecen ser un problema los valores atípicos. 
    """)

    # Análisis de la variable categórica 'Extracurricular Activities'
    st.subheader("Cantidad de Actividades Extracurriculares")
    st.write("""
    Esta información es valiosa ya que podemos confirmar que el dataset se encuentra balanceado con respecto a la variable Extracurricular Activities.
    """)

    # Obtener las cantidades de "Yes" y "No"
    cantidades = df['Extracurricular Activities'].value_counts()
    extracurriculares = pd.DataFrame({"Opcion": cantidades.index, "Cantidad": cantidades.values})

    # Usar st.bar_chart para mostrar el gráfico
    st.bar_chart(data=extracurriculares.set_index('Opcion'))

    # Correlacion
    st.subheader("Correlacion entre la performance y las horas estudiadas")
    st.write("Vemos que existe una pequeña correlación entre la variable target performance y las horas que un alumno estudia.")
    correlaction = df['Hours Studied'].corr(df['Performance Index'])
    st.write(f"Si calculamos la correlación entre ambas variables, observamos que la misma es de tan solo {round(correlaction,2)}.")

    fig = px.scatter(
        df,
        x='Hours Studied',
        y='Performance Index',
        color = "Extracurricular Activities",
        color_discrete_sequence=['red', 'blue']
    )
    st.plotly_chart(fig, theme = "streamlit", use_container_width = True)

    st.subheader("Correlacion entre la performance y las notas anteriores")
    correlationPractica = df['Previous Scores'].corr(df['Performance Index'])
    st.write(f"En el caso de la relación entre las notas anteriores y las performance, existe una clara correlación directa de aproximadamente ¡{round(correlationPractica,2)}!")
    fig = px.scatter(
        df,
        x='Previous Scores',
        y='Performance Index',
        color = "Extracurricular Activities",
        color_discrete_sequence=['red', 'blue']
    )
    st.plotly_chart(fig, theme = "streamlit", use_container_width = True)

    st.subheader("Correlacion entre la performance y los ejemplos de cuestionarios hechos")
    correlation = df['Sample Question Papers Practiced'].corr(df['Performance Index'])
    st.write(f"""Caso contrario cuando comparamos los cuestionarios pasados realizados
            con la performance, donde observamos una correlación de tan solo {round(correlation,2)}""")
    fig = px.scatter(
        df,
        x='Sample Question Papers Practiced',
        y='Performance Index',
        color = "Extracurricular Activities",
        color_discrete_sequence=['red', 'blue']
    )
    st.plotly_chart(fig, theme = "streamlit", use_container_width = True)

    st.subheader("Relación entre las variables más influyentes y la performance")
    st.write(f"""
        Las variables Hours Studied y Sample Question Papers Practiced fueron las más
        relacionadas con Performance Index, con una correlación de {round(correlaction,2)} y 
        {round(correlationPractica,2)}, respectivamente.
    """)
    st.write("""
        Por ello a continuación se graficará la correlación entre las tres variables:
    """)

    fig = px.scatter_3d(
        df,
        x='Sample Question Papers Practiced',
        y = 'Hours Studied',
        z= 'Performance Index',
        color = 'Extracurricular Activities',
        color_discrete_sequence=['red','blue']
    )
    st.plotly_chart(fig, theme = "streamlit")

    ##############  BOXPLOT ##################

    st.subheader("Boxplot actividades extracurriculares")

    st.write("""
        A continuación analizaremos la distribución de cada una de las variables
        para verificar que no existan valores atípicos.
    """)

    st.write("""
        \nEn el caso de Extracurricular Activities, vemos que los datos se encuentran 
        distribuidos simétricamente.
    """)
    fig = px.box(df, x='Extracurricular Activities',
                y= 'Performance Index')
    st.plotly_chart(fig, theme = "streamlit", use_container_width=True)

    st.write("""
        En el caso de las demás variables vemos un comportameiento similar:

    """)

    fig=make_subplots(rows = 2, cols= 2)

    fig.add_trace(
        go.Box(y=df['Hours Studied'], name = "Hours Studied"), row=1, col=1
    )

    fig.add_trace(
        go.Box(y = df['Previous Scores'], name = 'Previous Scores'), row=2, col=1
    )

    fig.add_trace(
        go.Box(y = df['Sleep Hours'], name = 'Sleep Hours'), row=1, col=2
    )

    fig.add_trace(
        go.Box(y = df['Sample Question Papers Practiced'], name = 'Sample Question Papers Practiced'), row=2,col=2
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, theme = "streamlit")

    ########## HEATMAP #################

    st.subheader("Heatmap de correlacion")
    st.write("""
        Para resumir, a continuación vemos un mapa de calor que nos muestra la correlación entre las distintas variables, 
        destacando especialmente Hours Studied y Previous Scores, como se mencionó anteriormente.
    """)


    fig = px.imshow(df[['Hours Studied','Previous Scores','Sleep Hours','Sample Question Papers Practiced','Performance Index']].corr(), text_auto=True)
    st.plotly_chart(fig, theme = "streamlit", use_container_width = True)

    ######## DUPLICADOS ###########
    st.write("## Duplicados")
    st.write("Un problema comun en los datasets es que hayan algunas filas duplicadas. Esto puede generar problemas a la hora de construir un buen modelo predictivo. Verifiquemos si nuestro dataset tiene duplicados:")
    codigo = '''
    df.duplicated().sum()
    '''
    st.code(codigo, language = 'python')
    st.write("El resultado es: ",df.duplicated().sum())
    st.write("Por lo tanto, hay 127 filas duplicadas en nuestro dataset. Procederemos a eliminarlas.")

    codigo = '''
    df.drop_duplicates()
    '''
    df.drop_duplicates()
    st.code(codigo, language = 'python')
    st.write("¡Listo! Esos duplicados ya no serán un problema.")
    # Footer
    st.write("Análisis Exploratorio Completo")

def training():
    import streamlit as st
    st.write("# Esta es la pagina de entrenamiento")
    st.markdown("aslajslj")

pages_names_to_funcs = {
    "Exploratory Data Analysis": exploratoryDataAnalysis,
    "Training": training
}
demo_name = st.sidebar.selectbox(
    "Choose a page",
    pages_names_to_funcs.keys()
)
pages_names_to_funcs[demo_name]()