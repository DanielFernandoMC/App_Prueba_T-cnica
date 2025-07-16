import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
import plotly.graph_objects as go


# ------------------ CONFIGURACIÓN DE LA PÁGINA ------------------
st.set_page_config(page_title="Dashboard Ejecutivo - Bancolombia", layout="wide")
st.title("Dashboard Ejecutivo - Análisis de Movimientos y Saldos Bancolombia")
st.markdown("Este panel permite visualizar y analizar los movimientos y saldos de cuentas para la toma de decisiones gerenciales.")

# ------------------ CARGA Y LIMPIEZA DE DATOS ------------------
@st.cache_data
def cargar_datos():
    movimientos = pd.read_excel(r"C:\Users\danie\Desktop\Prueba Bancolombia\data.xlsx", sheet_name="movimientos")
    saldos = pd.read_excel(r"C:\Users\danie\Desktop\Prueba Bancolombia\data.xlsx", sheet_name="saldos")

    # Conversión de fecha y tipos de dato
    movimientos['fecha'] = pd.to_datetime(movimientos['fecha'], format='%Y%m%d')
    saldos['fecha'] = pd.to_datetime(saldos['fecha'], format='%Y%m%d')
    movimientos['cuenta'] = movimientos['cuenta'].astype(str)
    saldos['cuenta'] = saldos['cuenta'].astype(str)
    movimientos['comentario'] = movimientos['comentario'].fillna('')
    movimientos['debitos'] = movimientos['debitos'].fillna(0)
    movimientos['creditos'] = movimientos['creditos'].fillna(0)

    return movimientos, saldos

movimientos, saldos = cargar_datos()

# ------------------ AJUSTES Y CÁLCULOS PERSONALIZADOS ------------------
movimientos_ajustados = pd.merge(movimientos, saldos, on=['cuenta', 'fecha'], how='inner', suffixes=('', '_saldo'))
movimientos_ajustados = movimientos_ajustados[~((movimientos_ajustados['debitos'] != 0) & (movimientos_ajustados['documento'] == '98'))]

# Aplicar reglas de negocio personalizadas
movimientos_ajustados['debitos_ajustados'] = movimientos_ajustados.apply(
    lambda x: x['debitos'] * 50 if 'INVE' in x['comentario'] and x['fecha'] <= pd.Timestamp('2024-12-15') else
    (x['debitos'] / 100 if x['documento'] in ['PRO', 'TCEP', 'RU'] and x['fecha'] > pd.Timestamp('2024-12-15') else x['debitos']),
    axis=1
)
movimientos_ajustados['creditos_ajustados'] = movimientos_ajustados.apply(
    lambda x: x['creditos'] * 500 if x['cuenta'].startswith('1') else
    (x['creditos'] / 800 if x['cuenta'].startswith('8') else x['creditos']),
    axis=1
)
movimientos_ajustados['neto'] = movimientos_ajustados['creditos_ajustados'] - movimientos_ajustados['debitos_ajustados']

# Base combinada para análisis cruzado
base = pd.merge(movimientos_ajustados, saldos, on=['negocio', 'cuenta', 'tercero', 'fecha'], how='inner')
base.rename(columns={'saldo_y': 'saldo'}, inplace=True)
base['porcentaje_debito_sobre_saldo'] = (base['debitos_ajustados'] / base['saldo']) * 100
anomalías = base[base['porcentaje_debito_sobre_saldo'] > 70]

# ------------------ MÉTRICAS CLAVE ------------------
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔁 Total Movimientos", f"{len(movimientos):,}")
    col2.metric("💰 Débitos Totales", f"${movimientos['debitos'].sum():,.0f}")
    col3.metric("💵 Créditos Totales", f"${movimientos['creditos'].sum():,.0f}")
    col4.metric("📈 Saldo Total", f"${saldos['saldo'].sum():,.0f}")

# ------------------ LAYOUT PRINCIPAL POR TABS ------------------
tab1, tab2, tab3 = st.tabs(["Visión General", "Análisis Detallado", "Exploración Avanzada"])

# ---------- TAB 1: VISIÓN GENERAL ----------
with tab1:
    st.header("📆 Evolución y Ranking General")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolución del Saldo por Negocio (Top 5 cuentas)")

        # Filtro por negocio
        negocio_id = st.selectbox("Selecciona un Negocio", saldos['negocio'].unique())

        # Filtrar solo ese negocio
        df = saldos[saldos['negocio'] == negocio_id]

        # Identificar las 5 cuentas con mayor saldo acumulado
        top5_cuentas = df.groupby('cuenta')['saldo'].sum().nlargest(5).index

        #Crear tabla pivot solo para esas cuentas
        df_top5 = df[df['cuenta'].isin(top5_cuentas)]
        pivot_top5 = df_top5.pivot_table(index='fecha', columns='cuenta', values='saldo', aggfunc='sum')

        # Graficar
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot_top5.plot(ax=ax)

        ax.set_title(f"Top 5 cuentas con mayor saldo - Negocio {negocio_id}")
        ax.set_ylabel("Saldo")
        ax.set_xlabel("Fecha")
        ax.grid(True)
        ax.legend().remove()  
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Top 10 Cuentas por Movimiento Neto")
        movimientos['neto'] = movimientos['creditos'] - movimientos['debitos']
        top_cuentas = movimientos.groupby('cuenta')['neto'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_cuentas)

    st.subheader("📅 Comparativo Semanal - Débitos, Créditos y Saldo")
    mov_semanal = movimientos.copy()
    mov_semanal['semana'] = mov_semanal['fecha'].dt.to_period('W').astype(str)
    resumen_semana = mov_semanal.groupby('semana')[['debitos', 'creditos']].sum().reset_index()
    saldos_semanal = saldos.copy()
    saldos_semanal['semana'] = saldos_semanal['fecha'].dt.to_period('W').astype(str)
    resumen_saldos_semana = saldos_semanal.groupby('semana')['saldo'].sum().reset_index()
    df_semana = pd.merge(resumen_semana, resumen_saldos_semana, on='semana', how='outer').fillna(0)
    df_semana.rename(columns={
        'debitos': 'Débitos',
        'creditos': 'Créditos',
        'saldo': 'Saldo'
    }, inplace=True)
    st.line_chart(df_semana.set_index('semana'))

   
    st.markdown("## 🧭 Visión General Complementaria")

    # Saldos diarios por cuenta (Heatmap simplificado por semana)
    st.subheader("📊 Tendencia de saldos por semana (Top 10 cuentas)")
    saldos['semana'] = saldos['fecha'].dt.to_period('W').astype(str)
    top_cuentas_saldo = saldos.groupby('cuenta')['saldo'].sum().nlargest(10).index
    heatmap_df = saldos[saldos['cuenta'].isin(top_cuentas_saldo)].groupby(['semana', 'cuenta'])['saldo'].sum().unstack().fillna(0)
    st.dataframe(heatmap_df.style.background_gradient(cmap='YlGnBu', axis=0))

    # Evolución de movimientos netos vs saldo total
    st.subheader("📈 Comparativo Neto Movimientos vs Saldos")
    diario = movimientos.groupby('fecha')[['debitos', 'creditos']].sum()
    diario['neto'] = diario['creditos'] - diario['debitos']
    saldo_diario = saldos.groupby('fecha')['saldo'].sum()

    fig_comp, ax = plt.subplots(figsize=(10, 5))
    ax.plot(diario.index, diario['neto'], label='Neto Movimientos', color='steelblue')
    ax.plot(saldo_diario.index, saldo_diario, label='Saldo Total', color='orange')
    ax.set_title("Evolución de Neto Movimientos vs Saldo Total")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Monto")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig_comp)

    # Comparativo mensual
    st.subheader("🗓️ Resumen Mensual de Movimientos")
    movimientos['mes'] = movimientos['fecha'].dt.to_period('M').astype(str)
    resumen_mes = movimientos.groupby('mes')[['debitos', 'creditos']].sum()
    st.line_chart(resumen_mes)

    # Ranking de negocios por saldo acumulado
    st.subheader("🏦 Top Negocios por Saldo Total")
    top_negocios = saldos.groupby('negocio')['saldo'].sum().sort_values(ascending=False).head(10)
    fig_negocios = go.Figure(data=[go.Bar(x=top_negocios.index, y=top_negocios.values)])
    fig_negocios.update_layout(title="Top 10 Negocios por Saldo Total", xaxis_title="Negocio", yaxis_title="Saldo Total")
    st.plotly_chart(fig_negocios)

# ---------- TAB 2: ANÁLISIS DETALLADO ----------
with tab2:
    st.header("🔍 Filtros y Visualizaciones Dinámicas")

    # Filtros interactivos
    negocios = st.multiselect("Selecciona Negocio", movimientos['negocio'].unique())
    cuentas = st.multiselect("Selecciona Cuenta", movimientos['cuenta'].unique())
    fecha_inicio = st.date_input("Fecha Inicial", movimientos["fecha"].min())
    fecha_fin = st.date_input("Fecha Final", movimientos["fecha"].max())

    df_filtrado = movimientos[
        (movimientos["negocio"].isin(negocios)) &
        (movimientos["cuenta"].isin(cuentas)) &
        (movimientos["fecha"] >= pd.to_datetime(fecha_inicio)) &
        (movimientos["fecha"] <= pd.to_datetime(fecha_fin))
    ]

    if not df_filtrado.empty:
        st.subheader("📈 Evolución de Saldos Filtrados")
        for negocio in negocios:
            df = saldos[
                (saldos['negocio'] == negocio) &
                (saldos['cuenta'].isin(cuentas)) &
                (saldos['fecha'] >= pd.to_datetime(fecha_inicio)) &
                (saldos['fecha'] <= pd.to_datetime(fecha_fin))
            ]
            if df.empty:
                st.warning(f"No hay datos para el negocio {negocio} en el periodo seleccionado.")
                continue
            pivot = df.pivot_table(index='fecha', columns='cuenta', values='saldo', aggfunc='sum')
            st.line_chart(pivot)

        st.subheader("📊 Top cuentas por movimiento neto filtrado")
        df_filtrado['neto'] = df_filtrado['creditos'] - df_filtrado['debitos']
        top_cuentas_filtrado = df_filtrado.groupby('cuenta')['neto'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_cuentas_filtrado)

        st.subheader("📊 Análisis de Movimientos")
        st.dataframe(df_filtrado[['fecha', 'cuenta', 'debitos', 'creditos', 'neto']])

        # Gráfica de débitos y créditos por cuenta
        st.subheader("📊 Débitos y Créditos por Cuenta")
        debitos_por_cuenta = df_filtrado.groupby('cuenta')['debitos'].sum()
        creditos_por_cuenta = df_filtrado.groupby('cuenta')['creditos'].sum()
        import plotly.graph_objects as go
        fig_dc = go.Figure()
        fig_dc.add_trace(go.Bar(x=debitos_por_cuenta.index, y=debitos_por_cuenta.values, name='Débitos'))
        fig_dc.add_trace(go.Bar(x=creditos_por_cuenta.index, y=creditos_por_cuenta.values, name='Créditos'))
        fig_dc.update_layout(barmode='group', title="Débitos vs Créditos por Cuenta")
        st.plotly_chart(fig_dc)

        # Gráfica de saldo total por fecha
        st.subheader("📈 Evolución del Saldo Total")
        saldo_total = saldos[
            (saldos['negocio'].isin(negocios)) &
            (saldos['cuenta'].isin(cuentas)) &
            (saldos['fecha'] >= pd.to_datetime(fecha_inicio)) &
            (saldos['fecha'] <= pd.to_datetime(fecha_fin))
        ].groupby('fecha')['saldo'].sum()
        st.line_chart(saldo_total)

        st.subheader("📥 Descargar Datos Filtrados")
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button("📂 Descargar CSV", data=csv, file_name='movimientos_filtrados.csv', mime='text/csv')

        st.subheader("📈 Análisis de Saldos")
        saldos_filtrados = saldos[
            (saldos['negocio'].isin(negocios)) &
            (saldos['cuenta'].isin(cuentas)) &
            (saldos['fecha'] >= pd.to_datetime(fecha_inicio)) &
            (saldos['fecha'] <= pd.to_datetime(fecha_fin))
        ]
        st.dataframe(saldos_filtrados[['fecha', 'cuenta', 'saldo']])

        # Gráfica de distribución de saldos por cuenta
        st.subheader("📊 Distribución de Saldos por Cuenta")
        saldo_por_cuenta = saldos_filtrados.groupby('cuenta')['saldo'].sum()
        fig_saldos = go.Figure(data=[go.Pie(labels=saldo_por_cuenta.index, values=saldo_por_cuenta.values)])
        fig_saldos.update_layout(title="Distribución de Saldos por Cuenta")
        st.plotly_chart(fig_saldos)

        st.subheader("📥 Descargar Saldos")
        csv_saldos = saldos_filtrados.to_csv(index=False).encode('utf-8')
        st.download_button("📂 Descargar CSV", data=csv_saldos, file_name='saldos.csv', mime='text/csv')

        # --- Análisis de Terceros ---
        st.subheader("📈 Análisis de Terceros")
        ranking_terceros = movimientos_ajustados.groupby('tercero')['neto'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(ranking_terceros)

        # Gráfica de neto por tercero
        st.subheader("📊 Neto por Tercero (Top 10)")
        fig_tercero = go.Figure(data=[go.Bar(x=ranking_terceros.index, y=ranking_terceros.values)])
        fig_tercero.update_layout(title="Neto por Tercero (Top 10)", xaxis_title="Tercero", yaxis_title="Neto")
        st.plotly_chart(fig_tercero)

        st.subheader("📥 Descargar Ranking de Terceros")
        csv_terceros = ranking_terceros.to_csv().encode('utf-8')
        st.download_button("📂 Descargar CSV", data=csv_terceros, file_name='ranking_terceros.csv', mime='text/csv')
    else:
        st.info("Seleccione al menos un negocio y una cuenta para visualizar los datos filtrados.")
        

        # ---------- TAB 3: EXPLORACIÓN AVANZADA ----------
        with tab3:
            st.header("📊 Análisis Avanzado de Riesgos, Predicciones y Volatilidad")

            # ==================== 🔍 Anomalías (>70% débito sobre saldo) ====================
            st.subheader("🔍 Anomalías de Débito Alto")
            base['porcentaje_debito_sobre_saldo'] = (base['debitos_ajustados'] / base['saldo']) * 100
            anomalías = base[base['porcentaje_debito_sobre_saldo'] > 70]
            st.dataframe(anomalías[['negocio', 'cuenta', 'fecha', 'debitos_ajustados', 'saldo', 'porcentaje_debito_sobre_saldo']])

            # ==================== 🔥 Heatmap de Correlaciones ====================
            st.subheader("🔥 Correlación entre Variables")
            import seaborn as sns
            import plotly.figure_factory as ff

            # Renombrar columnas 
            corr_base = base.rename(columns={
                'saldo': 'Saldo',
                'debitos_ajustados': 'Débitos Ajustados',
                'creditos_ajustados': 'Créditos Ajustados',
                'porcentaje_debito_sobre_saldo': 'Porcentaje del Débito sobre el Saldo'
            })
            corr_data = corr_base[['Saldo', 'Débitos Ajustados', 'Créditos Ajustados', 'Porcentaje del Débito sobre el Saldo']].corr()

            fig_corr, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr_data, annot=True, cmap="coolwarm", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            st.pyplot(fig_corr)

            # ==================== 🤖 Clasificación de Riesgo ====================
            st.subheader("🤖 Clasificación de Riesgo de Anomalía")
            from sklearn import tree
            import graphviz
            import plotly.graph_objects as go

            base['riesgo'] = base['porcentaje_debito_sobre_saldo'] > 70
            features = ['saldo', 'debitos_ajustados', 'creditos_ajustados']
            X = base[features]
            y = base['riesgo'].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            modelo = RandomForestClassifier(n_estimators=100, random_state=42)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            reporte = classification_report(y_test, y_pred, output_dict=True)

            # Mostrar métricas como gráfico de barras
            metrics = ['precision', 'recall', 'f1-score']
            classes = ['0', '1']
            fig = go.Figure()
            for metric in metrics:
                fig.add_trace(go.Bar(
                    x=classes,
                    y=[reporte[c][metric] for c in classes],
                    name=metric
                ))
            fig.update_layout(
                barmode='group',
                title="Métricas de Clasificación por Clase",
                xaxis_title="Clase (0=No Riesgo, 1=Riesgo)",
                yaxis_title="Valor",
                legend_title="Métrica"
            )
            st.plotly_chart(fig)

            # Árbol de decisión simplificado (solo para visualización explicativa)
            clf_tree = tree.DecisionTreeClassifier(max_depth=3)
            clf_tree.fit(X_train, y_train)
            dot_data = tree.export_graphviz(clf_tree, out_file=None, 
                                            feature_names=features, class_names=['No Riesgo', 'Riesgo'], 
                                            filled=True, rounded=True, special_characters=True)
            st.graphviz_chart(dot_data)

            # ==================== 📈 Predicción de Saldos (Holt-Winters) ====================
            st.subheader("📈 Predicción de Saldo por Cuenta (Holt-Winters)")

            cuenta_sel = st.selectbox("Selecciona una cuenta para predecir", saldos['cuenta'].unique())
            negocio_sel = st.selectbox("Selecciona el negocio de la cuenta", saldos[saldos['cuenta'] == cuenta_sel]['negocio'].unique())

            df_pred = saldos[(saldos['cuenta'] == cuenta_sel) & (saldos['negocio'] == negocio_sel)]
            df_pred = df_pred.sort_values('fecha').set_index('fecha')
            serie = df_pred['saldo'].resample('W').sum()

            if len(serie) > 10:
                modelo_forecast = ExponentialSmoothing(serie, trend='add', seasonal=None)
                ajuste_forecast = modelo_forecast.fit()
                prediccion = ajuste_forecast.forecast(8)

                fig, ax = plt.subplots(figsize=(10, 4))
                serie.plot(ax=ax, label='Histórico')
                prediccion.plot(ax=ax, label='Pronóstico', linestyle='--')
                ax.set_title(f"Predicción de Saldo Semanal - Cuenta {cuenta_sel}")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("No hay suficientes datos históricos para esta cuenta.")

            # ==================== 📈 Proyección de Saldo Histórico (Forecast) ====================
            st.markdown("## 📈 Proyección de Saldo Histórico (Forecast)")
            with st.expander("🔍 Realiza una proyección de saldo semanal usando Holt-Winters"):
                cuenta_filtro = st.selectbox("Selecciona la cuenta para predecir el saldo:", saldos['cuenta'].unique(), key="forecast_cuenta")
                negocio_filtro = st.selectbox("Selecciona el negocio asociado:", saldos[saldos['cuenta'] == cuenta_filtro]['negocio'].unique(), key="forecast_negocio")

                df_pred = saldos[(saldos['cuenta'] == cuenta_filtro) & (saldos['negocio'] == negocio_filtro)]
                df_pred = df_pred.sort_values('fecha').set_index('fecha')
                serie = df_pred['saldo'].resample('W').sum()

                if len(serie) > 10:
                    modelo_forecast = ExponentialSmoothing(serie, trend='add', seasonal=None)
                    ajuste_forecast = modelo_forecast.fit()
                    prediccion = ajuste_forecast.forecast(8)

                    fig_forecast, ax = plt.subplots(figsize=(10, 4))
                    serie.plot(ax=ax, label='Histórico')
                    prediccion.plot(ax=ax, label='Pronóstico', linestyle='--')
                    ax.set_title(f"📊 Predicción de Saldo Semanal - Cuenta {cuenta_filtro}")
                    ax.set_xlabel("Fecha")
                    ax.set_ylabel("Saldo")
                    ax.legend()
                    st.pyplot(fig_forecast)
                else:
                    st.warning("⚠️ No hay suficientes datos históricos para esta cuenta para aplicar el modelo de predicción.")

            # ==================== 📊 Volatilidad de Saldos ====================
            st.subheader("📊 Top 10 Cuentas más Volátiles")
            volatilidad = saldos.groupby('cuenta')['saldo'].std().sort_values(ascending=False).head(10)
            st.bar_chart(volatilidad)

            # ==================== 😴 Cuentas Dormidas ====================
            st.subheader("😴 Cuentas Dormidas (>30 días sin movimiento)")
            cuentas_activas = movimientos.groupby('cuenta')['fecha'].max()
            hoy = saldos['fecha'].max()
            cuentas_dormidas = cuentas_activas[cuentas_activas < hoy - timedelta(days=30)]

            if not cuentas_dormidas.empty:
                st.write("Se encontraron las siguientes cuentas inactivas:")
                st.dataframe(cuentas_dormidas.reset_index().rename(columns={'fecha': 'último_movimiento'}))
            else:
                st.success("✅ No se encontraron cuentas dormidas en los últimos 30 días.")