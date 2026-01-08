import pandas as pd
import os

def calculate_impact():
    # Load the anomalies file
    input_file = r'c:\Users\Kenny\PycharmProjects\AJEColombia\output\anomaly_detection_results\anomalies_full.csv'
    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
        return

    df = pd.read_csv(input_file)

    # --- 1. Pérdida por Descuentos Indebidos ---
    # Logic: Status='Retrasado' but Discount > 0
    delayed_discount_mask = (df['estado_pago'] == 'Retrasado') & (df['descuento_pronto_pago'] > 0)
    impact_discounts = (df.loc[delayed_discount_mask, 'monto'] * df.loc[delayed_discount_mask, 'descuento_pronto_pago'] / 100).sum()

    # --- 2. Sobrecosto por Penalizaciones Indebidas ---
    # Logic: Status='Pagado' (implies on time/ok?) but Penalty > 0.
    paid_penalty_mask = (df['estado_pago'] == 'Pagado') & (df['penalizacion_mora'] > 0)
    impact_penalties = (df.loc[paid_penalty_mask, 'monto'] * df.loc[paid_penalty_mask, 'penalizacion_mora'] / 100).sum()

    # --- 3. Riesgo de Liquidez / Pasivos Ocultos ---
    # Logic: Status='Pendiente' but Date < Now (Past Due)
    pending_overdue_mask = df['cross_consistency_description'].str.contains("pero ya venció", na=False)
    impact_liquidity = df.loc[pending_overdue_mask, 'monto'].sum()

    # --- 4. Exposición a Fraude / Montos Atípicos ---
    # Outliers
    outlier_mask = df['amount_has_anomaly'] == True
    # Impact is the total amount of these risky transactions
    impact_outliers = df.loc[outlier_mask, 'monto'].sum()

    # --- 5. Duplicados (Riesgo de Doble Pago) ---
    duplicate_mask = df['duplicate_has_anomaly'] == True
    impact_duplicates_total_exposure = df.loc[duplicate_mask, 'monto'].sum()
    impact_duplicates_risk = impact_duplicates_total_exposure / 2

    print("=" * 60)
    print("REPORTE DE IMPACTO FINANCIERO ESTIMADO")
    print("=" * 60)
    print(f"1. Fuga de Capital por Desc. Indebidos (Riesgo Cumplimiento): ${impact_discounts:,.2f}")
    print(f"   (Casos: {delayed_discount_mask.sum()})")
    print(f"2. Desperdicio en Penalizaciones Injustificadas:              ${impact_penalties:,.2f}")
    print(f"   (Casos: {paid_penalty_mask.sum()})")
    print(f"3. Deuda Vencida (Riesgo Liquidez/Legal):                     ${impact_liquidity:,.2f}")
    print(f"   (Casos: {pending_overdue_mask.sum()})")
    print(f"4. Exposición a Casos Atípicos (Posible Fraude):              ${impact_outliers:,.2f}")
    print(f"   (Casos: {outlier_mask.sum()})")
    print(f"5. Riesgo de Doble Pago (Duplicados Est.):                    ${impact_duplicates_risk:,.2f}")
    print(f"   (Casos: {duplicate_mask.sum()} registros involucrados)")
    print("-" * 60)
    
    total_impact = impact_discounts + impact_penalties + impact_liquidity + impact_outliers + impact_duplicates_risk
    print(f"IMPACTO FINANCIERO TOTAL DETECTADO (ROUGH EST.):          ${total_impact:,.2f}")
    print("=" * 60)

    # Save to CSV
    summary_df = pd.DataFrame({
        'Categoria': [
            'Fuga en Descuentos Indebidos', 
            'Penalizaciones Injustificadas', 
            'Deuda Vencida (Liquidez)', 
            'Exposicion Montos Atipicos', 
            'Riesgo Doble Pago'
        ],
        'Impacto_Estimado': [
            impact_discounts, 
            impact_penalties, 
            impact_liquidity, 
            impact_outliers, 
            impact_duplicates_risk
        ],
        'Cantidad_Casos': [
            delayed_discount_mask.sum(), 
            paid_penalty_mask.sum(), 
            pending_overdue_mask.sum(), 
            outlier_mask.sum(), 
            duplicate_mask.sum()/2  # approx pairs
        ]
    })
    output_path = r'c:\Users\Kenny\PycharmProjects\AJEColombia\output\impacto_financiero_estimado.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\nReporte guardado en: {output_path}")

if __name__ == "__main__":
    calculate_impact()
